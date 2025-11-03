module ReactionDiffusion

include("PseudoSpectral.jl")
using .PseudoSpectral
using Catalyst, Symbolics, OrdinaryDiffEqExponentialRK, OrdinaryDiffEqRosenbrock, SteadyStateDiffEq, LinearAlgebra, Combinatorics, StructArrays, Random, ProgressMeter, RecipesBase, ProgressLogging
using Makie, Printf # Plotting
# Methods and constructors to be extended:
import Random.seed!
import ModelingToolkit.ODESystem
import Catalyst.LatticeReactionSystem

export Model, species, parameters, reaction_parameters, diffusion_parameters,
    num_species, num_params, num_reaction_params, num_diffusion_params,
    domain_size, initial_conditions, noise
export simulate, filter_params, product
export @reaction_network, @transport_reaction # Re-export Catalyst DSL.
export @diffusion_system
export plot, interactive_plot

"Contains all information about the model which is independent of the parameter values and method of solution."
struct Model
    reaction
    diffusion
    initial_conditions
    initial_noise
    seed
end

function Model(reaction, diffusion; initial_conditions=Dict(), initial_noise=0.01, seed=nothing)
    seed = something(seed, rand(Int))
    Model(reaction, diffusion, Dict(initial_conditions), initial_noise, seed)
end

# Model getters
# TODO Eliminate unused getters.
species(model::Model) = Catalyst.species(model.reaction)
parameters(model::Model) = union(reaction_parameters(model), diffusion_parameters(model))

reaction_parameters(model::Model) = Catalyst.parameters(model.reaction)
diffusion_parameters(model::Model) = union(Catalyst.parameters.(model.diffusion.spatial_reactions)...)

reaction_rates(model) = Catalyst.assemble_oderhs(model.reaction, species(model))
function diffusion_rates(model::Model, default=0.0)
    dict = Dict(r.species => r.rate for r in model.diffusion.spatial_reactions)
    subst(species(model), dict, default)
end
num_species(model::Model) = numspecies(model.reaction)
num_params(model::Model) = num_reaction_params(model) + num_diffusion_params(model)
num_reaction_params(model::Model) = numparams(model.reaction)
num_diffusion_params(model::Model) = length(diffusion_parameters(model))

domain_size(model::Model) = model.diffusion.domain_size
domain_size(model::Model, params) = params[nameof(domain_size(model))] #TODO support default parameter.

is_fixed_size(model::Model) = typeof(domain_size(model)) != Num # TODO use type system. 
noise(model::Model) = model.initial_noise

reaction_parameters(model::Model, params, default=0.0) = subst(reaction_parameters(model), params, default)
#diffusion_parameters(model::Model, params, default=0.0) = get_vector(params, diffusion_parameters(model), default)

function diffusion_rates(model::Model, params::Dict{Symbol, Float64}, default=0.0) # wrong and bad
    syms = Dict(nameof(p) => p for p in parameters(model))
    params = Dict(syms[k] => v for (k,v) in params)
    [Symbolics.value(substitute(D, params)) for D in diffusion_rates(model,default)]
end

initial_conditions(model::Model, default=0.0) = subst(species(model), model.initial_conditions, default)

ModelingToolkit.ODESystem(model::Model) = convert(ODESystem, model.reaction)

Random.seed!(model::Model) = Random.seed!(model.seed)

"Return a vector of dictionaries containing the cartesian product of the given values."
product(;kwargs...) = vec([Dict(zip(keys(kwargs), vals)) for vals in Iterators.product(values(kwargs)...)])
product(dict::Dict) = product(;dict...)
"Replace each element of keys with either the corresponding value in dict or default."
function subst(keys, dict, default)
    v = Vector(undef,0)
    for k in keys
        val = get(dict, k, default)
        push!(v, val)
    end
    v
end


struct DiffusionSystem
    domain_size
    spatial_reactions
end


"""
    @diffusion_system L begin D, S;... end
Define a spatial domain of length and a set of diffusion rates. Values can be either fixed numbers or parameter symbols.
- `L`: Length of the domain.
- `D`: Diffusion rate.
- `S`: Species name.
"""
macro diffusion_system(L, body)
    DiffusionSystem(L,body)
end

macro diffusion_system(body)
    DiffusionSystem(1,body)
end

function DiffusionSystem(L, body::Expr)
    Base.remove_linenums!(body)
    trs_expr = Expr(:vect, (:(@transport_reaction $D/$L^2 $S) for (D,S) in getproperty.(body.args,:args))...)
    ds_expr = Expr(:call, DiffusionSystem, L, trs_expr)
    typeof(L) == Symbol ? Expr(:block, :(@parameters $L), ds_expr) : ds_expr
end


const nq = 100
const q2 = 10 .^(range(-2,stop=2,length=nq))
const n_denser = (range(0,stop=1,length=100))
const n_gridpoints = 128

        
function returnParam(type, ps,ics, i)
    n_params = length(ps)
    n_species = length(ics)
    p = zeros(type,n_params)
    u₀ = zeros(type,n_species)
    ind = CartesianIndices(Tuple([length.(ics); length.(ps)]))[Int(i)]
    for j in 1:n_species
        u₀[j] = ics[j][ind[j]]
    end
    for j in 1:n_params
        p[j] = ps[j][ind[j+n_species]]
    end
    return p,u₀
end

function returnD(ds, i)
    n_species = length(ds)
    D = zeros(Float64,n_species)
    ind = CartesianIndices(Tuple(length.(ds)))[Int(i)]
    for j in 1:n_species
        D[j] = ds[j][ind[j]]
    end
    return D
end

function computeStability(J)
    # compute general condition for Turing instability, from: https://doi.org/10.1103/PhysRevX.8.021071
    n_species = size(J)[1]
    if maximum(real(eigvals(J))) > 0
        return 2  ## steady state is unstable
    else
        for subset in combinations(1:n_species)
            if ((-1)^length(subset)*det(J[subset,subset])) < 0
                return 1 ## J is not a P_0 matrix
            end
        end
    end
    return 0 ## J is a P_0 matrix
end
    

function isTuring(J,D,q2_input)

    # determine whether any eigenvalue has a real positive part across range of wavevectors (q2_input)
    #       - if yes, then store relevant variables

    max_eig = zeros(length(q2_input))

    Threads.@threads for i in eachindex(q2_input)
        max_eig[i] = maximum(real(eigvals(J - diagm(q2_input[i]*D))))
    end

    index_max = findmax(max_eig)[2]

    if findmax(max_eig)[1] > 0 & index_max < nq 
        if max_eig[length(q2_input)] < 0
            real_max = max_eig[index_max]
            q2max = sqrt(q2_input[index_max])
            qmax = sqrt(q2max)
            eigenVals = eigen(J - diagm(q2_input[index_max]*D))
            phase = sign.(eigenVals.vectors[:,findmax(real(eigenVals.values))[2]])
            if real(phase[1]) < 0
                phase = phase * -1
            end
            M = eigvals(J - diagm(q2max*D))
            non_oscillatory = Bool(imag(M[findmax(real(M))[2]]) == 0)
            
            return qmax, phase, real_max, non_oscillatory
        else
            return 0,0,0,0
        end
    else
        return 0,0,0,0
    end
end



function identifyTuring(sol, ds, jacobian)

    # Returns turingParameters that satisfy diffusion-driven instability for each steady state in sol, and diffusion constants in ds. 
    #       - idx_ps refers to the parameter combinations (steady states [sol] are computed across parameter combinations, and do not depend on ds)
    #       - idx_ds refers to the diffusion constant combinations
    #       - idx_turing refers to the  combinations


    idx_turing = 1
    turingParameters = Array{save_turing, 1}(undef, 0)
    for solᵢ in sol
        if SciMLBase.successful_retcode(solᵢ) && minimum(solᵢ) >= 0
            J = jacobian(Array(solᵢ),solᵢ.prob.p,0.0)
            if computeStability(J) == 1
                for idx_ds in 1:prod(length.(ds))
                    qmax, phase, real_max, non_oscillatory = isTuring(J,returnD(ds,idx_ds),q2)
                    if qmax > 0
                        push!(turingParameters,save_turing(solᵢ,solᵢ.prob.p,returnD(ds,idx_ds),solᵢ.prob.u0,phase,2*pi/qmax, real_max, non_oscillatory,idx_turing))
                        idx_turing = idx_turing + 1
                    end
                end
            end
        end
    end
    return StructArray(turingParameters)
end

"""
    get_params(model, turing_param)

For a given `model` and a *single* pattern-forming parameter set, `turing_param`, this function creates a corresponding dictionary of parameter values.
"""
function get_params(model, turing_param)
    length(turing_param.wavelength) > 1 && error("Please input only a single parameter set, not multiple (e.g., turing_params[1] instead of turing_params)")
    rs = Dict(zip(nameof.(reaction_parameters(model)), turing_param.reaction_params))
    ds = Dict(zip(nameof.(diffusion_parameters(model)), turing_param.diffusion_constants))
    merge(rs, ds)
end


"""
    get_param(model, turing_params, name, type)

For a given `model` and a (potentially large) number of pattern-forming parameter sets, `turing_params`, this function extracts the parameter values prescribed by the input `name`. For reaction parameters, used `type="reaction"`, for diffusion constants, use `type="diffusion"`.

Example:

```julia
δ₁ = get_param(model, turing_params,"δ₁","reaction")
D_COMPLEX = get_param(model, turing_params,"COMPLEX","diffusion")
```
"""
function get_param(model, turing_params, name, type)
    output = Array{Float64,1}(undef,0)
    if type == "reaction"
        labels = []
        for p in parameters(model)
            push!(labels,nameof(p))
        end
        index = findall(labels .== name)
        if length(index) == 0
            error("Please be sure to enter the correct parameter name and/or parameter type (reaction or diffusion); it should match the original definition in model")
        end
        for i in eachindex(turing_params)
            push!(output,turing_params[i].reaction_params[index][1])
        end
    elseif type == "diffusion"
        labels = []
        for state in species(model)
            push!(labels,nameof(state))
        end
        index = findall(labels .== name)
        if length(index) == 0
            error("Please be sure to enter the correct parameter name and/or parameter type (reaction or diffusion); it should match the original definition in model")
        end
        for i in eachindex(turing_params)
            push!(output,turing_params[i].diffusion_constants[index][1])
        end
    end
    return output
end

"""
    returnTuringParams(model, params; maxiters = 1e3,alg=Rodas5(),abstol=1e-8, reltol=1e-6, tspan=1e4,ensemblealg=EnsembleThreads(),batch_size=1e4)

Return a `save_turing` object of parameters that are predicted to be pattern forming.

Required inputs:
- `model`: specified via the `@reaction_network` macro
- `params`: all reaction and diffusion parameters, in a Dict or collection of pairs.

Optional inputs:
- `batch_size`: the number of parameter sets to consider at once. Increasing/decreasing from the default value may improve speed.  

Inputs carried over from DifferentialEquations.jl; see [here](https://docs.sciml.ai/DiffEqDocs/stable/) for further details:
- `maxiters`: maximum number of iterations to reach steady state (otherwise simulation terminates)
- `alg`: ODE solver algorithm
- `abstol` and `reltol`: tolerance levels of solvers
- `tspan`: maximum time allowed to reach steady state (otherwise simulation terminates)
- `ensemblealg`: ensemble simulation method

"""
function returnTuringParams(model, params; maxiters = 1e3,alg=Rodas5(),abstol=1e-8, reltol=1e-6, tspan=1e4,ensemblealg=EnsembleThreads(),batch_size=1e4)
    params = Dict.(params)
    # read in parameters (ps), diffusion rates (ds), and initial conditions (ics)
    ps = [reaction_parameters(model,ps, 0.0) for ps in params]
    ds = [diffusion_rates(model, ps, 1.0) for ps in params]
    ics = initial_conditions(model, 1.0)
    n_params = num_reaction_params(model)
    n_species = num_species(model)
  

    # convert reaction network to ODESystem
    odesys = ODESystem(model)

    # build jacobian function
    jac = ModelingToolkit.generate_jacobian(odesys,expression = Val{false})[1] #false denotes function is compiled, world issues fixed
    J = ModelingToolkit.eval(jac)
    jacobian(u,p,t) = J(u,p,t)
    
    # build ODE function
    f_gen = ModelingToolkit.generate_function(odesys,expression = Val{false})[1] #false denotes function is compiled, world issues fixed
    f_oop = ModelingToolkit.eval(f_gen)
    f_ode(u,p,t) = f_oop(u,p,t)
    
    # Build optimized Jacobian and ODE functions using Symbolics.jl

    @variables uₛₛ[1:n_species]
    @parameters pₛₛ[1:n_params]
    
    duₛₛ = Symbolics.simplify.(f_ode(collect(uₛₛ),collect(pₛₛ),0.0))
    
    fₛₛ = eval(Symbolics.build_function(duₛₛ,vec(uₛₛ),pₛₛ;
                parallel=Symbolics.SerialForm(),expression = Val{false})[2]) #index [2] denotes in-place, mutating function
    jacₛₛ = Symbolics.sparsejacobian(vec(duₛₛ),vec(uₛₛ))
    fjacₛₛ = eval(Symbolics.build_function(jacₛₛ,vec(uₛₛ),pₛₛ,
                parallel=Symbolics.SerialForm(),expression = Val{false})[2]) #index [2] denotes in-place, mutating function


    # separate parameter screen into batches
    turing_params = Array{save_turing, 1}(undef, 0)
    n_total = length(ps)
    n_batches = Int(ceil(n_total/batch_size))
    progressMeter = Progress(n_batches; desc="Screening parameter sets: ",dt=0.1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)
 
    # Define the steady state problem
    prob_fn = ODEFunction((du,u,p,t)->fₛₛ(du,vec(u),p), jac = (du,u,p,t) -> fjacₛₛ(du,vec(u),p), jac_prototype = similar(jacₛₛ,Float64))

    function ss_condition(u,t,integrator)
            du = similar(u)
            fₛₛ(du,vec(u),integrator.p)
            return norm(du) < 1e-6   # steady state tolerance
    end
    
    if ensemblealg isa EnsembleThreads
        p = zeros(Float64,length(ps))
        u₀ = zeros(Float64,length(ics))
        prob = SteadyStateProblem(prob_fn,u₀,p)
        callback = nothing
        alg = DynamicSS(alg; tspan=tspan)
    else
        p = zeros(Float32,length(ps))
        u₀ = zeros(Float32,length(ics))
        tspan = Float32.(tspan)
        prob = ODEProblem(prob_fn,u₀,tspan, p)
        callback = DiscreteCallback(ss_condition, int -> terminate!(int))
    end

    for batch_number in 1:n_batches
        starting_index = (batch_number - 1)*batch_size
        final_index = min(batch_number*batch_size,n_total)
        n_batch = final_index - starting_index
        append!(turing_params,returnTuringParams_batch_single(n_batch, starting_index, ps, ds, ics, prob, jacobian, callback; maxiters = maxiters,alg=alg,abstol=abstol, reltol=reltol, tspan=tspan,ensemblealg=ensemblealg))
        next!(progressMeter)
    end
    println(string(length(turing_params),"/",prod([length.(ps); length.(ics); length.(ds)])," parameters are pattern forming"))

    return StructArray(turing_params)
end

function returnTuringParams_batch_single(n_batch, starting_index, ps, ds, ics, prob, jacobian, callback; maxiters = maxiters,alg=alg,abstol=abstol, reltol=reltol, tspan=tspan,ensemblealg=ensemblealg)

    # Construct function to screen through parameters
    function prob_func(prob,i,repeat)
      tmp1, tmp2 = returnParam(typeof(prob.u0[1]), ps,ics,(i+starting_index))
      remake(prob; u0=tmp2, p=tmp1)
    end
    
    ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)

    sol = solve(ensemble_prob,maxiters=maxiters,alg,ensemblealg,callback=callback, trajectories=n_batch,verbose=false,abstol=abstol, reltol=reltol, save_everystep = false, progress=true)

    # Determine whether the steady state undergoes a diffusion-driven instability
    return identifyTuring(sol, ds, jacobian)

end


function createIC(model, n)
    seed!(model)
    m = num_species(model)
    σ = noise(model)
    abs.(σ * randn(n, m) .+ initial_conditions(model)')
end

"""
    simulate(model,params; tspan=Inf, discretisation=:pseudospectral, alg=nothing, dt=0.01, dx=domain_size(model)/128, reltol=1e-6,abstol=1e-8, maxiters = 1e5)

Simulate `model` for a single parameter set `param`.

Required inputs:
- `model`: `Model`` object containg the system to be simulated.
- `param`: all reaction and diffusion parameters, in a Dict or collection of pairs. 

Inputs carried over from DifferentialEquations.jl; see [here](https://docs.sciml.ai/DiffEqDocs/stable/) for further details:
- `maxiters`: maximum number of iterations to reach steady state (otherwise simulation terminates)
- `alg`: solver algorithm
- `abstol` and `reltol`: tolerance levels of solvers
- `dt`: value for timestep

Additional Inputs
- `dx`: distance between points in spatial discretisation.
- `maxrepeats`: Number of times to halve dt and retry if the solver scheme proves unstable.
"""
function simulate(model, params; output_func=nothing, full_solution=false, tspan=Inf, alg=nothing, dt=0.1, num_verts=128, reltol=1e-6,abstol=1e-8, maxiters = 1e6, maxrepeats = 4)
    n = num_verts
    
    # Ensure params is a vector.
    if typeof(params) <: Vector
        single=false
    else
        params = [params]
        single = true # Unpack vector at the end if we only have one parameter set.
    end

    # Replace parameter names with actual Symbolics variables.
    ps = [Dict((@parameters $k)[1] => v for (k,v) in p) for p in params]

    u0 = createIC(model, n)
    steadystate = DiscreteCallback((u,t,integrator) -> isapprox(get_du(integrator), zero(u); rtol=reltol, atol=abstol), terminate!)
    make_prob, transform = pseudospectral_problem(species(model), reaction_rates(model), diffusion_rates(model), u0, tspan; callback=steadystate, maxiters=maxiters, dt=dt, abstol=abstol, reltol=reltol)
   
    progress = Progress(length(ps); desc="Simulating parameter sets: ",dt=0.1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    function output_func_(sol,i)
        repeat = sol.prob.p.state
        SciMLBase.successful_retcode(sol) || return ((missing, missing), repeat <= maxrepeats) # Rerun if solution failed.
        if full_solution
            t=sol.t
            u = stack(transform(u) for u in sol.u)
        else
            t = sol.t[end]
            u = transform(sol.u[end])
        end
        out = isnothing(output_func) ? (u,t) : output_func(u,t)
        (out, false)
    end
        
    function prob_func(prob, i, repeat)
        update!(progress, i)
        p = ps[i]
        dt = dt/2^(repeat-1) # halve dt if solve was unsuccessful.
        prob = make_prob(p, repeat; dt=dt)
    end

    ensemble_prob = EnsembleProblem(make_prob(ps[1]); output_func=output_func_, prob_func=prob_func)

    alg = something(alg, ETDRK4())
    sol = solve(ensemble_prob, alg; trajectories=length(params), progress=true)
    single ? sol[1] : sol
end

"""
    filter_params(f,model,params; kwargs...)

Return subset of `params` which satisfy the predicate `f(u)`.

Required inputs:
- `f(u,t)`: Takes a matrix of solution values and time points and returns true or false. Input is either a single steady state value, or with `full_solution=true`, vectors with values for each time point.
- `model`: `Model` object containg the system to be simulated.
- `params`: Parameter sets to be filtered. This should be a vector of dictionaries each containing a single parameter set.

- For `kwargs` see `simulate`.

Example:
Return parameters which produce a steady state less than 0.5.
```
filter_params(model,params) do u
    maximum(u) < 0.5
end
```
"""
function filter_params(f, model, params; kwargs...)
    sol = simulate(model,params; output_func=f, kwargs...)
    params[sol.u]
end


## Plotting functions

function plot(model, params; normalise=true, hide_y=true, autolimits=true, kwargs...)
    params = Dict(params)
    L = domain_size(model, params)
    labels = [string(s.f) for s in species(model)]
    u,t=simulate(model,params; full_solution=true, kwargs...)
    x_steps = size(u, 1)
    x = range(0,L,length=x_steps)
	r = normalise ? norm.(eachslice(u, dims=(2,3))) : ones(size(u)[2:3])
	fig=Figure()
	ax = Axis(fig[1,1])
	hide_y && hideydecorations!(ax)
    sg = SliderGrid(fig[2,1], (label="t",range=eachindex(t), format=i->@sprintf("%.2f",t[i])))
    sl=sg.sliders[1]
	T = lift(i -> t[i], sl.value)
	U = [lift(i -> u[:,i]/r[i], sl.value) for (u,r) in zip(eachslice(u, dims=2), eachrow(r))]
	for (U,label) in zip(U,labels)
		lines!(ax,x,U, label=label)
	end
	autolimits && on(sl.value) do _
	    autolimits!(ax)
	end
	axislegend(ax)
    display(fig)
end

# TODO Refactor so this shares code with simulate and plot.
function interactive_plot(model, param_ranges; tspan=Inf, alg=nothing, dt=0.1, num_verts=128, reltol=1e-6,abstol=1e-8, maxiters = 1e6, hide_y=true)
    n = num_verts
    alg = something(alg, ETDRK4())




    u0 = createIC(model, n)
    steadystate = DiscreteCallback((u,t,integrator) -> isapprox(get_du(integrator), zero(u); rtol=reltol, atol=abstol), terminate!)
    make_prob, transform = pseudospectral_problem(species(model), reaction_rates(model), diffusion_rates(model), u0, tspan; callback=steadystate, maxiters=maxiters, dt=dt, abstol=abstol, reltol=reltol)
   
	fig=Figure()
	ax = Axis(fig[1,1])
	hide_y && hideydecorations!(ax)

    # Replace parameter names with actual Symbolics variables.
    param_ranges = Dict((@parameters $k)[1] => v for (k,v) in param_ranges)
    param_ranges = PseudoSpectral.sort_params(param_ranges) ## Interface!
    slider_specs = [(label=string(k), range = v isa AbstractRange ? v : 1:length(v)) for (k,v) in param_ranges]

    sg = SliderGrid(fig[1,2], slider_specs...)
    function f(vals...)
        p = Dict(k => x isa Int ? v[x] : x for ((k,v), x) in zip(param_ranges,vals))
        prob = make_prob(p)
        sol =  solve(prob, alg)
        transform(sol.u[end])
    end
    U = lift(f, (sl.value for sl in sg.sliders)...)
    x = range(0,1,n)

    for i in eachindex(eachcol(U[]))
        lines!(ax, x, lift(u -> u[:,i], U))
    end

    display(fig)
end

end
