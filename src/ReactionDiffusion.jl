module ReactionDiffusion

include("PseudoSpectral.jl")
using .PseudoSpectral
using Catalyst, Symbolics, DifferentialEquations, LinearAlgebra, Combinatorics, StructArrays, Random, ProgressMeter, RecipesBase


export returnTuringParams, model_parameters, screen_values, get_params, get_param, simulate
export @reaction_network, @transport_reaction, LatticeReactionSystem, CartesianGrid # Re-export Catalyst interface.
export @transport_reactions # Extend Catalyst interface.
export endpoint, timepoint

mutable struct save_turing
    steady_state_values::Vector{Float64}
    reaction_params::Vector{Float64}
    diffusion_constants::Vector{Float64}
    initial_conditions::Vector{Float64}
    pattern_phase::Vector{Int64}
    wavelength::Float64
    max_real_eigval::Float64
    non_oscillatory::Bool
    idx_turing::Int64
end


struct Model
    reaction
    diffusion
    domain_size
    initial_noise
    seed
end

function Model(reaction, diffusion; domain_size=1.0, initial_noise=0.01, seed=nothing)
    seed = something(seed, rand(Int))
    Model(reaction, diffusion, domain_size, initial_noise, seed)
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

For a given `model` and a *single* pattern-forming parameter set, `turing_param`, this function creates a  corresponding `model_parameters` variable. This sets, by default, the `model_parameters` fields:
- `domain_size`: chosen to be 3x the computed pattern wavelength, i.e., `3*turing_param.wavelength`
- `initial_condition`: chosen to be the computed steady state values, i.e.,  `turing_param.steady_state_values`
- `initial_noise = 0.01`: the magnitude of noise (normally distributed random numbers) added to the steady state values to define the initial conditions.
"""
function get_params(model, turing_param)
    if length(turing_param.wavelength) > 1
        error("Please input only a single parameter set, not multiple (e.g., turing_params[1] instead of turing_params)")
    else
        param = model_parameters()
        param.reaction = Dict(zip(nameof.(vertex_parameters(model)), turing_param.reaction_params))
        param.diffusion = Dict(zip(nameof.(edge_parameters(model)), turing_param.diffusion_constants))
        param.domain_size = 3*turing_param.wavelength # default value
        param.initial_condition = Dict(zip([nameof(s.f) for s in species(model)], turing_param.steady_state_values))
        param.initial_noise = 0.01
        param.random_seed = 0
        return param
    end
end

function returnParameterSets(model, params)
    # read in parameters (ps), diffusion constants (ds), initial conditions (ics), domain sizes (ls), random seeds (seeds) and random noise (noise)
    ps = Vector{Vector{Float64}}(undef,0)
    for p in vertex_parameters(model) 
        val = get(params.reaction, nameof(p), [0]) # default is 0
        push!(ps,val)
    end

    ds = Vector{Vector{Float64}}(undef,0)
    for p in edge_parameters(model)
        val = get(params.diffusion, nameof(p), [1]) # default is 1
        push!(ds, val)
    end

    ics = Vector{Vector{Float64}}(undef,0)
    for s in species(model)
        val = get(params.initial_condition, nameof(s.f), [1]) # default is 1
        push!(ics, val)
    end
    seeds = Int64.(params.random_seed)
    noise = params.initial_noise
    ls = params.domain_size
    return ps,ds,ics, ls, seeds, noise
end

function returnSingleParameter(model, params)
    # read in parameters (ps), diffusion constants (ds), initial conditions (ics), domain sizes (ls), random seeds (seeds) and random noise (noise)
    ps = Vector(undef,0)
    for p in vertex_parameters(model) 
        val = get(params.reaction, nameof(p), 0) # default is 0
        push!(ps,val)
    end

    ds = Vector(undef,0)
    for p in edge_parameters(model)
        val = get(params.diffusion, nameof(p), 1) # default is 1
        push!(ds, val)
    end

    ics = Vector(undef,0)
    for s in species(model)
        val = get(params.initial_condition, nameof(s.f), 1) # default is 1
        push!(ics, val)
    end
 
    seeds = Int64(params.random_seed)
    noise = params.initial_noise
    ls = params.domain_size
    return ps,ds,ics, ls, seeds, noise
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
            push!(labels,chop(nameof(state.f), head=0,tail=3))
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
- `params`: all reaction and diffusion parameters, in a `model_parameters` object

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

    # read in parameters (ps), diffusion constants (ds), and initial conditions (ics)
    ps,ds,ics, _, _, _ = returnParameterSets(model, params)
    n_params = length(parameters(model))
    n_species = length(unknowns(model))

    # convert reaction network to ODESystem
    odesys = convert(ODESystem, Catalyst.reactionsystem(model))

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
    n_total = prod([length.(ps); length.(ics)])
    n_batches = Int(ceil(n_total/batch_size))
    progressMeter = Progress(n_batches; desc="Screening parameter sets: ",dt=0.1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)
 
    # Define the steady state problem

    p = zeros(Float64,length(ps))
    u₀ = zeros(Float64,length(ics))

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

    sol = solve(ensemble_prob,maxiters=maxiters,alg,ensemblealg,callback=callback, trajectories=n_batch,verbose=false,abstol=abstol, reltol=reltol, save_everystep = false)

    # Determine whether the steady state undergoes a diffusion-driven instability
    return identifyTuring(sol, ds, jacobian)

end



function createIC(model, n)
    Random.seed!(seed(model))
    m = numspecies(model)
    abs.(noise*randn(n, m) + 1)
end

"""
    simulate(model,param; discretisation=PseudoSpectralProblem, alg=nothing, reltol=1e-6,abstol=1e-8, dt = 0.1, maxiters = 1e3, save_everystep = true)

Simulate `model` for a single parameter set `param`.

Required inputs:
- `model`: specified via the `@reaction_network` macro
- `param`: all reaction and diffusion parameters, in a `model_parameters` object. *This must be a single parameter set only* 

Inputs carried over from DifferentialEquations.jl; see [here](https://docs.sciml.ai/DiffEqDocs/stable/) for further details:
- `maxiters`: maximum number of iterations to reach steady state (otherwise simulation terminates)
- `alg`: solver algorithm
- `abstol` and `reltol`: tolerance levels of solvers
- `dt`: initial value for timestep
- `save_everystep`: controls whether all timepoints are saved, defaults to `true`
 
"""
function simulate(model,params; tspan=Inf, discretisation=:pseudospectral, alg=nothing, dt=0.01, dx=0.01, reltol=1e-6,abstol=1e-8, maxiters = 1e5)
    n = Int(model.domain_size ÷ dx)
    lattice = CartesianGrid(n)
    lrs = LatticeReactionSystem(model.reaction, model.diffusion, lattice)
    u0 = createIC(model, n)
    if discretisation == :pseudospectral
        alg = something(alg, ETDRK4())
        prob, transform = pseudospectral_problem(lrs, u0, tspan, params, domain_size, dt)
        #prob = SteadyStateProblem(prob)
        #sol = solve(prob, DynamicSS(alg); maxiters=maxiters)
        steadystate = DiscreteCallback((u,t,integrator) -> maximum(abs.(u-integrator.uprev)) <= 1e-6, terminate!)
        sol = solve(prob, alg; callback=steadystate, maxiters=maxiters)
        u = stack(transform(u) for u in sol.u)
        t = sol.t
    elseif discretisation == :finitedifference
        alg = something(alg, FBDF())
        sps = species(model)
        U0 = Dict(zip(sps, eachcol(u0)))
        n = Catalyst.num_verts(model)
        d = Dict(s => d*n*2pi/domain_size for (s,d) in params.diffusion)
        p = merge(params.reaction, d)
        prob = ODEProblem(model,U0,tspan,p;reltol=reltol,abstol=abstol)
        prob = SteadyStateProblem(prob)
        sol = solve(prob, DynamicSS(alg); maxiters=maxiters)
        # Reshape solution into matrix of dimensions: n_verts x n_species x n_time_steps.
        u = reshape(stack(stack(lat_getu(sol.original,nameof(s.f),model)) for s in sps), n, length(sps), length(sol.original))
        t = sol.original.t
    end
    (u,t)
end



macro transport_reactions(ex)
    dx = gensym(dx)
    Expr(:block,
        :(@parameters $dx),
        (:(@transport_reaction $p/$dx $s) for (p,s) in ex.args)...)
end


        



struct endpoint end


@recipe function plot(::endpoint, model, u)
    pattern = u[:,:,end]
    x = range(0,1, length = size(pattern,1))
    labels = []
    for state in unknowns(model)
        push!(labels,chop(string(state), head=0,tail=3))
    end

    pattern = pattern ./ maximum(pattern,dims=1)

    labels --> reshape(labels,1,length(labels))
    ticks --> :none
    leg --> Symbol(:outer,:right)
    linewidth --> 2
    x, pattern
end



struct timepoint end

# Not yet working
@recipe function plot(::timepoint, model, u, t)
    if t > 1 || t < 0
        error("Time should be between 0 and 1, representing first and last simulation timepoints respectively)")
    end

    if t == 0
        t = 1e-20
    end

    x = range(0,1, length = size(u,1))    
    labels = []
    for state in unknowns(model)
        push!(labels,chop(string(state), head=0,tail=3))
    end
    labels = reshape(labels,1,length(labels))
    i = Int(ceil(t*size(u,3)))
    normalization_factor = reshape(maximum(u[:,:,1:i],dims=(1,3)), 1, :)

    pattern = u[:,:,i]

    pattern = pattern ./ normalization_factor

    labels --> reshape(labels,1,length(labels))
    ticks --> :none
    leg --> Symbol(:outer,:right)
    linewidth --> 2

    x,pattern

end


end