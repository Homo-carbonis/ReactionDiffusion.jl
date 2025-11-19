module ReactionDiffusion

include("PseudoSpectral.jl")
using .PseudoSpectral
using Catalyst, Symbolics, OrdinaryDiffEqExponentialRK, OrdinaryDiffEqRosenbrock, SteadyStateDiffEq, LinearAlgebra, Combinatorics, StructArrays, Random, ProgressMeter, RecipesBase, ProgressLogging
using Makie, Observables, Printf # Plotting
# Methods and constructors to be extended:
import Random.seed!
import ModelingToolkit.ODESystem
import Catalyst.LatticeReactionSystem

export Model, species, parameters, reaction_parameters, diffusion_parameters,
    num_species, num_params, num_reaction_params, num_diffusion_params,
    domain_size, initial_conditions, noise
export simulate, filter_params, product, dict
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
    Symbolics.Num.(subst(species(model), dict, default))
end
num_species(model::Model) = numspecies(model.reaction)
num_params(model::Model) = num_reaction_params(model) + num_diffusion_params(model)
num_reaction_params(model::Model) = numparams(model.reaction)
num_diffusion_params(model::Model) = length(diffusion_parameters(model))

domain_size(model::Model) = model.diffusion.domain_size
function domain_size(model::Model, params)
    L = domain_size(model)
    L isa Symbolics.Num ? params[nameof(L)] : L
end

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

"Convenience function to construct a dict using (k=v, ...) syntax"
dict(;kwargs...) = Dict(kwargs)


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
    L isa Symbol ? Expr(:block, :(@parameters $L), ds_expr) : ds_expr
end

function createIC(model, n)
    seed!(model)
    m = num_species(model)
    σ = noise(model)
    abs.(σ * randn(n, m) .+ initial_conditions(model)')
end

"""
    simulate(model,params; tspan=Inf, discretisation=:pseudospectral, alg=ETDRK4(), dt=0.01, dx=domain_size(model)/128, reltol=1e-6,abstol=1e-8, maxiters = 1e5)

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
function simulate(model, params; output_func=nothing, full_solution=false, tspan=Inf, alg=ETDRK4(), dt=0.1, num_verts=64, reltol=1e-4, abstol=1e-4, maxiters = 1e6, maxrepeats = 4, kwargs...)
    n = num_verts
    
    # Ensure params is a vector.
    if params isa Vector
        single=false
    else
        params = [params]
        single = true # Unpack vector at the end if we only have one parameter set.
    end

    # Replace parameter names with actual Symbolics variables.
    ps = lookup_params(params)

    u0 = createIC(model, n)
    make_prob, transform = pseudospectral_problem(species(model), reaction_rates(model), diffusion_rates(model), u0, tspan; callback=steady_state_callback(reltol,abstol), maxiters=maxiters, dt=dt)
   
    progress = Progress(length(ps); desc="Simulating parameter sets: ",dt=0.1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    function output_func_(sol,i)
        repeat = sol.prob.p.state
        SciMLBase.successful_retcode(sol) || return (missing, repeat <= maxrepeats) # Rerun if solution failed.
        next!(progress) # Advance progress bar.
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
        p = ps[i]
        dt = dt/2^(repeat-1) # halve dt if solve was unsuccessful.
        prob = make_prob(p, repeat; dt=dt)
    end

    ensemble_prob = EnsembleProblem(make_prob(ps[1]); output_func=output_func_, prob_func=prob_func)

    sol = solve(ensemble_prob, alg; trajectories=length(params), progress=true, kwargs...)
    single ? sol[1] : sol
end

function steady_state_callback(reltol=1e-4,abstol=1e-4)
    condition(u,t,integrator) = isapprox(get_du(integrator), zero(u); rtol=reltol, atol=abstol)
    DiscreteCallback(condition, terminate!)
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
    sol = simulate(model,params; output_func=f, verbose=false, kwargs...)
    pass = [ismissing(u) ? false : u for u in sol.u]
    params[pass]
end


function turing_wavenumbers(model, params; k=logrange(0.1,10,100), tspan=1e4, alg=Rodas5(), kwargs...)
    u0 = createIC(model,1)
    ps = lookup_params(params)

    du = reaction_rates(model)
    u = species(model)
    p = parameters(model)
    t = ()
    (f,f!) = Symbolics.build_function(du, u, p, t; expression=Val{false})
    jac = Symbolics.jacobian(du,u; simplify=true)
    (fjac,fjac!) = Symbolics.build_function(jac, u, p, t; expression=Val{false})

    R = ODEFunction(f!; jac=fjac!)
    prob = SteadyStateProblem(R, u0, [ps[1][k] for k in p])

    d = diffusion_rates(model)
    (D,D!) = Symbolics.build_function(diagm(d), p; expression=Val{false})

    k² = k.^2
    function output_func(sol,i)
        SciMLBase.successful_retcode(sol) || begin @show sol.original.errors; return (missing, false) end
        local p = sol.prob.p
        J = fjac(sol.u, p, 0.0)
        real_max,i = findmax(real(eigvals(J - D(p) * k²)[end]) for k² in k²)
        λ = real_max > 0.0 ? 2pi/k[i] : 0.0
        (λ, false)
    end

    function prob_func(prob,i,repeat)
        remake(prob, p=[ps[i][key] for key in p])
    end

    ensemble_prob = EnsembleProblem(prob; output_func=output_func, prob_func=prob_func)
    alg = DynamicSS(alg; tspan=tspan)
    sol = solve(ensemble_prob, alg; trajectories=length(ps), kwargs...)
    sol.u
end


"Replace parameter names with actual Symbolics variables."
lookup_params(params) = [Dict((@parameters $k)[1] => v for (k,v) in p) for p in params]

## Plotting functions

function plot(model, params; normalise=true, hide_y=true, autolimits=true, kwargs...)
    u,t=simulate(model,params; full_solution=true, kwargs...)
    plot(model, u,t)
end

function plot(model, u, t; normalise=true, hide_y=true, autolimits=true, kwargs...)
    labels = [string(s.f) for s in species(model)]
    x_steps = size(u, 1)
    x = range(0.0,1.0,length=x_steps)
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
function interactive_plot(model, param_ranges; tspan=Inf, alg=nothing, dt=0.1, num_verts=64, reltol=1e-6,abstol=1e-8, maxiters = 1e6, hide_y=true)
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
    U = throttle(1/120, U) # Limit update rate to 120Hz
    x = range(0,1,n)
    labels = [string(s.f) for s in species(model)]
    for i in eachindex(eachcol(U[]))
        lines!(ax, x, lift(u -> u[:,i], U); label=labels[i])
    end
    on(U) do _
	    autolimits!(ax)
	end
    axislegend(ax)
    display(fig)
end

end
