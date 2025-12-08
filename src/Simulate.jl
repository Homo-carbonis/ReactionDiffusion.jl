module Simulate
export simulate, turing_wavelength

using ..Models
using ..PseudoSpectral
using ..Util: ensure_params_vector, issingle, collect_params
using SciMLBase: solve, remake, successful_retcode, ODEFunction, SteadyStateProblem, EnsembleProblem, DiscreteCallback, terminate!, get_du
using SteadyStateDiffEq: DynamicSS
using OrdinaryDiffEqExponentialRK: ETDRK4
using OrdinaryDiffEqRosenbrock: Rodas5
using Symbolics: build_function, jacobian
using LinearAlgebra: diagm, eigvals
using ProgressMeter: Progress, BarGlyphs, next!


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
    u0 = createIC(model, num_verts)
    make_prob, transform = pseudospectral_problem(model, u0, tspan; callback=steady_state_callback(reltol,abstol), maxiters=maxiters, dt=dt)
    simulate(make_prob, transform, params; output_func=output_func, full_solution=full_solution, alg=alg, dt=dt, maxrepeats = maxrepeats, kwargs...)
end

function simulate(make_prob, transform, params; output_func=nothing, full_solution=false, alg=ETDRK4(), dt=0.1, maxrepeats = 4, kwargs...)
    single = issingle(params)
    params = ensure_params_vector(params) 

    progress = Progress(length(params); desc="Simulating parameter sets: ", dt=0.1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    function output_func_(sol,i)
        repeat = sol.prob.p.state
        successful_retcode(sol) || return (missing, repeat <= maxrepeats) # Rerun if solution failed.
        next!(progress) # Advance progress bar.
        if full_solution
            t = sol.t
            u = stack(transform(u) for u in sol.u)
        else
            t = sol.t[end]
            u = transform(sol.u[end])
        end
        out = isnothing(output_func) ? (u,t) : output_func(u,t)
        (out, false)
    end
        
    function prob_func(prob, i, repeat)
        p = params[i]
        dt = dt/2^(repeat-1) # halve dt if solve was unsuccessful.
        prob = make_prob(p, repeat; dt=dt)
    end

    ensemble_prob = EnsembleProblem(make_prob(params[1]); output_func=output_func_, prob_func=prob_func)
    sol = solve(ensemble_prob, alg; trajectories=length(params), kwargs...)
    single ? sol[1] : sol
end

function steady_state_callback(reltol=1e-4,abstol=1e-4)
    condition(u,t,integrator) = isapprox(get_du(integrator), zero(u); rtol=reltol, atol=abstol)
    DiscreteCallback(condition, terminate!)
end


function turing_wavelength(model, params; k=logrange(0.1,100,100), tspan=1e4, alg=Rodas5(), kwargs...)
    single = issingle(params)
    params = ensure_params_vector(params) 

    u0 = ones(num_species(model))

    du = reaction_rates(model)
    u = species(model)
    p = parameters(model)
    t = ()
    (f,f!) = build_function(du, u, p, t; expression=Val{false})
    jac = jacobian(du,u; simplify=true)
    (fjac,fjac!) = build_function(jac, u, p, t; expression=Val{false})

    R = ODEFunction(f!; jac=fjac!)
    prob = SteadyStateProblem(R, u0, [params[1][k] for k in p])

    d = diffusion_rates(model)
    (D,D!) = build_function(diagm(d), p; expression=Val{false})

    k² = k.^2
    function output_func(sol,i)
        successful_retcode(sol) || return (missing, false)
        local p = sol.prob.p

        J = fjac(sol.u, p, 0.0)
        any((!isfinite).(J)) && return (missing,false)
        real_max,i = findmax(real(eigvals(J - D(p) * k²)[end]) for k² in k²)
        λ = real_max > 0.0 ? 2pi/k[i] : 0.0
        (λ, false)
    end

    function prob_func(prob,i,repeat)
        remake(prob, p=[params[i][key] for key in p])
    end

    ensemble_prob = EnsembleProblem(prob; output_func=output_func, prob_func=prob_func)
    alg = DynamicSS(alg; tspan=tspan)
    sol = solve(ensemble_prob, alg; trajectories=length(params), verbose=true, kwargs...)
    single ? sol[1] : sol.u
end

end