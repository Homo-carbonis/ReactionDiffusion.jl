module Simulate
export simulate

using ..Models
using ..PseudoSpectral
using ..Util: issingle, lookup
using SciMLBase: solve, successful_retcode, EnsembleProblem, EnsembleSolution, DiscreteCallback, terminate!, get_du
using OrdinaryDiffEqExponentialRK: ETDRK4
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
simulate(model, params; kwargs...) = simulate(model; kwargs...)(params)

function simulate(model; output_func=nothing, full_solution=false, alg=ETDRK4(), dt=0.1, maxrepeats = 4, kwargs...)
    make_prob, transform = pseudospectral_problem(model, num_verts)

    f(params) = f([params]) |> only # Accept a single parameter set instead of a vector.

    function f(params::AbstractVector)
        isempty(params) && return EnsembleSolution([], 0.0, false)
        params = Dict(k => expand_spatial(v,n) for (k,v) in params)
        
        progress = Progress(length(params); desc="Simulating parameter sets: ", dt=0.1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

        function _output_func(sol,i)
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
            dt′ = dt/2^(repeat-1) # halve dt if solve was unsuccessful.
            prob = make_prob(p, repeat; dt=dt′)
        end

        ensemble_prob = EnsembleProblem(make_prob(params[1]); output_func=_output_func, prob_func=prob_func)
        sol = solve(ensemble_prob, alg; trajectories=length(params), callback=steady_state_callback(reltol,abstol), maxiters=maxiters, kwargs...)
        single ? sol[1] : sol
    end
end

expand_spatial(x::Function, n) = x.(range(0.0,1.0,n))
expand_spatial(x, n) = fill(x,n)


function steady_state_callback(reltol=1e-4,abstol=1e-4)
    condition(u,t,integrator) = isapprox(get_du(integrator), zero(u); rtol=reltol, atol=abstol)
    DiscreteCallback(condition, terminate!)
end

end


