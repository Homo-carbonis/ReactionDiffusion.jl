module Simulate
export simulate

using ..Models
using ..PseudoSpectral
using ..Util: issingle
using SciMLBase: solve, successful_retcode, EnsembleProblem, EnsembleSolution, DiscreteCallback, terminate!, get_du
using OrdinaryDiffEqExponentialRK: ETDRK4
using ProgressMeter: Progress, BarGlyphs, next!


"""
    simulate(model, params; output_func=nothing, full_solution=false, alg=ETDRK4(), num_verts=64, dt=0.1, maxrepeats = 4, reltol=1e-4, abstol=1e-4, kwargs...)


Simulate `model` for the parameters and initial conditions given in `params`, stopping when a steady state is reached. Returns `(u,t)` with the solution values and time.


# Arguments
- `model`: `Model` object containg the system to be simulated.
- `params`: Either a single parameter set or a vector of parameter sets to be solved as an ensemble. Parameter sets can be created manually with parameter_set or supplied as a dict or collection of pairs in which case defaults will be used for any missed values and low-level noise added to initial conditions. Parameters values may be either single numbers which are replicated homogenously over the domain, or functions mapping the interval [0.0,1.0] to values for the corresponding point in space. 
- `output_func(u,t)`: Function to transform output values. 
- `full_solution`: Return a vector of values at each time point if true, instead of just the steady-state solution.
- `max_repeats`: Number of times to retry with reduced dt before giving up if the solution fails to converge. 
- `num_verts`: Number of points in spatial discretisation.
For other keyword arguments see https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/.
"""
simulate(model, params; kwargs...) = simulate(model; kwargs...)(params)

"""
    function simulate(model; output_func=nothing, full_solution=false, alg=ETDRK4(), num_verts=64, dt=0.1, maxrepeats = 4, reltol=1e-4, abstol=1e-4, kwargs...)

Partially applied version of `simulate` to avoid repeating expensive setup when simulating the same model reapeatedly.
"""
function simulate(model; output_func=nothing, full_solution=false, alg=ETDRK4(), num_verts=64, dt=0.1, maxrepeats = 4, reltol=1e-4, abstol=1e-4, boundary_conditions=(0.0,0.0), kwargs...)
    make_prob, transform = pseudospectral_problem(model, num_verts, boundary_conditions)

    f(params) = f([params]) |> only # Accept a single parameter set instead of a vector.
    f(params::AbstractVector) = f(parameter_set.(model, params))
    function f(params::Vector{ParameterSet})
        isempty(params) && return EnsembleSolution([], 0.0, false)
        params = [Dict(k => expand_spatial(v,num_verts) for (k,v) in p) for p in params]
        
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
        solve(ensemble_prob, alg; trajectories=length(params), callback=steady_state_callback(reltol,abstol), kwargs...)
    end
end

expand_spatial(x::Function, n) = x.(range(0.0,1.0,n))
expand_spatial(x::Float64, n) = fill(x,n)


function steady_state_callback(reltol=1e-4,abstol=1e-4)
    condition(u,t,integrator) = isapprox(get_du(integrator), zero(u); rtol=reltol, atol=abstol)
    DiscreteCallback(condition, terminate!)
end

end


