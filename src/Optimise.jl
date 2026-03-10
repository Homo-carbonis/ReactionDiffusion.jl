module Optimise
export optimise, optimisation_problem, plot_cost

using ..Util: zip_dict, tmap
using ..Simulate
using FiniteDiff
using StatsBase: sample
using Pipe: @pipe
using OptimizationBase, OptimizationBBO

"""
    optimise(model, cost, vars, params_min, params_max, params0; in_domain=x->true, sample=nothing, optimiser=BBO_adaptive_de_rand_1_bin(), kwargs...)

    Choose parameters to minimise `cost` over the steady state of model.
"""
    function optimise(args...; optimiser=BBO_adaptive_de_rand_1_bin(), kwargs...)
    prob = optimisation_problem(args...; kwargs...)
    solve(prob, optimiser)
end


function optimisation_problem(model, cost, vars, params_min, params_max, params0; min_cost=0.0, in_domain=x->true, sample=nothing, kwargs...)
    _simulate = simulate(model; kwargs...)
    function _cost(sol)
        # (isempty(sol) || any(ismissing, sol.u)) && return 1.0
        isnothing(sample) ? cost(only(sol.u)...) : cost(sol.u)
    end

    _sample = something(sample, x->[x])

    __cost(p,_) = @pipe p |> zip_dict(vars,_) |> merge(params0, _) |> _sample |> filter(in_domain,_) |> _simulate |> _cost
    # __cost(p,_) = error("Called")
    p_min = [params_min[v] for v in vars]
    p_max = [params_max[v] for v in vars]
    p0 = [params0[v] for v in vars]
    function callback(state, cost)
        cost <= min_cost
    end
    OptimizationProblem((x,p)->__cost(x,p), p0; lb=p_min, ub=p_max, callback=callback)
    # OptimizationProblem((x,p)->error("CALLED"), p0; lb=p_min, ub=p_max, callback=callback)

end

function plot_cost(model, cost, vars, params; in_domain=x->true, sample=nothing, kwargs...)
    _simulate = simulate(model; kwargs...)
    
    function _cost(sol)
        # isempty(sol.u) || any(ismissing, sol.u) && return Inf
        isnothing(sample) ? cost(only(sol.u)...) : cost(sol.u)
    end

    _sample = something(sample, x->[x])

    __cost(p) = @pipe p |> zip_dict(vars,_) |> merge(params[1], _) |> _sample |> filter(in_domain,_) |> _simulate |> _cost

    p = [[params[v] for v in vars] for params in params]
    c = tmap(__cost, Float64, p)
    a = [first(x) for x in p]
    b = [last(x) for x in p] # TODO Proper unzip
    (a,b,c)
end

end