module GradientDescent
export optimise, plot_cost

using ..Util: defspecies, defparams, zip_dict, unzip_dict, tmap
using ..Models: pseudospectral_problem
using ..Simulate
using Symbolics, Catalyst
using FiniteDiff
using LinearAlgebra
using StatsBase: sample, Weights
using Pipe: @pipe


function optimise(model, cost, vars, params0; in_domain=x->true, sample=nothing, η=0.001, β₁ = 0.02, β₂=0.001, ϵ=0.001, maxsteps=100, savepath=false, kwargs...)
    _simulate = simulate(model; kwargs...)
    function _cost(sol)
        (isempty(sol) || any(ismissing, sol.u)) && return Inf
        isnothing(sample) ? cost(only(sol.u)...) : cost(sol.u)
    end

    _sample = something(sample, x->[x])

    __cost(p) = @pipe p |> zip_dict(vars,_) |> merge(params0, _) |> _sample |> filter(in_domain,_) |> _simulate |> _cost

    p = [params0[v] for v in vars]
    path = adam(__cost, p, η, β₁, β₂, ϵ; maxsteps=maxsteps)
    savepath ? path : path[end]
end


function adam(cost, p, η, β₁, β₂, ϵ; maxsteps=1000)
    m = zero(p)
    v = zero(p)
    path=[p]
    for i in 1:maxsteps
        print("$(i), ")
        J = vec(FiniteDiff.finite_difference_jacobian(cost, p))
        @show J
        norm(J) < ϵ && return path
        if !all(isfinite.(J))
            p = path[end] # Backtrack if p is unstable.
            η /= 2  # Reduce learning rate.
            @warn "Retrying with η = $(η)."
            continue
        end
        push!(path, p)
        m = β₁*m + (1-β₁) * J
        v = β₂*v + (1-β₂) * J.^2
        m̂ = m/(1-β₁^i)
        v̂ = v/(1-β₂^i)
        p = p - η * m̂./(sqrt.(v̂) .+ eps())
    end
    @warn "Maxiters exeeded. Optimum not found."
    path
end

function plot_cost(model, cost, vars, params; in_domain=x->true, sample=nothing, kwargs...)
    _simulate = simulate(model; kwargs...)
    
    function _cost(sol)
        isempty(sol.u) || any(ismissing, sol.u) && return Inf
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