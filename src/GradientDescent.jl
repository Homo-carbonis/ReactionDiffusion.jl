module GradientDescent
export optimise

using ..Util: defspecies, defparams, zip_dict, unzip_dict
using ..Models: createIC, pseudospectral_problem
using ..Simulate
using Symbolics, Catalyst
using FiniteDiff
using LinearAlgebra
using StatsBase: sample, Weights
using Pipe: @pipe

function show(p)
    @show p[1]
    p
end
function optimise(model, cost, vars, params0; in_domain=x->true, sample=p->[p], η=0.01, β₁ = 0.02, β₂=0.001, ϵ=0.001, maxsteps=100, savepath=false, kwargs...)
    _simulate = simulate(model; kwargs...)
    # params0= lookup(params0)
    # vars= lookup.(vars)
    p = [params0[v] for v in vars]
    _cost(p) = @pipe p |> zip_dict(vars,_) |> merge(params0, _) |> sample |> filter(in_domain,_) |> _simulate |> cost
    path = adam(_cost, p, η, β₁, β₂, ϵ; maxsteps=maxsteps)
    if savepath
        [merge(params0, zip_dict(vars, p)) for p in path]
    else
        merge(params0, zip_dict(vars, path[end]))
    end
end


function adam(cost, p, η, β₁, β₂, ϵ; maxsteps=100)
    m = zero(p)
    v = zero(p)
    path=[p]
    for i in 1:maxsteps
        J = vec(FiniteDiff.finite_difference_jacobian(cost, p))
        norm(J) < ϵ && return path
        if any(isnan.(J))
            p = path[end] # Backtrack if p is unstable.
            η /= 2  # Reduce learning rate.
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


end