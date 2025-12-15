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


function optimise(model, cost, params0; in_domain=x->true, sample=x->[x], η=0.01, β₁ = 0.02, β₂=0.001, ϵ=0.001, num_verts=16, tspan=Inf, maxiters=100)
    u0 = createIC(model,num_verts) #??
    make_prob, transform = pseudospectral_problem(model, u0, tspan)
    ps,p = unzip_dict(params0)
    _cost(p) = @pipe p |> zip_dict(ps,_) |> sample |> filter(in_domain,_) |> simulate(make_prob, transform, _; maxrepeats=0) |> cost
    p = adam(_cost, p, η, β₁, β₂, ϵ; maxiters=maxiters)
    zip_dict(ps,p)
end


function adam(cost, p, η, β₁, β₂, ϵ; maxiters=100)
    m = zero(p)
    v = zero(p)
    p_prev = p
    for i in 1:maxiters
        @show p
        J = vec(FiniteDiff.finite_difference_jacobian(cost, p))
        norm(J) < ϵ && return p
        if any(isnan.(J))
            p=p_prev
            η /= 2
            continue
        end
        p_prev = p
        m = β₁*m + (1-β₁) * J
        v = β₂*v + (1-β₂) * J.^2
        m̂ = m/(1-β₁^i)
        v̂ = v/(1-β₂^i)
        p .= p - η * m̂./(sqrt.(v̂) .+ eps())
        @show norm(J)
    end
    @warn "Maxiters exeeded. Optimum not found."
    p
end


end