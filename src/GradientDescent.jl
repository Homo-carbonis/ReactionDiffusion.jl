module GradientDescent
export optimise

using ..Util: defspecies, defparams 
using Symbolics, Catalyst
using FiniteDiff
using LinearAlgebra
using StatsBase: sample, Weights


"Return a vector of all polynomial terms in `vars`` up to degree `n``."
polynomial_terms(vars, n) = fill([1;vars], n) |> splat(Iterators.product) .|> prod |> vec

function rational_system(num_species, degree)
    t = default_t()
    X = defspecies(:X, t, num_species)
    terms = polynomial_terms(X, degree)
    R = defparams(:R, (2*num_species, length(terms)))
    D = defparams(:D, num_species)
    @parameters L
    rates = R*terms
    rxs₊ = [Reaction(r,nothing,[x]) for (r,x) in zip(rates[1:num_species], X)]
    rxs₋ = [Reaction(r,[x],nothing) for (r,x) in zip(rates[num_species+1:end], X)]
    rxs = [rxs₊ ; rxs₋]
    @named reaction = ReactionSystem(rxs,t)
    dxs = [TransportReaction(d/L^2, x) for (d,x) in zip(D, X)]
    diffusion = ReactionDiffusion.DiffusionSystem(L, dxs)
    Model(reaction,diffusion)
end

function hill_system(graph)
    num_params = size(graph,1)
    t = default_t()
    X = defspecies(:X, t, num_params)

    rates₊ = defparams(:μ, num_params)
    rates₋ = defparams(:δ, num_params)
    for i in eachindex(X)
        for j in eachindex(X)
            sign = graph[i,j]
            iszero(sign) && continue
            x = X[j]
            K = defparam(:K, i, j)
            n = defparam(:n, i, j)
            f = sign > 0 ? hill : hillr
            rates₊[i] *= f(x, 1, K, n)
        end
    end
    X1 = [[x] for x in X]
    rxs = [Reaction.(rates₊, nothing, X1) ; Reaction.(rates₋, X1, nothing)]
    @named reaction = ReactionSystem(rxs, t)
    D = defparams(:D, num_params)
    @parameters L
    dxs = [TransportReaction(d/L^2, x) for (d,x) in zip(D, X)]
    diffusion = ReactionDiffusion.DiffusionSystem(L, dxs)
    Model(reaction,diffusion)
end


function random_signed_digraph(n, sparsity)
    w = (1-sparsity)/2
    weights = Weights([w, sparsity, w], 1.0)
    sample([-1,0,1], weights, (n,n))
end



function optimise(model, cost, params0; dist=, pmap=identity, η=0.01, β₁ = 0.02, β₂=0.001)
    u0 = ReactionDiffusion.createIC(model,num_verts) #??
    make_prob, transform = ReactionDiffusion.pseudospectral_problem(model, u0, tspan)
    ps, p = unzip_params(params0)
    d = get.(dist, ps, Dirac)
    _simulate(p) = simulate(make_prob, transform, p)
    _dict(p) = Dict(zip(ps,p))
    _cost(p) = p |> _dict |> pmap |> _simulate |> cost
    p = adam(_cost, p, η, β₁, β₂)
    _dict(p)
end

function adam(cost, p, η, β₁, β₂; maxiters=100)
    m = zero(p)
    v = zero(p)
    p_prev = p
    for i in 1:maxiters
        J = vec(FiniteDiff.finite_difference_jacobian(cost, p))
        norm(J) ≈ 0 && return p
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



function find_turing(model, σ=100; batch_size=100, num_batches=10, kwargs...)
    ps = nameof.(ReactionDiffusion.parameters(model))
    for i in 1:num_batches
        params = [Dict(zip(ps, abs.(σ * rand(length(ps))))) for i in 1:batch_size]
        λ = turing_wavelength(model, params; kwargs...)
        println(i)
        nonzeros = isnonzero.(λ)
        any(nonzeros) && return p[nonzeros]
    end
    error("FAILURE!")
end


end