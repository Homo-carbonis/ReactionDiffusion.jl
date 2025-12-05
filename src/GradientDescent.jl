module GradientDescent
export optimise

using ReactionDiffusion
using Symbolics, Catalyst
using FiniteDiff
using LinearAlgebra
using Chain
using StatsBase: sample, Weights

"Map integers to subscript characters."
sub(i) = join(Char(0x2080 + d) for d in reverse!(digits(i)))
"Subscript a symbol with `i...` separated by `_`."
subscript(X, i...) = Symbol(X, join(sub.(i), "_"))
"Build an array of subscripted symbols."
function subscripts(name, dim)
    dim = tuple(dim...) # ensure tuple
    ixs = Iterators.product(range.(1,dim)...)
    [subscript(name, r...) for r in ixs]
end

"Define a set of subscripted species and return them as a vector."
function defspecies(name, t, n)
    names = subscripts(name, n)
    [only(@species $n(t)) for n in names]
end

function defparams(name, dim)
    names = subscripts(name, dim)
    [only(@parameters $n) for n in names]
end

function defparam(name, i...)
    name = subscript(name, i...)
    only(@parameters $name)
end

"Return a vector of all polynomial terms in `vars`` up to degree `n``."
polynomial_terms(vars, n) = @chain fill([1;vars], n) Iterators.product(_...) prod.(_) vec

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

#
function optimise(model, cost, params0; pmap=identity, η=0.01, β₁ = 0.02; β₂=0.001)
    u0 = ReactionDiffusion.createIC(model,num_verts) #??
    make_prob, transform = ReactionDiffusion.pseudospectral_problem(model, u0, tspan)
    params0 = sort(params0)
    ps, p = unzip(p0)

    _cost(p) = @chain p begin
        zipdict(ps,_)
        pmap
        simulate(make_prob,transform, _)
        cost
    end

    p = adam(_cost, _, η, β₁, β₂)
    zipdict(ps, p)
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

zipdict(keys,vals) = Dict(zip(keys,vals))
unzip(dict) = (keys(dict),values(dict)) .|> collect


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

isnonzero(x) = !(ismissing(x) || iszero(x))
end