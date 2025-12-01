includet("PseudoSpectral.jl")

using ReactionDiffusion
using .PseudoSpectral
using OrdinaryDiffEqExponentialRK
using Symbolics, Catalyst
using FiniteDiff
import ReactionDiffusion
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

function hill_system(num_species, sparsity=0.6)
    t = default_t()
    X = defspecies(:X, t, num_species)
    graph = random_signed_digraph(num_species, sparsity)

    rates₊ = defparams(:μ, num_species)
    rates₋ = defparams(:δ, num_species)
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
    D = defparams(:D, num_species)
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



function optimise_scale(num_species, degree; σ=0.01, η=0.01, λ=0.001, tspan=Inf, alg=ETDRK4(), dt=0.1, num_verts=64, reltol=1e-4, abstol=1e-4, maxiters = 1e6, kwargs...)
    model = rational_system(num_species, degree)
    u0 = createIC(model,1) #??
    make_prob,transform = pseudospectral_problem(model, u0, tspan)
    nr = num_reaction_params(model)
    nd = num_diffusion_params(model)
    R = σ * randn(nr) .* bernoulli(0.2, nr)
    D = σ * randn(nd)
    function cost(p)
        ps1 = Dict(zip(ps, p))
        ps2 = copy(ps1)
        ps2[L] = ps1[L]/2
        prob1 = make_prob(ps1)
        prob2 = make_prob(ps2)

        sol1 = solve(prob1; alg=alg, verbose=false)
        sol2 = solve(prob2; alg=alg, verbose=false)
        (SciMLBase.successful_retcode(sol1) && SciMLBase.successful_retcode(sol1)) || return NaN
        u1=transform(sol1.u[end])
        u2=transform(sol2.u[end])
        norm(u1-u2) + λ * sum(.!iszero.(p))/length(p)
    end

    p = find_convergent(make_prob,ps,p,100)
    # Adam

    m = zero(p)
    v = zero(p)
    β₁ = 0.02; β₂=0.001
    for i in 1:20
        J = vec(FiniteDiff.finite_difference_jacobian(cost, p))
        m = β₁*m + (1-β₁) * J
        v = β₂*v + (1-β₂) * J.^2
        m̂ = m/(1-β₁^i)
        v̂ = v/(1-β₂^i)
        p .= p - η * m̂./(sqrt.(v̂) .+ eps())
        @show norm(J)
    end

    Dict(zip(ps, p))
end

function find_convergent(make_prob, ps, μ, σ)
    dims = size(μ)
    p = similar(μ)
    for i in 1:1000000
        p .= abs.(μ + σ * randn(dims))
        prob=make_prob(Dict(zip(ps, p)))
        sol = solve(prob, alg=ETDRK4(), verbose=false)
        SciMLBase.successful_retcode(sol) && return p
        i%100 == 0 && println(i)
    end
    error("FAILURE!")
end


function find_turing(model, σ=100)
    n=100
    for i in 1:1000
        ps = nameof.(ReactionDiffusion.parameters(model))
        p = [Dict(zip(ps, abs.(σ * rand(length(ps))))) for i in 1:n]
        λ = turing_wavelength(model, p)
        println(i)
        nonzeros = λ.>0.0
        any(nonzeros) && return p[nonzeros]
    end
    error("FAILURE!")
end

