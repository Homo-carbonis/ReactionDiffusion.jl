includet("PseudoSpectral.jl")

using ReactionDiffusion
using .PseudoSpectral
using OrdinaryDiffEqExponentialRK
using Symbolics
using FiniteDiff
import ReactionDiffusion
using LinearAlgebra
using Chain
function rational_system(num_species, degree)
    t = default_t()
    @species X(t)
    names = @. Symbol(:X, Symbolics.map_subscripts(1:num_species))
    X = [X[i] for i in 1:num_species]
    terms = @chain fill([1;X],3) Iterators.product(_...) prod.(_) vec
    R = Symbolics.variables(:R, 1:num_species, 1:length(terms))
    D = Symbolics.variables(:D, 1:num_species)
    @variables L


function optimise_scale(R0,D0; σ=0.01, η=0.01, λ=0.001, tspan=Inf, alg=ETDRK4(), dt=0.1, num_verts=64, reltol=1e-4, abstol=1e-4, maxiters = 1e6, kwargs...)
    n = num_verts
    num_species,degree = size(R0)
    u0 = abs.(σ * randn(num_verts, num_species))
    X = Symbolics.variables(:X, 1:num_species)
    terms = @chain fill([1;X],3) Iterators.product(_...) prod.(_) vec
    R = Symbolics.variables(:R, 1:num_species, 1:length(terms))
    D = Symbolics.variables(:D, 1:num_species)
    @variables L

    t =default_t()
    @
    reaction_rates = R*terms
    diffusion_rates = D./L^2
    make_prob, transform = pseudospectral_problem(X, reaction_rates, diffusion_rates, u0, tspan; callback=ReactionDiffusion.steady_state_callback(reltol,abstol), maxiters=maxiters, dt=dt)

    ps = [vec(R) ; vec(D) ; L]
    p = [vec(R0); vec(D0); 1]
    
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