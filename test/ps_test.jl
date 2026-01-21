includet("../src/Util.jl")
includet("../src/PseudoSpectral.jl")
using .PseudoSpectral
using Symbolics: @variables
using OrdinaryDiffEqExponentialRK: ETDRK4
using SciMLBase: solve, successful_retcode
using Test

@testset "Heat equation" begin
    R = [0]
    D = [1/(pi)^2] # Divide by pi^2 for a domain of size pi.
    B = ([0],[0])
    n=64
    @variables U

    make_prob,transform = pseudospectral_problem([U], R, D, B, n)
    ϕ = [cos(x) for x in range(0.0,pi,n)]
    p = Dict(U=>ϕ)
    prob = make_prob(p)
    sol = solve(prob, ETDRK4(); tspan=(0.0,1.0), dt=0.01)
    @test successful_retcode(sol)
    u,t = transform(sol; full_solution=true)
    @testset for (u,t) in zip(eachcol(u[:,1,:]),t)
        @test u ≈ exp(-t)*ϕ rtol=1e-3
    end
end;
