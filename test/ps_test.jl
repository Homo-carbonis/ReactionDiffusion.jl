includet("../src/Util.jl")
includet("../src/PseudoSpectral.jl")
using .PseudoSpectral
using Symbolics: @variables
using OrdinaryDiffEqExponentialRK: ETDRK4
using SciMLBase: solve, successful_retcode
using Test

@testset "heat equation" begin
    @variables U,a,b,d

    R = [0]
    D = [d/(pi)^2] # Divide by pi^2 for a domain of size pi.
    n=128
    dt=0.001
    IC = [cos(pi*x)]

    make_prob,transform = pseudospectral_problem([U], R, D, IC, n)
    X = range(0,pi,n)
    @testset "zero flux" begin
        p = Dict(d=>1.0)
        prob = make_prob(p)
        sol = solve(prob, ETDRK4(); tspan=(0.0,2.0), dt=dt)
        @test successful_retcode(sol)
        u,t = transform(sol)
        @test u ≈ exp(-t)*cos.(X) rtol=1e-2;
    end
end;
