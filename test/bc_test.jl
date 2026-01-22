includet("../src/Util.jl")
includet("../src/PseudoSpectral.jl")    

using .PseudoSpectral
using WGLMakie
using Symbolics: @variables
using OrdinaryDiffEqExponentialRK: ETDRK4
using SciMLBase: solve, successful_retcode
using Test

@variables U,a,b,d

R = [0]
D = [d/(pi)^2] # Divide by pi^2 for a domain of size pi.
B = ([a/pi],[b/pi])
n=128
dt=0.001
x = range(0.0,pi,n)

make_prob,transform = pseudospectral_problem([U], R, D, B, n)

p = Dict(U=>sin.(x), a=>ones(n), b=>-ones(n), d=>ones(n))
prob = make_prob(p)
sol = solve(prob, ETDRK4(); tspan=(0.0,1.0), dt=dt)
@test successful_retcode(sol)
u,t = transform(sol)
# @test u ≈ exp(-t)*sin.(x) rtol=1e-3;

fig = lines(x,u[:,1])
lines!(x,exp(-t)*sin.(x))
fig