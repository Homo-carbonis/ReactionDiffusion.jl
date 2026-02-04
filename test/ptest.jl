includet("../src/Util.jl")
includet("../src/PseudoSpectral.jl")
using .PseudoSpectral
using Symbolics: @variables
using OrdinaryDiffEqExponentialRK: ETDRK4
using SciMLBase: solve, successful_retcode
using Test
using Makie, WGLMakie
@variables U,g0,g1,d,a,b
R = [0]
D = [1/(pi)^2] # Divide by pi^2 for a domain of size pi.
n=256
dt=0.001
X = range(0,pi,n)

B = [pi,pi]
IC = [pi*x]
make_prob,transform = pseudospectral_problem([U], R, D, B, IC, n)
prob = make_prob(Dict())
sol = solve(prob, ETDRK4(); tspan=(0.0,2.0), dt=dt)
@test successful_retcode(sol)
u,t = transform(sol, full_solution=false)
@test u ≈ X rtol=1e-2;

# lines(X,u[1,:,1])