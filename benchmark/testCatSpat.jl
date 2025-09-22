include("../src/PseudoSpectral.jl")
using .PseudoSpectral
using Catalyst, DifferentialEquations

includet("../src/Plot.jl")
using .Plot, WGLMakie
model = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 

u_diffusion = @transport_reaction 1.0 U
v_diffusion = @transport_reaction 50.0 V

n=128
lattice = CartesianGrid(n)
lrs = LatticeReactionSystem(model, [u_diffusion, v_diffusion], lattice)
##
u0 = [:U => randn(n).^2, :V => randn(n).^2]
tspan = (0.0, 10.0)
ps = [:γ => 1.0, :a => 0.2, :b => 2.0]

odeprob = ODEProblem(lrs,u0,tspan,ps; jac =true, sparse=true)
psprob = PseudoSpectralProblem(lrs, u0, tspan, l, d, ps; ss=false3)

@btime sol = solve(odeprob, KenCarp4())
plot_solutions([sol], ["u"])