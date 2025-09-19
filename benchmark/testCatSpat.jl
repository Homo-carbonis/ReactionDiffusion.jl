using Catalyst
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

u0 = [:U => rand(n), :V => rand(n)]
tspan = (0.0, 10.0)
ps = [:γ => 1.0, :a => 0.2, :b => 2.0]

odeprob = ODEProblem(lrs,u0,tspan,ps)

solve(odeprob)