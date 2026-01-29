using ReactionDiffusion
reaction = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 

diffusion = @diffusion_system L begin
    Dᵤ, U
    Dᵥ, V
end

inital = @initial_conditions begin
    U0, U
    V0, V
end

model = Model(reaction, diffusion, initial)
# model = Model(reaction, diffusion)
params = dict(a = 0.2, b = 2.0, γ = 0.0, Dᵤ = 50.0, Dᵥ = 50.0, L=100.0, U0=1.0, V0=2.0)
# params = dict(γ = 0.0, Dᵤ = 10.0, Dᵥ = 50.0, L=100.0)
# turing_wavelength(model,params)
parameter_set(model,params)
timeseries_plot(model,params; normalise=false, autolimits=false, num_verts=64)
# params = dict(γ = [0.0], aᵤ=[0.0], bᵤ=[0.0], L=[50.0])
# interactive_plot(model,params; tspan=100.0)