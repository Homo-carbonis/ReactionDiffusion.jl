using ReactionDiffusion
reaction = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 

diffusion = @diffusion_system LL begin
    Dᵤ, (aᵤ,bᵤ), U
    Dᵥ, (aᵥ,bᵥ), V
end

model = Model(reaction, diffusion)
params = dict(a = 0.2, b = 2.0, γ = 1.0, Dᵤ = 1.0, Dᵥ = 50.0, LL=100.0)
turing_wavelength(model,params)
# timeseries_plot(model,params)