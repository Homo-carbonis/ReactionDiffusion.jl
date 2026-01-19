using ReactionDiffusion, WGLMakie
reaction = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 

diffusion = @diffusion_system L begin
    Dᵤ, (aᵤ,bᵤ), U
    Dᵥ, (aᵥ,bᵥ), V
end

boundary1 = @reaction_network begin
    aᵤ, ∅ --> U
    aᵥ, ∅ --> V
end

boundary2 = @reaction_network begin
    bᵤ, ∅ --> U
    bᵥ, ∅ --> V
end

model = Model(reaction, diffusion, (boundary1, boundary2))
# params = dict(γ = 0.0, Dᵤ = 1.0, Dᵥ = 50.0,  aᵤ=10.0, L=100.0)
# turing_wavelength(model,params)
# timeseries_plot(model,params; num_verts=64)
params = dict(γ = [0.0], aᵤ=[-1.0,1.0], bᵤ=[-1.0,1.0], L=50.0)
interactive_plot(model,params; tspan=100.0)