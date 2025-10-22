# Example demonstrating the Schnakenburg model (a well-known reaction-diffusion system with analytical results for its Turing stability region)

module Schnakenberg
using ReactionDiffusion

reaction = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 

diffusion = [
    (@transport_reaction Dᵤ U),
    (@transport_reaction Dᵥ V)
]

model = Model(reaction, diffusion; domain_size=100.0)
params = (:a => 0.2, :b => 2.0, :γ => 1.0, :Dᵤ => 1.0, :Dᵥ => 50.0)

end