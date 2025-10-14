# Example demonstrating the Schnakenburg model (a well-known reaction-diffusion system with analytical results for its Turing stability region)

module Schnakenburg
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

params = (:a => range(0.0,0.6,4), :b =>range(0.0,3.0,4), :γ => [1.0], :Dᵤ => [1.0], :Dᵥ => [50.0])

end