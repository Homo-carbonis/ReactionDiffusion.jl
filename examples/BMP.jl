module BMP
using ReactionDiffusion

reaction = @reaction_network begin
    # complex formation
    (k₊, k₋),               GDF5 + NOG <--> COMPLEX 
    # degradation
    δ₁,                     GDF5 --> ∅
    δ₂,                     NOG --> ∅
    δ₃,                     pSMAD --> ∅
    # transcriptional feedbacks (here: repressive hill functions)
    hillr(pSMAD,μ₁,K₁,n₁),  ∅ --> GDF5
    hillr(pSMAD,μ₂,K₂,n₂),  ∅ --> NOG
    # signalling
    μ₃*GDF5,                ∅ --> pSMAD
end

diffusion = [
    (@transport_reaction D_GDF5 GDF5),
    (@transport_reaction D_NOG NOG),
    (@transport_reaction D_COMPLEX COMPLEX)
]

model = Model(reaction, diffusion)

params = (
    :μ₁ => 1.0,
    :μ₂ => 10.0,
    :k₊ => 40.0,
    :k₋ => 40.0,
    :μ₃ => 1.0,
    :δ₁ => 0.1,
    :δ₂ => 6.7,
    :δ₃ => 1.0,
    :K₁ => 0.01,
    :K₂ => 0.01,
    :n₁ => 8.0,
    :n₂ => 2.0,
    :D₁ => 1.0,
    :D₂ => 1.0,
    :D₃ => 30.0
)
# constant parameters
params.reaction["δ₃"] = [1.0]
params.reaction["μ₁"] = [1.0]
params.reaction["μ₃"] = [1.0]
params.reaction["n₁"] = [8]
params.reaction["n₂"] = [2]

# varying parameters
num_params = 5
params.reaction["δ₁"] = screen_values(min = 0.1,max = 10, number=num_params)
params.reaction["δ₂"] = screen_values(min = 0.1,max = 10,number=num_params)
params.reaction["μ₂"] = screen_values(min = 0.1,max = 10, number=num_params)
params.reaction["k₊"] = screen_values(min = 10.0,max = 100.0, number=num_params)
params.reaction["k₋"] = screen_values(min = 10.0,max = 100.0, number=num_params)
params.reaction["K₁"] = screen_values(min = 0.01,max = 1,number=num_params)
params.reaction["K₂"] = screen_values(min = 0.01,max = 1, number=num_params)

# diffusion coefficients
params.diffusion = Dict(
    "NOG"       => [1.0],
    "GDF5"      => screen_values(min = 0.1,max = 30, number=10),
    "COMPLEX"   => screen_values(min = 0.1,max = 30, number=10)
)


end