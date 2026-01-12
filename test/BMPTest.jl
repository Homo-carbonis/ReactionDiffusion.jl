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

diffusion = @diffusion_system L begin
    D_GDF5,     GDF5
    D_NOG,      NOG
    D_COMPLEX,  COMPLEX
end

model = Model(reaction, diffusion)

params1 = (
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

params = product(
    δ₃ = range(0.1,10.0,5),
    μ₁ = range(0.1,10.0,5),
    μ₃ = range(0.1,10.0,5),
    n₁ = range(1.0,20.0,5),
    n₂ = range(1.0,20.0,5),
    GDF5 = [0.1],
    NOG = [0.1,0.5],
)

tp = filter_turing(model,params)