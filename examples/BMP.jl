#using ReactionDiffusion#, DiffEqGPU, CUDA

reaction = @reaction_network begin
    # Transcription/secretion
    μ_GDF5 * hill(K,1,pSMAD,n₁), 0 --> GDF5
    μ_NOG * hill(K,1,pSMAD,n₂),  0 --> NOG
    # Complex formation
    k₊,             GDF5 + NOG --> COMPLEX
    k₋,             COMPLEX --> GDF5 + NOG
    # Signaling
    μ_pSMAD,        0 --> pSMAD
    # Degredation
    δ_GDF5,         GDF5 --> 0
    δ_NOG,          NOG --> 0
    δ_pSMAD,        pSMAD --> 0
end

diffusion = @transport_reactions begin
    D_GDF5, GDF5
    D_NOG, NOG
    D_COMPLEX, COMPLEX
end

model = Model(reaction, diffusion)

params = (
    :μ_GDF5 => 1.0,
    :μ_NOG => 10.0,
    :k₊ => 40.0,
    :k₋ => 40.0,
    :μ_pSMAD => 1.0,
    :δ_GDF5 => 0.1,
    :δ_NOG => 6.7,
    :δ_pSMAD => 1.0,
    :K => 0.01,
    :n₁ => 8.0,
    :n₂ => 2.0,
    :D_GDF5 => 1.0,
    :D_NOG => 1.0,
    :D_COMPLEX => 30.0
)

#turing_params = returnTuringParams(params)

u,t = simulate(model, params)