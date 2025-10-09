using ReactionDiffusion#, DiffEqGPU, CUDA

reaction = @reaction_network begin
    # Transcription/secretion
    μ_GDF5 * hill(K,1,pSMAD,n₁), 0 --> GDF5
    μ_NOG * hill(K,1,pSMAD,n₂),  0 --> NOG
    # Complex formation
    k₊,             GDF5 + NOG --> COMPLEX
    k₋,             COMPLEX --> GDF5 + NOG
    # Signaling
    μ_pSMAD,        0 --> pSMAD
    # Degredation]
    δ_GDF5,         GDF5 --> 0
    δ_NOG,          NOG --> 0
    δ_pSMAD,        pSMAD --> 0
end

diffusion = [
    (@transport_reaction D_GDF5 GDF5),
    (@transport_reaction D_NOG NOG),
    (@transport_reaction D_COMPLEX COMPLEX)
]

lattice = CartesianGrid(128)

model = LatticeReactionSystem(reaction, diffusion, lattice)

# D_GDF5 = 1
# D_NOG = 1
# D_COMPLEX = 30
# D_pSMAD =0

# δ_GDF5 = 0.1
# δ_NOG = 6.7
# δ_pSMAD = 1
# F_GDF5 = 1
# F_NOG = 10
# μ_pSMAD = 1
# k₊ = 40
# k₋ = 40
# K = 0.01
# n1 = 8
# n2 = 2

params = model_parameters()

params.reaction[:μ_GDF5] = screen_values(min = 0.0, max = 2.0, mode = "linear", number = 4)
params.reaction[:μ_NOG] = screen_values(min = 0.0, max = 2.0, mode = "linear", number = 4)
params.reaction[:k₊] = screen_values(min = 0.0, max = 80.0, mode = "linear", number = 4)
params.reaction[:k₋] = screen_values(min = 0.0, max = 80.0, mode = "linear", number = 4)
params.reaction[:μ_pSMAD] = screen_values(min = 0.0, max = 2.0, mode = "linear", number = 4)
params.reaction[:δ_GDF5] = screen_values(min = 0.0, max = 0.2, mode = "linear", number = 4)
params.reaction[:δ_NOG] = screen_values(min = 0.0, max = 20.0, mode = "linear", number = 4)
params.reaction[:δ_pSMAD] = screen_values(min = 0.0, max = 2.0, mode = "linear", number = 4)
params.reaction[:K] = screen_values(min = 0.0, max = 0.02, mode = "linear", number = 4)
params.reaction[:n₁] = screen_values(min = 0.0, max = 16, mode = "linear", number = 4)
params.reaction[:n₂] = screen_values(min = 0.0, max = 4, mode = "linear", number = 4)


params.diffusion[:D_GDF5] = [1.0] 
params.diffusion[:D_NOG] = [1.0]
params.diffusion[:D_COMPLEX] = [30.0]

ncores = length(Sys.cpu_info())
turing_params = returnTuringParams(model, params,batch_size=ncores);
# turing_params = returnTuringParams(model, params,batch_size=2);