using ReactionDiffusion
include("../examples/Schnakenberg.jl")
model = Schnakenberg.model
params = dict(a = 0.2, b = 2.0, γ = 1.0, Dᵤ = 1.0, Dᵥ = 50.0, L=100.0)
expected_periods = 1/turing_wavelength(model,params)
timeseries_plot(model, params; num_verts=128)
