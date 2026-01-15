    
using ReactionDiffusion, WGLMakie
include("../examples/Schnakenberg.jl")

reaction = @reaction_network begin
    r, ∅ --> U
    r, ∅ --> V
end 

diffusion = @diffusion_system L begin
    Dᵤ, (1.0,-1.0), U
    Dᵥ, (-1.0,-11.0), V
end

model = Model(reaction,diffusion)
params = dict(r=0.0,A=1.0, B=-200.0, Dᵤ = 1.0, Dᵥ = 1.0, L=10.0)

n = 128
h = params[:L]/n

timeseries_plot(model, params; normalise=true, hide_y=false, num_verts = n, dt= 0.01, tspan=1.0)
