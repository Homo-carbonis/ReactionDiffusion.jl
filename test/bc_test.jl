    
using ReactionDiffusion, WGLMakie
include("../examples/Schnakenberg.jl")

reaction = @reaction_network begin
    a, ∅ --> U
    b, ∅ --> V
end 

diffusion = @diffusion_system L begin
    Dᵤ, U
    Dᵥ, V
end

model = Model(reaction,diffusion)
params = dict(a=0.0,b=0.0,Dᵤ = 1.0, Dᵥ = 1.0, L=100.0, U=x->x, V= x->x)

n = 128
h = params[:L]/n
timeseries_plot(model, params; normalise=false, hide_y=false, boundary_conditions=(-0.1,-0.1), num_verts = n, tspan=100.0, dt= 0.01)
