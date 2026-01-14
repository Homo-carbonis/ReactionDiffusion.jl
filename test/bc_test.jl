    
using ReactionDiffusion, WGLMakie
include("../examples/Schnakenberg.jl")

reaction = @reaction_network begin
    r, ∅ --> U
    r, ∅ --> V
end 

diffusion = @diffusion_system L begin
    Dᵤ, (A,B), U
    Dᵥ, V
end

model = Model(reaction,diffusion)
params = dict(r=0.0,A=1.0, B=1.0, Dᵤ = 1.0, Dᵥ = 1.0, L=100.0, U=x->x, V= x->x)

n = 128
h = params[:L]/n
u,t=simulate(model, params; num_verts = n, tspan=100.0, dt= 0.01, full_solution=true)

timeseries_plot(model, params; normalise=false, hide_y=false, num_verts = n, tspan=100.0, dt= 0.01)
