using ReactionDiffusion
using ReactionDiffusion.Util: isnonzero
using Makie, WGLMakie

reaction = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 

diffusion = @diffusion_system L begin
    Dᵤ, U
    Dᵥ, V
end

model = Model(reaction, diffusion)

params = product(a = range(0.0,0.5,200), b = range(0.0,5.0,200), γ = [1.0], Dᵤ = [1.0], Dᵥ = [50.0], L=15*randn(10).+50)
const n = 16
function cost(u,t)
    sum(u[n÷2:end,1]) / sum(u[:,1])
end
with_logger(Base.NullLogger()) do
    sol = simulate(model, params; output_func=cost, num_verts=n, maxrepeats=0, maxiters = 1e4, verbose=false)
end
a = [p[:a] for p in params]
b = [p[:b] for p in params]
fig = Figure()
ax = Axis(fig[1,1], title="Schnakenberg Anterior Concentration", xlabel="a", ylabel="b")
    heatmap!(ax, a,b,sol.u, colormap=:grays)
x