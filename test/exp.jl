using ReactionDiffusion
using WGLMakie
using Statistics


reaction = @reaction_network begin
    (αₘ¹ + αₘ²*M)/(1+E),    M --> ∅
    αₑ¹,                    E --> ∅
    hillr(M,βₑ,T,h),        ∅ --> E
end


diffusion = @diffusion_system L begin
    Dₘ, M
    Dₑ, E
end

b0 = @reaction_network begin
    ηₘ, ∅ --> M
end

initial = @initial_conditions begin
    1.0, E
    1.0, M
end

boundary =  (b0,@reaction_network)

model = Model(reaction, diffusion, boundary, initial)

params_c = dict(L=100.0, Dₘ=10.0, Dₑ=0.1, αₘ¹ = 1e-5, αₘ² = 1.0, αₑ = 1e-4, ηₘ = 0.0001, βₑ = 1e-3, T=1e-3)

timeseries_plot(model,params_c; dt=0.1, tspan=100.0, noise=1e-10, num_verts=256, maxrepeats=0)

