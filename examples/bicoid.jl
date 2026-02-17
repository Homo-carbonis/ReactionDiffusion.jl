using ReactionDiffusion
using WGLMakie
using Catalyst

## SDD

reaction = @reaction_network begin
    μ, Bcd --> ∅
end

diffusion = @diffusion_system L begin
    D, Bcd
end

boundary = ((@reaction_network begin J, ∅ --> Bcd end), @reaction_network)

model = Model(reaction,diffusion,boundary)

params0 = dict(L=100, D=0.1, μ=0.001, J=0.01)
timeseries_plot(model,params0; dt=0.1, noise=0.01, max_attempts=1, num_verts=32, normalise=false, hide_y=false)
simulate(model,params0; dt=0.001, tspan=1.0)


##

##
reaction = @reaction_network begin
    @species T(t)
    @observables T ~ S + F

    (α, β₀ * T /(T + T₀)),    S <--> F
    μₛ,                       S --> ∅
    μᵥ,                       F --> ∅
end


diffusion = @diffusion_system L begin
    Dₛ, S
    Dᵥ, F
end

b0 = @reaction_network begin
    ηₘ, ∅ --> M
end

boundary =  (b0,@reaction_network)

model = Model(reaction, diffusion, boundary)
