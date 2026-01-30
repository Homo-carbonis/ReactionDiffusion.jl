using ReactionDiffusion
using WGLMakie
using Statistics
n=16
dorsal = >=((n-1)/n)

reaction = @reaction_network begin
    (k_Admp, λ_ChdAdmp * Xlr),              Chd + Admp <--> ChdAdmp
    (k_Bmp, λ_ChdBmp * Xlr),                Chd + Bmp <--> ChdBmp
    (η_Chd * dorsal(x), λ_Chd * Xlr),          ∅ <--> Chd
    hill(Admp+Bmp,1e-3,T_Admp,4) * dorsal,  ∅ --> Admp
end

diffusion = @diffusion_system L begin
    D_Chd,  Chd
    D_Lig,  Admp
    D_Lig,  Bmp
    D_Comp, ChdAdmp
    D_Comp, ChdBmp
end

model = Model(reaction,diffusion)


params = dict(k_Bmp = 0.1, k_Admp = 0.1, λ_Chd = 0.1, λ_ChdBmp = 0.1, λ_ChdAdmp = 0.1, η_Chd = 10^1.5, Xlr = 0.01, D_Chd = 1.0, D_Lig = 1.0, D_Comp = 1.0, T_Admp = 1e-4, L=1000.0)
param_ranges = dict(k_Bmp = [0.1], k_Admp = [0.1], λ_Chd = [0.1], λ_ChdBmp = [0.1], λ_ChdAdmp = [0.1], η_Chd = [10^1.5], Xlr = [0.01], D_Chd = [1.0], D_Lig = [1.0], D_Comp = [1.0], L=logrange(10.0,1000.0,100))

h=60.0*60.0
# u,t = simulate(model, [params,params]; num_verts=n, dt = 0.01, maxrepeats=3, tspan=(0.01*h), full_solution=true)
# s = timeseries_plot(model, u,t)

##

dynamic_range(S) = maximum(S)/minimum(S)

function validity_cost(S; a=1,b=1,c=1)
    x = range(0.0,1.0,n)
    polarity_cost = sum(S.*x)
    steepness_cost = 1/dynamic_range(S)
    continuous_increase_cost = dynamic_range(S[n÷2:end])
    a * polarity_cost + b * steepness_cost + c * continuous_increase_cost
end

function scaling_cost(S, threshold)
    X = [something(findfirst(>=(threshold), S), findfirst(>=(middle(S)), S)) for S in S]
    std(X)
end

function cost(sol; a=1,b=1)
    S = [u[:,2] + u[:,4] for (u,t) in sol]
    vc = sum(validity_cost, S) 
    sc = sum(T -> scaling_cost(S,T), [0.1,0.01])
    a * vc + b * sc
end

function sample(params)
    params2 = copy(params)
    params2[:L] /= 2
    @show params
    [params, params2]
end

vars = [:k_Bmp, :k_Admp, :λ_Chd, :λ_ChdBmp, :λ_ChdAdmp, :η_Chd, :D_Chd, :D_Lig, :D_Comp]


optimise(model, cost, vars, params; η=0.01, sample=sample, dt=0.001, tspan=1000.0, num_verts=n)

##


