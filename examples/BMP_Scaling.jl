using ReactionDiffusion
using WGLMakie

reaction = @reaction_network begin
    (k_Admp, λ_ChdAdmp * Xlr),              Chd + Admp <--> ChdAdmp
    (k_Bmp, λ_ChdBmp * Xlr),                Chd + Bmp <--> ChdBmp
    (η_Chd * dorsal, λ_Chd * Xlr),          ∅ <--> Chd
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

n=32

params = dict(dorsal = >=((n-1)/n), k_Bmp = 0.1, k_Admp = 0.1, λ_Chd = 0.1, λ_ChdBmp = 0.1, λ_ChdAdmp = 0.1, η_Chd = 10^1.5, Xlr = 0.01, D_Chd = 1.0, D_Lig = 1.0, D_Comp = 1.0, L=1000.0)
param_ranges = dict(dorsal = [>=((n-1)/n)], k_Bmp = [0.1], k_Admp = [0.1], λ_Chd = [0.1], λ_ChdBmp = [0.1], λ_ChdAdmp = [0.1], η_Chd = [10^1.5], Xlr = [0.01], D_Chd = [1.0], D_Lig = [1.0], D_Comp = [1.0], L=logrange(10.0,1000.0,100))

h=60.0*60.0
u,t = simulate(model, params; num_verts=n, dt = 0.01, maxrepeats=0, tspan=(2*h), full_solution=true)
s=timeseries_plot(model, u,t)

# s=interactive_plot(model, param_ranges; num_verts=n, dt = 0.01, maxrepeats=0, ts)
s
