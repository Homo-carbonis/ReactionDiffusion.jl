using ReactionDiffusion

reaction = @reaction_network begin
    anterior * μ_bcd,           ∅ --> BCD
    posterior * μ_nos,          ∅ --> NOS
    δ_bcd,                      BCD --> ∅
    δ_nos,                      NOS --> ∅
    hillar(BCD, NOS, μ_hb,K,n), ∅ --> HB
    δ_hb,                       HB --> ∅
end


diffusion = @diffusion_system L begin
    D_bcd,  BCD
    D_nos,  NOS
    D_hb,   HB
end

model = Model(reaction, diffusion)
params = product(
    L = [1.0],
    anterior = [x -> (x < 1/12) ? 1 : 0],
    posterior = [x -> (x > 11/12) ? 1 : 0],
    μ_bcd = range(0.1, 2.0, 2),
    μ_nos = range(0.1, 2.0, 2),
    δ_bcd = range(0.1, 2.0, 2),
    δ_nos = range(0.1, 2.0, 2),
    μ_hb = range(0.1, 2.0, 2),
    K = range(0.1, 1.0, 2),
    n = range(1.0, 8.0, 2),
    δ_hb = range(0.1, 2.0, 2),
    D_bcd = [0.1],
    D_nos = [0.1],
    D_hb = [0.1]
)

function hb_partition(u, t)
    u = u[:,3]
    n = length(u)
    u_c = maximum(u)/4
    all(>=(1-u_c), u[1:n÷3]) && all(<(u_c), u[n÷3+1:end])
end

good_params = filter_params(hb_partition, model, params)

u,t=simulate(model,good_params[1]; full_solution=true)

ReactionDiffusion.Plot.plot_solutions(u,t, string.(species(model)))