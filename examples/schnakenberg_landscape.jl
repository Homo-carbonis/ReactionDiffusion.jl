using Revise
using ReactionDiffusion
using ReactionDiffusion.Util: isnonzero, unzip_dict, lookup
using Makie, WGLMakie
using Logging: with_logger
using Unzip: unzip
const n = 16

##
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
params = product(a = range(0.0,0.5,200), b = range(0.0,5.0,200), γ = [1.0], Dᵤ = [1.0], Dᵥ = [50.0], L=[50.0])

##
function plot_turing(model, params)

    λ = turing_wavelength(model, params)
    a = [p[:a] for p in params]
    b = [p[:b] for p in params]

    fig = Figure()

    ax = Axis(fig[1,1], title="Schnakenberg Turing Instabilities", xlabel="a", ylabel="b")
    heatmap!(ax, a, b, λ, colormap=:grays)
    fig
end
##

function cost(sol)
    isempty(sol.u) && return Inf
    ut = only(sol.u)
    ismissing(ut) && return Inf
    (u,t) = ut
    sum(u[n÷2:end,1]) / sum(u[:,1])
end


function sample(params)
    local n = 20
    σ = 15
    params = copy.(fill(params,n))
    for p in params
        p[:L] += randn() * sigma
    end
    params
end
(asym,bsym) = lookup.((:a,:b)) 

function plot_cost(model, params)
    sol = with_logger(Base.NullLogger()) do
        simulate(model, params; output_func=cost, num_verts=n, maxrepeats=0, maxiters = 1e4, verbose=false)
    end

    vals = Dict()
    for (i,p) in enumerate(params)
        ismissing(sol.u[i]) && continue
        get!(vals, (p[asym],p[bsym]), 0.0)
        vals[(p[asym],p[bsym])] += sol.u[i]
    end

    ab,u = unzip_dict(vals)#
    a = [first(x) for x in ab]
    b = [last(x) for x in ab] # TODO Proper unzip


    fig = Figure()
    ax = Axis(fig[1,1], title="Schnakenberg Anterior Concentration", xlabel="a", ylabel="b")
    heatmap!(ax, a,b,u, colormap=:grays)
    fig
end

params = product(a = range(0.0,0.5,100), b = range(0.0,5.0,100), γ = [1.0], Dᵤ = [1.0], Dᵥ = [50.0], L=[50.0])
turing_params = filter_turing(model,params)

tp = lookup(Dict(:a => 0.2, :b => 1.6, :γ => 1.0, :Dᵤ => 1.0, :Dᵥ => 50.0, :L=>50.0))
vars = lookup.([:a,:b])
# plot_cost(model,turing_params)
path = optimise(model, cost, vars, tp; in_domain = is_turing(model), maxsteps=100, maxrepeats=1, maxiters=1e6, savepath=true)
path = [(p[asym], p[bsym]) for p in path]
lines(path)