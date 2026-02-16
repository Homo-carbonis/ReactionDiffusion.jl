using ReactionDiffusion
using WGLMakie
using Statistics
using Base.Iterators: flatmap, partition

using LinearAlgebra: norm

reaction = @reaction_network begin
    (αₘ¹ + αₘ²*M)/(1+E),    M --> ∅
    αₑ,                    E --> ∅
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

const L1 = 100.0
const tmax = 5e5

##

params0 = dict(L=L1, Dₘ=1e1, Dₑ=1e-1, αₘ¹ = 1e-5, αₘ² = 1e0, αₑ = 1e-4, ηₘ = 1e0, βₑ = 1e-3, T=1e-3, h=4.0)
timeseries_plot(model,params0; dt=3.0, tspan=tmax, noise=0.01, max_attempts=3, num_verts=16, normalise=false, hide_y=false, species=[:M])

##
function isvalid(sol)
    U,t=sol
    u=U[:,1]
    n=length(u)
    maxvalid = argmax(u) == 1
    minvalid = argmin(u) == n
    midvalid = u[n÷2] > 1e-6
    quartervalid = 1 < maximum(u)/u[n÷4] < 100
    tvalid = t < tmax
    all([maxvalid, minvalid, midvalid, quartervalid, tvalid])
end


function σ(sol)
    (U1,t1),(U2,t2) = sol
    u1=U1[:,1]
    u2=U2[:,1]
    n=length(u1)
    r = [0.25,0.5,0.75]
    i = n*r .|> floor .|> Int
    T = u1[i]
    i′ = [findfirst(<(t), u2) for t in T]
    any(isnothing, i′) && return 1.0
    x = i/n
    x′ = i′/n
    mean(abs.(x-x′))
end

function σ2(sol)
    (U1,t1),(U2,t2) = sol
    u1=U1[:,1]
    u2=U2[:,1]
    n=length(u1)
    r = [0.25,0.5,0.75]
    i = n*r .|> floor .|> Int
    norm(u1[i] - u2[i])
end


cost(sol) = all(isvalid,sol) ? σ(sol) : 1.0

function sample(params)
    params2 = copy(params)
    params2[:L] *= 2
    [params,params2]
end
decrange(start,stop) = logrange(10.0^start,10.0^stop,stop-start+1)

params = product(Dₘ=decrange(-1,1), Dₑ=decrange(-1,1), αₘ¹=decrange(-5,0), αₘ²=decrange(-3,2), αₑ=decrange(-5,0), ηₘ=decrange(-1,1), βₑ=decrange(-4,-2), T=decrange(-5,-3), h=[2.0,4.0], L=[100.0])

##
_params = rand(params, 100)
sol = simulate(model,_params; dt=3.0, tspan=tmax, noise=0.01, max_attempts=2, num_verts=8)
valid = findall(isvalid, sol.u)

##
vars = [:Dₘ,:Dₑ,:αₘ¹,:αₘ²,:αₑ,:ηₘ,:βₑ,:T]
params0=_params[valid[2]]
optimise(model,cost,vars,params0; η=1e-7, β₁=0.9, β₂=0.999, sample=sample, tspan=tmax, dt=0.5, num_verts=32, maxsteps=50, max_attempts=1)
##

##

params = product(Dₘ=decrange(-1,1), Dₑ=decrange(-1,1), αₘ¹=decrange(-5,0), αₘ²=decrange(-3,2), αₑ=decrange(-5,0), ηₘ=decrange(-1,1), βₑ=decrange(-4,-2), T=decrange(-5,-3), h=[2.0,4.0], L=[100.0])
# params = parameter_set.(model,params)

_params = [p for (i,p) in enumerate(params) if iszero(i%1000)]
params_s = flatmap(sample,_params) |> collect

sol = simulate(model,params_s; dt=5.0, tspan=5e5, noise=0.0, num_verts=8, max_attempts=5)
function _isvalid(sol)
    !ismissing(sol) && isvalid(sol)
end
valid = findall(_isvalid, sol.u)
s = σ.(partition(sol[valid],2),100.0)


