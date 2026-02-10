using ReactionDiffusion
using WGLMakie
using Statistics
using Base.Iterators: flatmap, partition

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
    10000000.0, E
    1.0, M
end

boundary =  (b0,@reaction_network)

model = Model(reaction, diffusion, boundary, initial)

params_c = dict(L=100.0, Dₘ=1e1, Dₑ=1e-1, αₘ¹ = 1e-5, αₘ² = 1e0, αₑ = 1e-4, ηₘ = 1e1, βₑ = 1e-3, T=1e-3, h=4.0)
timeseries_plot(model,params_c; dt=0.1, tspan=500.0, noise=1e-12, max_attempts=0, num_verts=64, normalise=false, hide_y=false)


# tmax=1e3
# decrange(start,stop) = logrange(10.0^start,10.0^stop,stop-start+1)
# params = product(Dₘ=decrange(-1,1), Dₑ=decrange(-1,1), αₘ¹=decrange(-5,0), αₘ²=decrange(-3,2), αₑ=decrange(-5,0), ηₘ=decrange(-1,1), βₑ=decrange(-4,-2), T=decrange(-5,-3), h=[2.0,4.0], L=[100.0])
# params = parameter_set.(model,params)

# function isvalid(u,t)
#     n=length(u)
#     maxvalid = argmax(u) == 1
#     minvalid = argmin(u) == n
#     midvalid = u[n÷2] > 1e-6
#     quartervalid = 1 < maximum(u)/u[n÷4] < 100
#     tvalid = t < tmax
#     @show u[n÷2]
#     all([maxvalid, minvalid, midvalid, quartervalid, tvalid])
# end

# function σ(u1,u2, L)
#     n=length(u)
#     r = [0.25,0.5,0.75]
#     i = n*r .|> floor .|> Int
#     T = u1[i]
#     i′ = [findfirst(<(t), u2) for t in T]
#     any(isnothing, i′) && return 1.0
#     x = L*i/n
#     x′ = L*i′/n
#     mean(abs.(x-x′))
# end

# function sample(params)
#     params2 = copy(params)
#     params2[domain_size(model)] *= 2
#     [params,params2]
# end

# _params = [p for (i,p) in enumerate(params) if iszero(i%1000)]
# params_s = flatmap(sample,_params) |> collect

# sol = simulate(model,params_s; dt=5.0, tspan=5e5, noise=0.0, num_verts=8, max_attempts=5)
# function _isvalid(sol)
#     ismissing(sol) && return false
#     u,t = sol
#     isvalid(u[:,1],t)
# end
# valid = findall(_isvalid, sol.u)
# s = σ.(partition(sol[valid],2),100.0)


