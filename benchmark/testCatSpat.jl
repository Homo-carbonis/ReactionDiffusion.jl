using Revise, Catalyst, DifferentialEquations, BenchmarkTools, Interpolations

includet("../src/PseudoSpectral.jl")
using .PseudoSpectral

includet("../src/Plot.jl")
using .Plot, WGLMakie

includet("benchmark.jl")

model = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 

v_diffusion = @transport_reaction Dᵥ V
u_diffusion = @transport_reaction Dᵤ U

n=256
lattice = CartesianGrid(n)
lrs = LatticeReactionSystem(model, [v_diffusion, u_diffusion], lattice)

##
tspan = (0.0, 10.0)
L=40
h = L/((n-1)*2pi)
ps1 = [:γ => 1.0, :a => 0.2, :b => 2.0, :Dᵤ => 1.0/h, :Dᵥ => 50.0/h]
ps = make_params(lrs; γ = 1.0, a = 0.2, b = 2.0, Dᵤ = 1.0, Dᵥ = 50.0)
u0 = 0.001 * randn(n,2).^2
U0=copy(u0[:,1]); V0=copy(u0[:,2])
dt = 0.01

##
prob, transform = pseudospectral_problem(lrs, u0, tspan,ps,L,dt)
sol = solve(prob, ETDRK4());


##


x=range(0,L, n)
u0_itp = cubic_spline_interpolation(x,u0)

function myinit(dx,dt)
    x = 0:dx:L
    u0 = stack(itp.(x) for itp in u0_itp) 
    n=size(u0,1)
    m=size(u0,2)

    lattice = CartesianGrid(n)
    lrs = LatticeReactionSystem(model, [v_diffusion, u_diffusion], lattice)
    prob,transform = PseudoSpectralProblem(lrs, u0, tspan,ps,L,dt)
    int = init(prob, ETDRK4())
    int,transform
end

function init_ref(dx)
    x = 0:dx:L
    u0 = stack(itp.(x) for itp in u0_itp) 
    n=length(x)
    U0=copy(u0[:,1]); V0=copy(u0[:,2])
    lattice = CartesianGrid(n)
    lrs = LatticeReactionSystem(model, [v_diffusion, u_diffusion], lattice)
    prob = ODEProblem(lrs, [:U =>U0, :V=>V0], tspan, ps1, jac=true, sparse=true)
    init(prob, FBDF())
end

tune(myinit,init_ref, 1.0,1.0)
    x = 0:dx:L
    u0 = stack(itp.(x) for itp in u0_itp) 
    n=size(u0,1)
    m=size(u0,2)

    lattice = CartesianGrid(n)
    lrs = LatticeReactionSystem(model, [v_diffusion, u_diffusion], lattice)
    prob,transform = PseudoSpectralProblem(lrs, u0, tspan,ps,L,dt)
    int = init(prob, ETDRK4())
    int,transform





# u = [reshape(u,n,2) for u in sol.u]
# for u in u
#     plan! * u
# end
# u=stack(u;dims=3)
# plot_solutions(u, sol.t, ["u", "v"]; autolimits=true)
##

# odeprob1 = ODEProblem(lrs, [:U =>U0, :V=>V0], tspan, ps1, jac=true, sparse=true)
# sol1 = solve(odeprob1, saveat=dt, abstol=1e-10, reltol=1e-10)
# U1 = reshape(stack(lat_getu(sol1, :U, lrs)), n, 1, length(sol1))
# V1 = reshape(stack(lat_getu(sol1, :V, lrs)), n, 1, length(sol1))
# u1=hcat(U1,V1)
# #plot_solutions(u1, sol1.t, ["u", "v"]; autolimits=true)
# plot_solutions(hcat(u,u1), sol.t, ["u_ps", "v_ps", "u_fd", "v_fd"]; autolimits=true)
##
# function fsolveref(u0)
#     n=size(u0,1)
#     U0=copy(u0[:,1]); V0=copy(u0[:,2])
#     lattice = CartesianGrid(n)
#     lrs = LatticeReactionSystem(model, [v_diffusion, u_diffusion], lattice)
#     prob = ODEProblem(lrs, [:U =>U0, :V=>V0], tspan, ps1, jac=true, sparse=true)
#     sol = solve(prob, FBDF(); abstol=1e-12, reltol=1e-12)
#     reshape(sol[end], n, 2)
# end

# function fsolve(u0, dt)
#     n=size(u0,1)
#     lattice = CartesianGrid(n)
#     lrs = LatticeReactionSystem(model, [v_diffusion, u_diffusion], lattice)
#     (prob, plan!) = pseudospectral_problem(lrs, u0, tspan,ps,L,dt)
#     sol = solve(prob, ETDRK4())
#     u = reshape(sol[end],n,2)
#     plan! * u
#     u
# end

# ##

# dxs = range(0.200001,1.00001, length=1)
# dts = range(0.001,0.2, length=8)
# x = 0:dxs[1]/2:L
# u0 = 0.0001*randn(length(x),2).^2
# u1 = fsolveref(u0)
# errgrid = errormap(fsolve, x, u0, u1, dxs, dts)
# fig,ax,hm = heatmap(dxs,dts,errgrid; colormap=:reds)
# Colorbar(fig[:, end+1], hm)
# fig
##

# x = range(0,L,n)
# u0_itp = [cubic_spline_interpolation(x, u) for u in eachcol(u0)]

# function myinit(dx,dt)
#     x = 0:dx:L
#     u0 = stack(itp.(x) for itp in u0_itp) 
#     n=size(u0,1)
#     m=size(u0,2)

#     lattice = CartesianGrid(n)
#     lrs = LatticeReactionSystem(model, [v_diffusion, u_diffusion], lattice)
#     (prob, plan!) = pseudospectral_problem(lrs, u0, tspan,ps,L,dt)
#     int = init(prob, ETDRK4())
# end

# function transform(dx)
#     x = 0:dx:L
#     u0 = stack(itp.(x) for itp in u0_itp) 
#     n=size(u0,1)
#     m=size(u0,2)
#     (prob, plan!) = pseudospectral_problem(lrs, u0, tspan,ps,L,dt)
#     function(u)
#         u = reshape(copy(u),n,m)
#         plan! * u
#         u
#     end
# end

# function init_ref(dx)
#     x = 0:dx:L
#     u0 = stack(itp.(x) for itp in u0_itp) 
#     n=length(x)
#     U0=copy(u0[:,1]); V0=copy(u0[:,2])
#     lattice = CartesianGrid(n)
#     lrs = LatticeReactionSystem(model, [v_diffusion, u_diffusion], lattice)
#     prob = ODEProblem(lrs, [:U =>U0, :V=>V0], tspan, ps1, jac=true, sparse=true)
#     int = init(prob, FBDF())
# end

# tune(myinit,init_ref, 1.0,1.0)
# ##
# @btime solve(odeprob1);

# @btime solve(odeprob1, FBDF());
# @btime solve(odeprob, ETDRK4());

# FFTW.set_num_threads(8)
# @btime solve(odeprob, ETDRK4());
# ##

# f_d = build_d!(lrs,L)
# f_r = build_r!(lrs,plan!)
# @profview solve(odeprob, ETDRK4())

# @benchmark solve(odeprob, ETDRK4())
# odeprob1 = ODEProblem(lrs, [:U =>U0, :V=>V0], tspan, ps1, jac=true, sparse=true)
# @benchmark solve(odeprob1)
