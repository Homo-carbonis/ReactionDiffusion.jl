#include("../src/PseudoSpectral.jl")
#using .PseudoSpectral
using Revise, Catalyst, DifferentialEquations, FFTW

includet("../src/Plot.jl")
using .Plot, WGLMakie



pseudospectral_problem(lrs, u0, tspan, p) = SplitODEProblem(build_d!(lrs), build_r!(lrs), u0,tspan, p)


function build_r!(lrs, plan!)
    n = Catalyst.num_verts(lrs)
    m = Catalyst.num_species(lrs)
    sps = species(lrs)
    p = parameters(lrs)
    rhs = Catalyst.assemble_oderhs(Catalyst.reactionsystem(lrs), sps)

    @variables u[1:n, 1:m]

    du = mapslices(u,dims=2) do v
            s = Dict(zip(sps, v))
            [substitute(expr, s) for expr in rhs]
    end
    jac = Symbolics.sparsejacobian(vec(du),vec(u))
    (f,f!) = Symbolics.build_function(du, u, p; expression=Val{false})
    (fjac,fjac!) = Symbolics.build_function(jac, vec(u), p, (); expression=Val{false})
    function f̂!(du,u,p,t)
        DU=reshape(du,n,m)
        U=reshape(copy(u),n,m)
        plan! * U
        f!(DU, U, p)
        plan! * DU
        nothing
    end
    f̂!
end

function build_d!(lrs, L=2pi)
    n = Catalyst.num_verts(lrs)
    m = Catalyst.num_species(lrs)
    p = parameters(lrs)
    dps = diffusion_parameters(lrs)
    k = 0:n-1
    h = L / (n-1)
    # Correction from -D(kh)^2 for the discrete transform.
    λ = vec([-D * (4/h^2) * sin(k*pi/(2*(n-1)))^2 for k in k, D in dps])
    (f,f!) = Symbolics.build_function(λ, p; expression=Val{false})

    λ0 = similar(λ, Float64)
    update!(λ,u,p,t) = f!(λ,p)
    DiagonalOperator(λ0; update_func! = update!)
end


function plan_dct1(u0)
    u = copy(u0)
    r = 1/sqrt(2*(n-1)) # normalisation factor
    plan = r * FFTW.plan_r2r(u, FFTW.REDFT00, 1; flags=FFTW.MEASURE)
    plan! = r * FFTW.plan_r2r!(u, FFTW.REDFT00, 1; flags=FFTW.MEASURE)
    (plan,plan!)
end


"Return diffusion rates for `lrs` in the same order as `species(lrs)`"
function diffusion_parameters(lrs::LatticeReactionSystem)
    sps = species(lrs)
    D::Vector{Num} = zeros(length(sps))
    srs = Catalyst.spatial_reactions(lrs)
    keys = getfield.(srs, :species)
    vals = getfield.(srs, :rate)
    for (k,v) in zip(keys,vals)
        i = findfirst(u -> u===k, sps)
        D[i] = v
    end
    D
end


"Build a vector of parameters in the right order from the given keyword values"
function make_params(network; params...)
    symbols = nameof.(parameters(network))
    [params[k] for k in symbols]
end

"Transform each value of `sol` from `u` to `f(u)"
function mapp!(f!, sol::ODESolution)
    for i in eachindex(sol.u)
        f!(sol.u[i])
    end
    for i in eachindex(sol.k)
        for j in eachindex(sol.k[i])
            f!(sol.k[i][j])
        end
    end
end

##


model = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 

v_diffusion = @transport_reaction Dᵥ V
u_diffusion = @transport_reaction Dᵤ U

n=64
lattice = CartesianGrid(n)
lrs = LatticeReactionSystem(model, [v_diffusion, u_diffusion], lattice)

tspan = (0.0, 1)
ps = make_params(lrs; γ = 1.0, a = 0.2, b = 2.0, Dᵤ = 1.0, Dᵥ = 20.0)
ps1 = [:γ => 1.0, :a => 0.2, :b => 2.0, :Dᵤ => 1.0, :Dᵥ => 50.0]

u0 = 0.0001 * randn(n, 2).^2
plan! = 1/sqrt(2*(n-1)) * FFTW.plan_r2r!(copy(u0), FFTW.REDFT00, 1; flags=FFTW.MEASURE)
plan! * u0
u0_ = copy(u0)
U0=copy(u0[:,1]); V0=copy(u0[:,2])

##
odeprob1 = ODEProblem(lrs, [:U =>U0, :V=>V0], tspan, ps1; jac=true, sparse=true)
#sol1 = solve(odeprob1)

#odeprob = pseudospectral_problem(lrs,u0,tspan,ps)
odeprob = SplitODEProblem(build_d!(lrs,10), build_r!(lrs,plan!), vec(u0),tspan, ps; dt=0.001)
sol = solve(odeprob, ETDRK3())
u = [reshape(u,n,2) for u in sol.u]
for u in u
    plan! * u
end
plot_solutions(u, sol.t, ["u", "v"]; autolimits=true)