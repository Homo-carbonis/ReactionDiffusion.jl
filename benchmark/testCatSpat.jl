#include("../src/PseudoSpectral.jl")
#using .PseudoSpectral
using Catalyst, DifferentialEquations, FFTW

include("../src/Plot.jl")
using .Plot, WGLMakie


function transform(f!,plan!, iplan! = plan!)
   u = Matrix{Float64}(undef, size(plan!))
    function (du,û,p,t)
        u .= û
        plan! * u
        f!(du,u,p,t)
        iplan! * du
        nothing
    end
end


"Return diffusion rates for `lrs` in the same order as `species(lrs)`"
function diffusion_rates(lrs::LatticeReactionSystem)
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


function make_params(network; params...)
    symbols = nameof.(parameters(network))
    [params[k] for k in symbols]
end


"Transform each value of `sol` from `u` to `f(u)"
function map!(f, sol::ODESolution)
    for i in eachindex(sol.u)
        sol.u[i] = f(sol.u[i])
    end
    for i in eachindex(sol.k)
        for j in eachindex(sol.k[i])
            sol.k[i][j] = f(sol.k[i][j])
        end
    end
end


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
##
tspan = (0.0, 1.0)

sps=species(lrs)

params = parameters(lrs)
reaction_params = vertex_parameters(lrs)
diffusion_params = diffusion_rates(lrs)

n_verts = Catalyst.num_verts(lrs)
n_species = length(sps)

ps = make_params(lrs; γ = 1.0, a = 0.2, b = 2.0, Dᵤ = 1.0, Dᵥ = 50.0)


u0 = 0.0001 * randn(n_verts, n_species).^2
u0_ = copy(u0)
U0=copy(u0[:,1]); V0=copy(u0[:,2])

L = 10


plan! = 1/sqrt(2*(n-1)) * FFTW.plan_r2r!(u0_, FFTW.REDFT00, 1; flags=FFTW.MEASURE) # Orthonormal DCT-I
jac_plan! = 1/sqrt(2*(n-1)) * FFTW.plan_r2r!(randn(n_verts*n_species,n_verts*n_species), FFTW.REDFT00, 1) # TODO Use real data to plan.
plan! * u0 # transform initial conditions


##

rhs = Catalyst.assemble_oderhs(Catalyst.reactionsystem(lrs), sps)

# Build optimized Jacobian and ODE functions using Symbolics.jl
@variables u[1:n_verts, 1:n_species]

du_r = mapslices(collect(u),dims=2) do u
        s = Dict(zip(sps, u))
        [substitute(expr, s) for expr in rhs]
end

jac_r = Symbolics.sparsejacobian(vec(du_r),vec(u))
f_r! = transform(eval(Symbolics.build_function(du_r,u,params,())[2]), plan!) # Index [2] denotes in-place function.
fjac_r! = transform(eval(Symbolics.build_function(jac_r,u,params,())[2]), plan!, jac_plan!)
jac_r_prototype = similar(jac_r, Float64)
fjac_r!(jac_r_protype, u0,ps,0.0)


f_r! = ODEFunction(f_r!; jac=fjac_r!, jac_prototype=jac_r_prototype)


k = 0:n_verts-1
h = L / (n_verts-1)

# Correction from -D(kh)^2 for the discrete transform.
λ = [-D * (4/h^2) * sin(k*pi/(2*(n-1)))^2 for k in k, D in diffusion_params]

du_d = collect(λ .* u)
jac_d = Symbolics.sparsejacobian(vec(du_d), vec(u))

f_d! = eval(Symbolics.build_function(du_d,u,params,())[2])


fjac_d! = eval(Symbolics.build_function(jac_d,u,params,())[2])
jac_d_prototype = similar(jac_d, Float64)
fjac_d!(jac_d_prototype, u0, ps, 0.0)


f_d! = ODEFunction(f_d!; jac=fjac_d!, jac_prototype=jac_d_prototype)


ps = make_params(lrs; γ = 1.0, a = 0.2, b = 2.0, Dᵤ = 1.0, Dᵥ = 50.0)
odeprob1 = SplitODEProblem(f_r!, f_d!, u0, tspan, ps)
odeprob2 = ODEProblem(lrs,[:U => U0, :V => V0],tspan,[:γ => 1.0, :a => 0.2, :b => 2.0, :Dᵤ => 1.0/L^2, :Dᵥ => 50.0/L^2]; jac=true, sparse=true)


@benchmark sol1 = solve(odeprob1, KenCarp3())
map!(sol1) do u
    plan! * u
end

sol2 = solve(odeprob2)

u2 = lat_getu(sol2, :U, lrs)
t = sol2.t
plot_solutions([sol1], ["u", "v"]; autolimits=true)