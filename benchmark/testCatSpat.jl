#include("../src/PseudoSpectral.jl")
#using .PseudoSpectral
using Catalyst, DifferentialEquations, FFTW

includet("../src/Plot.jl")
using .Plot, WGLMakie


function transform(f!,plan!)
    u = Matrix(undef, size(plan!)...)
    function (du,û,p,t)
        u .= û
        plan! * u
        f!(du,u,p,t)
        plan! * du
    end
end


model = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 

u_diffusion = @transport_reaction 1.0 U
v_diffusion = @transport_reaction 50.0 V

n=128
lattice = CartesianGrid(n)
lrs = LatticeReactionSystem(model, [u_diffusion, v_diffusion], lattice)
##
tspan = (0.0, 10.0)
ps = [:γ => 1.0, :a => 0.2, :b => 2.0]

sps=species(lrs)
params = parameters(lrs)

n_verts = Catalyst.num_verts(lrs)
n_species = length(sps)
n_params = length(params)
u0 = randn(n_species, n_verts)
l =1 


plan! = 1/sqrt(2*(n-1)) * FFTW.plan_r2r!(u0, FFTW.REDFT00,  2) # Orthonormal DCT-I
plan! * u0 # transform ICs


rhs = Catalyst.assemble_oderhs(Catalyst.reactionsystem(lrs), sps)

# Build optimized Jacobian and ODE functions using Symbolics.jl
@variables u[1:n_species, 1:n_verts]

du_r = mapslices(collect(u),dims=1) do u
        s = Dict(zip(sps, u))
        [substitute(expr, s) for expr in rhs]
end

jac_r = Symbolics.sparsejacobian(vec(du_r),vec(u))

f_r! = eval(Symbolics.build_function(du_r,u,params,())[2]) # Index [2] denotes in-place function.
fjac_r! = eval(Symbolics.build_function(jac_r,vec(u),params,())[2])

f_r! = ODEFunction(transform(f_r!, plan!); jac = transform(fjac_r!, plan!))

D = [r.rate for r in Catalyst.spatial_reactions(lrs)]

k = 0:n_verts-1
h = l / (n_verts-1)
λ = [-D * (4/h^2) * sin(k*pi/(2*(n-1)))^2 for D in D, k in k]

du_d = λ .* u
jac_d = Symbolics.sparsejacobian(vec(du_d),vec(u))

f_d! = eval(Symbolics.build_function(du_d,u,(),())[2]) # Index [1] denotes in-place function.
fjac_d! = eval(Symbolics.build_function(jac_d,vec(u),(),())[2])

f_d! = ODEFunction(f_d!; jac = fjac_d!)
odeprob = SplitODEProblem(f_d!, f_r!, u0, tspan, params)
sol = solve(odeprob)





# odeprob = ODEProblem(lrs,u0,tspan,ps; jac=true, sparse=true)

# sol = solve(odeprob, KenCarp4())
# plot_solutions([sol], ["u"])