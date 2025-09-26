#include("../src/PseudoSpectral.jl")
#using .PseudoSpectral
using Catalyst, DifferentialEquations, FFTW

includet("../src/Plot.jl")
using .Plot, WGLMakie


function transform(f!,plan!)
    u = Matrix(undef, size(plan!)...) #TODO Preallocate
    function (du,û,p,t)
        u .= û
        plan! * u
        f!(du,u,p,t)
        plan! * du
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



model = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 

v_diffusion = @transport_reaction Dᵥ V
u_diffusion = @transport_reaction Dᵤ U

n=128
lattice = CartesianGrid(n)



lrs = LatticeReactionSystem(model, [v_diffusion, u_diffusion], lattice)
##
tspan = (0.0, 10.0)

sps=species(lrs)

params = parameters(lrs)
reaction_params = vertex_parameters(lrs)
diffusion_params = diffusion_rates(lrs)

n_verts = Catalyst.num_verts(lrs)
n_species = length(sps)

u0 = randn(n_species, n_verts)
L = 1


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


k = 0:n_verts-1
h = L / (n_verts-1)
# Correction from -D k^2 for the discrete transform.
λ = [-D * (4/h^2) * sin(k*pi/(2*(n-1)))^2 for D in diffusion_params, k in k]

du_d = λ .* u
jac_d = Symbolics.sparsejacobian(vec(du_d),vec(u)) # f_d! is linear so supplying jac shouldn't make any differnce, but let's try anyway.

f_d! = eval(Symbolics.build_function(du_d,u,params,())[2])
fjac_d! = eval(Symbolics.build_function(jac_d,vec(u),params,())[2])

f_d! = ODEFunction(f_d!; jac = fjac_d!)

ps = make_params(lrs; γ = 1.0, a = 0.2, b = 2.0, Dᵤ = 1.0, Dᵥ = 50.0)
odeprob = SplitODEProblem(f_d!, f_r!, u0, tspan, ps)
sol = solve(odeprob)





# odeprob = ODEProblem(lrs,u0,tspan,ps; jac=true, sparse=true)

# sol = solve(odeprob, KenCarp4())
# plot_solutions([sol], ["u"])