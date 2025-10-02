#include("../src/PseudoSpectral.jl")
#using .PseudoSpectral
using Catalyst, DifferentialEquations, FFTW

include("../src/Plot.jl")
using .Plot, WGLMakie

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


function build_r!(lrs)
    sps = species(lrs)
    ps = parameters(lrs)
    rhs = Catalyst.assemble_oderhs(Catalyst.reactionsystem(lrs), sps)
    @variables u[1:n_species, 1:n_verts]

    du = mapslices(u,dims=2) do u
            s = Dict(zip(sps, u))
            [substitute(expr, s) for expr in rhs]
    end
    (f, f!) = Symbolics.build_function(du,sps,ps; expr=Val{false})
    (du,u,p,t) -> f!(du,u,p)
end

function build_d!(lrs, L=2pi)
    n = Catalyst.num_verts(lrs)
    m = Catalyst.num_species(lrs)
    ps = parameters(lrs)
    dps = diffusion_parameters(lrs)
    k = 0:n-1
    h = L / (n-1)
    # Correction from -D(kh)^2 for the discrete transform.
    λ = [-D * (4/h^2) * sin(k*pi/(2*(n-1)))^2 for k in k, D in dps]
    @variables u[1:n_species, 1:n_verts]
    (f,f!) = Symbolics.build_function(λ.*u, u, ps; expression=Val{false})
    u0 = Matrix{Float64}(undef,n,m)
    r = 1/sqrt(2*(n-1))
    plan = r * FFTW.plan_r2r(u0, FFTW.REDFT00; flags=FFTW.MEASURE)
    plan! = r * FFTW.plan_r2r!(u0, FFTW.REDFT00; flags=FFTW.MEASURE)
    function(du,u,p,t)
        û = plan * u
        f!(du,û,p,t)
        plan! * du
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

tspan = (0.0, 1.0)
ps = make_params(lrs; γ = 1.0, a = 0.2, b = 2.0, Dᵤ = 1.0, Dᵥ = 50.0)


u0 = 0.0001 * randn(n, 2).^2
u0_ = copy(u0)
U0=copy(u0[:,1]); V0=copy(u0[:,2])
odeprob1 = ODEProblem(lrs, U0, tspan, ps; jac=true, sparse=true)
sol1 = solve(odeprob1)

odeprob = SplitODEProblem(build_d!(lrs), build_r!(lrs), u0,tspan,ps; jac=true, sparse=true)
sol = solve(odeprob)

u2 = lat_getu(sol2, :U, lrs)
t = sol2.t
plot_solutions([sol], ["u", "v"]; autolimits=true)