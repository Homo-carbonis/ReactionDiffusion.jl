#include("../src/PseudoSpectral.jl")
#using .PseudoSpectral
using Catalyst, DifferentialEquations, FFTW

include("../src/Plot.jl")
using .Plot, WGLMakie

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

# Extend * for operator × matrix
function Base.:*(A::SciMLOperators.AbstractSciMLOperator, X::AbstractMatrix)
    Y = similar(X, eltype(A), size(A,1), size(X,2))
    for j in 1:size(X,2)
        mul!(view(Y, :, j), A, view(X, :, j))
    end
    return Y
end

function build_dct_op(lrs)
    n = Catalyst.num_verts(lrs)
    m = Catalyst.num_species(lrs)
    v = Vector{Float64}(undef,n)
    u = Matrix{Float64}(undef, n, m)
    params = parameters(lrs) 
    p = similar(params, Float64) # We have to pass p as a prototype even though it's not used because isconstant doesn't seem to do anything.
    plan = 1/sqrt(2*(n-1)) * FFTW.plan_r2r(v, FFTW.REDFT00, 1; flags=FFTW.MEASURE)
    w = plan * v
    op(v,u,p,t) = plan * v
    op(w,v,u,p,t) = mul!(w,plan,v)
    FunctionOperator(op, v, w; u = u, p=p, op_inverse=op, islinear=true, isconstant=true)
end

function build_reaction_op(lrs)
    n = Catalyst.num_verts(lrs)
    m = Catalyst.num_species(lrs)
    sps = species(lrs)
    params = parameters(lrs)
    rhs = Catalyst.assemble_oderhs(Catalyst.reactionsystem(lrs), sps)
    (f, f!) = eval.(Symbolics.build_function(rhs,sps,params))
    op(v,u,p,t) = f(v,p)
    op(w,v,u,p,t) = f!(w,v,p)
    v = w = similar(sps, Float64)
    u = Matrix{Float64}(undef, n, m)
    p = similar(params, Float64)

    FunctionOperator(op, v, w; u=u, p=p)
end

function build_diffusion_op(lrs)
    sps = species(lrs)
    params = parameters(lrs)
    rhs = Catalyst.assemble_oderhs(Catalyst.reactionsystem(lrs), sps)
    n_verts = Catalyst.num_verts(lrs)
    n_sps = Catalyst.num_species(lrs)

    k = 0:n_verts-1
    L = 2pi
    h = L / (n_verts-1)

    # Correction from -D(kh)^2 for the discrete transform.
    λ = [-D * (4/h^2) * sin(k*pi/(2*(n-1)))^2 for k in k, D in diffusion_params]
    (f, f!) = eval.(Symbolics.build_function(λ,params))
    update(diag,u,p,t) = f(p)
    update!(diag,u,p,t) = f!(diag,p)
    prototype = similar(λ, Float64)
    DiagonalOperator(prototype; update_func! = update!)
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


u0 = 0.0001 * randn(n_verts, n_species).^2
u0_ = copy(u0)
U0=copy(u0[:,1]); V0=copy(u0[:,2])

##

dct = build_dct_op(lrs)
R = build_reaction_op(lrs)
d = build_diffusion_op(lrs)

D = dct * d * dct
cache_operator(D,u0)
odeprob1 = SplitODEProblem(D, R, u0, tspan, ps)
# odeprob2 = ODEProblem(lrs,[:U => U0, :V => V0],tspan,[:γ => 1.0, :a => 0.2, :b => 2.0, :Dᵤ => 1.0/L^2, :Dᵥ => 50.0/L^2]; jac=true, sparse=true)


sol1 = solve(odeprob1, KenCarp3())
map!(sol1) do u
    plan! * u
end

sol2 = solve(odeprob2)

u2 = lat_getu(sol2, :U, lrs)
t = sol2.t
plot_solutions([sol1], ["u", "v"]; autolimits=true)