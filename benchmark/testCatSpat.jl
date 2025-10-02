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

function DCT(lrs)
    n = Catalyst.num_verts(lrs)
    u = Vector{Float64}(undef,n)
    plan = 1/sqrt(2*(n-1)) * FFTW.plan_r2r(u, FFTW.REDFT00; flags=FFTW.MEASURE)
    op(v,u,p,t) = plan * v
    op(w,v,u,p,t) = mul!(w,plan,v)
    FunctionOperator(op, u, u; op_inverse=op, islinear=true, isconstant=true)
end

function R(lrs)
    sps = species(lrs)
    ps = parameters(lrs)
    rhs = Catalyst.assemble_oderhs(Catalyst.reactionsystem(lrs), sps)
    (f, f!) = Symbolics.build_function(rhs,sps,ps; expr=Val{false})
    op(v,u,p,t) = f(v,p)
    op(w,v,u,p,t) = f!(w,v,p)
    u = similar(sps, Float64)
    p = similar(ps, Float64)
    FunctionOperator(op, u, u; p=p)
end

function Λ(lrs)
    n = Catalyst.num_verts(lrs)
    k = 0:n-1
    h = L / (n-1)
    # Correction from -D(kh)^2 for the discrete transform.
    λ = [-(4/h^2) * sin(k*pi/(2*(n-1)))^2 for k in k]
    DiagonalOperator(λ)
end
function D(lrs)
    ps = parameters(lrs)
    dps = diffusion_parameters(lrs)
    update! = Symbolics.build_function(dps,(), ps,(); expr=Val{false})[2]
    diag = similar(dps, Float64)
    DiagonalOperator(diag, update_func! = update!)
end

function broadcast_operator(F, V; reverse=Val{false})
    if reverse
        n = size(F,1); m = size(V,2)
    else
        n = size(V,1); m = size(F,2)
    end
    function op!(W,V,u,p,t)
        for i in axes(W,2)
            if reverse
                v = view(v, i, :); w = view(w, i, :)
            else
                v = view(v, :, i); w = view(w, :, i)
            end
            mul!(w,F,v)
        end
    end
    function op(V,u,p,t)
        W = similar(V, n, m)
        op!(W,V,u,p,t)
        W
    end

    W = similar(V,n,m)
    FunctionOperator(op, W, V; p = F.p, u = F.u, islinear=islinear(F), isconstant=isconstant(F))
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

##
R = reaction_operator(lrs)
D = diffusion_operator(lrs)

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