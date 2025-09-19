using DifferentialEquations, FFTW, Catalyst, Symbolics, BenchmarkTools
includet("../src/Plot.jl")
using .Plot
using WGLMakie

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


p=[0.2,2.0,1.0]
d=[1.0,50]
l = 1
n =128
# convert reaction network to ODESystem
odesys = convert(ODESystem, model)

# build ODE function
f_gen = ModelingToolkit.generate_function(odesys,expression = Val{false})[1] #false denotes function is compiled, world issues fixed
f = ModelingToolkit.eval(f_gen)



function f_reflective(u,p)
    mapslices(u,dims=1) do x
        f(x,p,0.0)
    end
end

# Build optimized Jacobian and ODE functions using Symbolics.jl
n_gridpoints = n
n_species = 2
n_params = 3

@variables uᵣ[1:n_species,1:n_gridpoints]
@parameters pᵣ[1:n_params]

duᵣ = Symbolics.simplify.(f_reflective(collect(uᵣ),collect(pᵣ)))



fᵣ = eval(Symbolics.build_function(duᵣ,vec(uᵣ),pᵣ;
            parallel=Symbolics.SerialForm(),expression = Val{false})[2]) #index [2] denotes in-place, mutating function
jacᵣ = Symbolics.sparsejacobian(vec(duᵣ),vec(uᵣ))
fjacᵣ = eval(Symbolics.build_function(jacᵣ,vec(uᵣ),pᵣ,
            parallel=Symbolics.SerialForm(),expression = Val{false})[2]) #index [2] denotes in-place, mutating function

            
prob_fn = ODEFunction((du,u,p,t)->fᵣ(du,vec(u),p), jac = (du,u,p,t) -> fjacᵣ(du,vec(u),p), jac_prototype = similar(jacᵣ,Float64))





u0 = randn(n,2).^2

tspan = (0,10)

u0 = permutedims(u0)

r = 1/sqrt(2*(n-1))
plan! = r * FFTW.plan_r2r!(u0, FFTW.REDFT00,  2) # Orthonormal DCT-I

plan! * u0

k = 0:n-1
h = l / (n-1)
λ = [-d * (4/h^2) * sin(k*pi/(2*(n-1)))^2 for d in d, k in k]

function f_d!(du,u,p,t)
    du .= λ .* u
end

uu=similar(u0)
function f_n!(du,u,p,t)
    uu .= u
    plan! * uu
    fᵣ(du,vec(uu),p)
    plan! * du
end

problem = SplitODEProblem(f_d!, f_n!, u0, tspan, p)

sol = solve(problem, KenCarp3())
#@btime solve(problem, KenCarp3())


map!(sol) do u
    plan! * u
    permutedims(u)
end

sol
plot_solutions([sol],["u","v"])