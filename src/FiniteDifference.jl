
module FiniteDifference

using DifferentialEquations, LinearAlgebra, ModelingToolkit, Symbolics
export FiniteDifferenceProblem, solve

struct FiniteDifferenceProblem
    problem
end

"Discretise the model using a central difference scheme"
function FiniteDifferenceProblem(f, u0, l, d, p, t_span=(0,Inf))
    n_gridpoints = size(u0)[1]
    n_params = length(p)
    n_species = length(d)

    q_ = [p; d/(l^2)]
    du = similar(u0)

    # build PDE function
    M = collect(Tridiagonal([1.0 for i in 1:n_gridpoints-1],[-2.0 for i in 1:n_gridpoints],[1.0 for i in 1:n_gridpoints-1]))
    M[1,2] = M[end,end-1] = 2.0 # Neumann BCs

    function f_reflective(u,q)
        du_rxn = mapslices(u,dims=2) do x
            f(x,q[1:n_params],0.0)
        end
        du_rxn + q[(n_params + 1):(n_params + n_species)]' .* (n_gridpoints^2*M * u)
    end
    
    # Build optimized Jacobian and ODE functions using Symbolics.jl

    @variables uᵣ[1:n_gridpoints,1:n_species]
    @parameters qᵣ[1:(n_params+n_species)]
    
    duᵣ = Symbolics.simplify.(f_reflective(collect(uᵣ),collect(qᵣ)))
    


    fᵣ = eval(Symbolics.build_function(duᵣ,vec(uᵣ),qᵣ;
                parallel=Symbolics.SerialForm(),expression = Val{false})[2]) #index [2] denotes in-place, mutating function
    jacᵣ = Symbolics.sparsejacobian(vec(duᵣ),vec(uᵣ))
    fjacᵣ = eval(Symbolics.build_function(jacᵣ,vec(uᵣ),qᵣ,
                parallel=Symbolics.SerialForm(),expression = Val{false})[2]) #index [2] denotes in-place, mutating function

                
    prob_fn = ODEFunction((du,u,q,t)->fᵣ(du,vec(u),q), jac = (du,u,q,t) -> fjacᵣ(du,vec(u),q), jac_prototype = similar(jacᵣ,Float64))

    
    prob = ODEProblem(prob_fn,u0,t_span,q_)
    FiniteDifferenceProblem(prob)
end

function DifferentialEquations.solve(problem::FiniteDifferenceProblem, alg=KenCarp4(); kwargs...)
    alg = something(alg, KenCarp4())
    problem = problem.problem

    if problem.tspan[end] < Inf
        solve(problem, alg; kwargs...)
    else
        solve(SteadyStateProblem(problem), DynamicSS(alg); kwargs...).original
    end
end

end