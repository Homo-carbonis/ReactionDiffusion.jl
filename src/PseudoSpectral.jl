module PseudoSpectral

export PseudoSpectralProblem, solve

using DifferentialEquations, FFTW, Symbolics

struct PseudoSpectralProblem
    problem
    plan!
end

function transform(f!,plan!)
    function (du,u,p,t)
        du .= u
        plan! * du
        for i in 1:n
            f!(du[:,i],du[:,i],p,t) # 
        end
        plan! * du
    end
end


function PseudoSpectralProblem(lrs, u0, tspan, l, d, p; ss=true)
    sps=species(lrs)
    params = parameters(lrs)
    
    n_verts = Catalyst.num_verts(lrs)
    n_species = length(sps)
    n_params = length(params)

    plan! = 1/sqrt(2*(n-1)) * FFTW.plan_r2r!(u0, FFTW.REDFT00,  2) # Orthonormal DCT-I
    plan! * u0 # transform ICs


    rhs = Catalyst.assemble_oderhs(Catalyst.reactionsystem(lrs), sps)

    # Build optimized Jacobian and ODE functions using Symbolics.jl
    @variables u[1:n_species, 1:n_verts]

    du_r = mapslices(collect(u),dims=1) do u
            s = Dict(zip(sps, u))
            [substitute(expr, s) for expr in rhs]
    end
    f_r! = eval(Symbolics.build_function(du_n,u,params)[2]) # Index [2] denotes in-place function.
    f_r! = transform(f_r!, plan!)

    jac_r = Symbolics.sparsejacobian(vec(du_n),vec(u))
    fjac_r! = eval(Symbolics.build_function(jac_r,vec(u),params)[2])
    fjac_r! = transform(fjac_r!, plan!)

    D = [r.rate for r in Catalyst.spatial_reactions(lrs)]

    k = 0:n_verts-1
    h = l / (n_verts-1)
    λ = [-d * (4/h^2) * sin(k*pi/(2*(n-1)))^2 for d in d, k in k]

    du_d = λ .* u
    f_d! = eval(Symbolics.build_function(du_d,u)[2])
    jac_d = Symbolics.sparsejacobian(vec(du_d),vec(u))
    fjac_d! = eval(Symbolics.build_function(jac_d,vec(u))[2])
    odeprob = SplitODEProblem(f_d!, f_n!, u0, tspan, p)
    if ss
        odeprob = SteadyStateProblem(odeprob)
    end
    PseudoSpectralProblem(odeprob, plan!)
end

function DifferentialEquations.solve(problem::PseudoSpectralProblem, alg=KenCarp3(); kwargs...)
    (;problem, plan!) = problem
    alg = something(alg, KenCarp3())
    if problem isa SteadyStateProblem
        sol = solve(problem, DynamicSS(alg); kwargs...).original
    else
        sol = solve(problem, alg; kwargs...)
    end

    map!(sol) do u
        plan! * u
        permutedims(u)
    end
    sol
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

end

