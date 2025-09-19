module PseudoSpectral

export PseudoSpectralProblem, solve

using DifferentialEquations, FFTW

struct PseudoSpectralProblem
    problem
    plan!
end

function PseudoSpectralProblem(f, u0, tspan, l, d, p; ss=true)
    n = size(u0)[1]
    u0 = permutedims(u0)

    plan! = 1/sqrt(2*(n-1)) * FFTW.plan_r2r!(u0, FFTW.REDFT00,  2) # Orthonormal DCT-I

    plan! * u0

    k = 0:n-1
    h = l / (n-1)
    λ = [-d * (4/h^2) * sin(k*pi/(2*(n-1)))^2 for d in d, k in k]

    function f_d!(du,u,p,t)
        du .= λ .* u
    end

    function f_n!(du,u,p,t)
        du .= u
        plan! * du

        for i in 1:n
            du[:,i] = f(du[:,i],p,t)
        end
        
        plan! * du
    end

    odeprob = SplitODEProblem(f_d!, f_n!, u0, tspan, p)
    if ss
        odeprob = SteadyStateProblem(odeprob)
    end
    PseudoSpectralProblem(odeprob, plan!)
end

function DifferentialEquations.solve(problem::PseudoSpectralProblem, alg=KenCarp3(autodiff=false); kwargs...)
    (;problem, plan!) = problem
    alg = something(alg, KenCarp3(autodiff=false))
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

