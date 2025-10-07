module PseudoSpectral

export PseudoSpectralProblem, solve

using DifferentialEquations, FFTW, Symbolics

struct PseudoSpectralProblem
    problem
    plan!
    num_verts
    num_species
end



function PseudoSpectralProblem(lrs, u0, tspan, p, L, dt; steady_state=false)
    u0 = copy(u0)
    n = size(u0,1)
    plan! = 1/sqrt(2*(n-1)) * FFTW.plan_r2r!(copy(u0), FFTW.REDFT00, 1; flags=FFTW.MEASURE)
    plan! * u0
    U = similar(u0)
    ps = (p...,U)
    D = build_d!(lrs,L)
    R = build_r!(lrs,plan!)
    update_coefficients!(D,u0,ps,0.0) # Must be called before first step.
    update_coefficients!(R,u0,ps,0.0)
    odeprob = SplitODEProblem(D, R, vec(u0),tspan, (ps...,U), dt=dt)
    prob = steady_state ? SteadyStateProblem(odeprob) : odeprob
    PseudoSpectralProblem(prob,plan!)
end


function solve(problem::PseudoSpectralProblem, alg=ETDRK4(); kwargs...)
    (;problem, plan!) = problem
    alg = something(alg, ETDRK4())
    if problem isa SteadyStateProblem
        sol = solve(problem, DynamicSS(alg); kwargs...).original
    else
        sol = solve(problem, alg; kwargs...)
    end

    map!(sol) do u
        reshape(u,)
        plan! * u
    end

    sol
end

"Build function for the reaction component."
function build_r!(lrs, plan!)
    n = Catalyst.num_verts(lrs)
    m = Catalyst.num_species(lrs)
    sps = species(lrs)
    p = parameters(lrs)
    rhs = Catalyst.assemble_oderhs(Catalyst.reactionsystem(lrs), sps)

    @variables u[1:n, 1:m]

    du = mapslices(u,dims=2) do v
            s = Dict(zip(sps, v))
            [substitute(expr, s) for expr in rhs]
    end
    jac = Symbolics.sparsejacobian(vec(du),vec(u); simplify=true)
    (f,f!) = Symbolics.build_function(du, u, p; expression=Val{false})
    (fjac,fjac!) = Symbolics.build_function(jac, vec(u), p, (); expression=Val{false})
    function f̂!(du,u,p,t)
        DU=reshape(du,n,m)
        U = p[end]
        U .= reshape(u,n,m)
        q = p[1:end-1]
        plan! * U
        f!(DU, U, q)
        plan! * DU
        nothing
    end
    ODEFunction(f̂!; jac=fjac!)
end

"Build linear operator for the diffusion component."
function build_d!(lrs, L=2pi)
    n = Catalyst.num_verts(lrs)
    m = Catalyst.num_species(lrs)
    p = parameters(lrs)
    dps = diffusion_parameters(lrs)
    k = 0:n-1
    h = L / (n-1)
    # Correction from -D(kh)^2 for the discrete transform.
    λ = vec([-D * (4/h^2) * sin(k*pi/(2*(n-1)))^2 for k in k, D in dps])
    (f,f!) = Symbolics.build_function(λ, p; expression=Val{false})
    λ0 = similar(λ, Float64)
    update!(λ,u,p,t) = f!(λ,p[1:end-1])
    DiagonalOperator(λ0; update_func! = update!)
end


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


"Build a vector of parameters from the given keyword values in the same order as `parameters(network)`"
function make_params(network; params...)
    symbols = nameof.(parameters(network))
    Tuple(params[k] for k in symbols)
end

"Transform each value in `sol` from `u` to `f(u)"
function map!(f!, sol::ODESolution)
    for i in eachindex(sol.u)
        f!(sol.u[i])
    end
    for i in eachindex(sol.k)
        for j in eachindex(sol.k[i])
            f!(sol.k[i][j])
        end
    end
end


end

