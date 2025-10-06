module PseudoSpectral

export PseudoSpectralProblem, solve

using DifferentialEquations, FFTW, Symbolics

struct PseudoSpectralProblem
    problem
    plan!
end





function PseudoSpectralProblem(lrs, u0, p, tspan; L=2pi, dt=0.01)
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
    prob = SplitODEProblem(D, R, vec(u0),tspan, (ps...,U), dt=dt)
    PseudoSpectralProblem(prob, plan!)
end


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


"Build a vector of parameters in the right order from the given keyword values"
function make_params(network; params...)
    symbols = nameof.(parameters(network))
    Tuple(params[k] for k in symbols)
end

"Transform each value of `sol` from `u` to `f(u)"
function mapp!(f!, sol::ODESolution)
    for i in eachindex(sol.u)
        f!(sol.u[i])
    end
    for i in eachindex(sol.k)
        for j in eachindex(sol.k[i])
            f!(sol.k[i][j])
        end
    end
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
    n_species = Catalyst.num_species(lrs)
    u0_ = copy(u0)
    plan! = 1/sqrt(2*(n-1)) * FFTW.plan_r2r!(u0_, FFTW.REDFT00,  2; flags=FFTW.MEASURE) # Orthonormal DCT-I
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

