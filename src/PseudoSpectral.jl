module PseudoSpectral

export pseudospectral_problem, remake_params

using SciMLBase, FFTW, Symbolics

"Construct a SplitODEProblem to solve `lrs` with reflective boundaries using a pseudo-spectral method.
Returns the SplitODEProblem with solutions in the frequency (DCT-1) domain and a FFTW plan to transform solutions back to the spatial domain."
function pseudospectral_problem(species, reaction_rates, diffusion_rates, u0, tspan, p; kwargs...)
    u0 = copy(u0)
    n = size(u0,1)
    m = size(u0,2)
    plan! = 1/sqrt(2*(n-1)) * FFTW.plan_r2r!(copy(u0), FFTW.REDFT00, 1; flags=FFTW.MEASURE)
    plan! * u0
    p = sort(p; by=nameof) # sort parameters
    R = build_r!(species, reaction_rates, u0, p.keys, plan!)
    D = build_d!(diffusion_rates, u0, p.keys)

    ps = make_params(u0, p) 
    update_coefficients!(D,u0,ps,0.0) # Must be called before first step.
    update_coefficients!(R,u0,ps,0.0)
    prob = SplitODEProblem(D, R, vec(u0), tspan, ps; kwargs...)
    function transform(u)
        u = reshape(copy(u),n,m)
        plan! * u
        u
    end
    prob, transform
end

remake_params(prob, u0, p) = remake(prob; p=make_params(u0, p))


"Build function for the reaction component."
function build_r!(species, reaction_rates, u0, params, plan!)
    (n,m) = size(u0)
    @variables u[1:n, 1:m]
    @variables p[1:n, 1:length(params)]
    # Do clever things to make only spatially varying parameters expand.
    for (v,q) in eachrow.((u,p))
        dict = Dict(zip(species,v)..., zip(params,q)...)
        [substitute(expr, dict) for expr in reaction_rates]
    end
    
    jac = Symbolics.sparsejacobian(vec(du),vec(u); simplify=true)
    (f,f!) = Symbolics.build_function(du, u, p; expression=Val{false})
    (fjac,fjac!) = Symbolics.build_function(jac, vec(u), p, (); expression=Val{false})
    function f̂!(du,u,p,t)
        DU=reshape(du,n,m)
        U = last(p)
        U .= reshape(u,n,m)
        q = first(p)
        plan! * U
        f!(DU, U, q)
        plan! * DU
        nothing
    end
    ODEFunction(f̂!; jac=fjac!)
end

"Build linear operator for the diffusion component."
function build_d!(diffusion_rates, u0, params, plan!)
    n = size(u0,1)
    @variables p[1:n, 1:length(params)]
    for q in eachrow(p)
        dict = Dict(zip(params,q)...)
        [substitute(D, dict) for D in diffusion_rates]
    end
    # Compute DCT of D *when params change*

    k = 0:n-1
    h = 1 / (n-1) # 2pi?

    
    # Correction from -D(k/2pi h)^2 for the discrete transform.
    λ = vec([-D * (4/h^2) * sin(k*pi/(2*(n-1)))^2 for k in k, D in diffusion_rates])

    (f,f!) = Symbolics.build_function(λ, params; expression=Val{false})
    λ0 = similar(λ, Float64)
    update!(λ,u,p,t) = f!(λ,p[1:end-1]) # Drop working memory at end of p.
    DiagonalOperator(λ0; update_func! = update!)
end


"Build an ordered tuple of parameters from the given keyword values."
function make_params(u0, params)
    U = similar(u0) # allocate working memory for computing dct.
    p = sort(params; by=nameof).vals
    (p...,U)
end

"Return diffusion rates for `lrs` in the same order as `species(lrs)` with a default of 0"
function diffusion_rates(lrs::LatticeReactionSystem)
    sps = species(lrs)
    D::Vector{Num} = zeros(length(sps))
    for r in Catalyst.spatial_reactions(lrs)
        i = findfirst(u -> u.f === r.species.f, sps)
        D[i] = r.rate
    end
    D
end


end

