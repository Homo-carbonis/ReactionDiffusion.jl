module PseudoSpectral

export pseudospectral_problem
using ..Util: collect_params
using SciMLBase: SplitODEProblem, DiagonalOperator, ODEFunction, update_coefficients!, remake
using FFTW: plan_r2r!, REDFT00, MEASURE
using Symbolics: @variables, sparsejacobian, build_function, substitute


"Construct a SplitODEProblem to solve a reaction diffusion system with reflective boundaries.
Returns the SplitODEProblem with solutions in the frequency (DCT-1) domain and a FFTW plan to transform solutions back to the spatial domain."
function pseudospectral_problem(species, reaction_rates, diffusion_rates, boundary_conditions, num_verts; kwargs...)
    n = num_verts
    m = length(species)
    u0 = Matrix{Float64}(undef, n, m)
    plan! = 1/sqrt(2*(n-1)) * plan_r2r!(u0, REDFT00, 1; flags=MEASURE)
  
    k = 0:n-1 # Wavenumbers
    h = 1/(n-1)

    ## 2nd order Fourier differentiation coefficients.
    # For a continuous FT this would be σ² = -(k/2pi h)^2, but corrected for
    # the discrete transform this becomes:
    σ² = @. -(4/h^2) * sin(k*pi/(2*(n-1)))^2 
    
    ## Offsets for non-zero-flux boundary conditions.
    # For u′(0) = a, u′(1) = b,
    # define ϕ as a smooth function so that ϕ′(0) = a, ϕ′(1) = b, and write v = u - ϕ.
    # Then v′(0) = 0, v′(1) = 0, so we can solve for v using DCT-I.
    a, b = .-boundary_conditions # Why the sign change? Something to do with ghost points of DCT?
    ϕ0 = [a * x + (b-a)/2 * x^2 for x in range(0.0,1.0,n)] 
    ϕ = stack(ϕ0 for i in 1:m) # Should change for each species, but repeat the same bcs for now.
    Φ = copy(ϕ)
    plan! * Φ
    Φ .*= σ² # Φ = Δϕ̃

    rs = collect_params(reaction_rates, species)
    ds = collect_params(diffusion_rates, species)
    R = reaction_operator(species, reaction_rates, rs, ϕ, Φ, plan!)
    D = diffusion_operator(diffusion_rates, ds, σ²)

    prob = SplitODEProblem(D, R, vec(u0), Inf, nothing; kwargs...)

    # Function to set parameter values.
    function make_problem(params, state=nothing; kwargs...)
        r = stack(params[k] for k in rs)
        d = [params[k][1] for k in ds] # Assume D is homogeneous for now.
        u0 = stack(params[k] for k in species)
        u0 .-= ϕ
        plan! * u0
        u0 = vec(u0)
        u = Matrix{Float64}(undef, n, m) # Allocate working memory for dct.
        p = Parameters(u,r,d,state)
        update_coefficients!(prob.f.f1.f, u0, p, 0.0) # Set parameter values in diffusion operator.
        remake(prob; p=p, u0=u0, kwargs...) # Set parameter values in SplitODEProblem.
    end     

    # Function to transform output back to spatial domain.
    function transform(u)
        u = reshape(copy(u),n,m)
        plan! * u
        u + ϕ
    end

    make_problem, transform
end

"Build function for the reaction component, with `f(v+ϕ) + Φ` offset for non-zero-flux BCs."
function reaction_operator(species, reaction_rates, ps, ϕ, Φ, plan!)
    (n,m) = size(plan!)
    @variables u[1:n, 1:m]
    @variables p[1:n, 1:length(ps)]
    # Do clever things to make only spatially varying parameters expand?
    # Build an nxm matrix of derivatives, substituting reactants for u[i,j] and parameters for p[k,l].
    du = stack([substitute(expr, Dict([zip(species,v)..., zip(ps,q)...])) for expr in reaction_rates] for (v,q) in zip(eachrow(u),eachrow(p)); dims=1)
    jac = sparsejacobian(vec(du),vec(u); simplify=true)
    (f,f!) = build_function(du, u, p; expression=Val{false})
    (fjac,fjac!) = build_function(jac, vec(u), p, (); expression=Val{false})

    function f̂!(du,u,p,t)
        du=reshape(du,n,m)
        p.u .= reshape(u,n,m)
        plan! * p.u
        p.u .+= ϕ
        f!(du, p.u, p.r)
        plan! * du
        du .+= Φ
        nothing
    end
    ODEFunction(f̂!; jac=fjac!)
end

"Build linear operator for the diffusion component."
function diffusion_operator(diffusion_rates, ps, σ²)
    λ = vec(σ² * diffusion_rates')
    (f,f!) = build_function(λ, ps; expression=Val{false})
    λ0 = similar(λ, Float64)
    update!(λ,u,p,t) = f!(λ, p.d)
    DiagonalOperator(λ0; update_func! = update!)
end


struct Parameters
    u # Working array for dct.
    r # Reaction parameter matrix.
    d # Diffusion parameter vector.
    state # Only used for metadata.
end

end

