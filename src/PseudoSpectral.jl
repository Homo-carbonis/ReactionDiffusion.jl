module PseudoSpectral

export pseudospectral_problem

using SciMLBase, FFTW, Symbolics

"Construct a SplitODEProblem to solve `lrs` with reflective boundaries using a pseudo-spectral method.
Returns the SplitODEProblem with solutions in the frequency (DCT-1) domain and a FFTW plan to transform solutions back to the spatial domain."
function pseudospectral_problem(species, reaction_rates, diffusion_rates, num_verts, tspan; kwargs...)
    n = num_verts
    m = length(species)
    u0 = Matrix{Float64}(undef, n,m)
    plan! = 1/sqrt(2*(n-1)) * FFTW.plan_r2r!(copy(u0), FFTW.REDFT00, 1; flags=FFTW.MEASURE)

    # Get arrays of Symbolics variables for reaction and diffusion parameters.
    rs = sort_params(setdiff(union(Symbolics.get_variables.(reaction_rates)...), species))
    ds = sort_params(union(Symbolics.get_variables.(diffusion_rates)...))
    
    R = build_r!(species, reaction_rates, n, rs, plan!)
    D = build_d!(diffusion_rates, n, ds)


    

    # update_coefficients!(D,u0,ps,0.0) # Must be called before first step. TODO Determine whether replaced by remake_params.
    # update_coefficients!(R,u0,ps,0.0)

    prob = SplitODEProblem(D, R, vec(u0), tspan, nothing; kwargs...)

    # Function to set parameter values.
    function make_problem(p, u0, state=nothing; kwargs...)
        r = [p[k] for k in rs]
        d = [p[k] for k in ds]
        u0 = [u0[k] for s in species]
        r = build_matrix(r,n)
        any(p isa Function for p in d) && error("Spatially dependent diffusion parameters are not supported.")
        u0 = build_matrix(u0,n)
        u = similar(u0) # Allocate working memory for dct.
        params = Parameters(u,r,d,state)
        update_coefficients!(prob.f.f1.f, u0, params, 0.0) # Set parameter values in diffusion operator.
        remake(prob; u0=u0, p=params, kwargs...) # Set parameter values in SplitODEProblem.
    end     

    # Function to transform output back to spatial domain.
    function transform(u)
        u = reshape(copy(u),n,m)
        plan! * u
        u
    end

    make_problem, transform
end


"Build function for the reaction component."
function build_r!(species, reaction_rates, n, ps, plan!)
    m = length(species)
    @variables u[1:n, 1:m]
    @variables p[1:n, 1:length(ps)]
    # Do clever things to make only spatially varying parameters expand?
    # Build an nxm matrix of derivatives, substituting reactants for u[i,j] and parameters for p[k,l].
    du = stack([substitute(expr, Dict([zip(species,v)..., zip(ps,q)...])) for expr in reaction_rates] for (v,q) in zip(eachrow(u),eachrow(p)); dims=1)
    jac = Symbolics.sparsejacobian(vec(du),vec(u); simplify=true)
    (f,f!) = Symbolics.build_function(du, u, p; expression=Val{false})
    (fjac,fjac!) = Symbolics.build_function(jac, vec(u), p, (); expression=Val{false})
    function f̂!(du,u,p,t)
        du=reshape(du,n,m)
        p.u .= reshape(u,n,m)
        plan! * p.u
        f!(du, p.u, p.r)
        plan! * du
        nothing
    end
    ODEFunction(f̂!; jac=fjac!)
end

"Build linear operator for the diffusion component."
function build_d!(diffusion_rates, n, ps)
    k = 0:n-1
    h = 1 / (n-1) # 2pi?
    k² = @. (4/h^2) * sin(k*pi/(2*(n-1)))^2  # Correction from (k/2pi h)^2 for the discrete transform.
    λ = vec(-k² * diffusion_rates')
    (f,f!) = Symbolics.build_function(λ, ps; expression=Val{false})
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

sort_params(p) = sort(p, by=nameof)

"Expand v into an n x length(v) matrix."
build_matrix(v, n) = stack(a isa Function ? a.(range(0.0,1.0,n)) : fill(a, n) for a in v)

end