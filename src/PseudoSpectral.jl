module PseudoSpectral

export pseudospectral_problem
using ..Util: collect_params
using SciMLBase: SplitODEProblem, DiagonalOperator, ODEFunction, update_coefficients!, remake
using FFTW: plan_r2r!, REDFT00, MEASURE
using Symbolics: @variables, sparsejacobian, build_function, substitute


"Construct a SplitODEProblem to solve a reaction diffusion system with reflective boundaries.
Returns the SplitODEProblem with solutions in the frequency (DCT-1) domain and a FFTW plan to transform solutions back to the spatial domain."
function pseudospectral_problem(species, reaction_rates, diffusion_rates, u0, tspan; kwargs...)
    u0 = copy(u0)
    n = size(u0,1)
    m = size(u0,2)

    plan! = 1/sqrt(2*(n-1)) * plan_r2r!(copy(u0), REDFT00, 1; flags=MEASURE)
    plan! * u0
    rs = collect_params(reaction_rates, species)
    ds = collect_params(diffusion_rates, species)
    R = reaction_operator(species, reaction_rates, u0, rs, plan!)
    D = diffusion_operator(diffusion_rates, u0, ds)

    prob = SplitODEProblem(D, R, vec(u0), tspan, nothing; kwargs...)

    # Function to set parameter values.
    function make_problem(params, state=nothing; kwargs...)
        r = [params[k] for k in rs]
        d = [params[k] for k in ds]
        r = stack(p isa Function ? p.(range(0.0,1.0,n)) : fill(p, n) for p in r) # Expand r into an n x length(r) matrix
        any(p isa Function for p in d) && error("Spatially dependent diffusion parameters are not supported.")
        u = similar(u0) # Allocate working memory for dct.
        p = Parameters(u,r,d,state)
        update_coefficients!(prob.f.f1.f, u0, p, 0.0) # Set parameter values in diffusion operator.
        remake(prob; p=p, kwargs...) # Set parameter values in SplitODEProblem.
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
function reaction_operator(species, reaction_rates, u0, ps, plan!)
    (n,m) = size(u0)
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
        f!(du, p.u, p.r)
        plan! * du
        nothing
    end
    ODEFunction(f̂!; jac=fjac!)
end

"Build linear operator for the diffusion component."
function diffusion_operator(diffusion_rates, u0, ps)
    n = size(u0,1)
    k = 0:n-1
    h = 1 / (n-1) # 2pi?
    k² = @. (4/h^2) * sin(k*pi/(2*(n-1)))^2  # Correction from (k/2pi h)^2 for the discrete transform.
    λ = vec(-k² * diffusion_rates')
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

