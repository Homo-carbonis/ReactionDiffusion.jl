module PseudoSpectral

export pseudospectral_problem
using ..Util: collect_variables, safe_stack
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
  
    rs = setdiff(collect_variables(reaction_rates), species)
    bs = setdiff(collect_variables(boundary_conditions), species)
    ds = collect_variables(diffusion_rates)
    
    @variables u[1:n, 1:m]
    α = [substitute(expr, Dict(zip(species,u[1,:]))) for expr in boundary_conditions[1]]'
    β = [substitute(expr, Dict(zip(species,u[end,:]))) for expr in boundary_conditions[2]]'
    x = range(0.0,1.0,n)
    # Use a cubic lifting function st. ϕ′(0) = α, ϕ′(1) = β and ϕ(0) = ϕ(1) = 0.
    ϕ = x.^3 * (α + β) - x.^2 * (2α + β) + x * α
    fϕ,fϕ! = build_function(ϕ, u, bs; expression=Val{false})

    R = reaction_operator(species, reaction_rates, ϕ, rs, bs, plan!)
    D = diffusion_operator(diffusion_rates, ds, n)

    prob = SplitODEProblem(D, R, vec(u0), Inf, nothing; kwargs...)


    # Function to set parameter values.
    function make_problem(params, state=nothing; kwargs...)
        r = safe_stack((params[k] for k in rs), n) # Use safe_stack to handle case with no reaction parameters.
        b = Float64[params[k] for k in bs]
        d = Float64[params[k] for k in ds]
        @show r; @show b; @show d
        u0 = safe_stack((params[k] for k in species), n)
        u0 .-= fϕ(u0,b)
        plan! * u0
        u0 = vec(u0)
        w = Matrix{Float64}(undef,n,m) # Allocate working memory for FFTW.
        p = Parameters(w,r,b,d,state)
        update_coefficients!(prob.f.f1.f, u0, p, 0.0) # Set parameter values in diffusion operator.
        remake(prob; p=p, u0=u0, kwargs...) # Set parameter values in SplitODEProblem.
    end     


    # Function to transform output back to spatial domain.
    # TODO: Avoid unnecessary allocation.
    function transform(sol; full_solution=false)
        function f(u)
            u = reshape(u,n,m)
            plan! * u
            u .+= fϕ(u, sol.prob.p.b)
        end
        if full_solution
            u = stack(f.(sol.u))
            t = sol.t
        else
            u = f(sol.u[end])
            t = sol.t[end]
        end
        (u,t)
    end

    make_problem, transform
end

"Build function for the reaction component, with `f(v+ϕ) + Φ` offset for non-zero-flux BCs."
function reaction_operator(species, reaction_rates, ϕ, rs, bs, plan!)
    (n,m) = size(plan!)
    @variables u[1:n, 1:m]
    @variables r[1:n, 1:length(rs)]
    # TODO: Clever things to make only spatially varying parameters expand?
    # Build an nxm matrix of derivatives, substituting reactants for u[i,j] and parameters for p[k,l].
    du = stack([substitute(expr, Dict([zip(species,v)..., zip(rs,q)...])) for expr in reaction_rates] for (v,q) in zip(eachrow(u), eachrow(r)); dims=1)
    @show collect(du[1])
    jac = sparsejacobian(vec(du),vec(u); simplify=true)
    f, f! = build_function(du, u, r, bs; expression=Val{false})
    _fjac, _fjac! = build_function(jac, vec(u), r, bs; expression=Val{false})
    fjac!(j,u,p,t) = _fjac!(j, u, p.r, p.b)
    function f̂!(du,u,p,t)
        du=reshape(du,n,m)
        p.u .= reshape(u,n,m)
        plan! * p.u
        f!(du, u, p.r, p.b)
        plan! * du
        nothing
    end
    ODEFunction(f̂!; jac=fjac!)
end 
"Build linear operator for the diffusion component."
function diffusion_operator(diffusion_rates, ps, n)
    k = 0:n-1 # Wavenumbers
    h = 1/(n-1)

    ## 2nd order Fourier differentiation coefficients.
    # For a continuous FT this would be σ² = -(k/2pi h)^2, but corrected for
    # the discrete transform this becomes:
    σ² = @. -(4/h^2) * sin(k*pi/(2*(n-1)))^2

    λ = vec(σ² * diffusion_rates')
    (f,f!) = build_function(λ, ps; expression=Val{false})
    λ0 = similar(λ, Float64)
    update!(λ,u,p,t) = f!(λ, p.d)
    DiagonalOperator(λ0; update_func! = update!)
end

struct Parameters
    u # Working array for dct.
    r # Reaction parameter matrix.
    b # Boundary reaction parameter vector.
    d # Diffusion parameter vector.
    state # Only used for metadata.
end

end

