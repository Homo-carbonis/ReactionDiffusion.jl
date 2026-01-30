module PseudoSpectral

export pseudospectral_problem, x
using ..Util: collect_variables, safe_stack
using SciMLBase: SplitODEProblem, DiagonalOperator, ODEFunction, update_coefficients!, remake
using FFTW: plan_r2r!, REDFT00, MEASURE
using Symbolics: @variables, sparsejacobian, build_function, substitute

x = only(@variables(x))

"Construct a SplitODEProblem to solve a reaction diffusion system with reflective boundaries.
Returns the SplitODEProblem with solutions in the frequency (DCT-1) domain and a FFTW plan to transform solutions back to the spatial domain."
function pseudospectral_problem(species, reaction_rates, diffusion_rates, initial_conditions, num_verts; noise=1e-4, kwargs...)
    # Collect parameter symbols. 
    rs,ds,is = (setdiff(collect_variables(exprs), x, species) for exprs in (reaction_rates, diffusion_rates, initial_conditions))

    n = num_verts
    m = length(species)
    
    u = Matrix{Float64}(undef, n, m)
    plan! = 1/sqrt(2*(n-1)) * plan_r2r!(u, REDFT00, 1; flags=MEASURE)

    R = reaction_operator(species, reaction_rates, rs, plan!)
    D = diffusion_operator(diffusion_rates, ds, n)

    prob = SplitODEProblem(D, R, vec(u), Inf, nothing; kwargs...)

    u0 = [substitute(ic, x=>X) for X in range(0,1,n), ic in initial_conditions]
    _fu0,_= build_function(u0, is; expression=Val{false})
    fu0(i) = _fu0(i) + noise*abs.(randn(n,m))

    # Function to set parameter values.
    function make_problem(params, state=nothing; kwargs...)
        r = Float64[params[k] for k in rs]
        d = Float64[params[k] for k in ds]
        i = Float64[params[k] for k in is]
        
        u0 = fu0(i)
        plan! * u0
        u0 = vec(u0)
        w = Matrix{Float64}(undef,n,m) # Allocate working memory for FFTW.
        p = Parameters(w,r,d,state)
        update_coefficients!(prob.f.f1.f, u0, p, 0.0) # Set parameter values in diffusion operator.
        remake(prob; p=p, u0=u0, kwargs...) # Set parameter values in SplitODEProblem.
    end     


    # Function to transform output back to spatial domain.
    # TODO: Avoid unnecessary allocation.
    function transform(sol; full_solution=false)
        function f(u)
            u = reshape(u,n,m)
            plan! * u
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
function reaction_operator(species, reaction_rates, rs, plan!)
    (n,m) = size(plan!)
    @variables u[1:n, 1:m]
    # TODO: Clever things to make only spatially varying parameters expand?
    # Build an nxm matrix of derivatives, substituting reactants for u[i,j] and parameters for p[k,l].
    du = [substitute(expr, Dict([x=>X, zip(species,v)...])) for (v,X) in zip(eachrow(u), range(0,1,n)), expr in reaction_rates]
    jac = sparsejacobian(vec(du),vec(u); simplify=true)
    _, f! = build_function(du, u, rs; expression=Val{false})
    _, _fjac! = build_function(jac, vec(u), rs; expression=Val{false})
    function f̂!(du,u,p,t)
        du=reshape(du,n,m)
        p.u .= reshape(u,n,m)
        plan! * p.u
        f!(du, p.u, p.r)
        plan! * du
        nothing
    end
    fjac!(j,u,p,t) = _fjac!(j, u, p.r)
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
    d # Diffusion parameter vector.
    state # Only used for metadata.
end

end

