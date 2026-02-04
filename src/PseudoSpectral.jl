module PseudoSpectral

export pseudospectral_problem, x
using ..Util: collect_variables, safe_stack
using SciMLBase: SplitODEProblem, DiagonalOperator, ODEFunction, update_coefficients!, remake
using FFTW: plan_r2r!, REDFT00, MEASURE
using Symbolics: @variables, sparsejacobian, build_function, substitute

x = only(@variables(x))

"Construct a SplitODEProblem to solve a reaction diffusion system with reflective boundaries.
Returns the SplitODEProblem with solutions in the frequency (DCT-1) domain and a FFTW plan to transform solutions back to the spatial domain."
function pseudospectral_problem(species, reaction_rates, diffusion_rates, boundary_conditions, initial_conditions, num_verts; noise=1e-4, kwargs...)
    n = num_verts
    N = 3n÷2
    m = length(species)
    
    # Collect parameter symbols. 
    rs,ds,bs,is = (setdiff(collect_variables(exprs), x, species) for exprs in (reaction_rates, diffusion_rates, vec(boundary_conditions), initial_conditions))

    u = Matrix{Float64}(undef, n, m)
    u2 = Matrix{Float64}(undef, N, m)

    plan! = 1/sqrt(2*(n-1)) * plan_r2r!(u, REDFT00, 1; flags=MEASURE)
    plan2! = 1/sqrt(2*(n-1)) * plan_r2r!(u2, REDFT00, 1; flags=MEASURE)



    ## Offsets for constant, non-zero flux boundary conditions.
    # For u′(0) = a, u′(1) = b,
    # define ϕ as a smooth function so that ϕ′(0) = a, ϕ′(1) = b, and write v = u - ϕ.
    # Then v′(0) = 0, v′(1) = 0, so we can solve for v using DCT-I.
    a,b = eachrow(boundary_conditions)'
    X = range(0.0,1.0,N)
    X1 = range(0.0,1.0,n)

    ϕ = X.^2 * (b-a)/2 + X * a
    ϕ1 = X1.^2 * (b-a)/2 + X1 * a

    # ϕ′′ = b-a, so Φ = DCT{ϕ′′} ∝ [(a-b), 0, 0...] 
    Φ = [-2*(N-1)/sqrt(2*(N-1))*(b-a) ; zeros(N-1,m)]
    fϕ,_ = build_function(ϕ, bs; expression=Val{false})
    fϕ1,_ = build_function(ϕ1, bs; expression=Val{false})

    fΦ,_ = build_function(Φ, bs; expression=Val{false})

    
    R = reaction_operator(species, reaction_rates, rs, plan2!, n)
    D = diffusion_operator(diffusion_rates, ds, n)
    prob = SplitODEProblem(D, R, vec(u), Inf, nothing; kwargs...)

    u0 = [substitute(ic, x=>X) for X in range(0,1,n), ic in initial_conditions]
    _fu0,_= build_function(u0, is; expression=Val{false})
    fu0(i) = _fu0(i) + noise*abs.(randn(n,m))


    # Function to set parameter values.
    function make_problem(params, state=nothing; kwargs...)
        r = Float64[params[k] for k in rs]
        d = Float64[params[k] for k in ds]
        b = Float64[params[k] for k in bs]
        i = Float64[params[k] for k in is]
        ϕ = fϕ(b)
        Φ = fΦ(b)
        
        u0 = fu0(i) - fϕ1(b)
        plan! * u0
        u0 = vec(u0)
        w = Matrix{Float64}(undef,N,m) # Allocate working memory for FFTW.
        dw = Matrix{Float64}(undef,N,m) # Allocate working memory for FFTW.

        p = Parameters(w,dw,r,d,ϕ,Φ,state)
        update_coefficients!(prob.f.f1.f, u0, p, 0.0) # Set parameter values in diffusion operator.
        remake(prob; p=p, u0=u0, kwargs...) # Set parameter values in SplitODEProblem.
    end     


    # Function to transform output back to spatial domain.
    # TODO: Avoid unnecessary allocation.
    function transform(sol; full_solution=false)
        ϕ = fϕ1(sol.prob.p.b) # no!
        function f(u)
            u = reshape(u,n,m)
            plan! * u
            u .+= ϕ
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
function reaction_operator(species, reaction_rates, rs, plan!,n)
    (N,m) = size(plan!)
    @variables u[1:N, 1:m]
    # TODO: Clever things to make only spatially varying parameters expand?
    # Build an nxm matrix of derivatives, substituting reactants for u[i,j] and parameters for p[k,l].
    du = [substitute(expr, Dict([x=>X, zip(species,v)...])) for (v,X) in zip(eachrow(u), range(0,1,N)), expr in reaction_rates]
    _, f! = build_function(du, u, rs; expression=Val{false})
    function f̂!(du,u,p,t)
        du=reshape(du,n,m)
        p.u[1:n,:] .= reshape(u,n,m)
        p.u[n+1:end,:] .= 0.0
        plan! * p.u
        p.u .+= p.ϕ
        f!(p.du, p.u, p.r)
        plan! * p.du
        p.du .+= p.Φ
        du .= p.du[1:n,:]
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
    du
    r # Reaction parameter matrix.
    d # Diffusion parameter vector.
    ϕ
    Φ
    state # Only used for metadata.
end

function zeropad(A,N)
    n,m=size(A)
    vcat(A, zeros(N-n,m))
end
end

