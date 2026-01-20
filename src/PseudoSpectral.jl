module PseudoSpectral

export pseudospectral_problem
using ..Util: collect_variables
using SciMLBase: SplitODEProblem, DiagonalOperator, ODEFunction, update_coefficients!, remake
using FFTW: plan_r2r!, REDFT00, MEASURE
using Symbolics: @variables, sparsejacobian, build_function, substitute
using Statistics: mean

"Construct a SplitODEProblem to solve a reaction diffusion system with reflective boundaries.
Returns the SplitODEProblem with solutions in the frequency (DCT-1) domain and a FFTW plan to transform solutions back to the spatial domain."
function pseudospectral_problem(species, reaction_rates, diffusion_rates, boundary_conditions, num_verts; kwargs...)
    n = num_verts
    m = length(species)
    u0 = Matrix{Float64}(undef, n+1, m)
    plan! = 1/sqrt(2*(n-1)) * plan_r2r!((@view u0[2:end,:]), REDFT00, 1; flags=MEASURE)
  

    rs = setdiff(collect_variables(reaction_rates,boundary_conditions), species)
    @show rs
    ds = collect_variables(diffusion_rates)
    R = reaction_operator(species, reaction_rates, boundary_conditions, rs, plan!)
    D = diffusion_operator(diffusion_rates, ds, n)

    prob = SplitODEProblem(D, R, vec(u0), Inf, nothing; kwargs...)


    # Function to set parameter values.
    function make_problem(params, state=nothing; kwargs...)
        r = stack(params[k] for k in rs)
        d = [params[k][1] for k in ds] # Assume D is homogeneous for now.
        u = stack(params[k] for k in species)
        ū = mean(u; dims=1)
        u .-= ū
        u0 = vcat(ū, u)
        plan! * (@view u0[2:end,:])
        u0 = vec(u0)
        v = Matrix{Float64}(undef,n+1,m) # Allocate working memory for FFTW.
        p = Parameters(v,r,d,state)
        update_coefficients!(prob.f.f1.f, u0, p, 0.0) # Set parameter values in diffusion operator.
        remake(prob; p=p, u0=u0, kwargs...) # Set parameter values in SplitODEProblem.
    end     


    # Function to transform output back to spatial domain.
    # TODO: Avoid unnecessary allocation.
    function transform(sol; full_solution=false)
        function f(u)
            u = reshape(u,n+1,m)
            plan! * (@view u[2:end,:])
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
function reaction_operator(species, reaction_rates, (boundary0, boundary1), ps, plan!)
    (n,m) = size(plan!)
    @variables u[1:n+1, 1:m]
    @variables p[1:n, 1:length(ps)]
    # TODO: Clever things to make only spatially varying parameters expand?
    # Build an nxm matrix of derivatives, substituting reactants for u[i,j] and parameters for p[k,l].
    du = stack([substitute(expr, Dict([zip(species,v)..., zip(ps,q)...])) for expr in reaction_rates] for (v,q) in zip(eachrow(u[2:end,:].+u[1:1,:]),eachrow(p)); dims=1)
    α = [substitute(expr, Dict([zip(species,u[1,:])..., zip(ps,p[1,:])...])) for expr in boundary0]'
    β = [substitute(expr, Dict([zip(species,u[end,:])..., zip(ps,p[end,:])...])) for expr in boundary1]'
    dū = β - α + mean(du; dims=1)
    du = vcat(dū, du) # Add extra point for zero mode.

    jac = sparsejacobian(vec(du),vec(u); simplify=true)
    (f,f!) = build_function(du, u, p; expression=Val{false})
    (fjac,fjac!) = build_function(jac, vec(u), p, (); expression=Val{false})
    function f̂!(du,u,p,t)
        du=reshape(du,n+1,m)
        p.u .= reshape(u,n+1,m)
        plan! * @view p.u[2:end,:]
        f!(du, p.u, p.r)
        plan! * (@view du[2:end,:])
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

    λ = vcat(0, σ² * diffusion_rates') |> vec # Add extra row of 0s to leave the zero mode unchanged.
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

