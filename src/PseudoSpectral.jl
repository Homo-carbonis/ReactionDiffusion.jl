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
    rps = setdiff(union(get_variables.(reaction_rates)...), species)
    dps = union(get_variables.(diffusion_rates)...)
    
    R = build_r!(species, reaction_rates, u0, r.keys, plan!)
    D = build_d!(diffusion_rates, u0, d.keys)

    ps = make_params(u0, p) 
    # update_coefficients!(D,u0,ps,0.0) # Must be called before first step. Replaced by remake_params?
    # update_coefficients!(R,u0,ps,0.0)
    prob = SplitODEProblem(D, R, vec(u0), tspan, ps; kwargs...)
    remake_params(prob, u0, r, d)
    function transform(u)
        u = reshape(copy(u),n,m)
        plan! * u
        u
    end
    prob, transform
end

remake_params(prob, u0, r, d) = remake(prob; p=make_parameters(u0, r, d))



"Build function for the reaction component."
function build_r!(species, reaction_rates, u0, p, plan!)
    p = sort_params(p)
    (n,m) = size(u0)
    @variables u[1:n, 1:m]
    @variables p[1:n, 1:length(params)]
    # Do clever things to make only spatially varying parameters expand?
    # Build an nxm matrix of derivatives, substituting reactants for u[i,j] and parameters for p[k,l].
    du = hcat([substitute(expr, Dict(zip(species,v)..., zip(params,q)...)) for expr in reaction_rates] for (v,q) in eachrow.((u,p)))'
    
    jac = Symbolics.sparsejacobian(vec(du),vec(u); simplify=true)
    (f,f!) = Symbolics.build_function(du, u, p; expression=Val{false})
    (fjac,fjac!) = Symbolics.build_function(jac, vec(u), p, (); expression=Val{false})
    function f̂!(du,u,p,t)
        du=reshape(du,n,m)
        p.u .= reshape(u,n,m)
        plan! * p.u
        f!(du, p.u, p.reaction)
        plan! * du
        nothing
    end
    ODEFunction(f̂!; jac=fjac!)
end

"Build linear operator for the diffusion component."
function build_d!(diffusion_rates, u0, p, plan!)
    p = sort_params(p)
    n = size(u0,1)
    @variables p[1:n, 1:length(params)]
    D = hcat([substitute(expr, Dict(zip(params,q))) for expr in reaction_rates] for q in eachrow(p))'
    (f,f!) = Symbolics.build_function(D, p; expression=Val{false})

    k = 0:n-1
    h = 1 / (n-1) # 2pi?
    k² = [(4/h^2) * sin(k*pi/(2*(n-1)))^2 for k in k]  # Correction from (k/2pi h)^2 for the discrete transform.

    λ0 = similar(λ, Float64)
    function update!(λ,u,p,t)
        f!(λ,p.d)
        plan! * λ
        λ .*= k²
    end
    DiagonalOperator(λ0; update_func! = update!)
end


struct Parameters
    u # Working array for dct.
    r
    d
end

function make_parameters(u0, r, d)
    r = sort_params(r).vals
    hcat(p isa Function ? p.(range(0.0,1.0,n)) : fill(v, n) for p in r) # jjnkn
    d = sort_params(d).vals
    u = similar(u0)
    Parameters(u,r,d)
end

sort_params(p) = sort(p, by=nameof)

end

