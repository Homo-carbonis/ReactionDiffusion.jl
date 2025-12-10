module Turing
export turing_wavelength
using Groebner

using ..Models
using ..Util: issingle, ensure_params_vector

@show isdefined(Main, :Groebner)
@show typeof(Groebner)
using Symbolics: jacobian, symbolic_solve
using SymbolicUtils: BasicSymbolic


function turing_wavelength(model, params; k=logrange(0.1,100,100))
    single = issingle(params)
    params = ensure_params_vector(params) 

    u0 = ones(num_species(model))

    du = reaction_rates(model)
    u = species(model)
    ps = parameters(model)
    p=[[params[k] for k in ps] for params in params]

    jac = jacobian(du,u; simplify=true)
    ss = filter(symbolic_solve(du, u)) do sol
        all(isrealsym, values(sol))
    end
    jac_ss = substitute(jac, only(ss)) # TODO handle multiple ss.
    (fjac,fjac!) = build_function(jac_ss, ps; expression=Val{false})

    d = diffusion_rates(model)
    (fd,fd!) = build_function(diagm(d), ps; expression=Val{false})

    k² = k.^2
    λ = map(params) do params
        p = [params[key] for key in ps]
        J = fjac(p)
        all(<(0.0), real(eigvals(J))) || return 0.0
        D = fd(p)
        real_max, i = findmax(real(eigvals(J - D * k²)[end]) for k² in k²)
        real_max > 0.0 ? 2pi/k[i] : 0.0
    end
    λ
end

isrealsym(::BasicSymbolic{Real}) = true
isrealsym(::BasicSymbolic{Complex{Real}}) = false


end
