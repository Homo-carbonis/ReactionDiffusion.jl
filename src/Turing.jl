module Turing
export turing_wavelength

using ..Models
using ..Util: issingle, ensure_params_vector

using Groebner
using Symbolics: jacobian, symbolic_solve, substitute, build_function, Num
using LinearAlgebra: diagm, eigvals


function turing_wavelength(model, params; k=logrange(0.1,100,100))
    single = issingle(params)
    params = ensure_params_vector(params) 

    u0 = ones(num_species(model))
   
    du = reaction_rates(model)
    u = Num.(species(model))  # For some unfathomable reason, symbolic_solve complains about missing Groebner if we don't convert u to Num.
    ps = parameters(model)
    p=[[params[k] for k in ps] for params in params]

    jac = jacobian(du,u; simplify=true)
    ss = symbolic_solve(du, u)
    jac_ss = substitute(jac, only(ss)) # TODO: Handle multiple steady states.
    (fjac,fjac!) = build_function(jac_ss, ps; expression=Val{false})

    d = diffusion_rates(model)
    (fd,fd!) = build_function(diagm(d), ps; expression=Val{false})

    k² = k.^2
    map(params) do params
        p = [params[key] for key in ps]
        J = fjac(p)
        all(<(eps(J[1])), real(eigvals(J))) || return 0.0
        D = fd(p)
        real_max, i = findmax(real(eigvals(J - D * k²)[end]) for k² in k²)
        real_max > 0.0 ? 2pi/k[i] : 0.0
    end
end
end
