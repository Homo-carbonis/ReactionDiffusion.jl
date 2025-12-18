module Turing
export turing_wavelength, is_turing, filter_turing, turing_wavelength_problem

using ..Models
using ..Util: issingle, lookup, tmap, tfilter

using Groebner
using Symbolics: jacobian, symbolic_solve, substitute, build_function, Num
using LinearAlgebra: diagm, eigvals
using Base.Threads: @threads

turing_wavelength(model, params; k=logrange(0.1,100,100)) = turing_wavelength(model;k=k)(params)

function turing_wavelength(model; k=logrange(0.1,100,100))   
    du = reaction_rates(model)
    u = Num.(species(model))  # For some unfathomable reason, symbolic_solve complains about missing Groebner if we don't convert u to Num.
    ps = parameters(model)
    jac = jacobian(du,u; simplify=true)
    ss = symbolic_solve(du, u)
    jac_ss = substitute(jac, only(ss)) # TODO: Handle multiple steady states.
    (fjac,fjac!) = build_function(jac_ss, ps; expression=Val{false})

    d = diffusion_rates(model)
    (fd,fd!) = build_function(diagm(d), ps; expression=Val{false})

    k² = k.^2

    function f(params)
        lookup(params)
        p = [params[key] for key in ps]
        J = fjac(p)
        all(<(eps(J[1])), real(eigvals(J))) || return 0.0
        D = fd(p)
        real_max, i = findmax(real(eigvals(J - D * k²)[end]) for k² in k²)
        real_max > 0.0 ? 2pi/k[i] : 0.0
    end
end


is_turing(model, params) = is_turing(model)(params)

function is_turing(model)
    f = turing_wavelength(model)
    params -> any(>(0.0), f(params))
end

function filter_turing(model,params)
    params = lookup.(params)
    f = turing_wavelength(model)
    tfilter(params) do p
        f(p) > 0.0
    end
end

end