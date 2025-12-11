module Simulate
export simulate

using ..Models
using ..PseudoSpectral
using ..Util: issingle, lookup
using SciMLBase: solve, successful_retcode, EnsembleProblem, DiscreteCallback, terminate!, get_du
using DiffEqGPU: EnsembleGPUArray
using CUDA: CUDABackend
using OrdinaryDiffEqExponentialRK: ETDRK4
using ProgressMeter: Progress, BarGlyphs, next!


"""
    simulate(model,params; tspan=Inf, discretisation=:pseudospectral, alg=ETDRK4(), dt=0.01, dx=domain_size(model)/128, reltol=1e-6,abstol=1e-8, maxiters = 1e5)

Simulate `model` for a single parameter set `param`.

Required inputs:
- `model`: `Model`` object containg the system to be simulated.
- `param`: all reaction and diffusion parameters, in a Dict or collection of pairs. 

Inputs carried over from DifferentialEquations.jl; see [here](https://docs.sciml.ai/DiffEqDocs/stable/) for further details:
- `maxiters`: maximum number of iterations to reach steady state (otherwise simulation terminates)
- `alg`: solver algorithm
- `abstol` and `reltol`: tolerance levels of solvers
- `dt`: value for timestep

Additional Inputs
- `dx`: distance between points in spatial discretisation.
- `maxrepeats`: Number of times to halve dt and retry if the solver scheme proves unstable.
"""
function simulate(model, params; output_func=nothing, full_solution=false, tspan=Inf, alg=ETDRK4(), dt=0.1, num_verts=64, reltol=1e-4, abstol=1e-4, maxiters = 1e6, maxrepeats = 4, kwargs...)
    u0 = createIC(model, num_verts)
    make_prob, transform = pseudospectral_problem(model, u0, tspan; callback=steady_state_callback(reltol,abstol), maxiters=maxiters, dt=dt)
    simulate(make_prob, transform, params; output_func=output_func, full_solution=full_solution, alg=alg, dt=dt, maxrepeats = maxrepeats, kwargs...)
end

function simulate(make_prob, transform, params; output_func=nothing, full_solution=false, alg=ETDRK4(), dt=0.1, maxrepeats = 4, kwargs...)
    single = issingle(params)
    params = single ? [lookup(params)] : lookup.(params)

    progress = Progress(length(params); desc="Simulating parameter sets: ", dt=0.1, barglyphs=BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    function _output_func(sol,i)
        repeat = sol.prob.p.state
        successful_retcode(sol) || return (missing, repeat <= maxrepeats) # Rerun if solution failed.
        next!(progress) # Advance progress bar.
        if full_solution
            t = sol.t
            u = stack(transform(u) for u in sol.u)
        else
            t = sol.t[end]
            u = transform(sol.u[end])
        end
        out = isnothing(output_func) ? (u,t) : output_func(u,t)
        (out, false)
    end
        
    function prob_func(prob, i, repeat)
        p = params[i]
        dt = dt/2^(repeat-1) # halve dt if solve was unsuccessful.
        prob = make_prob(p, repeat; dt=dt)
    end

    ensemble_prob = EnsembleProblem(make_prob(params[1]); output_func=_output_func, prob_func=prob_func)
    sol = solve(ensemble_prob, alg, EnsembleGPUArray(CUDA.CUDABackend()); trajectories=length(params), kwargs...)
    # sol = solve(ensemble_prob, alg; trajectories=length(params), kwargs...)

    single ? sol[1] : sol
end

function steady_state_callback(reltol=1e-4,abstol=1e-4)
    condition(u,t,integrator) = isapprox(get_du(integrator), zero(u); rtol=reltol, atol=abstol)
    DiscreteCallback(condition, terminate!)
end

# #TODO benchmark algs
# function turing_wavelength(model, params; k=logrange(0.1,100,100), tspan=1e4, alg=Rodas5(), kwargs...)
#     single = issingle(params)
#     params = ensure_params_vector(params) 

#     u0 = ones(num_species(model))

#     du = reaction_rates(model)
#     u = species(model)
#     ps = parameters(model)
#     p=[[params[k] for k in ps] for params in params]

#     jac = jacobian(du,u; simplify=true)
#     ss = filter(symbolic_solve(du, u)) do sol
#         all(isrealsym, values(sol))
#     end
#     jac_ss = substitute(jac, only(ss)) # TODO handle multiple ss.
#     (fjac,fjac!) = build_function(jac_ss, ps; expression=Val{false})

#     d = diffusion_rates(model)
#     (fd,fd!) = build_function(diagm(d), ps; expression=Val{false})

#     k² = k.^2
#     λ = map(params) do params
#         p = [params[key] for key in ps]
#         J = fjac(p)
#         all(<(0.0), real(eigvals(J))) || return 0.0
#         D = fd(p)
#         real_max, i = findmax(real(eigvals(J - D * k²)[end]) for k² in k²)
#         real_max > 0.0 ? 2pi/k[i] : 0.0
#     end
# end

# isrealsym(::BasicSymbolic{Real}) = true
# isrealsym(::BasicSymbolic{Complex{Real}}) = false

# function filter_turing(model, params)
#     turing_wavelength(model, params)
#     nonzeros = isnonzero.(λ)
#     params[nonzeros]
# end

end