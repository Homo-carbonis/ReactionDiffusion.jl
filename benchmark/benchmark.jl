using ReactionDiffusion, DifferentialEquations, Catalyst

function benchmark(model,param, tspan; discretisation=PseudoSpectralProblem, alg=nothing, reltol=1e-6,abstol=1e-8)
    p, d, ic, l, seed, noise = ReactionDiffusion.returnSingleParameter(model, param)

    # convert reaction network to ODESystem
    odesys = convert(ODESystem, model)

    # build ODE function
    f_gen = ModelingToolkit.generate_function(odesys,expression = Val{false})[1] #false denotes function is compiled, world issues fixed
    f_oop = ModelingToolkit.eval(f_gen)
    f_ode(u,p,t) = f_oop(u,p,t)

    u0 = ReactionDiffusion.createIC(ic, seed, noise)
    prob = discretisation(f_ode, u0, tspan, l, d, p; ss=false)
    @benchmark $sol = $solve($prob, $alg; reltol=$reltol,abstol=$abstol)
    sol
end

model = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 


params = model_parameters()

params.reaction["a"] = [0.2]
params.reaction["b"] = [2.0]
params.reaction["γ"] = [1.0]

params.diffusion["U"] = [1.0] 
params.diffusion["V"] = [50.0]


turing_params = returnTuringParams(model, params,batch_size=2);
a = get_param(model, turing_params,"a","reaction")
b = get_param(model, turing_params,"b","reaction")

param1 = get_params(model, turing_params[4])

sol1 = benchmark(model,param1, (0,10); discretisation=PseudoSpectralProblem, alg=KenCarp3(autodiff=false))
sol2 = benchmark(model,param1, (0,100); discretisation=FiniteDifferenceProblem, alg=KenCarp4())

sol1 = sol= simulate(model,param1; tspan=(0,10), discretisation=FiniteDifferenceProblem)
sol2 = sol= simulate(model,param1; tspan=(0,10), discretisation=PseudoSpectralProblem)