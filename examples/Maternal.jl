## Import the ReactionDiffusion library.
using ReactionDiffusion, WGLMakie
WGLMakie.activate!(resize_to=:body) # Make plots fill the available space.


## Define a reaction network.
reaction = @reaction_network begin
    anterior * μ_bcd,           ∅ --> BCD # Bicoid expression in anterior region
    posterior * μ_nos,          ∅ --> NOS  # Nanos expression in posterior region
    δ_bcd,                      BCD --> ∅   # Bicod degredation
    δ_nos,                      NOS --> ∅ # Nanos degredation
    hillar(BCD, NOS, μ_hb,K,n), ∅ --> HB  # Hunchback expression. Hill function with BCD activating and NOS inhibiting.
    δ_hb,                       HB --> ∅ # Hunchback degredation 
end

## Define a system of diffusing species on a 1D domain of size `L`.
diffusion = @diffusion_system L begin
    D_bcd,  BCD
    D_nos,  NOS
    D_hb,   HB
end

## Combine `reaction` and `diffusion` into a single `Model` object.
model = Model(reaction, diffusion)

## Pick some parameter sets to test.
params = product(
    L = range(10.0, 100.0, 3),
    # `anterior` and `posterior` are defined as step functions in space. 
    anterior = [x -> (x < 1/12) ? 1 : 0],   # 1 in the anterior 12th of the domain, 0 elsewhere.
    posterior = [x -> (x > 11/12) ? 1 : 0], # 1 in the posterior 12th of the domain, 0 elsewhere.
    μ_bcd = range(0.1, 2.0, 2),
    μ_nos = range(0.1, 2.0, 2),
    δ_bcd = range(0.1, 2.0, 2),
    δ_nos = range(0.1, 2.0, 2),
    μ_hb = range(0.1, 2.0, 2),
    K = [0.1],
    n = [1.0],
    δ_hb = range(0.1, 2.0, 2),
    D_bcd = [1],
    D_nos = [1],
    D_hb = [1]
)

## Define a predicate we want the steady-state value of our system to satisfy.
function hb_partition(u, t)
    u = u[:,3]
    n = length(u)
    # Divide domain into 3rds.
    anterior = u[1:n÷3] 
    posterior = u[n-n÷3:end]
    u_min = minimum(u)
    u_max = maximum(u)
    d = (u_max - u_min)/4
    all(anterior .> u_min + d) && all(posterior .< u_max - d)
end


## Find all the parameters which satisfy `hb_partition`.
good_params = filter_params(hb_partition, model, params; maxrepeats=2, maxiters=1e4, num_verts=64)

## Simulate the system with one of the good parameter sets and plot the results over time. 
ReactionDiffusion.plot(model, params[1])

## Define some plausible ranges of parameter values to explore.
param_ranges = dict(
    L = range(1.0, 50.0, 50),
    anterior = [x -> (x < 1/12) ? 1 : 0],
    posterior = [x -> (x > 11/12) ? 1 : 0],
    μ_bcd = range(0.1, 2.0, 50),
    μ_nos = range(0.1, 2.0, 50),
    δ_bcd = range(0.1, 2.0, 50),
    δ_nos = range(0.1, 2.0, 50),
    μ_hb = range(0.1, 2.0, 50),
    K = range(0.1, 2.0, 50),
    n = range(1.0, 8.0, 50),
    δ_hb = range(0.1, 2.0, 50),
    D_bcd = range(0.1, 2.0, 50),
    D_nos = range(0.1, 2.0, 50),
    D_hb = range(0.1, 2.0, 50)
)

## Create an interactive plot with sliders to change each parameter.
ReactionDiffusion.interactive_plot(model, param_ranges)