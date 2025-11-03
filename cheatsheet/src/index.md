# ReactionDiffusion.jl Cheatsheet
## Load module
```@example main
using ReactionDiffusion
``` 
## Reaction (Catalyst DSL)

### Define a reaction network
```@example main
reaction = @reaction_network begin
    μ, ∅ --> X
    δ, X --> ∅
    k, X --> Y + Z
end
```

#### Enter symbols using LaTeX macros + TAB
```
\mu<TAB>\_1<TAB> ===> μ₁
\emptyset<TAB>  ===> ∅
```

#### Alternative without extra symbols
```@example main
reaction = @reaction_network begin
    b, 0 --> X
    d, X --> 0
    k, X --> Y + Z
end
```
### Reactions
#### X created at rate μ
```julia
μ, ∅ --> X
```
#### X degraded at rate rδ
```julia
r*δ, X --> ∅
```
#### X and Y bind at rate k
```julia
k, X + Y --> XY
```
#### X and Y bind reversibly
```julia
k, X + Y <--> XY
```
#### Stoichiometries
```julia
k, 2X + Y --> 3Z
```
#### Production of Y depends upon binding of X (Hill Equation)
```julia
hill(X,v,k,n), ∅ --> Y
```

## Diffusion
#### Define a spatial domain and a set of diffusion parameters
```julia
diffusion = @diffusion_system L begin
    D₁, X
    D₂, Y
end
```

## Model
#### Define a model of a reaction-diffusion system.
```julia
model = Model(reaction, diffusion)
```

## Parameters
#### Single parameter sets
```julia
params = (
    :L => 10.0
    :μ => 1.0,
    :δ => 2.1
)
```

#### Multiple parameter sets
```julia
params = [
    (:μ => 1.0, :δ => 2.0),
    (:μ => 1.5, :δ => 3.0),
]
```

#### Cartesian product
```@example main
params = product(
    μ = [1.0,2.0],
    δ = range(0.0,5.0,3)
)
```

### Spatially dependent parameters
#### μ follows a decreasing gradient from left to right.
```julia
params = (
    :μ => x -> μ₀*exp(-rx),
    ...
)
```
#### X produced at rate μ₁ in anterior region and rate μ₂ in posterior region.
```julia
reaction = @reaction_network begin
    anterior * μ₁, ∅ --> X
    posterior * μ₁, ∅ --> X
end

margin = 0.1
params = (
    :anterior => <(margin),
    :posterior => >(1-margin)
)
```

## Simulation
#### Run simulations
```julia
u,t = simulate(model,params)
```

#### Save full time series
```julia
simulate(model,params; full_solution=true)
```

#### Stop at a fixed time
```julia
simulate(model,params; tspan=5.0)
```

### Filter parameters
#### Return parameter sets with steady state values < 0.5.
```julia
good_params = filter(model, params) do u,t
    maximum(u) < 0.5
```

## Plotting
#### Simulate and plot results over time for a particular parameter set.
```julia plot
plot(model, params)
```

#### Attach sliders to each parameter for an interactive plot.
```julia
params = (
    :μ => range(1.0, 5.0, 50),
    :δ => range(1.0, 5.0, 50),
    :k => [1.0,2.0]
)
interactive_plot(model, params)
```
