# ReactionDiffusion.jl

[![Build Status](https://github.com/twhiscock/ReactionDiffusion.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/twhiscock/ReactionDiffusion.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Latest Release (for users)](https://img.shields.io/badge/docs-stable-blue.svg)](https://hiscocklab.github.io/ReactionDiffusion.jl/stable)
[![Master (for developers)](https://img.shields.io/badge/docs-dev-blue.svg)](https://hiscocklab.github.io/ReactionDiffusion.jl/dev)

Reaction-diffusion dynamics are present across many areas of the physical and natural world, and allow complex spatiotemporal patterns to self-organize *de novo*. ReactionDiffusion.jl aims to be an easy-to-use and performant pipeline to simulate reaction-diffusion PDEs of arbitrary complexity, with a focus on pattern formation in biological systems. Leveraging computational approaches from [Catalyst.jl](https://github.com/SciML/Catalyst.jl), [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl), [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl), and [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl), ReactionDiffusion.jl enables complex, biologically-inspired reaction-diffusion models to be:

- specified using an intuitive, easy-to-understand syntax
- screened across millions of parameter combinations to identify pattern-forming networks (i.e., those that undergo a Turing instability)
- rapidly simulated to predict spatiotemporal patterns

## Tutorials and documentation

Documentation and worked examples for ReactionDiffusion.jl can be found in the [stable
documentation](https://hiscocklab.github.io/ReactionDiffusion.jl/stable/). The [in-development
documentation](https://hiscocklab.github.io/ReactionDiffusion.jl/dev/) describes unreleased features in
the current master branch.

## Illustrative example

Here we show how ReactionDiffusion.jl can be used to quickly simulate a biologically-inspired reaction-diffusion system that is responsible for generating evenly spaced joints along the length of your fingers and toes (from [Grall et el, 2024](https://www.pnas.org/doi/10.1073/pnas.2304470121)).

#### Step 1: Specify complex reaction-diffusion models with intuitive, easy-to-understand code

We begin by specifying the reaction-diffusion dynamics via the intuitive syntax developed in [Catalyst.jl](https://github.com/SciML/Catalyst.jl), which naturally mirrors biochemical feedbacks and interactions.

```julia
using ReactionDiffusion

reaction = @reaction_network begin
    # complex formation
    (k₊, k₋),               GDF5 + NOG <--> COMPLEX 
    # degradation
    δ₁,                     GDF5 --> ∅
    δ₂,                     NOG --> ∅
    δ₃,                     pSMAD --> ∅
    # transcriptional feedbacks (here: repressive hill functions)
    hillr(pSMAD,μ₁,K₁,n₁),  ∅ --> GDF5
    hillr(pSMAD,μ₂,K₂,n₂),  ∅ --> NOG
    # signalling
    μ₃*GDF5,                ∅ --> pSMAD
end  

diffusion = [
    (@transport_reaction D₁ GDF5),
    (@transport_reaction D₂ NOG),
    (@transport_reaction D₃ COMPLEX)
]

model = Model(reaction,diffusion)
```

#### Step 2: Specify value(s) for each parameter


```julia
params = (
    :μ₁ => [1.0],
    :μ₂ => range(0.1,10,5),
    :k₊ => range(10,100, num_params),
    :k₋ => range(10,100,num_params),
    :μ₃ => [1.0],
    :δ₁ => range(0.1,10,num_params),
    :δ₂ => range(0.1,10,num_params),
    :δ₃ => [1.0],
    :K₁ => range(0.01,1.0,num_params),
    :K₂ => range(0.01,1.0,num_params),
    :n₁ => [8.0],
    :n₂ => [2.0],
    :D₁ => [1.0],
    :D₂ => range(0.1,30,10.0),
    :D₃ => range(0.1,30,10.0)
)
```

#### Step 3 (Optional): Perform automated Turing instability analysis to find pattern-forming parameter sets

With a single line of code, we can perform a Turing instability analysis across all combinations of parameters:

```julia
turing_params = returnTuringParams(model, params);
```

which gives the following output:

```
Screening parameter sets:  25%[============>                                     ]  ETA: 0:04:23
Screening parameter sets:  38%[==================>                               ]  ETA: 0:03:47
Screening parameter sets:  50%[=========================>                        ]  ETA: 0:02:46
Screening parameter sets:  62%[===============================>                  ]  ETA: 0:02:05
Screening parameter sets:  75%[=====================================>            ]  ETA: 0:01:18
Screening parameter sets:  88%[===========================================>      ]  ETA: 0:00:38
Screening parameter sets: 100%[==================================================] Time: 0:04:49
3405/7812500 parameters are pattern forming
```

This returns all parameter combinations that break symmetry from a homogeneous initial condition. We take advantage of the highly performant numerical solvers in [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) to be able to simulate millions of parameter sets per minute on a standard laptop. 

#### Step 4: Simulate the spatiotemporal dynamics of the reaction-diffusion PDEs

We may then take a single parameter set and simulate its spatiotemporal dynamics directly, using [Plots.jl](https://github.com/JuliaPlots/Plots.jl) to visualize the resulting pattern:

```julia
# specify parameter values (e.g., here we select the 1000th pattern-forming parameter set)
param1 = get_params(model, turing_params[1000])

# simulate the reaction-diffusion PDE
u,t = simulate(model,param1)

# visualize the resulting pattern
using Plots
plot(endpoint(),model,u)
```
![final pattern](docs/src/assets/final_pattern.svg)


We may also view the full spatiotemporal dynamics:

```julia
@gif for t in 0.0:0.01:1
    plot(timepoint(),model,u,t)
end fps=20
```
![dynamics](docs/src/assets/dynamics.gif)



## Support, citation and future developments

If you find ReactionDiffusion.jl helpful in your research, teaching, or other activities, please star the repository and consider citing [this paper](https://www.biorxiv.org/content/10.1101/2025.05.27.656324v1).

We are a small team of academic researchers from the [Hiscock Lab](https://twhiscock.github.io/), who build mathematical models of developing embryos and tissues. We have found these scripts helpful in our own research, and make them available in case you find them helpful in your research too. We hope to extend the functionality of ReactionDiffusion.jl as our future projects, funding and time allows.

This work is supported by ERC grant SELFORG-101161207, and UK Research and Innovation (Biotechnology and Biological Sciences Research Council, grant number BB/W003619/1) 

*Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them*

![ERC_logo](docs/src/assets/LOGO_ERC-FLAG_FP.png)

