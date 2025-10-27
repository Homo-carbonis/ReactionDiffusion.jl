# ReactionDiffusion.jl for modelling pattern formation in biological systems

Reaction-diffusion dynamics are present across many areas of the physical and natural world, and allow complex spatiotemporal patterns to self-organize *de novo*. `ReactionDiffusion.jl` aims to be an easy-to-use and performant pipeline to simulate reaction-diffusion PDEs of arbitrary complexity, with a focus on pattern formation in biological systems. Using this package, complex, biologically-inspired reaction-diffusion models can be:

- specified using an intuitive syntax
- screened across millions of parameter combinations to identify pattern-forming networks (i.e., those that undergo a Turing instability)
- rapidly simulated to predict spatiotemporal patterns

## Quick start guide

Here we show how `ReactionDiffusion.jl` can be used to simulate a biologically-inspired reaction-diffusion system, responsible for generating evenly spaced joints along the length of your fingers and toes (from [Grall et el, 2024](https://www.pnas.org/doi/10.1073/pnas.2304470121)).

We begin by specifying the reaction-diffusion dynamics via the intuitive syntax developed in [Catalyst.jl](https://github.com/SciML/Catalyst.jl), which naturally mirrors biochemical feedbacks and interactions.

```@example quickstart
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

model = Model(reaction, diffusion)
```

We can then specify values for each parameter:

```@example quickstart
num_params=5
params = (
    :μ₁ => [1.0],
    :μ₂ => range(0.1,10,num_params),
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
    :D₂ => range(0.1,30,10),
    :D₃ => range(0.1,30,10)
)
```

Then, with a single line of code, we can perform a Turing instability analysis across all combinations of parameters:

```@example quickstart
turing_params = returnTuringParams(model, params);
```

This returns all parameter combinations that can break symmetry from a homogeneous initial condition. We take advantage of the highly performant numerical solvers in [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) to be able to simulate millions of parameter sets per minute on a standard laptop. 

We may then take a single parameter set and simulate its spatiotemporal dynamics directly, using `Plots.jl` to visualize the resulting pattern:

```@example quickstart
param1 = get_params(model, turing_params[1000])
u,t = simulate(model,param1)

using Plots
plot(endpoint(),model,u)
```

We may also view the full spatiotemporal dynamics:

```@example quickstart
@gif for t in 0.0:0.01:1
    plot(timepoint(),model,u,t)
end fps=20
```


## Support, citation and future developments

If you find `ReactionDiffusion.jl` helpful in your research, teaching, or other activities, please star the repository and consider citing this paper:

```
@article{TBD,
 doi = {TBD},
 author = {Muzatko, Daniel AND Daga, Bijoy AND Hiscock, Tom W.},
 journal = {biorXiv},
 publisher = {TBD},
 title = {TBD},
 year = {TBD},
 month = {TBD},
 volume = {TBD},
 url = {TBD},
 pages = {TBD},
 number = {TBD},
}
```

We are a small team of academic researchers from the [Hiscock Lab](https://twhiscock.github.io/), who build mathematical models of developing embryos and tissues. We have found these scripts helpful in our own research, and make them available in case you find them helpful in your research too. We hope to extend the functionality of `ReactionDiffusion.jl` as our future projects, funding and time allows.

This work is supported by ERC grant SELFORG-101161207, and UK Research and Innovation (Biotechnology and Biological Sciences Research Council, grant number BB/W003619/1) 

*Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them*

![ERC_logo](./assets/LOGO_ERC-FLAG_FP.png)

