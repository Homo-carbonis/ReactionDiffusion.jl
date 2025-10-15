# API for ReactionDiffusion.jl

ReactionDiffusion aims to be an easy-to-use *and* computationally-efficient pipeline to simulate biologically-inspired reaction-diffusion models. It is our hope that models can be built with just a few lines of code and solved without the user having any knowledge of PDE solver methods. 

This is achieved by drawing from a range of numerical routines from the SciML packages, including [Catalyst.jl](https://github.com/SciML/Catalyst.jl), [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl), [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl), and [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl). 

For users more familiar with PDE solvers, it is possible to specify optional arguments to e.g., control the method of discretisation or the specific solver algorithm used (see the API below). However, ReactionDiffusion does not aim to be a fully customizable PDE solving package that covers all bases; our focus is to make something that is relatively easy-to-use *and* performant but only for a restricted number of use cases.

If you require more customization or a more flexible PDE-solving framework, we highly recommend [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) and/or [Catalyst.jl](https://github.com/SciML/Catalyst.jl).

## Data structures
- `save_turing` is an object that records parameter sets that undergo a Turing instability. It has the following fields:

    - `steady_state_values`: The computed steady state values of each variable
    - `reaction_params`: The reaction parameters
    - `diffusion_constants`: The diffusion constants
    - `initial_conditions`: The initial conditions of each variable used to compute the steady state values
    - `pattern_phase: The predicted phase of each of the variables in the final pattern. `[1 1]` would be in-phase, `[1 -1]` would be out-of-phase
    - `wavelength`: The wavelength that is maximally unstable
    - `max_real_eigval`: The maximum real eigenvalue associated with the Turing instability
    - `non_oscillatory`: If `true`, this parameter set represents a stationary Turing pattern. If `false`, the unstable mode has complex eigenvalues and thus may be oscillatory.

## Functions

```@docs
get_params
get_param
returnTuringParams
simulate
```



