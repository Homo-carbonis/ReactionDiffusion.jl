# Simulate pattern formation in 1D

Having now screened through parameter sets, we can now pick a single one and simulate the corresponding PDE. The simulation is performed on a 1D domain with reflective boundary conditions.  

We first select the parameter values from `turing_params`; here we choose the 1000th parameter set found. 

```julia
param1 = get_params(model, turing_params[1000])
```

Optionally, we can manually change the parameters too:

```julia
param1.reaction["n1"] = 10
```

To simulate the script, you then simply run:

```julia
u,t = simulate(model,param1)
```
This will return the solution as an array containg the concentration values and a vector of corresponding time points. The first dimension of u corresponds to points in space, the second to reactant and the third to points in time.

For instance if `species(model) == [GDF5, NOG, COMPLEX, pSMAD]`, then u[1,3,end] is the concetration of `COMPLEX` at the left-most end of the domain at the conclusion of the simulation, and t[end] will be the time at which this occurs.

*Note: as with all Julia functions, there is a significant compilation time associated with the first time you run this function. This overhead time will not be present if you re-run the simulation e.g., for different parameters*

To visualize the results, you can use a variety of plotting packages (e.g., Makie.jl, Plots.jl). We provide simple helper functions for the Plots.jl package.

To visualize the final pattern (once steady state has been reached), use: `endpoint()` 

```julia
using Plots
plot(endpoint(),model,u)
```

To visualize intermediate timepoints (e.g., for making a movie of the dynamics), use `timepoint()`, e.g.,:

```julia
@gif for t in 0.0:0.01:1
    plot(timepoint(),model,u,t)
end fps=20
```

Here, for example, `plot(timepoint(),model,u,0.1)` plots the solution at a time that is 10% through the simulation. 