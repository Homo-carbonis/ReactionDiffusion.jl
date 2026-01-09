# Simulate pattern formation in 1D

Having now screened through parameter sets, we can now simulate the corresponding PDEs. The simulation is performed on a 1D domain with reflective boundary conditions.  

To simulate the system for a single parameters set, simply run:

```julia
u,t = simulate(model, turing_params[1])
```
This will return the solution as an array containg the concentration values, and the time at which steady state was reached. The first dimension of u corresponds to points in space and the second to reactants.

For instance if `species(model) == [GDF5, NOG, COMPLEX, pSMAD]`, then `u[1,3]` is the concetration of `COMPLEX` at the left-most end of the domain at the conclusion of the simulation, and `t` will be the time at which this occurs.

To save the complete time series rather than just the steady state value, we can use the `full_solution` option.
```julia
u,t = simulate(model, turing_params[1]; full_solution=true)
```

Here `u[1,3,i]` is the concetration of `COMPLEX` at the left-most end of the domain after the ith time step, and `t[i]` is the time at which this occurs.

A collection of parameter sets can be simulated as an ensemble
```julia
sols = simulate(model, turing_params)
```
This will run in parallel and avoids repeating expensive setup computation for each parameter set. 

