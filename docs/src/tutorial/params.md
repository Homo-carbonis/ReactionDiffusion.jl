# Specify the parameter values and initial conditions.

We must now define parameters and inital conditions.

We may specify single values:

```julia
params1 = dict(
    # parameter values
    δ₃ = 1.0,
    μ₁ = 1.0,
    μ₃ = 1.0,
    n₁ = 8,
    n₂ = 2
    # initial conditions
    GDF5 = 0.1, 
    NOG = 0.5
)
```

or ranges of parameter values:

```julia
num_params = 5
params2 = product(
    δ₃ = range(0.1,10.0,5),
    μ₁ = range(0.1,10.0,5),
    μ₃ = range(0.1,10.0,5),
    n₁ = range(1.0,20.0,5),
    n₂ = range(1.0,20.0,5),
    GDF5 = [0.1],
    NOG = [0.1,0.5],
)

params = [params1 ; params2]
```

Here, the function `range(min, max, n)` returns a sequence of `n` parameters that linearly spaced between the `min` and `max` limits. The function `logrange ` may be used to sample in log-space instead. 

The `product` function generates a cartesian product over the values given for each parameter.

Spatially heteregeneous values for reaction parameters or initial conditions can be expressed as functions on the interval [0.0, 1.0]. Here the initial concentration of GDF5 is an exponential gradient.
```julia 
GDF5 = x -> exp(-0.2x)
```

