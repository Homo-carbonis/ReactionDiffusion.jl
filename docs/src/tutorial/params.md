# Specify the parameter values

We must now define parameters.

We may specify single parameter values:

```julia
# constant parameters
params1 = Dict(
    :δ₃ => [1.0],
    :μ₁ => [1.0],
    :μ₃ => [1.0],
    :n₁ => [8],
    :n₂ => [2]
)
```

or ranges of parameter values:

```julia
num_params = 5
params2 = Dict(
    :μ₂ => range(0.1,10,5),
    :k₊ => range(10,100, num_params),
    :k₋ => range(10,100,num_params),
    :δ₁ => range(0.1,10,num_params),
    :δ₂ => range(0.1,10,num_params),
    :K₁ => range(0.01,1.0,num_params),
    :K₂ => range(0.01,1.0,num_params),
    :D₂ => range(0.1,30,10.0),
    :D₃ => range(0.1,30,10.0)
)

params = merge(params1, params2)
```

Here, the function `range` returns a series of `num_params` parameters that are linearly spaced between the `min` and `max` limits. The function `logrange ` may be used to sample in log-space instead. 

Arbitrary collections of parameters may also be specified, e.g.,

```julia
params[:n₁] = [2; 4; 8]
```