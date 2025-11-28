includet("../src/GradientDescent.jl")

model = rational_system(2,2)
find_turing(model,20)
good_params = optimise_scale(randn(2,27), [1.0, 50.0]; η=0.1, maxiters=1e7)