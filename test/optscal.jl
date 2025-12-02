includet("../src/GradientDescent.jl")
using OrdinaryDiffEqTsit5

graph = random_signed_digraph(3, 0.0)

model = hill_system(graph)
params = find_turing(model,20; num_batches=100, alg=Tsit5())
good_params = optimise_scale(model,params[1]; η=0.1, maxiters=1e7)