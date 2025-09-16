module ReactionDiffusion

include("PseudoSpectral.jl")
include("FiniteDifference.jl")
include("Plot.jl")
using .PseudoSpectral, .FiniteDifference, .Plot

using Catalyst, Combinatorics, Random, StructArrays
using LinearAlgebra, DifferentialEquations
using JLD2
using ProgressMeter
using RecipesBase

include("package_scripts.jl")

export FiniteDifferenceProblem, PseudoSpectralProblem
export returnTuringParams, @reaction_network, model_parameters, screen_values
export get_params, get_param
export simulate
export @save, @load
export endpoint, timepoint

end