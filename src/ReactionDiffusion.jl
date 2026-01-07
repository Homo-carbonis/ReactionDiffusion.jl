module ReactionDiffusion
include("Util.jl")
include("PseudoSpectral.jl")
include("Models.jl")
include("Simulate.jl")
include("Turing.jl")
include("GradientDescent.jl")
include("Plot.jl")

using Reexport
@reexport using .Models
@reexport using .Simulate
@reexport using .Turing
@reexport using .Util: dict, product
@reexport using Catalyst: @reaction_network
@reexport using .GradientDescent
@reexport using .Plot

end
