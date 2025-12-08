module ReactionDiffusion
include("Util.jl")
include("PseudoSpectral.jl")
include("Models.jl")
include("Simulate.jl")
#include("GradientDescent.jl")

using Reexport
@reexport using .Models
@reexport using .Simulate
@reexport using .Util: dict, product
@reexport using Catalyst: @reaction_network
#@reexport using .GradientDescent

end
