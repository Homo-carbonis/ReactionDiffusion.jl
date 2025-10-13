using ReactionDiffusion

include("../examples/Schnakenberg.jl")


model = Schnakenburg.model
params = Schnakenburg.params

turing_params = returnTuringParams(model, params, batch_size=2);
param1 = get_params(model, turing_params[4])

u,t= simulate(model,param1)

U_final = u[:,1,end]
dynamicRange = maximum(U_final)/minimum(U_final) 
deviation = sign.(U_final.- 0.5*(maximum(U_final) .+ minimum(U_final)))
halfMaxSignChanges = length(findall(!iszero,diff(deviation)))

# Test whether simulated PDE is 'sensible'; we evaluate the max/min value of the final pattern, and also the number of sign changes about the half maximum (both for U)
#       note:   we give a range for both test values as we are using random initial conditions, and thus variations are to be expected
#               (even when setting seeds, it's not clear that Pkg updates to random will conserve values).
# @test dynamicRange > 1.5 && dynamicRange < 4
# @test halfMaxSignChanges > 3 && halfMaxSignChanges < 7
