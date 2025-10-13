using ReactionDiffusion
using Test





include("../examples/Schnakenberg.jl")

@testset "Schnakenberg" begin
    model = Schnakenburg.model
    params = Schnakenburg.params

    turing_params = returnTuringParams(model, params, batch_size=2);
    a = get_param(model, turing_params,:a,"reaction")
    b = get_param(model, turing_params,:b,"reaction")

    # Test whether the computed Turing parameters match the ground-truth Turing instability region
    @test a == [0.0; 0.2; 0.0; 0.2]
    @test b == [1.0; 1.0; 2.0; 2.0]

    param1 = get_params(model, turing_params[4])

    @testset "Simulate using $discretisation" for discretisation in [:finitedifference, :pseudospectral]
        u,t= simulate(model,param1; discretisation=discretisation)

        U_final = u[:,1,end]
        dynamicRange = maximum(U_final)/minimum(U_final) 
        deviation = sign.(U_final.- 0.5*(maximum(U_final) .+ minimum(U_final)))
        halfMaxSignChanges = length(findall(!iszero,diff(deviation)))

        # Test whether simulated PDE is 'sensible'; we evaluate the max/min value of the final pattern, and also the number of sign changes about the half maximum (both for U)
        #       note:   we give a range for both test values as we are using random initial conditions, and thus variations are to be expected
        #               (even when setting seeds, it's not clear that Pkg updates to random will conserve values).
        @test dynamicRange > 1.5 && dynamicRange < 4
        @test halfMaxSignChanges > 3 && halfMaxSignChanges < 7
    end
end
