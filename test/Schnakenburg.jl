# Test scripts on Schnakenburg model (a well-known reaction-diffusion system with analytical results for its Turing stability region)
reaction = @reaction_network begin
    γ*a + γ*U^2*V,  ∅ --> U
    γ,              U --> ∅
    γ*b,            ∅ --> V
    γ*U^2,          V --> ∅
end 

diffusion = [
    (@transport_reaction Dᵤ U),
    (@transport_reaction Dᵥ V)
]

lattice = CartesianGrid(128)

model = LatticeReactionSystem(reaction, diffusion, lattice)

params = model_parameters()

params.reaction[:a] = screen_values(min = 0, max = 0.6, mode = "linear", number = 4)
params.reaction[:b] = screen_values(min = 0.0, max = 3.0, mode = "linear", number = 4)
params.reaction[:γ] = [1.0]

params.diffusion[:Dᵤ] = [1.0]
params.diffusion[:Dᵥ] = [50.0]

#params = (:a => range(0.0,0.6,4), :b =>range(0.0,3.0,4), :γ => [1.0], :Dᵤ => [1.0], :Dᵥ => [50.0])

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