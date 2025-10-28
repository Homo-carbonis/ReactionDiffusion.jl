using ReactionDiffusion
using Test

include("../examples/Schnakenberg.jl")


function num_periods(u)
    deviation = sign.(u.- 0.5*(maximum(u) .+ minimum(u)))
    length(findall(!iszero,diff(deviation))) / 2
end

dynamic_range(u) = maximum(u)/minimum(u) 


# @testset "returnTuringParams" begin
#     model = Schnakenberg.model
#     params = product(a = range(0.0,0.6,4), b = range(0.0,3.0,4), γ = [1.0], Dᵤ = [1.0], Dᵥ = [50.0])

#     turing_params = returnTuringParams(model, params, batch_size=2);
#     a = get_param(model, turing_params,:a,"reaction")
#     b = get_param(model, turing_params,:b,"reaction")

#     # Test whether the computed Turing parameters match the ground-truth Turing instability region
#     @test a == [0.0; 0.2; 0.0; 0.2]
#     @test b == [1.0; 1.0; 2.0; 2.0]
# end

@testset "simulate" begin
    model = Schnakenberg.model
    params = Schnakenberg.params
    expected_periods = 12 # TODO: Write wavelength() function so we can compute this separately from turing params.
    u,t = simulate(model, params)
    u = u[:,1,end]

    # Test whether simulated PDE is 'sensible'; we evaluate the max/min value of the final pattern, and also the number of sign changes about the half maximum (both for U)
    #       note:   we give a range for both test values as we are using random initial conditions, and thus variations are to be expected
    #               (even when setting seeds, it's not clear that Pkg updates to random will conserve values).
    @test 1.5 < dynamic_range(u) < 4
    @test expected_periods/2 < num_periods(u) <= expected_periods
end

# TODO: Write a better test
@testset "filter_params" begin
    model = Schnakenberg.model
    params = product(a = [0.2,0.5,10.0], b = [1.0,2.0,10.0], γ = [1.0], Dᵤ = [1.0], Dᵥ = [50.0])
    expected_periods = 12

    ps = filter_params(model,params) do u,t
        isapprox(num_periods(u[:,1]), expected_periods; rtol=0.5)
    end

    @test length(ps) == 6
end
