using ReactionDiffusion
using Test

include("../examples/Schnakenberg.jl")


function num_periods(u)
    deviation = sign.(u.- 0.5*(maximum(u) .+ minimum(u)))
    length(findall(!iszero,diff(deviation))) / 2
end

dynamic_range(u) = maximum(u)/minimum(u) 


@testset "turing_wavelength" begin
    model = Schnakenberg.model
    params = product(a = range(0.0,0.6,4), b = range(0.0,3.0,4), γ = [1.0], Dᵤ = [1.0], Dᵥ = [50.0])

    λ = turing_wavelength(model, params);
    turing_params = params[λ.>0]

    a = get.(turing_params, :a, missing)
    b = get.(turing_params, :b, missing)
    # Test whether the computed Turing parameters match the ground-truth Turing instability region
    @test a == [0.0, 0.2, 0.0, 0.2]
    @test b == [1.0, 1.0, 2.0, 2.0]
end

@testset "simulate" begin
    model = Schnakenberg.model
    params = Schnakenberg.params
    expected_periods = 1/turing_wavelength(model,params)
    u,t = simulate(model, params)

    u = u[:,1,end]

    # Test whether simulated PDE is 'sensible'; we evaluate the max/min value of the final pattern, and also the number of sign changes about the half maximum (both for U)
    #       note:   we give a range for both test values as we are using random initial conditions, and thus variations are to be expected
    #               (even when setting seeds, it's not clear that Pkg updates to random will conserve values).
    @test 1.5 < dynamic_range(u) < 4
    @test num_periods(u) ≈ expected_periods rtol=0.1
end

# TODO: Write a better test
@testset "filter_params" begin
    model = Schnakenberg.model
    params = product(a = [0.2,0.5,10.0], b = [1.0,2.0,10.0], γ = [1.0], Dᵤ = [1.0], Dᵥ = [50.0])
    expected_periods = 1/turing_wavelength(model,params)

    ps = filter_params(model,params) do u,t
        isapprox(num_periods(u[:,1]), expected_periods; rtol=0.5)
    end

    @test length(ps) == 6
end
