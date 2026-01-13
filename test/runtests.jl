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
    params = product(a = range(0.0,0.6,4), b = range(0.0,3.0,4), γ = [1.0], Dᵤ = [1.0], Dᵥ = [50.0], L = [100.0])

    λ = turing_wavelength.(model, params);
    turing_params = params[λ.>0]

    a = get.(turing_params, :a, missing)
    b = get.(turing_params, :b, missing)
    # Test whether the computed Turing parameters match the ground-truth Turing instability region
    @test a == [0.0, 0.2, 0.0, 0.2]
    @test b == [1.0, 1.0, 2.0, 2.0]
end

@testset "simulate" begin
    model = Schnakenberg.model
    params = dict(a = 0.2, b = 2.0, γ = 1.0, Dᵤ = 1.0, Dᵥ = 50.0, L=100.0)
    expected_periods = 1/turing_wavelength(model,params)
    u,t = simulate(model, params)

    u = u[:,1]

    # Test whether simulated PDE is 'sensible'; we evaluate the max/min value of the final pattern, and also the number of sign changes about the half maximum (both for U)
    #       note:   we give a range for both test values as we are using random initial conditions, and thus variations are to be expected
    #               (even when setting seeds, it's not clear that Pkg updates to random will conserve values).
    @test 1.5 < dynamic_range(u) < 4
    @test num_periods(u) ≈ expected_periods rtol=0.1
end

@testset "simulate ics" begin
    model = Schnakenberg.model
    params = dict(a = 0.2, b = 2.0, γ = 1.0, Dᵤ = 1.0, Dᵥ = 50.0, U=0.0, V = x->0.01*x, L=100.0)
    expected_periods = 1/turing_wavelength(model,params)
    u,t = simulate(model, params)

    u = u[:,1]

    # Test whether simulated PDE is 'sensible'; we evaluate the max/min value of the final pattern, and also the number of sign changes about the half maximum (both for U)
    #       note:   we give a range for both test values as we are using random initial conditions, and thus variations are to be expected
    #               (even when setting seeds, it's not clear that Pkg updates to random will conserve values).
    @test 1.5 < dynamic_range(u) < 4
    @test num_periods(u) ≈ expected_periods rtol=0.1
end

@testset "Boundary Conditions" begin
    model = Schnakenberg.model
    params = dict(a = 0.2, b = 2.0, γ = 1.0, Dᵤ = 1.0, Dᵥ = 50.0, L=100.0)
    n = 128
    h = params[:L]/n
    u,t = simulate(model, params;boundary_conditions=(0.01,0.01), num_verts = n, tspan=100.0, dt= 0.01)

    u = u[:,1]
    @test (u[2]-u[1])/h ≈ 0.01
end