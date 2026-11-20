using LinearAlgebra
BLAS.set_num_threads(1)

@testset "e–D nonlinear relaxation from analytic Maxwellians" begin
    deck = load_test_case("ed_relaxation")
    function check_relaxation(expected_slope;kwargs...)
        model = king_case1(deck;kwargs...)
        start = model.diagnostics(model.evaluate(model.initial,0.0,false))
        @test size(model.basis,2)==42 && length(model.initial)==51
        @test start.heat_keV_per_tau/10 ≈ 0.19020734633065575 atol=3e-10
        state = evolve!(StepWorkspace(),model,model.initial,deck.times_tau;method=:sdirk2)
        @test state.slope == expected_slope
        @test state.diagnostic.number <= 2e-12
        @test state.diagnostic.momentum == 0
        @test state.diagnostic.energy <= 2e-12
        @test state.diagnostic.H < start.H
        @test state.diagnostic.cpe <= 2e-12 && state.residual <= 1e-12
        @test state.diagnostic.electron_temperature_keV < start.electron_temperature_keV
        @test state.diagnostic.deuteron_temperature_keV > start.deuteron_temperature_keV
    end
    check_relaxation(0.0;width_rule=:common)
    check_relaxation(0.005)
    check_relaxation(0.01;slope=linear_time_slope([0.0,last(deck.times_tau)],[0.005,0.01]))
end

@testset "e–D common-temperature equilibrium" begin
    deck = load_test_case("ed_equilibrium")
    problem = king_case1(deck;width_rule=:common)
    start = problem.diagnostics(problem.evaluate(problem.initial,0.0,false))
    @test abs(start.heat_keV_per_tau) <= 5e-10
    result = evolve!(StepWorkspace(),problem,problem.initial,deck.times_tau)
    @test result.diagnostic.electron_temperature_keV ≈ 12.5 atol=2e-12
    @test result.diagnostic.deuteron_temperature_keV ≈ 12.5 atol=2e-12
    @test abs(result.diagnostic.H-start.H)<=1e-11
    @test maximum(abs,result.z-problem.initial)<=1e-11
    @test result.diagnostic.number<=2e-12 && result.diagnostic.energy<=2e-12
    @test result.diagnostic.momentum==0 && result.diagnostic.cpe<=2e-12
end
