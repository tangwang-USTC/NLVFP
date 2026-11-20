using TOML

@testset "Two-file physical authority and unit checks" begin
    relaxation = case1_smoke_input()
    equilibrium = load_test_case("ed_equilibrium")
    @test relaxation.id == "ed_relaxation" && equilibrium.id == "ed_equilibrium"
    @test relaxation.scales.time_s ≈ 0.25966523492993543 rtol=2e-14
    @test relaxation.scales.velocity_m_per_s ≈ 5.930969580731296e7 rtol=2e-14
    @test relaxation.times_tau == [0.0,2.0^-31,2.0^-30]
    physical = test_case("ed_relaxation","physical-input.toml")
    manifest = test_case("ed_relaxation","dimensionless-input.toml")
    @test_throws ArgumentError load_case(physical,physical)
    @test_throws ArgumentError load_case(physical,test_case("ed_equilibrium","dimensionless-input.toml"))
    mktempdir() do dir
        changed = joinpath(dir,"dimensionless-input.toml")
        settings = TOML.parsefile(manifest)
        settings["assertions"]["thermal_widths"][1] *= 1.001
        open(io->TOML.print(io,settings),changed,"w")
        err = try
            load_case(physical,changed)
            nothing
        catch problem
            problem
        end
        @test err isa ArgumentError
        @test occursin("thermal_widths",sprint(showerror,err))
        invalid_physical = joinpath(dir,"physical-input.toml")
        parameters = TOML.parsefile(physical)
        parameters["plasma"]["electron_edges_in_initial_thermal_speed"] = [0.0, 3.0]
        open(io->TOML.print(io,parameters),invalid_physical,"w")
        @test_throws ArgumentError load_case(invalid_physical,manifest)
        settings = TOML.parsefile(manifest)
        settings["discretization"]["temperature_unit_keV"] = 5.0
        open(io->TOML.print(io,settings),changed,"w")
        @test_throws ArgumentError load_case(physical,changed)
    end
end
