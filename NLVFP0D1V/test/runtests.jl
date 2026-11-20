using NLVFP0D1V, LinearAlgebra, Test

@testset "Radial rule, owned columns, and signed projection" begin
    radius, weights = [1.0, 2.0], [1.0, 1.0]
    measure = radial_measure(radius, weights; convention=:volume, coordinate="r / v_ref")
    basis = [1.0 1.0; 1.0 4.0]
    space = power_space(:even, 2)
    sample = bind_basis(space,measure,basis)
    radius[1] = 7
    weights[1] = -1
    basis[:,1] .= -2
    @test measure.radius == [1.0, 2.0]
    @test collect(measure.weights) == [1.0, 1.0]
    @test_throws Base.CanonicalIndexError measure.weights[1] = -1
    @test observe(sample,[2.0,1.0]) == [3.0,6.0]
    @test observe_rate(sample,[1.0,-1.0]) == [0.0,-3.0]
    @test observe_rate(sample,[1.0,-1.0];basis_time_derivative=ones(2,2),density=[2.0,1.0]) == [3.0,0.0]
    @test_throws ArgumentError observe(sample,[-1.0,1.0])
    @test_throws ArgumentError observe_rate(sample,[Inf,0.0])
    @test_throws ArgumentError observe_rate(sample,[1.0,-1.0];density=[1.0,1.0])
    @test_throws ArgumentError radial_measure([0.0,1.0],[-1.0,1.0];convention=:dr,coordinate="r")
    @test_throws ArgumentError NLVFP0D1V.RadialMeasure([1.0],[-1.0],"r")
    @test_throws DimensionMismatch bind_basis(space,measure,ones(1,2))
    @test observation_metadata(sample).rank == 2
    @test observation_metadata(bind_basis(space,measure,ones(2,2))).rank == 1
    scratch = ObservationWorkspace(sample)
    result = zeros(2)
    @test observe!(result,scratch,sample,[2.0,1.0]) === result
    @test result == [3.0,6.0]
    @test_throws ArgumentError observe!(scratch.weighted,scratch,sample,[2.0,1.0])
    @test_throws ArgumentError observe!(result,scratch,sample,scratch.weighted)
    big_rule = radial_measure(BigFloat[1,2],BigFloat[1,1];convention=:volume,coordinate="r")
    @test eltype(observe(power_space(:odd,3),big_rule,BigFloat[2,1])) == BigFloat
end

@testset "Speed moment target metadata" begin
    stage = resolve_slope(constant_slope())
    g = affine_geometry(stage,[0.25,0.75],[0.1,0.4])
    values = king_observations(power_space(:even,6;normalization=:maxwell),[0.25,0.75],
        g.squared_widths,[0.1,0.4];thermal_speed=1.0)
    target(orders, data; norm=:thermal_maxwell) =
        speed_moment_target(orders,data;normalization=norm,velocity_scale=1,coordinate="r / v_ref")
    m = target(0:2:6,values)
    saved = values[3]
    values[3] = 4.0
    @test m.values[3] == saved
    @test shell_targets(stage,m,g.d) ≈ [sum([0.25,0.75].*[0.1,0.4].^j) for j in 0:3]
    @test_throws MethodError shell_targets(stage,values,g.d)
    @test_throws ArgumentError shell_targets(stage,target(0:3,collect(m.values)),g.d)
    @test_throws ArgumentError shell_targets(stage,target([0,2,6,8],collect(m.values)),g.d)
    @test_throws ArgumentError shell_targets(stage,target(0:2:6,collect(m.values);norm=:maxwell),g.d)
    @test_throws ArgumentError shell_targets(stage,target(0:2:6,[1.0,0.8,1.0,1.0]),g.d)
    @test_throws ArgumentError target([0,4,2],[1.0,1.0,1.0])
    @test_throws DimensionMismatch target(0:2:6,[1.0,1.0])
    @test power_space(:integer,4).parameters.orders == (0,1,2,3,4)
    @test power_space(:odd,5;conserved=true).parameters.orders == (0,1,2,3,5)
end
include("evolution.jl")
include("execution.jl")
include("analytic_case.jl")
include("case_inputs.jl")
include("radial_moments.jl")
include("kinetic_cases.jl")
