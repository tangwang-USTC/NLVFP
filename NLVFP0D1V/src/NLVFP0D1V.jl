module NLVFP0D1V

using LinearAlgebra
using QuadGK: quadgk
using TOML
using SHA: sha256

const LEGACY = joinpath(@__DIR__, "internal", "legacy_jcp")
include(joinpath(LEGACY, "algorithms/EDCase1Interfaces.jl"))
using .EDCase1Interfaces: SlopePolicy, ResolvedSlope, constant_slope, linear_time_slope,
    checkpoint_slope, functional_slope, supplied_slope, slope_policy, resolve_slope,
    affine_geometry, RadialSpace, radial_space, power_space, king_observations

export constant_slope, linear_time_slope, checkpoint_slope, functional_slope,
    supplied_slope, slope_policy, resolve_slope, affine_geometry, radial_space, power_space,
    king_observations, radial_measure, bind_basis, sample_basis, observe, observe!,
    observe_rate, observation_metadata, ObservationWorkspace, speed_moment_target,
    shell_targets, WeakEvolution, StepOptions, StepWorkspace, StepRejected,
    advance_step!, evolve!, king_case1, execution_metadata, invert_zero_offset, load_case

include("Observations.jl")
include("Evolution.jl")
include("KingCase.jl")
include("CaseInputs.jl")
include("Execution.jl")

end
