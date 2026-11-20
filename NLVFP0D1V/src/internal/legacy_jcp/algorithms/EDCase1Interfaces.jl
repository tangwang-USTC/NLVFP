module EDCase1Interfaces

using LinearAlgebra: svdvals
include("EDKingObservations.jl")
using .EDKingObservations: speed_orders, king_moments, maxwell_moment_factor,
    generalized_observations, affine_widths, affine_shell_moments

export DEFAULT_AFFINE_SLOPE, constant_slope, linear_time_slope, checkpoint_slope,
    functional_slope, supplied_slope, slope_policy, resolve_slope, affine_geometry, shell_targets,
    RadialMeasure, radial_measure, radial_space, power_space, sample_basis, bind_basis,
    observe, observe_rate, observation_metadata, king_observations

const DEFAULT_AFFINE_SLOPE = 0.005

struct SlopePolicy{F,P}
    mode::Symbol
    id::String
    evaluate::F
    parameters::P
end

struct ResolvedSlope{T<:Real,P}
    value::T
    time_tau::Float64
    mode::Symbol
    id::String
    parameters::P
end

function checked_slope(x)
    x isa Real && isfinite(x) || throw(ArgumentError("An affine slope must be a finite real scalar"))
    float(x)
end

function constant_slope(value=DEFAULT_AFFINE_SLOPE)
    x = checked_slope(value)
    SlopePolicy(:constant, "constant", (t, state) -> x, (; value=x))
end

function scheduled_slope(mode, times_tau, values)
    times = Tuple(Float64.(times_tau))
    slopes = Tuple(checked_slope(v) for v in values)
    !isempty(times) && length(times)==length(slopes) || throw(DimensionMismatch("Schedule knots and slope values"))
    all(t -> isfinite(t) && t>=0, times) && all(j -> times[j-1]<times[j], 2:length(times)) ||
        throw(ArgumentError("Schedule knots must be finite, nonnegative and strictly increasing"))
    function evaluate(t, state)
        # Piecewise-constant checkpoints are right-continuous. Hold the end
        # values outside the schedule for either interpolation rule.
        j = findlast(k -> k<=t, times)
        j === nothing && return first(slopes)
        (mode === :checkpoint || j == length(times)) && return slopes[j]
        fraction = (t-times[j])/(times[j+1]-times[j])
        (1-fraction)*slopes[j] + fraction*slopes[j+1]
    end
    SlopePolicy(mode, String(mode), evaluate, (; times_tau=times, values=slopes))
end

linear_time_slope(times_tau, values) = scheduled_slope(:linear_time, times_tau, values)
checkpoint_slope(times_tau, values) = scheduled_slope(:checkpoint, times_tau, values)

"""Evaluate a named, deterministic law f(time_tau, state) at the supplied state."""
function functional_slope(id::AbstractString, law)
    isempty(strip(id)) && throw(ArgumentError("A slope law requires a stable identifier"))
    SlopePolicy(:function, String(id), law, NamedTuple())
end

"""Accept the slope supplied by an outer CPE solve or a time-evolution state."""
function supplied_slope(source::Symbol)
    source in (:cpe, :evolved) || throw(ArgumentError("Expected :cpe or :evolved slope source"))
    function evaluate(t, state)
        state !== nothing && hasproperty(state, :lambda) ||
            throw(ArgumentError("Supply a state with a lambda field"))
        state.lambda
    end
    SlopePolicy(source, String(source), evaluate, NamedTuple())
end

function check_keys(spec, keys_allowed)
    all(key -> key in keys_allowed, keys(spec)) || throw(ArgumentError("Unexpected configuration keys"))
end

"""Read a constant, linear-time or checkpoint slope from a TOML table."""
function slope_policy(spec::AbstractDict)
    mode = get(spec, "mode", "constant")
    if mode == "constant"
        check_keys(spec, ("mode", "value"))
        return constant_slope(get(spec, "value", DEFAULT_AFFINE_SLOPE))
    end
    check_keys(spec, ("mode", "times_tau", "values"))
    mode == "linear-time" && return linear_time_slope(spec["times_tau"], spec["values"])
    mode == "checkpoint" && return checkpoint_slope(spec["times_tau"], spec["values"])
    throw(ArgumentError("Unknown affine slope mode $mode"))
end

function resolve_slope(policy::SlopePolicy; time_tau::Real=0, state=nothing)
    t = Float64(time_tau)
    isfinite(t) && t>=0 || throw(ArgumentError("time_tau must be finite and nonnegative"))
    value = checked_slope(policy.evaluate(t, state))
    ResolvedSlope(value, t, policy.mode, policy.id, policy.parameters)
end

"""Use the resolved stage value to construct temperature-normalized King widths."""
function affine_geometry(slope::ResolvedSlope, weights, squared_offsets)
    geometry = affine_widths(weights, squared_offsets, slope.value)
    all(isfinite, geometry.squared_widths) || throw(DomainError(slope.value, "Affine widths overflow"))
    (; geometry..., slope)
end

"""Convert Maxwell-normalized targets M0,M2,... using the same resolved stage value."""
function shell_targets(slope::ResolvedSlope, targets, d; precision_bits=256, diagonal_rtol=1e-12)
    0 < diagonal_rtol < 1 || throw(ArgumentError("Expected a relative diagonal threshold in (0,1)"))
    setprecision(BigFloat, precision_bits) do
        x = BigFloat(slope.value)
        for k in 1:length(targets)-1
            terms = BigFloat[]
            coefficient = big"1.0"
            for j in 0:k
                push!(terms, coefficient*x^(k-j))
                coefficient *= BigFloat(k-j)/((j+1)*(BigFloat(j)+big"1.5"))
            end
            abs(sum(terms)) > diagonal_rtol*sum(abs, terms) ||
                throw(DomainError(slope.value, "Affine transform diagonal is ill-conditioned for M$(2k)"))
        end
        affine_shell_moments(targets, d, x; precision_bits)
    end
end

struct RadialMeasure{T<:Real,W<:Real}
    radius::Vector{T}
    weights::Vector{W}
    coordinate::String
end

"""Bind nonnegative nodes to volume weights, or explicitly convert dr to 4πr²dr."""
function radial_measure(radius, weights; convention::Symbol, coordinate::AbstractString)
    r, w = collect(float.(radius)), collect(float.(weights))
    length(r) == length(w) && !isempty(r) || throw(DimensionMismatch("Radial nodes and weights"))
    all(v -> isfinite(v) && v>=0, r) && all(v -> isfinite(v) && v>=0, w) ||
        throw(ArgumentError("Radial nodes and weights must be finite and nonnegative"))
    isempty(strip(coordinate)) && throw(ArgumentError("Specify the radius coordinate or unit"))
    if convention === :dr
        w = (4 .* oftype.(r, π)) .* r.^2 .* w
    elseif convention !== :volume
        throw(ArgumentError("Expected :dr or :volume quadrature weights"))
    end
    all(isfinite, w) && any(>(0), w) || throw(ArgumentError("Invalid spherical volume weights"))
    RadialMeasure(r, w, String(coordinate))
end

struct RadialSpace{F,P}
    id::String
    labels::Vector{String}
    sampler::F
    parameters::P
end

struct PowerWeights{T}
    orders::Vector{Int}
    velocity_scale::T
    normalization::Symbol
end

function (p::PowerWeights)(radius)
    r = radius ./ p.velocity_scale
    [x^order/divisor(order, p.normalization, typeof(x)) for x in r, order in p.orders]
end

divisor(p, mode, T) = mode === :maxwell ? maxwell_moment_factor(p, T) : one(T)

"""Describe radial weights by a stable identifier, ordered labels, and matrix sampler."""
function radial_space(id::AbstractString, labels, sampler)
    names = String[labels...]
    !isempty(strip(id)) && !isempty(names) && all(n -> !isempty(strip(n)), names) &&
        allunique(names) || throw(ArgumentError("Specify a stable id and unique ordered basis labels"))
    RadialSpace(String(id), names, sampler, (; mode=:general))
end

"""Power modes are radial spaces with p-specific weights and explicit scaling."""
function power_space(mode::Symbol, highest_order::Integer; conserved::Bool=false,
                     velocity_scale::Real=1.0, normalization::Symbol=:raw)
    orders = Int.(speed_orders(mode, highest_order; conserved))
    !isempty(orders) || throw(ArgumentError("Power mode contains no observations"))
    isfinite(velocity_scale) && velocity_scale>0 || throw(ArgumentError("Expected positive velocity scale"))
    normalization in (:raw, :maxwell) || throw(ArgumentError("Expected :raw or :maxwell"))
    scale = float(velocity_scale)
    labels = ["M$p" for p in orders]
    RadialSpace("speed-power-" * String(mode), labels, PowerWeights(orders, scale, normalization),
        (; mode, highest_order, conserved, orders=Tuple(orders), velocity_scale=scale, normalization))
end

"""Read a concrete integer power mode from a TOML table."""
function power_space(spec::AbstractDict)
    check_keys(spec, ("mode", "highest_order", "conserved", "velocity_scale", "normalization"))
    power_space(Symbol(spec["mode"]), spec["highest_order"]; conserved=get(spec,"conserved",false),
        velocity_scale=get(spec,"velocity_scale",1.0), normalization=Symbol(get(spec,"normalization","raw")))
end

struct SampledSpace{S,M,B}
    space::S
    measure::M
    basis::B
end

"""Bind an existing basis matrix, with rows on this exact rule and columns in label order."""
function bind_basis(space::RadialSpace, measure::RadialMeasure, basis::AbstractMatrix)
    size(basis) == (length(measure.radius), length(space.labels)) ||
        throw(DimensionMismatch("Basis rows must follow radial nodes and columns must follow observation labels"))
    all(v -> v isa Real && isfinite(v), basis) || throw(ArgumentError("Nonfinite or nonreal basis values"))
    SampledSpace(space, measure, Matrix(basis))
end

sample_basis(space::RadialSpace, measure::RadialMeasure) = bind_basis(space, measure, space.sampler(measure.radius))

observe(sample::SampledSpace, density) = generalized_observations(sample.basis, sample.measure.weights, density)
observe(space::RadialSpace, measure::RadialMeasure, density) = observe(sample_basis(space, measure), density)

"""Integrate a signed collision source. Supply ∂tψ and f as a pair for a moving basis."""
function observe_rate(sample::SampledSpace, source; basis_time_derivative=nothing, density=nothing)
    w = sample.measure.weights
    length(source)==length(w) || throw(DimensionMismatch("Signed source and radial nodes"))
    all(v -> v isa Real && isfinite(v), source) || throw(ArgumentError("Nonfinite or nonreal signed source"))
    rate = sample.basis'*(w.*source)
    basis_time_derivative === nothing && density === nothing && return rate
    basis_time_derivative !== nothing && density !== nothing ||
        throw(ArgumentError("Moving weights require both basis_time_derivative and density"))
    derivative = bind_basis(sample.space, sample.measure, basis_time_derivative)
    rate + observe(derivative, density)
end

"""Report the sampled numerical rank using volume weights and column equilibration."""
function observation_metadata(sample::SampledSpace; rank_rtol::Real=1e-12)
    isfinite(rank_rtol) && 0<rank_rtol<1 || throw(ArgumentError("Expected rank threshold in (0,1)"))
    weighted = Float64.(sample.basis).*sqrt.(Float64.(sample.measure.weights))
    all(isfinite, weighted) || throw(ArgumentError("Weighted basis exceeds Float64 rank range"))
    for column in eachcol(weighted)
        scale = maximum(abs, column)
        iszero(scale) && continue
        column ./= scale
        column ./= sqrt(sum(abs2, column))
    end
    values = svdvals(weighted)
    rank = count(s -> s>rank_rtol*first(values), values)
    (; id=sample.space.id, labels=copy(sample.space.labels), sample.space.parameters,
        coordinate=sample.measure.coordinate, nodes=length(sample.measure.radius),
        dimension=length(sample.space.labels), rank, rank_rtol, singular_values=values)
end

"""Read power-space coordinates of a King mixture without radial quadrature."""
function king_observations(space::RadialSpace{<:PowerWeights}, weights, squared_widths, squared_offsets;
                           thermal_speed::Real, number::Real=1.0)
    isfinite(thermal_speed) && thermal_speed>0 && isfinite(number) && number>0 ||
        throw(ArgumentError("Expected positive thermal speed and number"))
    p = space.sampler
    m = king_moments(weights, squared_widths, squared_offsets, p.orders)
    scale = thermal_speed/p.velocity_scale
    [number*y*scale^order*(p.normalization === :raw ? maxwell_moment_factor(order, typeof(y)) : one(y))
        for (order,y) in zip(p.orders,m)]
end

end
