# The owning vector is copied on construction. The public array contract is read-only.
struct ReadOnlyVector{T} <: AbstractVector{T}
    storage::Vector{T}
end

Base.size(v::ReadOnlyVector) = size(getfield(v, :storage))
Base.getindex(v::ReadOnlyVector, i::Int) = getfield(v, :storage)[i]
Base.IndexStyle(::Type{<:ReadOnlyVector}) = IndexLinear()

struct RadialMeasure{R,W}
    radius::ReadOnlyVector{R}
    weights::ReadOnlyVector{W}
    coordinate::String

    function RadialMeasure(radius, weights, coordinate)
        r = collect(float.(radius))
        w = collect(float.(weights))
        length(r) == length(w) && !isempty(r) ||
            throw(DimensionMismatch("Radial nodes and weights must be nonempty and the same length"))
        all(is_nonnegative_finite, r) && all(is_nonnegative_finite, w) && any(>(0), w) ||
            throw(ArgumentError("Radial nodes and weights must be real, finite and nonnegative, with positive total weight"))
        isempty(strip(coordinate)) && throw(ArgumentError("Specify a velocity coordinate and unit"))
        new{eltype(r),eltype(w)}(ReadOnlyVector(r), ReadOnlyVector(w), String(coordinate))
    end
end

is_nonnegative_finite(v) = v isa Real && isfinite(v) && v >= 0
is_real_finite(v) = v isa Real && isfinite(v)

"""
    radial_measure(radius, weights; convention, coordinate)

Construct a radial rule with an explicit coordinate label. Use :dr for 1D
quadrature weights or :volume for weights that already include 4πr².
"""
function radial_measure(radius, weights; convention::Symbol, coordinate::AbstractString)
    length(radius) == length(weights) || throw(DimensionMismatch("Radial nodes and weights"))
    all(is_nonnegative_finite, radius) && all(is_nonnegative_finite, weights) ||
        throw(ArgumentError("Invalid raw radial nodes or weights"))
    if convention === :dr
        r = float.(radius)
        return RadialMeasure(r, 4 .* oftype.(r, π) .* r.^2 .* weights, coordinate)
    end
    convention === :volume || throw(ArgumentError("Expected :dr or :volume quadrature"))
    RadialMeasure(radius, weights, coordinate)
end

struct BoundObservations{M,T,P}
    measure::M
    _basis::Matrix{T}
    labels::Tuple
    id::String
    parameters::P

    function BoundObservations(space::RadialSpace, measure::RadialMeasure, basis::AbstractMatrix)
        size(basis) == (length(measure.radius), length(space.labels)) ||
            throw(DimensionMismatch("Basis rows are radial nodes and columns are labelled observables"))
        all(is_real_finite, basis) || throw(ArgumentError("Basis must contain finite real values"))
        labels = Tuple(space.labels)
        !isempty(labels) && allunique(labels) || throw(ArgumentError("Expected unique, nonempty basis labels"))
        copied = Matrix(float.(basis))
        new{typeof(measure),eltype(copied),typeof(space.parameters)}(
            measure, copied, labels, space.id, space.parameters)
    end
end

"""Bind sampled columns by copying the input matrix, in the declared label order."""
bind_basis(space::RadialSpace, measure::RadialMeasure, basis::AbstractMatrix) =
    BoundObservations(space, measure, basis)

sample_basis(space::RadialSpace, measure::RadialMeasure) =
    bind_basis(space, measure, space.sampler(measure.radius))

"""Scratch buffer for projections on a fixed radial rule."""
struct ObservationWorkspace{T}
    weighted::Vector{T}
end

ObservationWorkspace(sample::BoundObservations, ::Type{T}) where {T<:AbstractFloat} =
    ObservationWorkspace(zeros(T, length(sample.measure.radius)))

ObservationWorkspace(sample::BoundObservations) =
    ObservationWorkspace(sample, promote_type(eltype(sample._basis),eltype(sample.measure.weights)))

function project!(output, workspace, sample, density, check)
    n, p = size(sample._basis)
    length(density) == length(workspace.weighted) == n && length(output) == p ||
        throw(DimensionMismatch("Projection buffer, coordinates or input data"))
    (Base.mightalias(output, workspace.weighted) || Base.mightalias(density, workspace.weighted)) &&
        throw(ArgumentError("Projection input and output must not alias workspace storage"))
    all(check, density) || throw(ArgumentError("Invalid radial input values for projection"))
    workspace.weighted .= sample.measure.weights .* density
    mul!(output, transpose(sample._basis), workspace.weighted)
    all(isfinite, output) || throw(OverflowError("Radial projection exceeded numeric range"))
    output
end

"""Project a nonnegative density into a caller-owned output and scratch buffer."""
observe!(output, workspace::ObservationWorkspace, sample::BoundObservations, density) =
    project!(output, workspace, sample, density, is_nonnegative_finite)

function projection_buffer(sample, values)
    T = promote_type(eltype(sample._basis),eltype(sample.measure.weights),float(eltype(values)))
    zeros(T, length(sample.labels)), ObservationWorkspace(sample, T)
end

function observe(sample::BoundObservations, density)
    output, scratch = projection_buffer(sample,density)
    observe!(output, scratch, sample, density)
end

observe(space::RadialSpace, measure::RadialMeasure, density) = observe(sample_basis(space,measure),density)

"""Project a signed source and optionally add the moving-observable term Ḃ'Wf."""
function observe_rate(sample::BoundObservations, source; basis_time_derivative=nothing, density=nothing)
    output, scratch = projection_buffer(sample,source)
    project!(output, scratch, sample, source, is_real_finite)
    if basis_time_derivative === nothing && density === nothing
        return output
    end
    basis_time_derivative !== nothing && density !== nothing ||
        throw(ArgumentError("Supply both basis_time_derivative and density"))
    size(basis_time_derivative) == size(sample._basis) ||
        throw(DimensionMismatch("Moving weight derivative must have the same grid and columns"))
    length(density) == length(sample.measure.weights) && all(is_nonnegative_finite, density) &&
        all(is_real_finite, basis_time_derivative) || throw(ArgumentError("Invalid moving-weight integrand"))
    result = output + transpose(basis_time_derivative)*(sample.measure.weights.*density)
    all(isfinite, result) || throw(OverflowError("Moving-weight projection exceeded numeric range"))
    result
end

"""Return the column-equilibrated, volume-weighted Float64 rank diagnostic."""
function observation_metadata(sample::BoundObservations; rank_rtol=1e-12)
    0 < rank_rtol < 1 || throw(ArgumentError("Expected a relative rank threshold in (0,1)"))
    weighted = Float64.(sample._basis) .* sqrt.(Float64.(sample.measure.weights))
    all(isfinite, weighted) || throw(ArgumentError("Weighted basis exceeds Float64 rank range"))
    for col in eachcol(weighted)
        scale = maximum(abs, col)
        if !iszero(scale)
            col ./= scale
            col ./= norm(col)
        end
    end
    σ = svdvals(weighted)
    (; sample.id, sample.labels, sample.parameters, sample.measure.coordinate,
       nodes=size(weighted,1), dimension=size(weighted,2), rank=count(>(rank_rtol*first(σ)),σ),
       rank_rtol, singular_values=σ, rank_precision=Float64)
end

struct SpeedMomentTarget{T,S}
    orders::Tuple
    values::ReadOnlyVector{T}
    normalization::Symbol
    velocity_scale::S
    coordinate::String
    function SpeedMomentTarget(orders, values, normalization::Symbol, scale::Real, coordinate::AbstractString)
        orders = Tuple(orders)
        v = collect(float.(values))
        length(v) == length(orders) && !isempty(v) || throw(DimensionMismatch("Moment labels and target values"))
        all(p -> p isa Integer && p >= 0, orders) && allunique(orders) && issorted(orders) ||
            throw(ArgumentError("Moment orders must increase strictly and be nonnegative integers"))
        all(x -> is_real_finite(x) && x > 0, v) || throw(ArgumentError("Moment targets must be finite and positive"))
        normalization in (:raw,:maxwell,:thermal_maxwell) || throw(ArgumentError("Unknown moment normalization"))
        isfinite(scale) && scale > 0 && !isempty(strip(coordinate)) ||
            throw(ArgumentError("Supply a positive velocity scale and coordinate/unit"))
        new{eltype(v),typeof(float(scale))}(orders,ReadOnlyVector(v),normalization,float(scale),String(coordinate))
    end
end

"""
    speed_moment_target(orders, values; normalization, velocity_scale, coordinate)

:thermal_maxwell means divided by each same-temperature Maxwell moment,
including the species number. The other conventions denote raw or Cₚ-scaled
power integrals in the recorded coordinate with the given velocity scale.
"""
speed_moment_target(orders, values; normalization, velocity_scale, coordinate) =
    SpeedMomentTarget(orders,values,normalization,velocity_scale,coordinate)

"""Transform only unit-number, same-temperature Maxwell-normalized consecutive even moments."""
function shell_targets(slope::ResolvedSlope, target::SpeedMomentTarget, d; normalization_rtol=2e-12, kwargs...)
    target.normalization === :thermal_maxwell ||
        throw(ArgumentError("Shell inversion requires :thermal_maxwell normalization"))
    n = length(target.orders)
    n>=2 && target.orders==Tuple(0:2:2(n-1)) ||
        throw(ArgumentError("Shell inversion requires orders M0,M2,… without gaps"))
    0 < normalization_rtol < 1 || throw(ArgumentError("Invalid normalization tolerance"))
    max(abs(target.values[1]-1),abs(target.values[2]-1)) <= normalization_rtol ||
        throw(ArgumentError("Same-temperature normalization requires M0=M2=1"))
    EDCase1Interfaces.shell_targets(slope,target.values,d;kwargs...)
end
