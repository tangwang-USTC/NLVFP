module KingFPR

export modified_spherical_bessel_i, king_physical, king_scaled, shifted_gaussian, gaussian_monopole_direct

"""
    modified_spherical_bessel_i(l, z)

Positive-real modified spherical Bessel function for integer `l ≥ 0`, evaluated
by its nonnegative power series. Intended for finite benchmark arguments; this
is not yet an asymptotically scaled large-argument implementation.
"""
function modified_spherical_bessel_i(l::Integer, z::T) where {T<:AbstractFloat}
    l >= 0 || throw(DomainError(l, "angular degree must be nonnegative"))
    isfinite(z) && z >= 0 || throw(DomainError(z, "argument must be finite and nonnegative"))
    z == 0 && return l == 0 ? one(T) : zero(T)
    z <= 80 || throw(DomainError(z, "series benchmark is restricted to z <= 80"))
    term = z^l
    for q in 1:l
        term /= T(2q + 1)
    end
    total = term
    for n in 0:9999
        term *= z^2 / (T(2(n + 1)) * T(2l + 2n + 3))
        next = total + term
        next == total && return total
        total = next
    end
    error("modified spherical Bessel series did not converge")
end

"""Physical King radial kernel of a shifted isotropic 3V Gaussian.
`v`, `u`, and `width` have the same speed unit; output has inverse speed-cubed.
The `l=0,u=0` limit is the centered Gaussian monopole.
"""
function king_physical(l::Integer, v::T, u::T, width::T) where {T<:AbstractFloat}
    l >= 0 || throw(DomainError(l))
    isfinite(v) && v >= 0 || throw(DomainError(v))
    isfinite(u) && u >= 0 || throw(DomainError(u))
    isfinite(width) && width > 0 || throw(DomainError(width))
    z = 2u * v / width^2
    prefactor = T(2l + 1) / (sqrt(T(pi))^3 * width^3)
    prefactor * exp(-(v^2 + u^2) / width^2) * modified_spherical_bessel_i(l, z)
end

"""King kernel in `x=v/reference_width`, retaining physical density amplitude.
`rho=component_width/reference_width` and `k=2u/component_width`.
"""
function king_scaled(l::Integer, x::T, k::T, reference_width::T, rho::T) where {T<:AbstractFloat}
    isfinite(rho) && rho > 0 || throw(DomainError(rho))
    isfinite(reference_width) && reference_width > 0 || throw(DomainError(reference_width))
    isfinite(k) && k >= 0 || throw(DomainError(k))
    king_physical(l, x * reference_width, k * rho * reference_width / 2, rho * reference_width)
end

"""Direct shifted 3V Gaussian at speed `v` and cosine to the shift axis `mu`."""
function shifted_gaussian(v::T, mu::T, u::T, width::T) where {T<:AbstractFloat}
    -1 <= mu <= 1 || throw(DomainError(mu))
    isfinite(width) && width > 0 || throw(DomainError(width))
    exp(-(v^2 + u^2 - 2v * u * mu) / width^2) / (sqrt(T(pi))^3 * width^3)
end

"""Independent angular average of the direct Gaussian using composite Simpson quadrature."""
function gaussian_monopole_direct(v::T, u::T, width::T; panels::Int=256) where {T<:AbstractFloat}
    panels > 0 && iseven(panels) || throw(ArgumentError("panels must be positive and even"))
    h = T(2) / panels
    total = shifted_gaussian(v, -one(T), u, width) + shifted_gaussian(v, one(T), u, width)
    for j in 1:panels-1
        mu = -one(T) + j*h
        total += (iseven(j) ? 2 : 4) * shifted_gaussian(v, mu, u, width)
    end
    total * h / 6
end

end
