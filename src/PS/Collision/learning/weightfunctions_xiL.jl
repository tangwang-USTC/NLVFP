"""
  The `ℓᵗʰ`-order coefficients of normalized distribution function will be:

    f̂ₗᵐ(v̂,ℓ,m=0;μᵤ=1) = (2ℓ+1)//2 * (μᵤ)^ℓ * (n̂ / v̂ₜₕ³) * exp((-û²-v̂²)/v̂ₜₕ²) * superRBFv(ℓ,ξ)

    superRBFv(ℓ,ξ) = 1 / ξ^(ℓ+1/2) * besseli(1/2 + ℓ, ξ)

  where

    ξ = 2 * û * v̂ / v̂ₜₕ²
    n̂ = nₛ / n₀
    v̂ₜₕ = vₜₕₛ / vₜₕ₀
    û = uₛ / vₜₕ₀
    v̂ = v / vₜₕ₀
    μᵤ = ± 1

  and `n₀`, `vₜₕ₀` are the effective values (or named as experimental values) of the specified total distribution function.

"""

# # Gaussian models for Maxwellian distribution
# modelMexp(v,p::AbstractVector{Float64}) = exp.(- ((v .- p[2]) / p[1]).^2)
modelMexp(v,p::AbstractVector{Float64}) = exp.(- (v / p[1]).^2) + 0v * p[2]
modelMexp(v,p::Float64) = exp.(- (v / p).^2)

# # Normalized Harmonic models for drift-Maxwellian distribution
# f̂ₗ(v̂) / (2u * v)ᴸ
modelDMc(μu,p,ℓ::Int) = μu^ℓ * (ℓ + 0.5)  * sqrt2pi / p^3
modelDMexp(v,μu,p,ℓ::Int) = modelDMc(μu,p[1],ℓ) / (2p[2] / p[1]^2).^(ℓ+0.5) *
        exp.(- (v.^2 / p[1]^2 .+ p[2]^2 / p[1]^2)) ./ v.^(ℓ+0.5) .* besseli.(0.5+ℓ,(2p[2] / p[1]^2 * v))

# when `v → 0`   # modelDMc(μu,p[1],ℓ) * √(2 / π)
modelDMv0c(μu,p,ℓ::Int) = μu^ℓ / prod(3:2:(2ℓ-1)) / p^3
modelDMexpv0(v::Vector{Float64},μu,p,ℓ) = modelDMv0c(μu,p[1],ℓ) * exp.(- ((v.^2 .+ p[2]^2) / p[1]^2))
modelDMexpv0(v::Float64,μu,p,ℓ) = modelDMv0c(μu,p[1],ℓ) * exp(- ((v^2 + p[2]^2) / p[1]^2))
modelDMexpv0(v::Float64,μu,p) = exp.(- (p[2] / p[1])^2) / p[1]^3
