# """
#   The `ℓᵗʰ`-order coefficients of normalized distribution function will be:
#
#     f̂ₗᵐ(v̂,ℓ,m=0;μᵤ=1) = (2ℓ+1)//2 * (μᵤ)^ℓ * (n̂ / v̂ₜₕ³) * exp((-û²-v̂²)/v̂ₜₕ²) * superRBFv(ℓ,ξ)
#
#     superRBFv(ℓ,ξ) = 1 / √ξ * besseli(1/2 + ℓ, ξ)
#
#   where
#
#     ξ = 2 * û * v̂ / v̂ₜₕ²
#     n̂ = nₛ / n₀
#     v̂ₜₕ = vₜₕₛ / vₜₕ₀
#     û = uₛ / vₜₕ₀
#     v̂ = v / vₜₕ₀
#     μᵤ = ± 1
#
#   and `n₀`, `vₜₕ₀` are the effective values (or named as experimental values) of
#   the specified total distribution function `f̂ₗᵐ(v̂)`.
#
#   When 'ξ → 0' leads to:
#
#     f̂ₗ(v̂) = (μᵤ)^ℓ * (n̂ / v̂ₜₕ³) / (2ℓ-1)!! * exp((-û²-v̂²)/v̂ₜₕ²) * superRBFv(ℓ,ξ)
#
#   where
#
#     superRBFv(ℓ,ξ) = ξ^ℓ * (1 + ∑ₙ(cnL * ξ^(2n))), n ∈ 1:1:nM - ℓ
#
#     cnL = 1 / (2ⁿn!) * (2ℓ+1)!!/(2ℓ+2n+1)!!
#
#   Especially when`û = 0` yields:
#
#     f̂₀(v̂) = (n̂ / v̂ₜₕ³) * exp(- v̂² / v̂ₜₕ²)
#
#   Coefficient `1 / π^(3/2)` is not included in this step for `f̂ₗᵐ(v̂)`.
# """
#
# # # Gaussian models for Maxwellian distribution
# modelMexp(v,p::AbstractVector{Float64}) = exp.(- ((v .- p[2]) / p[1]).^2)
# modelMexp(v,p::Float64) = exp.(- (v / p).^2)
#
# # # Normalized Harmonic models for drift-Maxwellian distribution
#  # p = [vth, ua]
# modelDMc(p,ℓ::Int) = (ℓ + 0.5) * sqrt2pi / p^3
# modelDMexp(v,p,ℓ::Int) = modelDMc(p[1],ℓ) * p[1] / (2p[2]).^0.5 ./ v.^0.5 .*
#           exp.(- (v.^2 / p[1]^2 .+ p[2]^2 / p[1]^2)) .* besseli.(0.5+ℓ,(2p[2] / p[1]^2 * v))
# # modelDMexp(v,p,ℓ::Int) = modelDMc(p[1],ℓ) * exp.(- (v.^2 / p[1]^2 .+ p[2]^2 / p[1]^2)) ./
# #                            (2p[2] / p[1]^2 * v).^0.5 .* besseli.(0.5+ℓ,(2p[2] / p[1]^2 * v))
#
# # when `v → 0`   # modelDMc(p[1],ℓ) * √(2 / π)
# modelDMv0c(p,ℓ::Int) = 1 / prod(3:2:(2ℓ-1)) / p^3
# modelDMexpv0(v::Vector{Float64},p,ℓ::Int) = modelDMv0c(p[1],ℓ) * (2p[2] / p[1]^2)^ℓ *
#                                          exp.(- ((v.^2 .+ p[2]^2) / p[1]^2)) .* v.^ℓ
# modelDMexpv0(v::Float64,p,ℓ::Int) = modelDMv0c(p[1],ℓ) * exp(- ((v^2 + p[2]^2) / p[1]^2)) * (2p[2] / p[1]^2 * v)^ℓ
# modelDMexpv0(v::Float64,p) = exp.(- (p[2] / p[1])^2) / p[1]^3
#
# # modelDMexpv0(v,p,L) = (1.0 + (2 / (2L+3) * u^2 - p[1]) * v.^2) .* v.^L
#
# """
#
#   Outputs:
#     fL0DMt = fL0DMmodel(L,na,vth,ua)
#     fL0DMt = fL0DMmodel(L,na,ua)
#     fL0DMt = fL0DMmodel(L,ua)
#     fL0DMv0t = fL0DMv0model(L,na,vth,ua)
# """
#
# function fL0DMmodel(L::Int,na::T,vth::T,u::T) where{T}
#
#     if vth == 1.0
#         if na == 1.0
#             return fL0DMmodel(L,u)
#         else
#             return fL0DMmodel(L,na,u)
#         end
#     else
#         if L == 0
#             if u == 0
#                 return v -> na/vth^3 * exp.(- (v.^2 / vth^2))
#             else
#                 return v -> na / vth / (2u) ./ v .* exp.(- (v.^2 / vth^2 .+ (u / vth)^2)) .* sinh.(2u / vth^2 * v)
#                 # return v -> na/vth^3 * (L + 1/2)  * sqrt2pi ./ (2u / vth^2 * v).^0.5 .*
#                 #             exp.(- (v.^2 / vth^2 .+ (u / vth)^2)) .* besseli.(0.5+L,(2u / vth^2 * v))
#                 # return v -> na/vth^3 ./ (2u / vth^2 * v) .* exp.(- (v.^2 / vth^2 .+ (u / vth)^2)) .* sinh.(2u / vth^2 * v)
#             end
#         else
#             if L == 1
#                 return v -> na/vth * 3 / (2u) ./ v .* exp.(- (v.^2 / vth^2 .+ (u / vth)^2)) .*
#                             (cosh.(2u / vth^2 * v) - sinh.(2u / vth^2 * v) ./ (2u / vth^2 * v))
#                 # return v -> 3 * na/vth^3 ./ (2u / vth^2 * v) .* exp.(- (v.^2 / vth^2 .+ (u / vth)^2)) .*
#                 #             (cosh.(2u / vth^2 * v) - sinh.(2u / vth^2 * v) ./ (2u / vth^2 * v))
#             else
#                 if isodd(L)
#                     return v -> na/vth^2 * (L + 1/2)  * sqrt2pi / (2u).^0.5 ./ v.^0.5 .*
#                                 exp.(- (v.^2 / vth^2 .+ (u / vth)^2)) .* besseli.(0.5+L,(2u / vth^2 * v))
#                 else
#                     return v -> abs(μu) * na/vth^2 * (L + 1/2)  * sqrt2pi / (2u).^0.5 ./ v.^0.5 .*
#                                 exp.(- (v.^2 / vth^2 .+ (u / vth)^2)) .* besseli.(0.5+L,(2u / vth^2 * v))
#                     # return v -> na/vth^3 * (L + 1/2)  * sqrt2pi ./ (2u / vth^2 * v).^0.5 .*
#                     #             exp.(- (v.^2 / vth^2 .+ (u / vth)^2)) .* besseli.(0.5+L,(2u / vth^2 * v))
#                 end
#             end
#         end
#     end
#
# end
#
# # `vth = 1`
# function fL0DMmodel(L::Int,na::T,u::T) where{T}
#
#     if L == 0
#         if u == 0
#             return v -> na * exp.(- v.^2)
#         else
#             return v -> na / (2u) ./ v .* exp.(- (v.^2 .+ u^2)) .* sinh.(2u * v)
#             # return v -> na * (L + 1/2)  * sqrt2pi ./ (2u * v).^0.5 .*
#             #               exp.(- (v.^2 .+ u^2)) .* besseli.(0.5+L,(2u * v))
#             # return v -> na ./ (2u * v) .* exp.(- (v.^2 .+ u^2)) .* sinh.(2u * v)
#         end
#     else
#         if L == 1
#             return v -> na * 3 / (2u) ./ v .* exp.(- (v.^2 .+ u^2)) .*
#                         (cosh.(2u * v) - sinh.(2u * v) ./ (2u * v))
#             # return v -> na * 3 ./ (2u * v) .* exp.(- (v.^2 .+ u^2)) .*
#             #             (cosh.(2u * v) - sinh.(2u * v) ./ (2u * v))
#         else
#             if isodd(L)
#                 return v -> na * (L + 1/2)  * sqrt2pi / (2u).^0.5 ./ v.^0.5 .*
#                             exp.(- (v.^2 .+ u^2)) .* besseli.(0.5+L,(2u * v))
#             else
#                 return v -> na * abs(μu) * (L + 1/2)  * sqrt2pi / (2u).^0.5 ./ v.^0.5 .*
#                             exp.(- (v.^2 .+ u^2)) .* besseli.(0.5+L,(2u * v))
#                 # return v -> na * (L + 1/2)  * sqrt2pi ./ (2u * v).^0.5 .*
#                 #             exp.(- (v.^2 .+ u^2)) .* besseli.(0.5+L,(2u * v))
#             end
#         end
#     end
# end
#
# # `na = 1` and `vth = 1`
# function fL0DMmodel(L::Int,u::T) where{T}
#
#     if L == 0
#         if u == 0
#             return v -> exp.(- v.^2)
#         else
#             return v -> 1.0 / (2u) ./ v .* exp.(- (v.^2 .+ u^2)) .* sinh.(2u * v)
#             # return v -> (L + 1/2)  * sqrt2pi ./ (2u * v).^0.5 .*
#             #             exp.(- (v.^2 .+ u^2)) .* besseli.(0.5+L,(2u * v))
#             # return v -> 1.0 ./ (2u * v) .* exp.(- (v.^2 .+ u^2)) .* sinh.(2u * v)
#         end
#     else
#         if L == 1
#             return v -> 3 / (2u) ./ v .* exp.(- (v.^2 .+ u^2)) .*
#                         (cosh.(2u * v) - sinh.(2u * v) ./ (2u * v))
#             # return v -> 3 ./ (2u * v) .* exp.(- (v.^2 .+ u^2)) .*
#             #             (cosh.(2u * v) - sinh.(2u * v) ./ (2u * v))
#         else
#             if isodd(L)
#                 return v -> (L + 1/2)  * sqrt2pi / (2u).^0.5 ./ v.^0.5 .*
#                             exp.(- (v.^2 .+ u^2)) .* besseli.(0.5+L,(2u * v))
#             else
#                 return v -> abs(μu) * (L + 1/2)  * sqrt2pi / (2u).^0.5 ./ v.^0.5 .*
#                             exp.(- (v.^2 .+ u^2)) .* besseli.(0.5+L,(2u * v))
#                 # return v -> (L + 1/2)  * sqrt2pi ./ (2u * v).^0.5 .*
#                 #             exp.(- (v.^2 .+ u^2)) .* besseli.(0.5+L,(2u * v))
#             end
#         end
#     end
# end
#
# """
#   When 'ξ → 0' leads to:
#
#      f̂ₗ(v̂) = (μᵤ)^ℓ * (n̂ / v̂ₜₕ³) / (2ℓ-1)!! * exp((-û²-v̂²)/v̂ₜₕ²) * superRBFv(ℓ,ξ)
#
#    where
#
#      superRBFv(ℓ,ξ) = ξ^ℓ * (1 + ∑ₙ(cnL * ξ^(2n))), n ∈ 1:1:nM - ℓ
#
#      cnL = 1 / (2ⁿn!) * (2ℓ+1)!!/(2ℓ+2n+1)!!
#
#    Especially when`û = 0` yields:
#
#      f̂₀(v̂) = (n̂ / v̂ₜₕ³) * exp(- v̂² / v̂ₜₕ²)
#
# """
#
#
# function fL0DMv0model(L::Int,na::Float64,vth::Float64,u::Float64)
#
#     if L == 0
#         if vth == 1.0
#             if u == 0.0
#                 return v -> na * exp(- v^2)
#             else
#                 return v -> na * exp(- (v^2 + u^2))
#             end
#         else
#             if u == 0.0
#                 return v -> na / vth^3 * exp(- (v^2 / vth^2))
#             else
#                 return v ->  na / vth^3 * exp(- ((v^2 + u^2) / vth^2))
#             end
#         end
#     else
#         return v -> 0
#         # if L == 1
#         #     return v -> na/vth * (2u * v) * exp(- ((v^2 + u^2) / vth^2))
#         #     # return v -> na/vth^3 * (2u / vth^2 * v) * exp(- ((v^2 + u^2) / vth^2))
#         # else
#         #     if isodd(L)
#         #         return v -> na/vth^3 * (2u / vth^2 * v)^L / prod(3:2:2L+1) * exp(- ((v^2 + u^2) / vth^2))
#         #     else
#         #         return v -> na/vth^3 * (2u / vth^2 * v)^L / prod(3:2:2L+1) * exp(- ((v^2 + u^2) / vth^2))
#         #     end
#         # end
#     end
# end
