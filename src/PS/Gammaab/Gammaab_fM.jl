
"""
  Computing the collision coefficients of Fokker-Planck collision operator, including the Coulomb algorithm.

    lnAg = cf3[isp] * 4π * ( ZₐZᵦ / (4πεᵣmₐ))^2 * lnA()

         = cf3[isp] / 4π * ( ZₐZᵦ / (εᵣmₐ))^2 * lnA()

  if `is_normδtf=false`, coefficient `cf3[isp]` is also taken into accout when the integrals process omitting the coefficient `cf3[isp]`.

    lnAg = 4π * lnAg

         = cf3[isp] * ( ZₐZᵦ / (εᵣmₐ))^2 * lnA()

  The factor `4π` is owing to the background distribution function `FLm` and `HLM`, `GLM`
  
    `CΓ = 1 / (n₀kₚ₀³)`, 
  the dimensionless parameter is included in other codes.

  Inputs:
    ma: = ma[isp] / Dₐ
    Zqab: = Za * Zb
    na: = na[isp] / n20
    nb: = na[iFv] / n20
    vath: = vth[isp] / Mms
    vbth: = vth[iFv] / Mms
    is_normδtf: (default: true) owing to the integrals process has taken into account the coefficients `cf3[isp]` acquiescently.

  Outputs:
    lnAg = lnAgamma_fM(ma,Zq,spices,na,vath,εᵣ,isp,iFv;is_normδtf=is_normδtf)
    lnAg = lnAgamma_fM(ma,Zq,spices,na,vath,εᵣ;is_normδtf=is_normδtf)
"""
# [ns], nMod = 1
function lnAgamma_fM(ma::AbstractVector{T},Zq::Vector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},vath::AbstractVector{T},
    εᵣ::T,isp::Int64,iFv::Int64;is_normδtf::Bool=false) where{T}
    
    Ta = 0.5 * ma .* vath.^2
    lnAab = lnAab_fM(ma,Zq,spices,na,Ta,T_unit,logT_unit)
    if is_normδtf
        return (na[iFv] / vath[iFv]^3) / sqrtpi3 * (Zq[isp] * Zq[iFv] / ma[isp] / εᵣ)^2 * lnAab
    else
        return (na[isp] / vath[isp]^3) * (na[iFv] / vath[iFv]^3) / pi3 * (Zq[isp] * Zq[iFv] / ma[isp] / εᵣ)^2 * lnAab
    end
end

# [nMod=2], ns = 1
function lnAgamma_fM(ma::T,Zq::Int64,spices::Symbol,
  na::AbstractVector{T},vath::AbstractVector{T},
  εᵣ::T,isp::Int64,iFv::Int64;is_normδtf::Bool=false) where{T}
  
  Ta = 0.5 * ma .* vath.^2
  lnAab = lnAab_fM(Zq,spices,na,Ta,T_unit,logT_unit)
  if is_normδtf
      return (na[iFv] / vath[iFv]^3) / sqrtpi3 * (Zq^2 / ma / εᵣ)^2 * lnAab
  else
      return (na[isp] / vath[isp]^3) * (na[iFv] / vath[iFv]^3) / pi3 * (Zq^2 / ma / εᵣ)^2 * lnAab
  end
end

# [], ns = 1, isSelf = true
function lnAgamma_fM(ma::T,Zq::Int64,spices::Symbol,
  na::T,vath::T,εᵣ::T;is_normδtf::Bool=false) where{T}
    
    Ta = 0.5 * ma * vath^2
    lnAab = lnAab_fM(Zq,spices,na,Ta,T_unit,logT_unit)
    if is_normδtf
        return (na / vath^3) / sqrtpi3 * (Zq^2 / ma / εᵣ)^2 * lnAab
    else
        return (na / vath^3)^2 / pi3 * (Zq^2 / ma / εᵣ)^2 * lnAab
    end
end