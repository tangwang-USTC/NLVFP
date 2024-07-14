
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
    lnAg = lnAgamma(εᵣ,ma,Zqab,spices,na,nb,vath,vbth;is_normδtf=is_normδtf)
    lnAg = lnAgamma(εᵣ,ma,Zqab,spices,na,vath;is_normδtf=is_normδtf)
"""

function lnAgamma(εᵣ::T,ma::T,Zqab::Int64,spices::Vector{Symbol},
  na::T,nb::T,vath::T,vbth::T;is_normδtf::Bool=false) where{T}
    
    lnAab = lnA_const(spices)
    # lnAab = lnA_const()
    if is_normδtf
        return (nb / vbth^3) / sqrtpi3 * (Zqab / ma / εᵣ)^2 * lnAab
    else
        return (na / vath^3) * (nb / vbth^3) / pi3 * (Zqab / ma / εᵣ)^2 * lnAab
    end
end

# isSelf = true
function lnAgamma(εᵣ::T,ma::T,Zqab::Int64,spices::Symbol,
  nb::T,vbth::T;is_normδtf::Bool=false) where{T}
    
  lnAab = lnA_const(spices)
  # lnAab = lnA_const()
    if is_normδtf
        return (nb / vbth^3) / sqrtpi3 * (Zqab / ma / εᵣ)^2 * lnAab
    else
        return (nb / vbth^3)^2 / pi3 * (Zqab / ma / εᵣ)^2 * lnAab
    end
end

