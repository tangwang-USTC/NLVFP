

"""
  Coefficients for analytical results of Shkarofsky integrals:

    Cₗᵉʳᶠ = √π / 2³ * (ûᵦ)ᴸ

    Cₗ₊₂ᵉʳᶠ = Cₗᵉʳᶠ × [(L + 3/2) * (v̂ᵦₜₕₛ)² + (ûᵦ)²]

    Cₗ₋₁ᵉʳᶠ = √π / 2³ * (v̂ / ûᵦ)ᴸ

    Cₗ₊₁ᵉʳᶠ = √π / 2³ * [1 - (L - 1/2) * (v̂ᵦₜₕₛ / ûᵦ)²] * (v̂ / ûᵦ)ᴸ⁻¹

    Cᵢᵉˣᵖ = 1 / 2² v̂ᵦₜₕₛ * (2 ûᵦ)ᴸ⁺¹
 
            1 / 2² * [v̂ᵦₜₕₛ / (2 ûᵦ)ᴸ⁺¹] * (1 / v̂),   L = 0,
    Cᵢᵉˣᵖ = 1 / 2 * [v̂ᵦₜₕₛ / (2 ûᵦ)ᴸ⁺¹],              1 ≤ L ≤ 2,
            1 / 2 * [v̂ᵦₜₕₛ / (2 ûᵦ)ᴸ⁺¹] * (1 / v̂ᴸ⁻²), L ≥ 3.
    
    Cⱼᵖ = ℂ𝕏puⱼⁿᵐ [v̂⁰, v̂¹, v̂², ⋯, v̂ᵐ]ᵀ,  j ∈ [L, L + 2, L + 1, L - 1],
    Cⱼⁿ = ℂ𝕏nuⱼⁿᵐ [v̂⁰, v̂¹, v̂², ⋯, v̂ᵐ]ᵀ.

  where

    ℂ𝕏puⱼⁿᵐ = [ûᵦ⁰, ûᵦ¹, ûᵦ², ⋯, ûᵦⁿ] (ℂⱼᵖ × 𝕏ⱼⁿᵐ)
  
  Notes:
    
    Factor `v̂ᵏ` in above equations will be transformed the main body of the Shkarofsky integrals.

  Inputs:

  Outputs:
    M = 
"""

"""
  Intputs:
    L:

  Outputs:
    CLerf = CLerfL(uhbs,vhbths,L)
    CL2erf = CL2erfL(uhbs,vhbths,L)
    CLn1erf = CLn1erfL(uhbs,vhbths,L)
    CL1erf = CL1erfL(uhbs,vhbths,L)
    CIexp = CIexpL(uhbs,vhbths,L)
    CJexp = CJexpL(uhbs,vhbths,L)

"""

function CLerfL(uhbs::T,vhbths::T,L::Int64) where{T}

    if abs(uhbs) ≤ 2epsT
      if L == 0
        return sqrtpi / 8
      else 
        return 0.0
      end
    else
      if L == 0
        return sqrtpi / 8
      elseif L == 1
        return sqrtpi / 8 * uhbs
      else 
        return sqrtpi / 8 * uhbs^L
      end
    end
end

function CL2erfL(uhbs::T,vhbths::T,L::Int64) where{T}

  if abs(uhbs) ≤ 2epsT
    if L == 0
      return sqrtpi / 8 * vhbths^2 * (3/2)
    else
      return 0.0
    end
  else
    if L == 0
      return sqrtpi / 8 * vhbths^2 * (3/2 + (uhbs / vhbths)^2)
    elseif L == 1
      return sqrtpi / 8 * uhbs * vhbths^2 * (5/2 + (uhbs / vhbths)^2)
    else 
      return sqrtpi / 8 * uhbs^L * vhbths^2 * (L + 3/2 + (uhbs / vhbths)^2)
    end
  end
end


function CLn1erfL(uhbs::T,vhbths::T,L::Int64) where{T}

  if abs(uhbs) ≤ 2epsT
    return error("`uhbs = 0` is not convergent, please applying the form of Tayler expansion!")
  else
    return sqrtpi / 8 * uhbs^L
  end
end

function CL1erfL(uhbs::T,vhbths::T,L::Int64) where{T}

  if abs(uhbs) ≤ 2epsT
    return error("`uhbs = 0` is not convergent, please applying the form of Tayler expansion!")
  else
    if L == 0
      return sqrtpi / 8 * (uhbs + (1/2) * (vhbths^2 / uhbs))
    elseif L == 1
      return sqrtpi / 8 * (1 - (1/2) * (vhbths / uhbs)^2)
    elseif L == 2
      return sqrtpi / 8 / uhbs * (1 - (3/2) * (vhbths / uhbs)^2)
    else 
      return sqrtpi / 8 / uhbs^(L-1) * (1 - (L - 1/2) * (vhbths / uhbs)^2)
    end
  end
end

function CIexpL(uhbs::T,vhbths::T,L::Int64) where{T}

  return 0.25 * vhbths * (2uhbs)^(L + 1)
end


function CJexpL(uhbs::T,vhbths::T,L::Int64) where{T}

  if abs(uhbs) ≤ 2epsT
    return error("`uhbs = 0` is not convergent, please applying the form of Tayler expansion!")
  else
    if L == 0
      return 0.125 * vhbths / uhbs
    else
      return 0.5 * vhbths / (2uhbs)^(L + 1)
    end
  end
end
