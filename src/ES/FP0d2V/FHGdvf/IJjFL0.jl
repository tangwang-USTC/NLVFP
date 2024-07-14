

"""
  Coefficients for analytical results of Shkarofsky integrals:

    Iₗ,ₗ⁰   = CL * [Cₗᵉʳᶠ *   (erfvuTp + erfvuTn) + Cₗᵉˣᵖ *  (Cₗᵖ * expvuTp   + Cₗⁿ * expvuTn)]   / (v̂/v̂ᵦₜₕₛ)ᴸ
    Iₗ₊₂,ₗ⁰ = CL * [Cₗ₊₂ᵉʳᶠ * (erfvuTp + erfvuTn) + Cₗᵉˣᵖ *  (Cₗ₊₂ᵖ * expvuTp + Cₗ₊₂ⁿ * expvuTn)] / (v̂/v̂ᵦₜₕₛ)ᴸ⁺²
    Jₗ₋₁,ₗ⁰ = CL * [Cₗ₋₁ᵉʳᶠ * (erfvuTp - erfvuTn) + Cⱼᵉˣᵖ * (Cₗ₋₁ᵖ * expvuTp + Cₗ₋₁ⁿ * expvuTn)]
    Jₗ₊₁,ₗ⁰ = CL * [Cₗ₊₁ᵉʳᶠ * (erfvuTp - erfvuTn) + Cⱼᵉˣᵖ * (Cₗ₊₁ᵖ * expvuTp + Cₗ₊₁ⁿ * expvuTn)]

  where

    CL = (2L + 1) / π^(3/2)
    erfvuTp = erf(vuTp)
    erfvuTn = erf(vuTn)
    expvuTp = exp(-vuTp^2)
    expvuTn = exp(-vuTn^2)
    vuTp = (v̂ + ûᵦ)/v̂ᵦₜₕₛ
    vuTn = (v̂ - ûᵦ)/v̂ᵦₜₕₛ

  Inputs:

  Outputs:
    M = 
"""

"""
  Intputs:
    L:

  Outputs:
    mCM, CXuL0::Matrix = CXuL0nm(uhbs,vhbths,L)

"""

function shkarofskyIL0(ILn1FL0::AbstractVector{T},IL1FL0::AbstractVector{T},
    vhe::AbstractVector{T},uhbs::T,vhbths::T,L::Int64) where{T}

    CLerf = CLerfL(uhbs,vhbths,L)
    CL2erf = CL2erfL(uhbs,vhbths,L)
    CIexp = CIexpL(uhbs,vhbths,L)
    CJexp = CJexpL(uhbs,vhbths,L)

    CLn1erf = CLn1erfL(uhbs,vhbths,L)
    CL1erf = CL1erfL(uhbs,vhbths,L)
    CIexp = CIexpL(uhbs,vhbths,L)
    CJexp = CJexpL(uhbs,vhbths,L)
    if uhbs ≤ 2epsT
        if vhbths == 1.0
        else
        end
    else
        if vhbths == 1.0
        else
            erfnp = erf.(())
            if L == 0
                ILn1FL0[:] = 1 / sqrtpi3 * (CLerf )
            else
            end
        end
    end
end

