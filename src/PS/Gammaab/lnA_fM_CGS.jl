
"""
  lnAab = lnAab_fM_CGS(ma,Za,spices,na,Ta)

    Coulomb logarithm for fM and fDM with Gaussian System of Units (CGS) with `cmgs` between different spices `a` and `b`.

      `ma`: [mp]
      `Zq`: [e]
      `na`: [cm⁻³]
      `Ta`: [eV]
      `v`: [cm/s],  such as `ua` and `vath`
      `t`: [s]

    * Copy from `NRL Plasma Formulary, 2016`, 2020/9/2
"""

# [], νT
function FPTaTb_fM_CGS(isp::Int64,iFv::Int64,ma::AbstractVector{T},
  Zq::Vector{Int64},spices::Vector{Symbol},
  na::AbstractVector{T},Ta::AbstractVector{T}) where{T}

  if is_lnA_const
      lnAab = lnA_const(spices)
  else
      lnAab = lnAab_fM_CGS(ma,Zq,spices,na,Ta)     # m[mp]
  end
  
  # # m[g]
  # cT = (ma[isp] * ma[iFv])^0.5 / (ma[isp] * Ta[iFv] + ma[iFv] * Ta[isp])^1.5
  # νT = 1.8e-19 * cT * (Zq[isp]* Zq[iFv])^2 * na[iFv] * lnAab

  # m[mp]
  # ma = ma * 1000mp
  # cT = 1 / m^0.5 = 1 / (1000mp)^0.5
  cT = (ma[isp] * ma[iFv])^0.5 / (ma[isp] * Ta[iFv] + ma[iFv] * Ta[isp])^1.5
  # 1.8e-19 / (1000mp)^0.5
  # 1.8e-20 / (10mp)^0.5
  return 1.391789693257197e-7 * cT * (Zq[isp]* Zq[iFv])^2 * na[iFv] * lnAab
end

"""
  lnA = lnAab_fM_CGS(m0,Zq,n0,T0,spices)
"""


# [ns=2]
function lnAab_fM_CGS(m0::AbstractVector{T},
  Zq::AbstractVector{Int64},spices::Vector{Symbol},
    n0::AbstractVector{T},T0::AbstractVector{T}) where{T}

    if spices[1] == :e && spices[2] == :e                  # Cee
        ne = n0[1] + n0[2]
        Te = sum(n0 .* T0) / ne
        return lnAee_fM_CGS(ne,Te)
    else 
        if spices[1] ≠ :e && spices[2] ≠ :e                   # Cii
            return lnAii_fM_CGS(m0,Zq,n0,T0)
        else                                          # Cei || Cie
            # return lnAie(mi,Zi,ni,Ti,ne,Te)
            if spices[1] == :e
                return lnAie_fM_CGS(m0[2],Zq[2],n0[2],T0[2],n0[1],T0[1])
            else
                return lnAie_fM_CGS(m0[1],Zq[1],n0[1],T0[1],n0[2],T0[2])
            end
        end
    end
end

"""
  Thermal e-e collisions

  which means
    
    `ue ≪ vₑₜₕ`

  Outputs:
    lnA = lnAee_fM_CGS(ne,Te)

"""

# [ns=2], a, Cee , fM-FM
function lnAee_fM_CGS(ne::T,Te::T) where{T}

    return 23.5 - log(ne^0.5 / Te^1.25) - (1e-5 + (log(Te) - 2)^2 / 16)^0.5  # Tb[eV]
end

"""
  Thermal ion-ion collisions

  which means
    
    `uii ≪ vᵢₜₕ`

  Inputs:
    m0: = m0SI / mp

  Outputs:
    lnA = lnAii_fM_CGS(m0,Zq,n0,T0)

"""
# [ns=2], c, Cii , fM-FM
function lnAii_fM_CGS(m0::AbstractVector{T},Zq::AbstractVector{Int64},
    n0::AbstractVector{T}, T0::AbstractVector{T}) where{T}

    mabT = (m0[1] + m0[2]) / (m0[1] * T0[2] + m0[2] * T0[1])
    nTab = Zq[1]^2 * n0[1] / T0[1] + Zq[2]^2 * n0[2] / T0[2]

    return 23 - log(Zq[1] * Zq[2] * mabT * nTab^0.5)
end

"""
  Thermal ion-ion collisions when `ue, ui ≪ vᵢₜₕ, vₑₜₕ`
    
    if Te ≤ Ti * (me/mi)                   # c
    else
        if Te ≤ 10 * Zi² [eV]                # a
        else
            if Ti * (me/mi) ≤ 10 * Zi² [eV]  # b
            else
                sdfgbhnjm
            end
        end
    end

  which means
    
    if vₑₜₕ ≤ vᵢₜₕ                           # c
    else
        if vₑₜₕ ≤ Zi * vₜₕ₀                  # a
        else
            if vᵢₜₕ ≤ Zi * vₜₕ₀              # b
            else
                sdfgbhnjm
            end
        end
    end

    where

      vₜₕ₀ = (10eV/me)^0.5

  Inputs:
    mi: = miSI / mp
    Te: = TeSI / eV

  Outputs:
    lnA = lnAie_fM_CGS(m0,Zq,n0,T0,uab)

"""
# [ns=2], b, Cei = Cie , fM-FM
function lnAie_fM_CGS(mi::T,Zi::Int64,ni::T,Ti::T,ne::T,Te::T) where{T}

    if Te ≤ Ti * (me/mi)                                      # c
        return 16 - log(Zi^2 * mi * (ni^0.5 / Ti^1.5))
    else
        if Te ≤ 10 * Zi^2                                     # a
            return 23 - log(Zi * (ne^0.5 / Te^1.5))
        else
            if Ti * (me/mi) ≤ 10 * Zi^2                       # b
                return 24 - log(ne^0.5 / Te)
            else
                println("`lnAie` when `Ti > 10(mi / me) * Z^2` ")
                sdfgbhnjm
            end
        end
    end
end

"""
  Counterstreaming ions (with relative velocity `uab = βₐᵦ * c₀`) 
  in the presence of warm electrons, `Tᵢ / mᵢ < uab^2 < Te / me`.

  which means
    
    `vᵢₜₕ < uii < vₑₜₕ`

  Inputs:
    m0: = m0SI / mp

  Outputs:
    lnA = lnAii_fM_CGS(m0,Zq,n0,T0)

"""
# [ns=3], d, Ciie , fDM-FDM-fMe
function lnAiie_fDM_CGS(ne::T,Te::T,m0::AbstractVector{T},
    Zq::AbstractVector{Int64},T0::AbstractVector{T},uab::T) where{T}

    uab = 100uab                         # uab = |ua - ub|
    if uab^2 ≤ (Te/ 1000me) && uab^2 ≥ 1e4 * maximum(T0./m0)
        # ma = 1000m0[1]   # Gaussian Units
        # mb = 1000m0[2]

        Te = Te / e
        ne = ne / 1e6
        # Ta = T0[1] / e
        # Tb = T0[2] / e
        # ma = 1000m0[1]   # Gaussian Units
        # mb = 1000m0[2]
        # na = n0[1] / 1e6
        # nb = n0[2] / 1e6
        

        mab = (m0[1] * m0[2]) / (m0[1] + m0[2]) * 1000   # Gaussian Units
        βD = uab / c₀

        return 43 - log(Zq[1] * Zq[2]/ mab / βD^2 * (ne / Te)^0.5)
    else
        println("`lnAiie` has been given only when `vith < uab < veth`!")
    end
end
