
"""
  lnAab = lnAab_fM_CGS(ma,Za,spices,na,Ta)

  Coulomb logarithm for fM and fDM with Gaussian System of Units (CGS) with `cmgs` between different spices `a` and `b`.
          CGS    →   SI
    `ma`: [mp]   →   [mp]
    `Zq`: [e]    →   [e]
    `na`: [cm⁻³] →   [m⁻³] / 1e6
    `Ta`: [eV]   →   [eV]
    `v` : [cm/s] →   [m/s] / 100
    `t` : [s]    →   [s]

    * Copy from `NRL Plasma Formulary, 2016`, 2020/9/2
"""

# [], νT
function FPTaTb_fM_SI(isp::Int64,iFv::Int64,ma::AbstractVector{T},
    Zq::Vector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},Ta::AbstractVector{T}) where{T}

    # lnAab = lnA_const(spices)
    if is_lnA_const
        lnAab = lnA_const(spices)
    else
        lnAab = lnAab_fM_SI(ma,Zq,spices,na,Ta)
    end
  
    cT = (ma[isp] * ma[iFv])^0.5 / (ma[isp] * Ta[iFv] + ma[iFv] * Ta[isp])^1.5
    # 1.8e-19 / (1000mp)^0.5 / 1e6
    # 1.8e-20 / (10mp)^0.5 / 1e6
    return 1.391789693257197e-13 * cT * (Zq[isp]* Zq[iFv])^2 * na[iFv] * lnAab
end

"""
  Imputs:

  Outputs:
    lnAab = lnAab_fM_SI(ma*(md/mp),Zq,n0*n20,1000T0,spices0)
    lnAab = lnAab_fM_SI(ma*(md/mp),Zq,na*nd,Ta*(Td/e),spices0)


"""

# [ns=2]
function lnAab_fM_SI(m0::AbstractVector{T},
    Zq::AbstractVector{Int64},spices::Vector{Symbol},
    n0::AbstractVector{T},T0::AbstractVector{T}) where{T}

    if spices[1] == :e && spices[2] == :e                  # Cee
        ne = n0[1] + n0[2]
        Te = sum(n0 .* T0) / ne
        return lnAee_fM_SI(ne,Te)
    else 
        if spices[1] ≠ :e && spices[2] ≠ :e                   # Cii
            return lnAii_fM_SI(m0,Zq,n0,T0)
        else                                          # Cei || Cie
            # return lnAie(mi,Zi,ni,Ti,ne,Te)
            if spices[1] == :e
                return lnAie_fM_SI(m0[2],Zq[2],n0[2],T0[2],m0[1],n0[1],T0[1])
            else
                return lnAie_fM_SI(m0[1],Zq[1],n0[1],T0[1],m0[2],n0[2],T0[2])
            end
        end
    end
end

"""
  Thermal e-e collisions

  which means
    
    `ue ≪ vₑₜₕ`

  Outputs:
    lnAab = lnAee_fM_SI(ne,Te)

  Testting:
    lnAab = lnAee_fM_SI(na[1]*nd,Ta[1]*Td/e)

"""

# [ns=2], a, Cee , fM-FM
function lnAee_fM_SI(ne::T,Te::T) where{T}
    
    # 23.5 - log(1e-3)
    return 30.407755278982137 - log(ne^0.5 / Te^1.25) - (1e-5 + (log(Te) - 2)^2 / 16)^0.5  # Tb[eV]
end

"""
  Thermal ion-ion collisions

  which means
    
    `uii ≪ vᵢₜₕ`

  Outputs:
    lnAab = lnAii_fM_SI(m0,Zq,n0,T0)

  Testting:
    lnAab = lnAii_fM_SI(ma*(md/mp),Zq,na*nd,Ta*(Td/e))

"""
# [ns=2], c, Cii , fM-FM
function lnAii_fM_SI(m0::AbstractVector{T},Zq::AbstractVector{Int64},
    n0::AbstractVector{T}, T0::AbstractVector{T}) where{T}

    # nTab = 1 / 1e6
    mabT = (m0[1] + m0[2]) / (m0[1] * T0[2] + m0[2] * T0[1])
    nTab = (Zq[1]^2 * n0[1] / T0[1] + Zq[2]^2 * n0[2] / T0[2])

    # 23 - log(1e-3)
    return 29.907755278982137 - log(Zq[1] * Zq[2] * mabT * nTab^0.5)
end

"""
  Thermal electron-ion collisions when `ue, ui ≪ vᵢₜₕ, vₑₜₕ`
    
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
    me: = meSI / mp
    mi: = miSI / mp

  Outputs:
    lnAab = lnAie_fM_SI(m0,Zq,n0,T0)

  Testting:
    lnAab = lnAie_fM_SI(ma[2]*(md/mp),Zq[2],na[2]*nd,Ta[2]*(Td/e),ma[1]*(md/mp),na[1]*nd,Ta[1]*(Td/e))

"""
# [ns=2], b, Cei = Cie , fM-FM
function lnAie_fM_SI(mi::T,Zi::Int64,ni::T,Ti::T,me::T,ne::T,Te::T) where{T}

    if Te ≤ Ti * (me/mi)                                                      # c
        # 16 - log(1e-3)
        return 22.907755278982137 - log(Zi^2 * mi * (ni^0.5 / Ti^1.5))
    else
        if Te ≤ 10 * Zi^2                                                     # a
            # 23 - log(1e-3)
            return 29.907755278982137 - log(Zi * (ne^0.5 / Te^1.5))
        else
            if Ti * (me/mi) ≤ 10 * Zi^2                                       # b
                # 24 - log(1e-3)
                return 30.907755278982137 - log(ne^0.5 / Te)
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

  Outputs:
    lnAab = lnAiie_fDM_SI(ne,Te,m0,Zq,n0,T0,uab)

"""
# [ns=3], d, Ciie , fDM-FDM-fMe
function lnAiie_fDM_SI(ne::T,Te::T,m0::AbstractVector{T},
    Zq::AbstractVector{Int64},T0::AbstractVector{T},uab::T,c₀::T) where{T}

    efrff
end
