
"""
  lnAab = lnAab_fM_Tk(ma,Za,spices,na,Ta)

  Coulomb logarithm for fM and fDM with Gaussian System of Units (CGS) with `cmgs` between different spices `a` and `b`.
           SI    →    PS
    `ma`: [mp]   →   [mp]
    `Zq`: [e]    →   [e]
    `na`: [m⁻³]  →   [n20] * 1e20
    `Ta`: [eV]   →   [Tk=keV] * 1000
    `v` : [m/s]  →   [Mms]
    `t` : [s]    →   [s]

    * Copy from `NRL Plasma Formulary, 2016`, 2020/9/2
"""

# [], νT
function FPTaTb_fM_Tk(isp::Int64,iFv::Int64,ma::AbstractVector{T},
    Zq::Vector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},Ta::AbstractVector{T}) where{T}
  
    # cT = 1 / 1000^1.5 = 10^-4.5
  
    if is_lnA_const
        lnAab = lnA_const(spices)
    else
        lnAab = lnAab_fM_Tk(ma,Zq,spices,na,Ta)
    end
    
    # cT = 1 / T^1.5 = 1 / 1000^1.5
    cT = (ma[isp] * ma[iFv])^0.5 / (ma[isp] * Ta[iFv] + ma[iFv] * Ta[isp])^1.5
  
    # 1.8e-20 / (10mp)^0.5 / 1e6 * 1e20 / 1000^1.5
    # 1.8e-6 / (10mp)^0.5 * 10^-4.5
    return 440.1225454639835 * cT * (Zq[isp]* Zq[iFv])^2 * na[iFv] * lnAab
    # return 1.391789693257197e-13 * 1e20 / 1000^1.5 * cT * (Zq[isp]* Zq[iFv])^2 * na[iFv] * lnAab
end
function FPTaTb_fM_n1_Tk(isp::Int64,iFv::Int64,ma::AbstractVector{T},
    Zq::Vector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},Ta::AbstractVector{T}) where{T}

    if is_lnA_const
        lnAab = lnA_const(spices)
    else
        lnAab = lnAab_fM_Tk(ma,Zq,spices,na,Ta)
    end
    cT = (ma[isp] * ma[iFv])^0.5 / (ma[isp] * Ta[iFv] + ma[iFv] * Ta[isp])^1.5
    return 440.1225454639835 * cT * (Zq[isp]* Zq[iFv])^2 * lnAab
end

"""
  Imputs:

  Outputs:
    lnAab = lnAab_fM_Tk(ma*(md/mp),Zq,spices0,n0*n20,1000T0)
    lnAab = lnAab_fM_Tk(ma*(md/mp),Zq,spices0,na*nd,Ta*(Td/e))


"""

# [ns=2]
function lnAab_fM_Tk(m0::AbstractVector{T},
    Zq::AbstractVector{Int64},spices::Vector{Symbol},
    n0::AbstractVector{T},T0::AbstractVector{T}) where{T}

    if spices[1] == :e && spices[2] == :e                  # Cee
        ne = n0[1] + n0[2]
        Te = sum(n0 .* T0) / ne
        return lnAee_fM_Tk(ne,Te)
    else 
        if spices[1] ≠ :e && spices[2] ≠ :e                   # Cii
            return lnAii_fM_Tk(m0,Zq,n0,T0)
        else                                          # Cei || Cie
            # return lnAie(mi,Zi,ni,Ti,ne,Te)
            if spices[1] == :e
                return lnAie_fM_Tk(m0[2],Zq[2],n0[2],T0[2],m0[1],n0[1],T0[1])
            else
                return lnAie_fM_Tk(m0[1],Zq[1],n0[1],T0[1],m0[2],n0[2],T0[2])
            end
        end
    end
end

"""
  Thermal e-e collisions

  which means
    
    `ue ≪ vₑₜₕ`

  Outputs:
    lnAab = lnAee_fM_Tk(ne,Te)

  Testting:
    lnAab = lnAee_fM_Tk(na[1]*nd,Ta[1]*Td/e)

"""

# [ns=2], a, Cee , fM-FM
function lnAee_fM_Tk(ne::T,Te::T) where{T}
    
    # 23.5 - log(1e-3) - log(n20^0.5) - log(1000^-1.25)
    # 23.5 - log(1e-3) - log(1e10) + 1.25 * log(1000)
    # 23.5 + log(1e3) - 6.25 * log(10)
    return 16.01659844776935 - log(ne^0.5 / Te^1.25) - (1e-5 + (log(Te) + 4.907755278982138)^2 / 16)^0.5  # Tb[Tk]
    return 16.01659844776935 - log(ne^0.5 / Te^1.25) - (1e-5 + (log(Te) + 3log(10) - 2)^2 / 16)^0.5  # Tb[Tk]
end

"""
  Thermal ion-ion collisions

  which means
    
    `uii ≪ vᵢₜₕ`

  Outputs:
    lnAab = lnAii_fM_Tk(m0,Zq,n0,T0)

  Testting:
    lnAab = lnAii_fM_Tk(ma*(md/mp),Zq,na*nd,Ta*(Td/e))

"""
# [ns=2], c, Cii , fM-FM
function lnAii_fM_Tk(m0::AbstractVector{T},Zq::AbstractVector{Int64},
    n0::AbstractVector{T}, T0::AbstractVector{T}) where{T}

    # mabT = 1 / T = 1 / 1000
    # nTab = n0[1] / T0[1] = 1e17

    mabT = (m0[1] + m0[2]) / (m0[1] * T0[2] + m0[2] * T0[1])
    nTab = (Zq[1]^2 * n0[1] / T0[1] + Zq[2]^2 * n0[2] / T0[2])

    # 23 - log(1e-3) - log(1e11^0.5)
    # 23 - log(1e-3) - log(10^5.5)
    # 23 - log(1e-3) - 5.5 * log(10)
    return 17.243537267514885 - log(Zq[1] * Zq[2] * mabT * nTab^0.5)
end

"""
  Thermal electron-ion collisions when `ue, ui ≪ vᵢₜₕ, vₑₜₕ`
    
    if Te ≤ Ti * (me/mi)                   # c: e-α, burning plasma
    else
        if Te ≤ 10 * Zi² [eV]              # a: low temperature plasma
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
    lnAab = lnAie_fM_Tk(m0,Zq,n0,T0)

  Testting:
    lnAab = lnAie_fM_Tk(ma[2]*(md/mp),Zq[2],na[2]*nd,Ta[2]*(Td/e),ma[1]*(md/mp),na[1]*nd,Ta[1]*(Td/e))

"""
# [ns=2], b, Cei = Cie , fM-FM
function lnAie_fM_Tk(mi::T,Zi::Int64,ni::T,Ti::T,me::T,ne::T,Te::T) where{T}

    if Te ≤ Ti * (me/mi)                                  # c: Te ≪ Ti, e-α, burning plasma
        # 16 - log(1e-3) - log(1e10 / 10^4.5)
        return 10.243537267514885 - log(Zi^2 * mi * (ni^0.5 / Ti^1.5))
    else
        if Te ≤ (Zi/10)^2                                 # a: low electron temperature
            # 23 - log(1e-3) - log(1e10 / 10^4.5)              Te < 50 eV
            return 17.243537267514885  - log(Zi * (ne^0.5 / Te^1.5))
        else                                              
            if Ti * (me/mi) ≤ (Zi/10)^2                   # b: Ti < 50 keV, Te > 50 eV
                # 24 - log(1e-3) - log(1e7)
                return 14.789659628023816  - log(ne^0.5 / Te)
            else                                          # d: 50 keV < Ti < 2000Te, Te > 50 eV
              # return 14.789659628023816  - log(ne^0.5 / Ti)
              return 17
                println("`lnAie` when `Ti > 10(mi / me) * Z^2` ")
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
function lnAiie_fDM_Tk(ne::T,Te::T,m0::AbstractVector{T},
    Zq::AbstractVector{Int64},T0::AbstractVector{T},uab::T,c₀::T) where{T}

    efrff
end
