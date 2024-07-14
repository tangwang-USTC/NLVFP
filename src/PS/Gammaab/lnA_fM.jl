
"""
  lnAab = lnAab_fM(ma,Za,spices,na,Ta)

    Cooulomb logarithm for fM and fDM with Gaussian System of Units (CGS) with `cmgs` between different spices `a` and `b`.
                   PS    →   dimmensionless
            `ma`: [mp]   →   [md=Dₐ]
            `Zq`: [e]    →   [qd=e] 
            `na`: [n20]  →   [nd=n20]
            `Ta`: [Tk]   →   [Td=0.5md*vd^2] / T_unit, where `T_unit = Tk / Td`
            `v` : [Mms]  →   [vd=Mms]
            `t` : [s]    →   [tdd] * (tdd / 1[s])

    * Copy from `NRL Plasma Formulary, 2016`, 2023/9/2
"""

"""
  Outputs:
    νT = FPTaTb_fM(isp,iFv,ma,Zq,spices,na,Ta,T_unit,logT_unit) 
"""

# [], νT
function FPTaTb_fM(isp::Int64,iFv::Int64,ma::AbstractVector{T},
    Zq::Vector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},Ta::AbstractVector{T},T_unit::T,logT_unit::T) where{T}
    
    if is_lnA_const
        lnAab = lnA_const(spices)
    else
        lnAab = lnAab_fM(ma,Zq,spices,na,Ta,T_unit,logT_unit)
    end
    @show is_lnA_const
    
    # cT = 1 / T^1.5 = T_unit^1.5
    cT = (ma[isp] * ma[iFv])^0.5 / (ma[isp] * Ta[iFv] + ma[iFv] * Ta[isp])^1.5
  
    # 1.8e-20 / (10mp)^0.5 / 1e6 * 1e20 / 1000^1.5 * T_unit^1.5
    # 1.8e-6 / (10mp)^0.5 * 10^-4.5 * T_unit^1.5
    return 440.1225454639835 * T_unit^1.5 / tdd * cT * (Zq[isp]* Zq[iFv])^2 * na[iFv] * lnAab
end

function FPTaTb_fM_n1(isp::Int64,iFv::Int64,ma::AbstractVector{T},Zq::Vector{Int64},
    na::AbstractVector{T},Ta::AbstractVector{T},spices::Vector{Symbol},T_unit::T,logT_unit::T) where{T}
    
    if is_lnA_const
        lnAab = lnA_const(spices)
    else
        lnAab = lnAab_fM(ma,Zq,spices,na,Ta,T_unit,logT_unit)
    end
    
    # cT = 1 / T^1.5 = T_unit^1.5
    cT = (ma[isp] * ma[iFv])^0.5 / (ma[isp] * Ta[iFv] + ma[iFv] * Ta[isp])^1.5
  
    # 1.8e-20 / (10mp)^0.5 / 1e6 * 1e20 / 1000^1.5 * T_unit^1.5
    # 1.8e-6 / (10mp)^0.5 * 10^-4.5 * T_unit^1.5
    return 440.1225454639835 * T_unit^1.5 / tdd * cT * (Zq[isp]* Zq[iFv])^2* lnAab
end

"""
  Imputs:

  Outputs:
    lnAab = lnAab_fM(ma,Zq,na,Ta,spices0,T_unit,logT_unit)
    lnAab = lnAab_fM(ma[1],Zq[1],na[1],Ta[1],spices0[1],T_unit,logT_unit)
    lnAab = lnAab_fM(ma[2],Zq[2],na[2],Ta[2],spices0[2],T_unit,logT_unit)

"""

# [ns=2]
function lnAab_fM(ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},Ta::AbstractVector{T},T_unit::T,logT_unit::T) where{T}

    if spices[1] == :e && spices[2] == :e                  # Cee
        return lnAee_fM(na,Ta,logT_unit)
    else 
        if spices[1] ≠ :e && spices[2] ≠ :e                # Cii
            return lnAii_fM(ma,Zq,na,Ta,logT_unit)
        else                                               # Cei || Cie
            # return lnAie(mi,Zi,ni,Ti,ne,Te)
            if spices[1] == :e
                return lnAie_fM(ma[2],Zq[2],na[2],Ta[2],ma[1],na[1],Ta[1],T_unit,logT_unit)
            else
                return lnAie_fM(ma[1],Zq[1],na[1],Ta[1],ma[2],na[2],Ta[2],T_unit,logT_unit)
            end
        end
    end
end

# [nMod=2], ns = 1
function lnAab_fM(Zq::Int64,spices::Symbol,na::AbstractVector{T},Ta::AbstractVector{T},logT_unit::T) where{T}

    if spices == :e
        return lnAee_fM(na,Ta,logT_unit)
    else 
        return lnAii_fM(Zq,na,Ta,logT_unit)
    end
end

# [], nMod=1, ns=1
function lnAab_fM(Zq::Int64,spices::Symbol,na::T,Ta::T,T_unit::T,logT_unit::T) where{T}

    if spices == :e                  # Cee
        return lnAee_fM(na,Ta,logT_unit)
    else 
        return lnAii_fM(Zq,na,Ta,T_unit,logT_unit)
    end
end

"""
  Thermal e-e collisions

  which means
    
    `ue ≪ vₑₜₕ`

  Outputs:
    lnAab = lnAee_fM(ne,Te)

  Testting:
    lnAab = lnAee_fM(na,Ta)
    lnAab = lnAee_fM(na[1],Ta[1])

"""
# [ns=2], a, Cee , fM-FM 
function lnAee_fM(na::AbstractVector{T},Ta::AbstractVector{T},logT_unit::T) where{T}
    
    ne = na[1] + na[2]
    Te = sum(na .* Ta) / ne
    return lnAee_fM(ne,Te,logT_unit)
end

# [ns=1]
function lnAee_fM(ne::T,Te::T,logT_unit::T) where{T}
    
    return 16.016598447769354 - 1.25 * logT_unit - log(ne^0.5 / Te^1.25) - (1e-5 + (4.907755278982138 - logT_unit + log(Te))^2 / 16)^0.5  # Tb[Td]
    
    # 23.5 - log(1e-3) - log(n20^0.5) - log(1000^-1.25)
    # 23.5 - log(1e-3) - log(1e10) + 1.25 * log(1000)
    # 23.5 + log(1e3) - 6.25 * log(10)
    # return 16.01659844776935 - log(ne^0.5 / (Te / T_unit)^1.25) - (1e-5 + (log(Te / T_unit) + 4.907755278982138)^2 / 16)^0.5  # Tb[Td]
end

"""
  Thermal ion-ion collisions

  which means
    
    `uii ≪ vᵢₜₕ`

  Outputs:
    lnAab = lnAii_fM(ma,Zq,na,Ta)
    lnAab = lnAii_fM(Zq[2],na[2],Ta[2])
    lnAab = lnAii_fM([ma[2],ma[2]],[Zq[2],Zq[2]],[na[2],na[2]],[Ta[2],Ta[2]])

"""
# [ns=2], c, Cii , fM-FM
function lnAii_fM(ma::AbstractVector{T},Zq::AbstractVector{Int64},
    na::AbstractVector{T}, Ta::AbstractVector{T},logT_unit::T) where{T}

    # mabT = nTab = 1 / T = 1 / (1 / T_unit) = T_unit

    mabT = (ma[1] + ma[2]) / (ma[1] * Ta[2] + ma[2] * Ta[1])
    nTab = (Zq[1]^2 * na[1] / Ta[1] + Zq[2]^2 * na[2] / Ta[2])
    return 17.243537267514885 - 1.5 * logT_unit - log(Zq[1] * Zq[2] * mabT * nTab^0.5)

    # 23 - log(1e-3) - 5.5 * log(10)
    # return 17.243537267514885 - log(T_unit^1.5) - log(Zq[1] * Zq[2] * mabT * nTab^0.5)
end

# [nMod=2], ns=1
function lnAii_fM(Zq::Int64,na::AbstractVector{T}, Ta::AbstractVector{T},logT_unit::T) where{T}

    mabT = 2 / (Ta[2] + Ta[1])  # 1 or 2
    nTab = Zq^2 * (na[1] / Ta[1] + na[2] / Ta[2])
    return 17.243537267514885 - 1.5 * logT_unit - log(Zq^2 * mabT * nTab^0.5)
end

# [ns=1],
function lnAii_fM(Zq::Int64,na::T, Ta::T,T_unit::T,logT_unit::T) where{T}

    # mabT = 1 / Ta
    # nTab = Zq^2 * na / Ta

    nTab = Zq^2 * na / Ta * (T_unit / n_unit)
    return 17.24353726751488 - 1.5 * logT_unit - log(Zq^2 / Ta * nTab^0.5)
end

"""
  Thermal electron-ion collisions when `ue, ui ≪ vᵢₜₕ, vₑₜₕ`
    
    if Te ≤ Ti * (me/mi)                     # c
    else
        if Te ≤ 10 * Zi² [eV]                # a, low temperature Plasma
        else
            if Ti * (me/mi) ≤ 10 * Zi² [eV]  # b
            else                             # d, (α-e)
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
            else                           # d, (α-e)
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
    lnAab = lnAie_fM(mi,Zi,ni,Ti,spice_i,me,ne,Te)

  Testting:
    lnAab = lnAie_fM(ma[2],Zq[2],na[2],Ta[2],ma[1],na[1],Ta[1])
  
"""
# [ns=2], b, Cei = Cie , fM-FM
function lnAie_fM(mi::T,Zi::Int64,ni::T,Ti::T,me::T,ne::T,Te::T,T_unit::T,logT_unit::T) where{T}

    # ne = ne / n_unit
    # Te = Te / T_unit

    if Te ≤ Ti * (me/mi)                                               # c: Te ≪ Ti, e-α, burning plasma
        # 16 - log(1e-3) - log(1000^-1.5) - log(1e10)
        return 10.243537267514885 - 1.5 * logT_unit - log(Zi^2 * mi * (ni^0.5 / Ti^1.5))
        # return 10.243537267514885 - log(T_unit^1.5 / n_unit^0.5) - log(Zi^2 * mi * (ni^0.5 / Ti^1.5))
    else
        if Te ≤ Zi^2 / 100 * T_unit                                    # a: low electron temperature
            # 23 - log(1e-3) - log(1000^-1.5) - log(1e10)                   Te < 50 eV
            return 17.243537267514885 - 1.5 * logT_unit - log(Zi * (ne^0.5 / Te^1.5))
            # return 17.243537267514885 - log(T_unit^1.5 / n_unit^0.5) - log(Zi * (ne^0.5 / Te^1.5))
        else
            if Ti * (me/mi) ≤ Zi^2 / 100 * T_unit                      # b: Ti < 50 keV, Te > 50 eV
                # 24 - log(1e-3) - log(1000^-1) - log(1e10)
                return 14.789659628023816 - logT_unit - log(ne^0.5 / Te)
                # return 14.789659628023816 - log(T_unit / n_unit^0.5) - log(ne^0.5 / Te)
            else                                                       # d: 50 keV < Ti < 2000Te, Te > 50 eV
                return 14.789659628023816 - logT_unit - log(ne^0.5 / Ti)
                @warn(" Checking `lnAie` when `Ti > (mi / me) * Z^2 / 100`")
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
    lnAab = lnAiie_fDM(ne,Te,ma,Zq,na,Ta,uab)

"""
# [ns=3], d, Ciie , fDM-FDM-fMe
function lnAiie_fDM(ne::T,Te::T,ma::AbstractVector{T},
    Zq::AbstractVector{Int64},Ta::AbstractVector{T},uab::T,c₀::T) where{T}

    sdfgnjm
end
