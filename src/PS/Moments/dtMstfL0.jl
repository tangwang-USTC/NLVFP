"""
  The change of the `jᵗʰ`-order moment of the `zeroth`-order coefficient
  of the distribution function which is Maxwellian (`û = 0`).
      δₜᵃᵇ𝓜ⱼ(f₀) = (π^(-3/2) × nₐmₐvₜₕʲv̂ₜₕʲ) × (n̂ₐn̂ᵦ/v̂ᵦₜₕ³) × 4π∫₀^∞(v̂ₐᵢ²⁺ʲδₜᵃᵇf̂₀ᵢ(v̂))dv̂ₐᵢ
                 = - (nₐmₐvₜₕʲv̂ₜₕʲ) (nᵦvᵦₜₕ⁻³Γₐᵦ) × (n̂ₐn̂ᵦ/v̂ᵦₜₕ³)
                   × 8√π / v̂ₐᵦ² × (mM -1/v̂ₐᵦ²) / (1 + v̂ₐᵦ²)^((j+1)/2)
                   × Γ((j+3)/2) × (hypergometric2F1(1,-j/2,3/2,-v̂ₐᵦ²) - 1)

  When `n̂ₐ = n̂ᵦ = v̂ₜₕ = v̂ᵦₜₕ = 1`,

      δₜᵃᵇ𝓜ⱼ(f₀) = (π^(-3/2) × nₐmₐvₜₕʲ) × 4π × ∫₀^∞(v̂²⁺ʲδₜᵃᵇf̂₀(v̂))dv̂
                 = - (nₐmₐvₜₕʲ) (nᵦvᵦₜₕ⁻³Γₐᵦ) × 8√π / vₐᵦ² × (mM -1/vₐᵦ²) / (1 + vₐᵦ²)^((j+1)/2)
                   × Γ((j+3)/2) × (hypergometric2F1(1,-j/2,3/2,-vₐᵦ²) - 1)

  where

    v̂ₐᵦ = vₐᵦ * v̂ₐᵦᵢ
    vₐᵦ = vₐₜₕ / vᵦₜₕ
    v̂ₐᵦᵢ = v̂ₐₜₕᵢ / v̂ᵦₜₕᵢ.
    δₜf̂₀(v̂) = π^(3/2) × nₐ⁻¹vₜₕ³ × δₜf₀(v)

  which is including the coefficient `cF`:

    cF = π^(-3/2) × nᵦvₜₕ⁻³.

  Especially,

    hypergometric2F1(1,-0/2,3/2,-vₐᵦ²) = 1
    hypergometric2F1(1,-2/2,3/2,-vₐᵦ²) - 1 = 2 /3 × vₐᵦ²

  gives:

    δₜᵃᵇnₐ = δₜᵃᵇ𝓜₀(f₀) = 0
    δₜᵃᵇKₐ = δₜᵃᵇ𝓜₀(f₀) / 2 = - 4πnₐTₐ × (nᵦvᵦₜₕ⁻³Γₐᵦ) × (mM - 1/vₐᵦ²) / (1 + vₐᵦ²)^(3/2)

  The total energy satisfies:

    δₜᵃᵇKₐ / δₜᵇᵃKᵦ = - 1

  which means:

    δₜᵃᵇKₐ + δₜᵇᵃKᵦ = 0.

  if `is_renorm = true`

    Ms = Ms / CjLL2(j,0) = Ms / CjLL2(j) = Ms * (√π / 2) / gamma((3+j)/2)
      CjLL2(0,0) = 1
      CjLL2(1,1) = 3
      CjLL2(2,0) = 3/2
"""

"""
  Inputs:
    dtMs:
    j:
    na: na / n20
    ma: ma[isp] / Dₐ
    vth: vth / Mms

  Outputs:
    dtMs = dtMsrnt(dtMs,j,ma,mM,Zq,na,nb,vath,vbth,nai,nbi,vathi,vbthi;is_renorm=is_renorm)
    dtMs = dtMsrnt(dtMs,j,ma,mM,Zq,na,nb,vath,vbth;is_renorm=is_renorm)
"""
# 0D,
function dtMsrnt(dtMs::T,j::Int64,ma::T,mM::T,Zq::AbstractVector{Int},
    na::T,nb::T,vath::T,vbth::T,nai::T,nbi::T,vathi::T,vbthi::T;is_renorm::Bool=false) where{T}

    if j == 0
        dtMs = 0.0
        return dtMs
    else
        vabth = vath / vbth * (vathi / vbthi)
        vabth2 = vabth^2
        # cF3 = na[iFv] ./ vth[iFv].^3 / Pi^(3/2)
        # lnAg = 4Pi / Pi^(3/2) * (4Pi * qma^2 * lnAab)  # `4π` owing the `F(v)` and `H(F)`
        qma = qma0 / ma * (Zq[1] * Zq[2])
        lnAab = lnA() * (nb ./ vbth.^3) * (nai * nbi / vbthi^3)
        if is_renorm
            if j == 2
                # lnAg = (32 * qma^2 * lnAab) # `Pi^(3/2)
                # dtMs = - (sqrtpi / 4) * Cn2DMms * Mms^(j-1) * (na * ma * vath^j) * lnAg *
                #       (mM - 1.0 / vabth2) / vabth2 / (1 + vabth2)^((j+1)/2) *
                #       (_₂F₁(1,-j/2,3/2,-vabth2) - 1.0)
                # dtKa = dtM2f0 / 2
                lnAg = (16/2 * sqrtpi * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - (3/2) * Cn2DMms * Mms * (na * ma * (vath * vathi)^j) * lnAg *
                        (mM - 1.0 / vabth2) / (1 + vabth2)^(3/2)
            elseif j == 4
                lnAg = (16 * sqrtpi * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - (15/4) * Cn2DMms * Mms^3 * (na * ma * (vath * vathi)^j) * lnAg *
                      (mM - 1.0 / vabth2) / (1 + vabth2)^(5/2) * (5 + 2vabth2)
            else
                lnAg = (32 * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - (sqrtpi / 2) * Cn2DMms * Mms^(j-1) * (na * ma * (vath * vathi)^j) * lnAg *
                      (mM - 1.0 / vabth2) / vabth2 / (1 + vabth2)^((j+1)/2) *
                      gamma((j+3)/2) * (_₂F₁(1,-j/2,3/2,-vabth2) - 1.0)
            end
            return dtMs
        else
            if j == 2
                lnAg = (16/2 * sqrtpi * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - Cn2DMms * Mms * (na * ma * (vath * vathi)^j) * lnAg *
                        (mM - 1.0 / vabth2) / (1 + vabth2)^(3/2)
            elseif j == 4
                lnAg = (16 * sqrtpi * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - Cn2DMms * Mms^3 * (na * ma * (vath * vathi)^j) * lnAg *
                      (mM - 1.0 / vabth2) / (1 + vabth2)^(5/2) * (5 + 2vabth2)
            else
                lnAg = (32 * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - Cn2DMms * Mms^(j-1) * (na * ma * (vath * vathi)^j) * lnAg *
                      (mM - 1.0 / vabth2) / vabth2 / (1 + vabth2)^((j+1)/2) *
                      gamma((j+3)/2) * (_₂F₁(1,-j/2,3/2,-vabth2) - 1.0)
            end
            # lnAg = (32 * qma^2 * lnAab) # `Pi^(3/2)
            # dtMs = - Cn2DMms * Mms^(j-1) * (na * ma * (vath * vathi)^j) * lnAg *
            #       (mM - 1.0 / vabth2) / (1 + vabth2)^((j+1)/2) / vabth2 *
            #       gamma((j+3)/2) * (_₂F₁(1,-j/2,3/2,-vabth2) - 1.0)
            #       @show vath/vbth,vathi,vbthi, dtMs
            return dtMs
        end
    end
end

# 0D, nai = vthi = 1
function dtMsrnt(dtMs::T,j::Int64,ma::T,mM::T,Zq::AbstractVector{Int},
    na::T,nb::T,vath::T,vbth::T;is_renorm::Bool=false) where{T}

    if j == 0
        dtMs = 0.0
        return dtMs
    else
        vabth = vath / vbth
        # cF3 = na[iFv] ./ vth[iFv].^3 / Pi^(3/2)
        # lnAg = 4Pi * cF3 * (4Pi * qma^2 * lnAab)  # `4π` owing the `F(v)` and `H(F)`
        qma = qma0 / ma * (Zq[1] * Zq[2])
        lnAab = lnA()
        # lnAg = (nb ./ vbth.^3) * (16.0 * sqrtpi * qma^2 * lnAab)  / Pi^(3/2) # `Pi^(3/2)` owing the `f(v)`
        # lnAg = (nb ./ vbth.^3) * (16.0 / π * qma^2 * lnAab) # `Pi^(3/2)` owing the `f(v)`
        if is_renorm
            if j == 2
                # dtKa = dtM2f0 / 2
                lnAg = (nb ./ vbth.^3) * (8.0 * sqrtpi * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - (3/2) * Cn2DMms * Mms * na * ma * vath^2 * lnAg *
                        (mM - 1.0 / vabth^2) / (1 + vabth^2)^(3/2)
            elseif j == 4
                lnAg = (nb ./ vbth.^3) * (16 * sqrtpi * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - (15/4) * Cn2DMms * Mms^3 * (na * ma * vath^4) * lnAg *
                      (mM - 1.0 / vabth^2) / (1 + vabth^2)^(5/2) * (5 + 2vabth^2)
            else
                lnAg = (nb ./ vbth.^3) * (32 * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - (sqrtpi / 2) * Cn2DMms * Mms^(j-1) * (na * ma * vath^j) * lnAg *
                      (mM - 1.0 / vabth^2) / vabth^2 / (1 + vabth^2)^((j+1)/2) *
                      (_₂F₁(1,-j/2,3/2,-vabth^2) - 1.0)
            end
            return dtMs
        else
            if j == 2
                # dtKa = dtM2f0 / 2
                lnAg = (nb ./ vbth.^3) * (8.0 * sqrtpi * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - Cn2DMms * Mms * na * ma * vath^2 * lnAg *
                        (mM - 1.0 / vabth^2) / (1 + vabth^2)^(3/2)
            elseif j == 4
                lnAg = (nb ./ vbth.^3) * (16 * sqrtpi * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - Cn2DMms * Mms^3 * (na * ma * vath^4) * lnAg *
                      (mM - 1.0 / vabth^2) / (1 + vabth^2)^(5/2) * (5 + 2vabth^2)
            else
                lnAg = (nb ./ vbth.^3) * (32 * qma^2 * lnAab) # `Pi^(3/2)
                dtMs = - Cn2DMms * Mms^(j-1) * (na * ma * vath^j) * lnAg *
                      (mM - 1.0 / vabth^2) / vabth^2 / (1 + vabth^2)^((j+1)/2) *
                      gamma((j+3)/2) * (_₂F₁(1,-j/2,3/2,-vabth^2) - 1.0)
            end
            return dtMs
        end
    end
end

"""
  Inputs:
    dtMs:
    j:
    na: na / n20
    ma: ma / Dₐ
    vth: vth / Mms

  Outputs:
    dtMs = dtMsrnt(dtMs,j,ma,Zq,na,vth,ns,nai,vthi,nMod;is_renorm=is_renorm)
    dtMs = dtMsrnt(dtMs,j,ma,Zq,na,vth,ns;is_renorm=is_renorm)
"""

# 1.5D, [nMod,ns]
function dtMsrnt(dtMs::AbstractVector{T},j::Int64,ma::AbstractVector{T},
    Zq::AbstractVector{Int},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64,
    nai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int};is_renorm::Bool=false) where{T}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec .≠ isp]
        iFv = nspF[1]
        maa = ma[isp]
        mM = maa / ma[iFv]
        na20, vath = na[isp], vth[isp]
        nb20, vbth = na[iFv], vth[iFv]
        ka = 1
        kb = 1
        naik, vathi = nai[isp][ka], vthi[isp][ka]
        nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
        dtMs[isp] = dtMsrnt(0.0,j,maa,mM,Zq,na20,nb20,vath,vbth,naik,nbi,vathi,vbthi;is_renorm=is_renorm)
        if nMod[iFv] == 1
            for ka in 2:nMod[isp]
                naik, vathi = nai[isp][ka], vthi[isp][ka]
                dtMs[isp] += dtMsrnt(0.0,j,maa,mM,Zq,na20,nb20,vath,vbth,naik,nbi,vathi,vbthi;is_renorm=is_renorm)
            end
        else
            for kb in 2:nMod[iFv]
                nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
                dtMs[isp] += dtMsrnt(0.0,j,maa,mM,Zq,na20,nb20,vath,vbth,naik,nbi,vathi,vbthi;is_renorm=is_renorm)
            end
            for ka in 2:nMod[isp]
                kb = 1
                naik, vathi = nai[isp][ka], vthi[isp][ka]
                nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
                dtMs[isp] += dtMsrnt(0.0,j,maa,mM,Zq,na20,nb20,vath,vbth,naik,nbi,vathi,vbthi;is_renorm=is_renorm)
                for kb in 2:nMod[iFv]
                    nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
                    dtMs[isp] += dtMsrnt(0.0,j,maa,mM,Zq,na20,nb20,vath,vbth,naik,nbi,vathi,vbthi;is_renorm=is_renorm)
                end
            end
        end
    end
    return dtMs
end

# 1D, [ns]
function dtMsrnt(dtMs::AbstractVector{T},j::Int64,ma::AbstractVector{T},Zq::AbstractVector{Int},
    na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;is_renorm::Bool=false) where{T}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec .≠ isp]
        iFv = nspF[1]
        mM = ma[isp] / ma[iFv]
        na20, nb20 = na[isp], na[iFv]
        vath, vbth = vth[isp], vth[iFv]
        dtMs[isp] = dtMsrnt(dtMs[isp],j,ma[isp],mM,Zq,na20,nb20,vath,vbth;is_renorm=is_renorm)
    end
    return dtMs
end

"""
  Inputs:

  Outputs:
    dtMs = dtMsrnt(dtMs,njMs,ma,mM,Zq,na,vth,nai,vathi,nMod;is_renorm=is_renorm)
    dtMs = dtMsrnt(dtMs,njMs,ma,mM,Zq,na,vth;is_renorm=is_renorm)
"""

# 1D, [njMs], jvec = 0:2:N⁺
function dtMsrnt(dtMs::AbstractVector{T},njMs::Int64,ma::T,mM::T,Zq::AbstractVector{Int},
    na::T,nb::T,vath::T,vbth::T;is_renorm::Bool=false) where{T}

    for k in 1:njMs
        j = 2(k - 1)
        dtMs[k] = dtMsrnt(dtMs[k],j,ma,mM,Zq,na,nb,vath,vbth;is_renorm=is_renorm)
    end
    return dtMs
end

"""
  Inputs:

  Outputs:
    dtMs = dtMsrnt(dtMs,njMs,ma,Zq,na,vth,ns;is_renorm=is_renorm)
"""
# 2D, [njMs,ns]
function dtMsrnt(dtMs::AbstractArray{T,N},njMs::Int64,ma::AbstractVector{T},Zq::AbstractVector{Int},
    na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;is_renorm::Bool=false) where{T,N}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec .≠ isp]
        iFv = nspF[1]
        mM = ma[isp] / ma[iFv]
        na20, nb20 = na[isp], na[iFv]
        vath, vbth = vth[isp], vth[iFv]
        dtMs[:,isp] = dtMsrnt(dtMs[:,isp],njMs,ma[isp],mM,Zq,na20,nb20,vath,vbth;is_renorm=is_renorm)
    end
    return dtMs
end

"""
  Inputs:

  Outputs:
    dtMs = dtMsrnt(dtMs,njMs,ma,Zq,na,vth,ns,nai,vthi,nMod;is_renorm=is_renorm)
"""

# 2.5D, [nMod,njMs,ns]

function dtMsrnt(dtMs::AbstractArray{T,N},njMs::Int64,ma::AbstractVector{T},
    Zq::AbstractVector{Int},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64,
    nai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int};is_renorm::Bool=false) where{T,N}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec .≠ isp]
        iFv = nspF[1]
        maa = ma[isp]
        mM = maa / ma[iFv]
        na20, vath = na[isp], vth[isp]
        nb20, vbth = na[iFv], vth[iFv]
        for k in 1:njMs
            j = 2(k - 1)
            ka = 1
            kb = 1
            naik, vathi = nai[isp][ka], vthi[isp][ka]
            nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
            dtMs[k,isp] = dtMsrnt(dtMs[k,isp],j,maa,mM,Zq,na20,nb20,vath,vbth,naik,nbi,vathi,vbthi;is_renorm=is_renorm)
            if nMod[iFv] == 1
                for ka in 2:nMod[isp]
                    naik, vathi = nai[isp][ka], vthi[isp][ka]
                    dtMs[k,isp] += dtMsrnt(dtMs[k,isp],j,maa,mM,Zq,na20,nb20,vath,vbth,naik,nbi,vathi,vbthi;is_renorm=is_renorm)
                end
            else
                for kb in 2:nMod[iFv]
                    nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
                    dtMs[k,isp] += dtMsrnt(dtMs[k,isp],j,maa,mM,Zq,na20,nb20,vath,vbth,naik,nbi,vathi,vbthi;is_renorm=is_renorm)
                end
                for ka in 2:nMod[isp]
                    kb = 1
                    naik, vathi = nai[isp][ka], vthi[isp][ka]
                    nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
                    dtMs[k,isp] += dtMsrnt(dtMs[k,isp],j,maa,mM,Zq,na20,nb20,vath,vbth,naik,nbi,vathi,vbthi;is_renorm=is_renorm)
                    for kb in 2:nMod[iFv]
                        nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
                        dtMs[k,isp] += dtMsrnt(dtMs[k,isp],j,maa,mM,Zq,na20,nb20,vath,vbth,naik,nbi,vathi,vbthi;is_renorm=is_renorm)
                    end
                end
            end
        end
    end
    return dtMs
end

# function dtMsrnt(dtMs::AbstractArray{T,N},njMs::Int64,ma::AbstractVector{T},
#     Zq::AbstractVector{Int},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64,
#     nai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int};is_renorm::Bool=false) where{T,N}
#
#     for k in 1:njMs
#         j = 2(k - 1)
#         dtMs[k,isp] = dtMsrnt(dtMs[k,isp],j,ma,Zq,na,vth,ns,nai,vthi,nMod;is_renorm=is_renorm)
#     end
#     return dtMs
# end
