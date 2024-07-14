
"""
  δₜfLn = 4π / π^(3/2) * na[iFv] / vth[iFv]^3 * Γₐᵦ * (∑ᵢŜᵢ)

       ∝ c1 * exp(- (1+vabth^2) * v²) - c2 / v * exp(- v²) * erf(vabth * v)

  which is not include the coefficient:

    cf3 = 1 / π^(3/2) * na[isp] / vth[isp]^3

  Inputs:
    nb: na[iFv] / n20
    vth: vth[iFv] / Mms
    ma: ma[isp] / Dₐ
    Zqab: Zq[isp] * Zq[iFv]

  Outputs:
    dtfLn = dtfMab(dtfLn,ma,Zq,na20,vth6,ns,vG0,εᵣ,nai,vthi,nMod)
    dtfLn = dtfMab(dtfLn,ma,Zq,na20,vth6,ns,vG0,εᵣ)
"""

# [nMod,ns], nSdtf = 1
function dtfMab(dtf::AbstractArray{T,N}, ma::AbstractVector{T}, Zq::Vector{Int},
    na::AbstractVector{T}, vth::AbstractVector{T}, ns::Int64, v::AbstractVector{T}, εᵣ::T,
    nai::AbstractVector{TA}, vthi::AbstractVector{TA}, nMod::Vector{Int}) where {T,TA,N}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec.≠isp]
        iFv = nspF[1]
        mM = ma[isp] / ma[iFv]
        Zqab = Zq[isp] * Zq[iFv]
        ka = 1
        naik, vathi = nai[isp][ka], vthi[isp][ka]
        kb = 1
        nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
        dtf[:, isp] = dtfMab(ma[isp], mM, Zqab, na[iFv],
            vth[isp], vth[iFv], v, εᵣ, naik, nbi, vathi, vbthi)
        if nMod[iFv] == 1
            for ka in 2:nMod[isp]
                naik, vathi = nai[isp][ka], vthi[isp][ka]
                dtf[:, isp] += dtfMab(ma[isp], mM, Zqab, na[iFv],
                    vth[isp], vth[iFv], v, εᵣ, naik, nbi, vathi, vbthi)
            end
        else
            for kb in 2:nMod[iFv]
                nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
                dtf[:, isp] += dtfMab(ma[isp], mM, Zqab, na[iFv],
                    vth[isp], vth[iFv], v, εᵣ, naik, nbi, vathi, vbthi)
            end
            for ka in 2:nMod[isp]
                naik, vathi = nai[isp][ka], vthi[isp][ka]
                kb = 1
                nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
                dtf[:, isp] += dtfMab(ma[isp], mM, Zqab, na[iFv],
                    vth[isp], vth[iFv], v, εᵣ, naik, nbi, vathi, vbthi)
                for kb in 2:nMod[iFv]
                    nbi, vbthi = nai[iFv][kb], vthi[iFv][kb]
                    dtf[:, isp] += dtfMab(ma[isp], mM, Zqab, na[iFv],
                        vth[isp], vth[iFv], v, εᵣ, naik, nbi, vathi, vbthi)
                end
            end
        end
    end
    return dtf
end

# [ns], nSdtf = 1, nai = nbi = vathi = vbthi = 1
function dtfMab(dtf::AbstractArray{T,N}, ma::AbstractVector{T}, Zq::Vector{Int},
    na::AbstractVector{T}, vth::AbstractVector{T}, ns::Int64, v::AbstractVector{Tb}, εᵣ::T) where {T,Tb,N}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec.≠isp]
        iFv = nspF[1]
        mM = ma[isp] / ma[iFv]
        Zqab = Zq[isp] * Zq[iFv]
        vabth = vth[isp] / vth[iFv]
        dtf[:, isp] = dtfMab(ma[isp], mM, Zqab, na[iFv], vth[iFv], vabth, v, εᵣ)
    end
    return dtf
end

"""

  Inputs:
    nb: na[iFv] / n20
    vth: vth[iFv] / Mms
    ma: ma[isp] / Dₐ
    Zqab: Zq[isp] * Zq[iFv]

  Outputs:
    dtfLn = dtfMab(ma,mM,Zqab,nb,vath,vbth,vG0, spices,εᵣ,nai,nbi,vathi,vbthi)
    dtfLn = dtfMab(ma,mM,Zqab,nb,vath,vbth,vG0, spices,εᵣ)
"""

# nSdtf = 1
function dtfMab(ma::T, mM::T, Zqab::Int64, nb::T, vath::T, vbth::T, 
    v::AbstractVector{Tb},spices::Vector{Symbol}, εᵣ::T, nai::T, nbi::T, vathi::T, vbthi::T) where {T,Tb}

    lnAg = lnAgamma(ma, Zqab, 1.0, nb, 1.0, vbth, spices, εᵣ) # is_normdtf = true
    # lnAg = lnAgamma(ma,Zqab,na,nb,vath,vbth, spices,εᵣ)
    lnAg *= CΓ        # CΓ is owing to the dimensionless process
    vabth = vath / vbth / vbthi
    vabth2 = vabth^2
    vathi2 = vathi^2
    Sf = (1 + 1 / vabth2 / vathi2) * exp.(-(1 / vathi2 + vabth2) * v .^ 2) -
            sqrtpi / 2 / vabth2 / vabth / vathi2 * erf.(vabth * v) ./ v .* exp.(-v .^ 2 / vathi2)
    if v[1] == 0
        Sf[1] = lnAg * (mM - 1 / vabth2 / vathi2) * (nai * nbi / (vathi * vbthi)^3)
        Sf[2:end] *= Sf[1]
    else
        Sf *= (lnAg * (mM - 1 / vabth2 / vathi2) * (nai * nbi / (vathi * vbthi)^3))
    end
    wedfgbn
    return Sf
end

# nSdtf = 1, nai = nbi = vathi = vbthi = 1
function dtfMab(ma::T, mM::T, Zqab::Int64, nb::T, 
    vbth::T, vabth::T, v::AbstractVector{Tb},spices::Vector{Symbol}, εᵣ::T) where {T,Tb}

    lnAg = lnAgamma(ma, Zqab, 1.0, nb, 1.0, vbth, spices, εᵣ) # ;is_normdtf=true
    vabth2 = vabth^2
    Sf = (mM - 1 / vabth2) * exp.(- (1 + vabth2) * v .^ 2) 
    Sf .*= ((1 + 1 / vabth2) .- sqrtpi / 2 / vabth^3 * exp.(vabth2 * v .^ 2) ./ v .* erf.(vabth * v))
    if mM == 1 && vabth == 1
        Sf[1:2] .= 0.0
    else
        count0 = 0
        for vv in v
            vv < 2e-6 ? count0 += 1 : break
        end
        av0(v) = (mM / vabth2 - 1.0 / vabth2^2) * ((1.0 + vabth2) * exp.(-(1 + vabth2) * v .^ 2)
                    .- (1.0 .- vabth2 / 3.0 * v .^ 2 + vabth2^2 / 10 * v .^ 4) .* exp.(vabth2 * v .^ 2))
        if count0 > 0
            # @show Sf[1:3]
            # @show av0(v[1:count0+1])
            # sdfvgbjh
            Sf[1:count0] = av0(v[1:count0])
        end
    end
    Sf *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
    return Sf
end
# normalized
function dtfMab(mM::T, vabth::T, v::AbstractVector{Tb}) where {T,Tb}

    vabth2 = vabth^2
    Sf = (mM - 1 / vabth2) * exp.(- (1 + vabth2) * v .^ 2) 
    Sf .*= ((1 + 1 / vabth2) .- sqrtpi / 2 / vabth^3 * exp.(vabth2 * v .^ 2) ./ v .* erf.(vabth * v))
    if mM == 1 && vabth == 1
        Sf[1:2] .= 0.0
    else
        count0 = 0
        for vv in v
            vv < 2e-6 ? count0 += 1 : break
        end
        av0(v) = (mM / vabth2 - 1.0 / vabth2^2) * ((1.0 + vabth2) * exp.(-(1 + vabth2) * v .^ 2)
                                                  .-
                                                  (1.0 .- vabth2 / 3.0 * v .^ 2 + vabth2^2 / 10 * v .^ 4) .* exp.(vabth2 * v .^ 2))
        if count0 > 0
            # @show Sf[1:3]
            # @show av0(v[1:count0])
            Sf[1:count0] = av0(v[1:count0])
        end
    end
    Sf *= CΓ   # CΓ is owing to the dimensionless process
    return Sf
end

# Sf = zeros(T,length(v),nSdtf)
function dtfMab(Sf::AbstractArray{T,N}, ma::T, mM::T, Zqab::Int64, nb::T, 
    vbth::T, vabth::T, v::AbstractVector{Tb},spices::Vector{Symbol}, εᵣ::T) where {T,Tb,N}

    lnAg = lnAgamma(ma, Zqab, 1.0, nb, 1.0, vbth, spices, εᵣ) # ;is_normdtf=true
    # lnAg = lnAgamma(ma[isp],Zqab,na,nb,vath,vbth, spices,εᵣ)
    vabth2 = vabth^2
    CF = mM
    CH = (1 - mM) / vabth
    CG = 0.5 / vabth2
    va = vabth * v
    expv2 = exp.(-(1 + vabth2) * v .^ 2)
    expv2erfv = sqrtpi / 2 * exp.(-v .^ 2) .* erf.(va)
    # Sf1
    (abs(CF - 1.0) + abs(vabth - 1.0) ≤ epsT10) ? Sf[:, 1] = CF * expv2 : Sf[:, 1] = expv2
    # Sf2
    if CH ≠ 0.0
        Sf[:, 2] = -CH / vabth * (expv2 - (va) .* expv2erfv)
        Sf[1, 2] = CH * 2 / 3 * va[1]
    end
    # Sf5
    Sf[:, 4] = - CG / vabth^3 * (2 ./ v - 1 ./ v .^ 3) .* (va .* expv2 - expv2erfv)
    Sf[1, 4] = CG * (-2 / 3 + 2 / 5 * ((5 + vabth2) * v[1]))
    # Sf8, Sf10
    Sf[:, 6] = -CG ./ (2vabth^3 .* v .^ 3) .* (va .* expv2 + (2vabth^2 * v .^ 2 .- 1) .* expv2erfv)
    Sf[1, 6] = CG * (-2 / 3 + 2 / 15 * ((5 + vabth2) * v[1]))
    Sf[:, 7] = Sf[:, 6]
    Sf *= lnAg
    # FLn = exp.(-va.^2)
    # HLn = sqrtpi / 4 * erf.(va) ./ va
    # HLn[1] = 0.5 - va[1]^2 / 6
    # GLn = sqrtpi / 4 * (0.5 ./va + va) .* erf.(va) + exp.(-va.^2) / 4
    # GLn[1] = (0.5 + va[1]^2) * (0.5 - va[1]^2 / 6) + exp(-va[1]^2) / 4
    # dHLn = exp.(-va.^2) ./ 2va - sqrtpi / 4 * erf.(va) ./ va.^2
    # dHLn[1] = - va[1] / 3 + va[1]^3 / 5
    # dGLn = exp.(-va.^2) ./ 4va + sqrtpi / 4 * erf.(va) .* (1 .- 1 ./ 2va.^2)
    # dGLn[1] = va[1] / 3 - va[1]^3 / 15
    # ddGLn = - exp.(-va.^2) ./ 2va.^2 + sqrtpi / 4 * erf.(va) ./ va.^3
    # ddGLn[1] = 1 / 3 - va[1]^2 / 5
    # label = "HLnt"
    # ph = plot(va,HLn,label=label)
    # label = "FLnt"
    # pF = plot(va,FLn,label=label)
    # display(plot(pF,ph,layout=(2,1)))
    # @show mM, vabth, fmtf4.(FLn[1:4])
    # @show mM, vabth, fmtf4.(HLn[1:4])
    # @show mM, vabth, fmtf4.(GLn[1:4])
    # @show mM, vabth, fmtf4.(dHLn[1:4])
    # @show mM, vabth, fmtf4.(dGLn[1:4])
    return Sf
end
