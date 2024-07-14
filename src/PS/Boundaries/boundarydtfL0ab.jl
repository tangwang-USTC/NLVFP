
"""
  When `v̂ = 0`: boundary conditions of collision terms:

    Sf[:,3] = Sf3
    Sf[:,6] = Sf6
    Sf[:,8] = Sf8
    Sf[:,10] = Sf10

  The boundary conditions of the Fokker-Planck collison in `collisionsLSab.jl`.

  Warning: coefficient `na / vth^3 / π^(3/2)` of `f̂ₗ(v̂)` is not included in following codes.

  Warning: transformation of the inverse spherical harmonics is not implemented in the following codes.

  Inputs:
    ua = uai[isp] / vth[isp] = [uai[1],uai[2],⋯] with number `nMod[isp]`
    ub = uai[iFv] / vth[iFv] = [ubi[1],ubi[2],⋯] with number `nMod[iFv]`

  Outputs:
    Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CH,CG,nai,nbi,vathi,vbthi,uai,ubi,LM1,nModa,nModb)
    Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CG,nai,nbi,vathi,vbthi,uai,ubi,LM1,nModa,nModb)
    # Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CH,CG,nai,nbi,vathi,vbthi,uai,ubi,LM1)
    # Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CG,nai,nbi,vathi,vbthi,uai,ubi,LM1)
    Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CH,CG,uai,ubi,LM1)
    Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CG,uai,ubi,LM1)
"""

"""

  Outputs:
    Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CH,CG,nai,nbi,vathi,vbthi,uai,ubi,LM1,nModa,nModb)
"""

function dtfvLDMabv0(Sf1::AbstractArray{T,N},mu::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    CH::T,CG::T,na::AbstractVector{T},nb::AbstractVector{T},vath::AbstractVector{T},
    vbth::AbstractVector{T},ua::AbstractVector{T},ub::AbstractVector{T},
    LM1::Int64,nModa::Int64,nModb::Int64) where{T,N,NM1,NM2}

    sedrfg
    cotmu = mu ./ (1 .-mu.^2).^0.5
    ##
    dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
    fvLv = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]
    fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
    ka = 1
    nai = na[ka]
    vathi, uai = vath[ka], ua[ka]
    if uai == 0
        if vathi == 1
            dfvLv[1,1] = - nai * 2
            # if CH ≠ 0.0
            #     fvLv[1,2] = 0.0
            # end
            # if NM2 ≥ 3
            #     fvLv2[1,3] = 0.0
            #     dfvLv[1,3] = 0.0
            # end
        else
            dfvLv[1,1] = - nai/vathi^3/vathi * 2
            # if CH ≠ 0.0
            #     fvLv[1,2] = 0.0
            # end
            # if NM2 ≥ 3
            #     fvLv2[1,3] = 0.0
            #     dfvLv[1,3] = 0.0
            # end
        end
    else
        uai2 = uai^2
        if vathi == 1
            dfvLv[1,1] = - nai * 2(1.0 - 2/3 * uai2) * exp(-uai2)
            if CH ≠ 0.0
                fvLv[1,2] = nai * 2uai * exp(-uai2)
            end
            if NM2 ≥ 3
                a = 4/3 * nai * uai2 * exp(-uai2)
                fvLv2[1,3] = a
                dfvLv[1,3] = 2a
            end
        else
            # L = 0
            dfvLv[1,1] = - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
            # L = 1
            if CH ≠ 0.0
                fvLv[1,2] = nai/vathi^3/vathi * 2uai * exp(-uai2)
            end
            # L = 2
            if NM2 ≥ 3
                a = 4/3 * nai/vathi^3 / vathi * uai2 * exp(-uai2)
                fvLv2[1,3] = a / vathi
                dfvLv[1,3] = 2a
            end
        end
    end
    for ka in 2:nModa
        nai = na[ka]
        if nai > 0
            vathi, uai = vath[ka], ua[ka]
            if uai == 0
                if vathi == 1
                    dfvLv[1,1] += - nai * 2
                    # if CH ≠ 0.0
                    #     fvLv[1,2] += 0.0
                    # end
                    # if NM2 ≥ 3
                    #     fvLv2[1,3] += 0.0
                    #     dfvLv[1,3] += 0.0
                    # end
                else
                    dfvLv[1,1] += - nai/vathi^3/vathi * 2
                    # if CH ≠ 0.0
                    #     fvLv[1,2] += 0.0
                    # end
                    # if NM2 ≥ 3
                    #     fvLv2[1,3] += 0.0
                    #     dfvLv[1,3] += 0.0
                    # end
                end
            else
                uai2 = uai^2
                if vathi == 1
                    dfvLv[1,1] += - nai * 2(1.0 - 2/3 * uai2) * exp(-uai2)
                    if CH ≠ 0.0
                        fvLv[1,2] += nai * 2uai * exp(-uai2)
                    end
                    if NM2 ≥ 3
                        a = 4/3 * nai * uai2 * exp(-uai2)
                        fvLv2[1,3] += a
                        dfvLv[1,3] += 2a
                    end
                else
                    dfvLv[1,1] += - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
                    if CH ≠ 0.0
                        fvLv[1,2] += nai/vathi^3/vathi * 2uai * exp(-uai2)
                    end
                    if NM2 ≥ 3
                        a = 4/3 * nai/vathi^3 / vathi * uai2 * exp(-uai2)
                        fvLv2[1,3] += a / vathi
                        dfvLv[1,3] += 2a
                    end
                end
            end
        end
    end
    dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
    HvLv = zeros(T,2,LM1) # = HvL[1:2,2:end] ./ va0[1:2]
    GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
    kb = 1
    nbi = nb[kb]
    vbthi, ubi = vbth[kb], ub[kb]
    if ubi == 0
        if vbthi == 1
            dGvLv[1,1] = nbi / 3
            # if CH ≠ 0.0
            #     HvLv[1,2] = 0.0
            # end
            # if NM2 ≥ 3
            #     GvLv2[1,3] = 0.0
            #     dGvLv[1,3] = 0.0
            # end
        else
            dGvLv[1,1] = nbi/vbthi / 3
            # if CH ≠ 0.0
            #     HvLv[1,2] = 0.0
            # end
            # if NM2 ≥ 3
            #     GvLv2[1,3] = 0.0
            #     dGvLv[1,3] = 0.0
            # end
        end
    else
        ubi2 = ubi^2
        if vbthi == 1
            dGvLv[1,1] = sqrtpi / 6 * nbi * erf(ubi) / ubi
            if CH ≠ 0.0
                HvLv[1,2] = nbi / 2ubi * (sqrtpi / 2 * erf(ubi) / ubi2 - exp(-ubi2))
            end
            if NM2 ≥ 3
                a = nbi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                GvLv2[1,3] = a
                dGvLv[1,3] = 2a
            end
        else
            # L = 0
            dGvLv[1,1] = sqrtpi / 6 * nbi/vbthi * erf(ubi) / ubi
            # L = 1
            if CH ≠ 0.0
                HvLv[1,2] = nbi/vbthi / 2ubi * (sqrtpi / 2 * erf(ubi) / ubi2 - exp(-ubi2))
            end
            # L = 2
            if NM2 ≥ 3
                a = nbi/vbthi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                GvLv2[1,3] = a / vbthi
                dGvLv[1,3] = 2a
            end
        end
    end
    for kb in 2:nModb
        nbi = nb[kb]
        if nbi > 0
            vbthi, ubi = vbth[kb], ub[kb]
            if ubi == 0
                if vbthi == 1
                    dGvLv[1,1] += nbi / 3
                    # if CH ≠ 0.0
                    #     HvLv[1,2] += 0.0
                    # end
                    # if NM2 ≥ 3
                    #     GvLv2[1,3] += 0.0
                    #     dGvLv[1,3] += 0.0
                    # end
                else
                    dGvLv[1,1] += nbi/vbthi / 3
                    # if CH ≠ 0.0
                    #     HvLv[1,2] += 0.0
                    # end
                    # if NM2 ≥ 3
                    #     GvLv2[1,3] += 0.0
                    #     dGvLv[1,3] += 0.0
                    # end
                end
            else
                ubi2 = ubi^2
                if vbthi == 1
                    dGvLv[1,1] += sqrtpi / 6 * nbi * erf(ubi) / ubi
                    if CH ≠ 0.0
                        HvLv[1,2] += nbi / 2ubi * (sqrtpi / 2 * erf(ubi) / ubi2 - exp(-ubi2))
                    end
                    if NM2 ≥ 3
                        a = nbi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                        GvLv2[1,3] += a
                        dGvLv[1,3] += 2a
                    end
                else
                    # L = 0
                    dGvLv[1,1] += sqrtpi / 6 * nbi/vbthi * erf(ubi) / ubi
                    # L = 1
                    if CH ≠ 0.0
                        HvLv[1,2] += nbi/vbthi / 2ubi * (sqrtpi / 2 * erf(ubi) / ubi2 - exp(-ubi2))
                    end
                    # L = 2
                    if NM2 ≥ 3
                        a = nbi/vbthi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                        GvLv2[1,3] += a / vbthi
                        dGvLv[1,3] += 2a
                    end
                end
            end
        end
    end
    # Computing `Sf1[1,:] += Sf3[1,:] + Sf6[1,:] + Sf8[1,:] + Sf10[1,:]`

    # ############ 3,  Sf3
    if CH ≠ 0.0
        Sf1[1,:] += CH * ((fvLv[:,2:end] * Mun1) .* (HvLv[:,2:end] * Mun1))[1,:]
    end
    ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = (1 - 2vbthi) × GvLv2[:,2:end]
    # GG1 = dX01 * Mun1
    GG1 = (GvLv2[:,2:end] - dGvLv[:,2:end]) * Mun1
    # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = (1 - 2vathi) × fvLv2[:,2:end]
    # GG1 .*= 2(dX01 * Mun1)
    GG1 .*= 2((fvLv2[:,2:end] - dfvLv[:,2:end]) * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = ((dGvLv * Mun) + (GvLv2[:,2:end] * Mun1) .* cotmu)
    GG7 = copy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
        GG7 .*= dX01
        GG1 += 2GG7
    else
        GG6 = (dX01 + (GvLv2[:,3:end] * Mun2))
        dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvLv2[:,3:end] * Mun2))
        GG1 += (GG7 + GG6)
    end
    Sf1[1,:] += CG * GG1[1,:]
    return Sf1
end

"""
  Outputs:
    Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CG,nai,nbi,vathi,vbthi,uai,ubi,LM1,nModa,nModb)
"""

# mM = 1   →    CH = 0.0
function dtfvLDMabv0(Sf1::AbstractArray{T,N},mu::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    CG::T,na::AbstractVector{T},nb::AbstractVector{T},vath::AbstractVector{T},
    vbth::AbstractVector{T},ua::AbstractVector{T},ub::AbstractVector{T},
    LM1::Int64,nModa::Int64,nModb::Int64) where{T,N,NM1,NM2}

    cotmu = mu ./ (1 .-mu.^2).^0.5
    ##
    dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
    fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
    ka = 1
    nai = na[ka]
    vathi, uai = vath[ka], ua[ka]
    if uai == 0
        if vathi == 1
            dfvLv[1,1] = - nai * 2
            # if NM2 ≥ 3
            #     fvLv2[1,3] = 0.0
            #     dfvLv[1,3] = 0.0
            # end
        else
            dfvLv[1,1] = - nai/vathi^3/vathi * 2
            # if NM2 ≥ 3
            #     fvLv2[1,3] = 0.0
            #     dfvLv[1,3] = 0.0
            # end
        end
    else
        uai2 = uai^2
        if vathi == 1
            dfvLv[1,1] = - nai * 2(1.0 - 2/3 * uai2) * exp(-uai2)
            if NM2 ≥ 3
                a = 4/3 * nai * uai2 * exp(-uai2)
                fvLv2[1,3] = a
                dfvLv[1,3] = 2a
            end
        else
            # L = 0
            dfvLv[1,1] = - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
            # L = 1
            # L = 2
            if NM2 ≥ 3
                a = 4/3 * nai/vathi^3 / vathi * uai2 * exp(-uai2)
                fvLv2[1,3] = a / vathi
                dfvLv[1,3] = 2a
            end
        end
    end
    for ka in 2:nModa
        nai = na[ka]
        if nai > 0
            vathi, uai = vath[ka], ua[ka]
            if uai == 0
                if vathi == 1
                    dfvLv[1,1] += - nai * 2
                    # if NM2 ≥ 3
                    #     fvLv2[1,3] += 0.0
                    #     dfvLv[1,3] += 0.0
                    # end
                else
                    dfvLv[1,1] += - nai/vathi^3/vathi * 2
                    # if NM2 ≥ 3
                    #     fvLv2[1,3] += 0.0
                    #     dfvLv[1,3] += 0.0
                    # end
                end
            else
                uai2 = uai^2
                if vathi == 1
                    dfvLv[1,1] += - nai * 2(1.0 - 2/3 * uai2) * exp(-uai2)
                    if NM2 ≥ 3
                        a = 4/3 * nai * uai2 * exp(-uai2)
                        fvLv2[1,3] += a
                        dfvLv[1,3] += 2a
                    end
                else
                    dfvLv[1,1] += - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
                    if NM2 ≥ 3
                        a = 4/3 * nai/vathi^3 / vathi * uai2 * exp(-uai2)
                        fvLv2[1,3] += a / vathi
                        dfvLv[1,3] += 2a
                    end
                end
            end
        end
    end
    dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
    GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
    kb = 1
    nbi = nb[kb]
    vbthi, ubi = vbth[kb], ub[kb]
    if ubi == 0
        if vbthi == 1
            dGvLv[1,1] = nbi / 3
            # if NM2 ≥ 3
            #     GvLv2[1,3] = 0.0
            #     dGvLv[1,3] = 0.0
            # end
        else
            dGvLv[1,1] = nbi/vbthi / 3
            # if NM2 ≥ 3
            #     GvLv2[1,3] = 0.0
            #     dGvLv[1,3] = 0.0
            # end
        end
    else
        ubi2 = ubi^2
        if vbthi == 1
            dGvLv[1,1] = sqrtpi / 6 * nbi * erf(ubi) / ubi
            if NM2 ≥ 3
                a = nbi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                GvLv2[1,3] = a
                dGvLv[1,3] = 2 * GvLv2[1,3]
            end
        else
            # L = 0
            dGvLv[1,1] = sqrtpi / 6 * nbi/vbthi * erf(ubi) / ubi
            # L = 1
            # L = 2
            if NM2 ≥ 3
                a = nbi/vbthi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                GvLv2[1,3] = a / vbthi
                dGvLv[1,3] = 2 * GvLv2[1,3]
            end
        end
    end
    for kb in 2:nModb
        nbi = nb[kb]
        if nbi > 0
            vbthi, ubi = vbth[kb], ub[kb]
            if ubi == 0
                if vbthi == 1
                    dGvLv[1,1] += nbi / 3
                    # if NM2 ≥ 3
                    #     GvLv2[1,3] += 0.0
                    #     dGvLv[1,3] += 0.0
                    # end
                else
                    dGvLv[1,1] += nbi/vbthi / 3
                    # if NM2 ≥ 3
                    #     GvLv2[1,3] += 0.0
                    #     dGvLv[1,3] += 0.0
                    # end
                end
            else
                ubi2 = ubi^2
                if vbthi == 1
                    dGvLv[1,1] += sqrtpi / 6 * nbi * erf(ubi) / ubi
                    if NM2 ≥ 3
                        a = nbi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                        GvLv2[1,3] += a
                        dGvLv[1,3] += 2a
                    end
                else
                    dGvLv[1,1] += sqrtpi / 6 * nbi/vbthi * erf(ubi) / ubi
                    if NM2 ≥ 3
                        a = nbi/vbthi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                        GvLv2[1,3] += a / vbthi
                        dGvLv[1,3] += 2a
                    end
                end
            end
        end
    end
    # Computing `Sf1[1,:] += Sf3[1,:] + Sf6[1,:] + Sf8[1,:] + Sf10[1,:]`

    # ############ 3,  Sf3
    ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = (1 - 2vbthi) × GvLv2[:,2:end]
    # GG1 = dX01 * Mun1
    GG1 = - GvLv2[:,2:end] * Mun1
    # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = (1 - 2vathi) × fvLv2[:,2:end]
    # GG1 .*= 2(dX01 * Mun1)
    GG1 .*= - 2 * (fvLv2[:,2:end] * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = ((dGvLv * Mun) + (GvLv2[:,2:end] * Mun1) .* cotmu)
    GG7 = copy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
        GG7 .*= dX01
        GG1 += 2GG7
    else
        GG6 = (dX01 + (GvLv2[:,3:end] * Mun2))
        dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvLv2[:,3:end] * Mun2))
        GG1 += (GG7 + GG6)
    end
    Sf1[1,:] += CG * GG1[1,:]
    return Sf1
end

## nModa = nModb == 1 means `nai = vthi = 1`

"""

  Outputs:
    Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CH,CG,uai,ubi,LM1)
"""

function dtfvLDMabv0(Sf1::AbstractArray{T,N},mu::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    CH::T,CG::T,uai::T,ubi::T,LM1::Int64) where{T,N,NM1,NM2}

    cotmu = mu ./ (1 .-mu.^2).^0.5
    ##
    dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
    fvLv = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]
    fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
    if uai == 0
        dfvLv[1,1] = - 2.0
    else
        uai2 = uai^2
        dfvLv[1,1] = - 2(1.0 - 2/3 * uai2) * exp(-uai2)
        if CH ≠ 0.0
            fvLv[1,2] = 2uai * exp(-uai2)
        end
        if NM2 ≥ 3
            fvLv2[1,3] = 4/3 * uai2 * exp(-uai2)
            dfvLv[1,3] = 2fvLv2[1,3]
        end
    end
    dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
    HvLv = zeros(T,2,LM1) # = HvL[1:2,2:end] ./ va0[1:2]
    GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
    if ubi == 0
        dGvLv[1,1] = 1 / 3
    else
        ubi2 = ubi^2
        dGvLv[1,1] = sqrtpi / 6 * erf(ubi) / ubi
        if CH ≠ 0.0
            HvLv[1,2] = (sqrtpi / 2 * erf(ubi) / ubi2 - exp(-ubi2)) / 2ubi
        end
        if NM2 ≥ 3
            GvLv2[1,3] = (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2)) / 4ubi2
            dGvLv[1,3] = 2GvLv2[1,3]
        end
    end
    # Computing `Sf1[1,:] += Sf3[1,:] + Sf6[1,:] + Sf8[1,:] + Sf10[1,:]`

    # ############ 3,  Sf3
    if CH ≠ 0.0
        Sf1[1,:] += CH * ((fvLv[:,2:end] * Mun1) .* (HvLv[:,2:end] * Mun1))[1,:]
    end
    ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = (1 - 2vbthi) × GvLv2[:,2:end]
    # GG1 = dX01 * Mun1
    GG1 = (GvLv2[:,2:end] - dGvLv[:,2:end]) * Mun1
    # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = (1 - 2vathi) × fvLv2[:,2:end]
    # GG1 .*= 2(dX01 * Mun1)
    GG1 .*= 2((fvLv2[:,2:end] - dfvLv[:,2:end]) * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = ((dGvLv * Mun) + (GvLv2[:,2:end] * Mun1) .* cotmu)
    GG7 = copy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
        GG7 .*= dX01
        GG1 += 2GG7
    else
        GG6 = (dX01 + (GvLv2[:,3:end] * Mun2))
        dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvLv2[:,3:end] * Mun2))
        GG1 += (GG7 + GG6)
    end
    Sf1[1,:] += CG * GG1[1,:]
    return Sf1
end

"""
  Outputs:
    Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CG,uai,ubi,LM1)
"""

# mM = 1   →    CH = 0.0
function dtfvLDMabv0(Sf1::AbstractArray{T,N},mu::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    CG::T,uai::T,ubi::T,LM1::Int64) where{T,N,NM1,NM2}

    cotmu = mu ./ (1 .-mu.^2).^0.5
    ##
    dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
    fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
    if uai == 0
        dfvLv[1,1] = - 2.0
    else
        uai2 = uai^2
        dfvLv[1,1] = - 2(1.0 - 2/3 * uai2) * exp(-uai2)
        if NM2 ≥ 3
            fvLv2[1,3] = 4/3 * uai2 * exp(-uai2)
            dfvLv[1,3] = 2fvLv2[1,3]
        end
    end
    dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
    GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
    if ubi == 0
        dGvLv[1,1] = 1 / 3
    else
        ubi2 = ubi^2
        dGvLv[1,1] = sqrtpi / 6 * erf(ubi) / ubi
        if NM2 ≥ 3
            GvLv2[1,3] = (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2)) / 4ubi2
            dGvLv[1,3] = 2 * GvLv2[1,3]
        end
    end
    # Computing `Sf1[1,:] += Sf3[1,:] + Sf6[1,:] + Sf8[1,:] + Sf10[1,:]`

    # ############ 3,  Sf3
    ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = (1 - 2vbthi) × GvLv2[:,2:end]
    # GG1 = dX01 * Mun1
    GG1 = (GvLv2[:,2:end] - dGvLv[:,2:end]) * Mun1
    # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = (1 - 2vathi) × fvLv2[:,2:end]
    # GG1 .*= 2(dX01 * Mun1)
    GG1 .*= 2((fvLv2[:,2:end] - dfvLv[:,2:end]) * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = ((dGvLv * Mun) + (GvLv2[:,2:end] * Mun1) .* cotmu)
    GG7 = copy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
        GG7 .*= dX01
        GG1 += 2GG7
    else
        GG6 = (dX01 + (GvLv2[:,3:end] * Mun2))
        dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvLv2[:,3:end] * Mun2))
        GG1 += (GG7 + GG6)
    end
    Sf1[1,:] += CG * GG1[1,:]
    return Sf1
end


# ## nModa = nModb == 1
#
# """
#
#   Outputs:
#     Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CH,CG,nai,nbi,vathi,vbthi,uai,ubi,LM1)
# """
#
# function dtfvLDMabv0(Sf1::AbstractArray{T,N},mu::AbstractArray{T,N},
#     Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
#     CH::T,CG::T,nai::T,nbi::T,vathi::T,vbthi::T,uai::T,ubi::T,LM1::Int64) where{T,N,NM1,NM2}
#
#     cotmu = mu ./ (1 .-mu.^2).^0.5
#     ##
#     dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
#     fvLv = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]
#     fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
#     if uai == 0
#         if vathi == 1
#             dfvLv[1,1] = - nai * 2
#         else
#             dfvLv[1,1] = - nai/vathi^3/vathi * 2
#         end
#     else
#         uai2 = uai^2
#         if vathi == 1
#             dfvLv[1,1] = - nai * 2(1.0 - 2/3 * uai2) * exp(-uai2)
#             if CH ≠ 0.0
#                 fvLv[1,2] = nai * 2uai * exp(-uai2)
#             end
#             if NM2 ≥ 3
#                 a = 4/3 * nai * uai2 * exp(-uai2)
#                 fvLv2[1,3] = a
#                 dfvLv[1,3] = 2a
#             end
#         else
#             # L = 0
#             dfvLv[1,1] = - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
#             # L = 1
#             if CH ≠ 0.0
#                 fvLv[1,2] = nai/vathi^3/vathi * 2uai * exp(-uai2)
#             end
#             # L = 2
#             if NM2 ≥ 3
#                 a = 4/3 * nai/vathi^3 / vathi * uai2 * exp(-uai2)
#                 fvLv2[1,3] = a / vathi
#                 dfvLv[1,3] = 2a
#             end
#         end
#     end
#     dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
#     HvLv = zeros(T,2,LM1) # = HvL[1:2,2:end] ./ va0[1:2]
#     GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
#     if ubi == 0
#         if vbthi == 1
#             dGvLv[1,1] = nbi / 3
#         else
#             dGvLv[1,1] = nbi/vbthi / 3
#         end
#     else
#         ubi2 = ubi^2
#         if vbthi == 1
#             dGvLv[1,1] = sqrtpi / 6 * nbi * erf(ubi) / ubi
#             if CH ≠ 0.0
#                 HvLv[1,2] = nbi / 2ubi * (sqrtpi / 2 * erf(ubi) / ubi2 - exp(-ubi2))
#             end
#             if NM2 ≥ 3
#                 a = nbi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
#                 GvLv2[1,3] = a
#                 dGvLv[1,3] = 2a
#             end
#         else
#             # L = 0
#             dGvLv[1,1] = sqrtpi / 6 * nbi/vbthi * erf(ubi) / ubi
#             # L = 1
#             if CH ≠ 0.0
#                 HvLv[1,2] = nbi/vbthi / 2ubi * (sqrtpi / 2 * erf(ubi) / ubi2 - exp(-ubi2))
#             end
#             # L = 2
#             if NM2 ≥ 3
#                 a = nbi/vbthi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
#                 GvLv2[1,3] = a / vbthi
#                 dGvLv[1,3] = 2a
#             end
#         end
#     end
#     # Computing `Sf1[1,:] += Sf3[1,:] + Sf6[1,:] + Sf8[1,:] + Sf10[1,:]`
#
#     # ############ 3,  Sf3
#     if CH ≠ 0.0
#         Sf1[1,:] += CH * ((fvLv[:,2:end] * Mun1) .* (HvLv[:,2:end] * Mun1))[1,:]
#     end
#     ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
#     # ############ 5,  Sf6
#     # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = (1 - 2vbthi) × GvLv2[:,2:end]
#     # GG1 = dX01 * Mun1
#     GG1 = (GvLv2[:,2:end] - dGvLv[:,2:end]) * Mun1
#     # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = (1 - 2vathi) × fvLv2[:,2:end]
#     # GG1 .*= 2(dX01 * Mun1)
#     GG1 .*= 2((fvLv2[:,2:end] - dfvLv[:,2:end]) * Mun1)
#     ############ (6,7), (Sf8, Sf10)
#     dX01 = ((dGvLv * Mun) + (GvLv2[:,2:end] * Mun1) .* cotmu)
#     GG7 = copy(dX01)
#     if NM2 == 0
#         # GG6 = GG7 = Sf8 = Sf10
#         dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
#         GG7 .*= dX01
#         GG1 += 2GG7
#     else
#         GG6 = (dX01 + (GvLv2[:,3:end] * Mun2))
#         dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
#         GG7 .*= dX01
#         GG6 .*= (dX01 + (fvLv2[:,3:end] * Mun2))
#         GG1 += (GG7 + GG6)
#     end
#     Sf1[1,:] += CG * GG1[1,:]
#     return Sf1
# end
#
# """
#   Outputs:
#     Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CG,nai,nbi,vathi,vbthi,uai,ubi,LM1)
# """
#
# # mM = 1   →    CH = 0.0
# function dtfvLDMabv0(Sf1::AbstractArray{T,N},mu::AbstractArray{T,N},
#     Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
#     CG::T,nai::T,nbi::T,vathi::T,vbthi::T,uai::T,ubi::T,LM1::Int64) where{T,N,NM1,NM2}
#
#     cotmu = mu ./ (1 .-mu.^2).^0.5
#     ##
#     dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
#     fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
#     if uai == 0
#         if vathi == 1
#             dfvLv[1,1] = - nai * 2
#             # if NM2 ≥ 3
#             #     fvLv2[1,3] = 0.0
#             #     dfvLv[1,3] = 0.0
#             # end
#         else
#             dfvLv[1,1] = - nai/vathi^3/vathi * 2
#             # if NM2 ≥ 3
#             #     fvLv2[1,3] = 0.0
#             #     dfvLv[1,3] = 0.0
#             # end
#         end
#     else
#         uai2 = uai^2
#         if vathi == 1
#             dfvLv[1,1] = - nai * 2(1.0 - 2/3 * uai2) * exp(-uai2)
#             if NM2 ≥ 3
#                 a = 4/3 * nai * uai2 * exp(-uai2)
#                 fvLv2[1,3] = a
#                 dfvLv[1,3] = 2a
#             end
#         else
#             # L = 0
#             dfvLv[1,1] = - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
#             # L = 1
#             # L = 2
#             if NM2 ≥ 3
#                 a = 4/3 * nai/vathi^3 / vathi * uai2 * exp(-uai2)
#                 fvLv2[1,3] = a / vathi
#                 dfvLv[1,3] = 2a
#             end
#         end
#     end
#     dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
#     GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
#     if ubi == 0
#         if vbthi == 1
#             dGvLv[1,1] = nbi / 3
#             # if NM2 ≥ 3
#             #     GvLv2[1,3] = 0.0
#             #     dGvLv[1,3] = 0.0
#             # end
#         else
#             dGvLv[1,1] = nbi/vbthi / 3
#             # if NM2 ≥ 3
#             #     GvLv2[1,3] = 0.0
#             #     dGvLv[1,3] = 0.0
#             # end
#         end
#     else
#         ubi2 = ubi^2
#         if vbthi == 1
#             dGvLv[1,1] = sqrtpi / 6 * nbi * erf(ubi) / ubi
#             if NM2 ≥ 3
#                 a = nbi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
#                 GvLv2[1,3] = a
#                 dGvLv[1,3] = 2 * GvLv2[1,3]
#             end
#         else
#             # L = 0
#             dGvLv[1,1] = sqrtpi / 6 * nbi/vbthi * erf(ubi) / ubi
#             # L = 1
#             # L = 2
#             if NM2 ≥ 3
#                 a = nbi/vbthi / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
#                 GvLv2[1,3] = a / vbthi
#                 dGvLv[1,3] = 2 * GvLv2[1,3]
#             end
#         end
#     end
#     # Computing `Sf1[1,:] += Sf3[1,:] + Sf6[1,:] + Sf8[1,:] + Sf10[1,:]`
#
#     # ############ 3,  Sf3
#     ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
#     # ############ 5,  Sf6
#     # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = (1 - 2vbthi) × GvLv2[:,2:end]
#     # GG1 = dX01 * Mun1
#     GG1 = (GvLv2[:,2:end] - dGvLv[:,2:end]) * Mun1
#     # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = (1 - 2vathi) × fvLv2[:,2:end]
#     # GG1 .*= 2(dX01 * Mun1)
#     GG1 .*= 2((fvLv2[:,2:end] - dfvLv[:,2:end]) * Mun1)
#     ############ (6,7), (Sf8, Sf10)
#     dX01 = ((dGvLv * Mun) + (GvLv2[:,2:end] * Mun1) .* cotmu)
#     GG7 = copy(dX01)
#     if NM2 == 0
#         # GG6 = GG7 = Sf8 = Sf10
#         dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
#         GG7 .*= dX01
#         GG1 += 2GG7
#     else
#         GG6 = (dX01 + (GvLv2[:,3:end] * Mun2))
#         dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
#         GG7 .*= dX01
#         GG6 .*= (dX01 + (fvLv2[:,3:end] * Mun2))
#         GG1 += (GG7 + GG6)
#     end
#     Sf1[1,:] += CG * GG1[1,:]
#     return Sf1
# end
