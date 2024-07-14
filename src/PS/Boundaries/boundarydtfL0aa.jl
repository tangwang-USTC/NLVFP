

"""
  When `v̂ = 0`: boundary conditions of seif-collision terms:

    Sf[:,3] = Sf3
    Sf[:,6] = Sf6
    Sf[:,8] = Sf8
    Sf[:,10] = Sf10

  The boundary conditions of the Fokker-Planck self-collison in `collisionsLSaa.jl`.

  Warning: coefficient `na / vth^3 / π^(3/2)` of `f̂ₗ(v̂)` is not included in following codes.

  Warning: transformation of the inverse spherical harmonics is not implemented in the following codes.

  Inputs:
    uai = uai[isp] / vth[isp]
    XvL: XvL[1:2,:], X ∈ [f,F,H,G,df,ddf,dH,dG,ddG]

  Outputs:
    Sf1 = dtfvLDMaav0(Sf1,mu,Mun,Mun1,Mun2,nai,vathi,uai,LM1,nMod)
    Sf1 = dtfvLDMaav0(Sf1,mu,Mun,Mun1,Mun2,uai,LM1)
    Sf1 = dtfvLDMaav0(Sf1,Mun,LM1)
"""
# mM = 1     →  CH = 0.0
# vabth = 1  →  CG = 1/2

#  [nMod,LM1]
function dtfvLDMaav0(Sf1::AbstractArray{T,N},mu::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    na::AbstractVector{T},vath::AbstractVector{T},ua::AbstractVector{T},
    LM1::Int64,nModa::Int64) where{T,N,NM1,NM2}

    cotmu = mu ./ (1 .-mu.^2).^0.5
    ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
    dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
    dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
    GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
    fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
    ka = 1
    nai = na[ka]
    vathi, uai = vath[ka], ua[ka]
    if uai == 0
        if vathi == 1
            dfvLv[1,1] = - nai * 2
            dGvLv[1,1] = nai / 3
            # if NM2 ≥ 3
            #     fvLv2[1,3] = 0.0
            #     dfvLv[1,3] = 0.0
            #     GvLv2[1,3] = 0.0
            #     dGvLv[1,3] = 0.0
            # end
        else
            dfvLv[1,1] = - nai/vathi^3/vathi * 2
            dGvLv[1,1] = nai/vathi / 3
            # if NM2 ≥ 3
            #     fvLv2[1,3] = 0.0
            #     dfvLv[1,3] = 0.0
            #     GvLv2[1,3] = 0.0
            #     dGvLv[1,3] = 0.0
            # end
        end
    else
        uai2 = uai^2
        if vathi == 1
            dfvLv[1,1] = - nai * 2(1.0 - 2/3 * uai2) * exp(-uai2)
            dGvLv[1,1] = sqrtpi / 6 * nai * erf(uai) / uai
            if NM2 ≥ 3
                a = 4/3 * nai * uai2 * exp(-uai2)
                fvLv2[1,3] = a
                dfvLv[1,3] = 2a
                a = nai / 4uai2 * (sqrtpi/2 * (1/uai - 2uai / 3) * erf(uai) - exp(-uai2))
                GvLv2[1,3] = a
                dGvLv[1,3] = 2a
            end
        else
            dfvLv[1,1] = - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
            dGvLv[1,1] = sqrtpi / 6 * nai/vathi * erf(uai) / uai
            if NM2 ≥ 3
                a = 4/3 * nai/vathi^3 / vathi * uai2 * exp(-uai2)
                fvLv2[1,3] = a / vathi
                dfvLv[1,3] = 2a
                a = nai/vathi / 4uai2 * (sqrtpi/2 * (1/uai - 2uai / 3) * erf(uai) - exp(-uai2))
                GvLv2[1,3] = a / vathi
                dGvLv[1,3] = 2a
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
                    dGvLv[1,1] += nai / 3
                else
                    dfvLv[1,1] += - nai/vathi^3/vathi * 2
                    dGvLv[1,1] += nai/vathi / 3
                end
            else
                uai2 = uai^2
                if vathi == 1
                    dfvLv[1,1] += - nai * 2(1.0 - 2/3 * uai2) * exp(-uai2)
                    dGvLv[1,1] += sqrtpi / 6 * nai * erf(uai) / uai
                    if NM2 ≥ 3
                        a = 4/3 * nai * uai2 * exp(-uai2)
                        fvLv2[1,3] += a
                        dfvLv[1,3] += 2a
                        a = nai / 4uai2 * (sqrtpi/2 * (1/uai - 2uai / 3) * erf(uai) - exp(-uai2))
                        GvLv2[1,3] += a
                        dGvLv[1,3] += 2a
                    end
                else
                    dfvLv[1,1] += - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
                    dGvLv[1,1] += sqrtpi / 6 * nai/vathi * erf(uai) / uai
                    if NM2 ≥ 3
                        a = 4/3 * nai/vathi^3 / vathi * uai2 * exp(-uai2)
                        fvLv2[1,3] += a / vathi
                        dfvLv[1,3] += 2a
                        a = nai/vathi / 4uai2 * (sqrtpi/2 * (1/uai - 2uai / 3) * erf(uai) - exp(-uai2))
                        GvLv2[1,3] += a / vathi
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
    Sf1[1,:] += GG1[1,:] / 2   # CG = 1 / 2
    return Sf1
end

"""
  Outputs:
    Sf1 = dtfvLDMaav0(Sf1,mu,Mun,Mun1,Mun2,uai,LM1)
    Sf1 = dtfvLDMaav0(Sf1,Mun,LM1)
"""

## `nMod == 1` means `nai = vthi = 1`
#  [LM1]
function dtfvLDMaav0(Sf1::AbstractArray{T,N},mu::AbstractArray{T,N},Mun::AbstractArray{T,N},
    Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},uai::T,LM1::Int64) where{T,N,NM1,NM2}

    # if uai == 0
    #     Sf1 = dtfvLDMaav0(Sf1,Mun,LM1)
    #     return Sf1
    # end
    cotmu = mu ./ (1 .-mu.^2).^0.5
    ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
    dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
    dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
    GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
    fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
    uai2 = uai^2
    dfvLv[1,1] = - 2(1.0 - 2/3 * uai2) * exp(-uai2)
    dGvLv[1,1] = sqrtpi / 6 * erf(uai) / uai
    if NM2 ≥ 3
        fvLv2[1,3] = 4/3 * uai2 * exp(-uai2)
        dfvLv[1,3] = 2fvLv2[1,3]
        GvLv2[1,3] = (sqrtpi/2 * (1/uai - 2uai / 3) * erf(uai) - exp(-uai2)) / 4uai2
        dGvLv[1,3] = 2GvLv2[1,3]
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
    GG1 .*= - 2(fvLv2[:,2:end] * Mun1)
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
    Sf1[1,:] += GG1[1,:] / 2   # CG = 1 / 2
    return Sf1
end


#  [LM1] and `uai == 0`
function dtfvLDMaav0(Sf1::AbstractArray{T,N},Mun::AbstractArray{T,N},LM1::Int64) where{T,N}

    ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
    dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
    dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
    dfvLv[1,1] = - 2.0
    dGvLv[1,1] = 1 / 3
    # Computing `Sf1[1,:] += Sf3[1,:] + Sf6[1,:] + Sf8[1,:] + Sf10[1,:]`

    # ############ 3,  Sf3
    ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    ############ (6,7), (Sf8, Sf10)
    # GG1 = 2(dGvLv * Mun) .* (dfvLv * Mun)
    # Sf1[1,:] += GG1[1,:] / 2   # CG = 1 / 2
    Sf1 += (dGvLv * Mun) .* (dfvLv * Mun)
    return Sf1
end
