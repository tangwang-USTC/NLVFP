
"""
  When `v̂ = 0`: boundary conditions of collision terms:

    Sf[:,1] = Sf1
    Sf[:,2] = Sf2
    Sf[:,3] = Sf3
    Sf[:,5] = Sf5
    Sf[:,6] = Sf6
    Sf[:,8] = Sf8
    Sf[:,10] = Sf10

  The boundary conditions of the Fokker-Planck collison in `collisionsLSab.jl`.

  Warning: coefficient `na / vth^3 / π^(3/2)` of `f̂ₗ(v̂)` is not included in following codes.

  Warning: transformation of the inverse spherical harmonics is not implemented in the following codes.

  Inputs:
    XvL: XvL[1:2,:], X ∈ [f,F,H,G,df,ddf,dH,dG,ddG]
    uabi: [uai, ubi]
      uai = uai[isp] / vth[isp]
      ubi = uai[iFv] / vth[iFv]

  Outputs:
    Sf1 = dtfvLDMabv0(ddfvL,dfvL,fvL,FvL,dHvL,ddGvL,mu,Mun,Mun1,Mun2,CF,CH,CG,nabi,vabthi,uabi,LM1)
    Sf1 = dtfvLDMabv0(ddfvL,dfvL,fvL,FvL,dHvL,ddGvL,Mun,CF,CH,CG,nabi,vabthi,LM1)
    Sf1 = dtfvLDMabv0(ddfvL,dfvL,fvL,FvL,dHvL,ddGvL,mu,Mun,Mun1,Mun2,CF,CH,CG,uabi,LM1)
    Sf1 = dtfvLDMabv0(ddfvL,dfvL,fvL,FvL,dHvL,ddGvL,Mun,CF,CH,CG,LM1)
    Sf1 = dtfvLDMabv0(ddfvL,fvL,FvL,ddGvL,mu,Mun,Mun1,Mun2,CG,nabi,vabthi,uabi,LM1)
    Sf1 = dtfvLDMabv0(ddfvL,fvL,FvL,ddGvL,Mun,CG,nabi,vabthi,LM1)
    Sf1 = dtfvLDMabv0(ddfvL,fvL,FvL,ddGvL,mu,Mun,Mun1,Mun2,CG,uabi,LM1)
    Sf1 = dtfvLDMabv0(ddfvL,fvL,FvL,ddGvL,Mun,CG,LM1)
"""

"""

  Outputs:
    Sf1 = dtfvLDMabv0(ddfvL,dfvL,fvL,FvL,dHvL,ddGvL,mu,Mun,Mun1,Mun2,CF,CH,CG,nabi,vabthi,uabi,LM1)
    Sf1 = dtfvLDMabv0(ddfvL,dfvL,fvL,FvL,dHvL,ddGvL,Mun,CF,CH,CG,nabi,vabthi,LM1)
"""

function dtfvLDMabv0(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},
    fvL::AbstractArray{T,N},FvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},mu::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    CF::T,CH::T,CG::T,nabi::AbstractVector{T},vabthi::AbstractVector{T},
    uabi::AbstractVector{T},LM1::Int) where{T,N,NM1,NM2}

    nai, nbi = nabi[1], nabi[2]
    vathi, vbthi = vabthi[1], vabthi[2]
    uai, ubi = uabi[1], uabi[2]
    cotmu = mu ./ (1 .-mu.^2).^0.5
    ############ 1, Sf1 = CF * f * F
    Sf1 = CF * (fvL * Mun) .* (FvL * Mun)
    ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
    dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
    dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
    if uai == 0
        if ubi == 0
            # L = 0
            dfvLv[1,1] = - nai/vathi^3/vathi * 2
            dGvLv[1,1] = nbi/vbthi / 3
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            if CH ≠ 0.0
                Sf1 += CH * (dfvL * Mun) .* (dHvL * Mun)
            end
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            # GG6 = GG7 = Sf8 = Sf10
            # GG7 = (dGvLv * Mun) .* (dfvLv * Mun)
            GG1 = 2(dGvLv * Mun) .* (dfvLv * Mun)
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        else
            # L = 0
            dfvLv[1,1] = - nai/vathi^3/vathi * 2
            dGvLv[1,1] = sqrtpi / 6 * nbi/vbthi * erf(ubi) / ubi
            # L = 1
            ubi2 = ubi^2
            # L = 2
            GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
            if NM2 ≥ 3
                GvLv2[1,3] = nbi/vbthi^2 / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                dGvLv[1,3] = 2vbthi * GvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            if CH ≠ 0.0
                GG1 = (dfvL * Mun) .* (dHvL * Mun)
                Sf1 += CH * GG1
            end
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            dX01 = ((dGvLv * Mun) + (GvLv2[:,2:end] * Mun1) .* cotmu)
            GG7 = copy(dX01)
            if NM2 == 0
                # GG6 = GG7 = Sf8 = Sf10
                dX01 = (dfvLv * Mun)
                GG7 .*= dX01
                GG1 = 2GG7
            else
                GG6 = (dX01 + (GvLv2[:,3:end] * Mun2))
                dX01 = (dfvLv * Mun)
                GG7 .*= dX01
                GG6 .*= dX01
                GG1 = (GG7 + GG6)
            end
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        end
    else
        if ubi == 0
            uai2 = uai^2
            # L = 0
            dfvLv[1,1] = - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
            dGvLv[1,1] = nbi/vbthi / 3
            # L = 1
            # L = 2
            fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
            if NM2 ≥ 3
                # fvLv[1,3] = 0.0
                fvLv2[1,3] = 4/3 * nai/vathi^3 / vathi^2 * uai2 * exp(-uai2)
                dfvLv[1,3] = 2vathi * fvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            if CH ≠ 0.0
                Sf1 += CH * (dfvL * Mun) .* (dHvL * Mun)
            end
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            GG7 = (dGvLv * Mun)
            if NM2 == 0
                # GG6 = GG7 = Sf8 = Sf10
                GG7 .*= ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
                GG1 = 2GG7
            else
                GG6 = (dGvLv * Mun)
                dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
                GG7 .*= dX01
                GG6 .*= (dX01 + (fvLv2[:,3:end] * Mun2))
                GG1 = (GG7 + GG6)
            end
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        else
            uai2 = uai^2
            # L = 0
            dfvLv[1,1] = - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
            dGvLv[1,1] = sqrtpi / 6 * nbi/vbthi * erf(ubi) / ubi
            # L = 1
            # dfvLv[1,2] = 0.0
            # dGvLv[1,2] = 0.0
            ubi2 = ubi^2
            if CH ≠ 0.0
                fvLv = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]
                HvLv = zeros(T,2,LM1) # = HvL[1:2,2:end] ./ va0[1:2]
                fvLv[1,2] = nai/vathi^3/vathi * 2uai * exp(-uai2)
                HvLv[1,2] = nbi/vbthi / 2ubi * (sqrtpi / 2 * erf(ubi) / ubi2 - exp(-ubi2))
            end
            # L = 2
            GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
            fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
            if NM2 ≥ 3
                # fvLv[1,3] = 0.0
                # HvLv[1,3] = 0.0
                fvLv2[1,3] = 4/3 * nai/vathi^3 / vathi^2 * uai2 * exp(-uai2)
                GvLv2[1,3] = nbi/vbthi^2 / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                dfvLv[1,3] = 2vathi * fvLv2[1,3]
                dGvLv[1,3] = 2vbthi * GvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            if CH ≠ 0.0
                GG1 = (dfvL * Mun) .* (dHvL * Mun)
                GG1 += (fvLv[:,2:end] * Mun1) .* (HvLv[:,2:end] * Mun1)
                Sf1 += CH * GG1
            end
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = (1 - 2vbthi) × GvLv2[:,2:end]
            # GG1 = dX01 * Mun1
            GG1 = (1 - 2vbthi) * GvLv2[:,2:end] * Mun1
            # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = (1 - 2vathi) × fvLv2[:,2:end]
            # GG1 .*= 2(dX01 * Mun1)
            GG1 .*= 2((1 - 2vathi) * fvLv2[:,2:end] * Mun1)
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
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        end
    end
end

# uai = ubi = 0
function dtfvLDMabv0(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},
    fvL::AbstractArray{T,N},FvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},Mun::AbstractArray{T,N},CF::T,CH::T,CG::T,
    nabi::AbstractVector{T},vabthi::AbstractVector{T},LM1::Int) where{T,N}

   nai, nbi = nabi[1], nabi[2]
   vathi, vbthi = vabthi[1], vabthi[2]
   ############ 1, Sf1 = CF * f * F
   Sf1 = CF * (fvL * Mun) .* (FvL * Mun)
   ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
   dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
   dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
   # L = 0
   dfvLv[1,1] = - nai/vathi^3/vathi * 2
   dGvLv[1,1] = nbi/vbthi / 3
   # ############ 2,  Sf2
   # ############ 3,  Sf3
   if CH ≠ 0.0
       Sf1 += CH * (dfvL * Mun) .* (dHvL * Mun)
   end
   # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
   ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
   # ############ 5,  Sf6
   ############ (6,7), (Sf8, Sf10)
   # GG6 = GG7 = Sf8 = Sf10
   # GG7 = (dGvLv * Mun) .* (dfvLv * Mun)
   GG1 = 2(dGvLv * Mun) .* (dfvLv * Mun)
   ############ 4, Sf5
   GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
   Sf1 += CG * GG1
   return Sf1
end

## nai = 1, vthi = 1

"""
  Outputs:
    Sf1 = dtfvLDMabv0(ddfvL,dfvL,fvL,FvL,dHvL,ddGvL,mu,Mun,Mun1,Mun2,CF,CH,CG,uabi,LM1)
    Sf1 = dtfvLDMabv0(ddfvL,dfvL,fvL,FvL,dHvL,ddGvL,Mun,CF,CH,CG,LM1)
"""

function dtfvLDMabv0(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},
    fvL::AbstractArray{T,N},FvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},mu::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    CF::T,CH::T,CG::T,uabi::AbstractVector{T},LM1::Int) where{T,N,NM1,NM2}

    uai, ubi = uabi[1], uabi[2]
    cotmu = mu ./ (1 .-mu.^2).^0.5
    ############ 1, Sf1 = CF * f * F
    Sf1 = CF * (fvL * Mun) .* (FvL * Mun)
    ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
    dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
    dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
    if uai == 0
        if ubi == 0
            # L = 0
            dfvLv[1,1] = - 2.0
            dGvLv[1,1] = 1 / 3
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            if CH ≠ 0.0
                Sf1 += CH * (dfvL * Mun) .* (dHvL * Mun)
            end
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            # GG6 = GG7 = Sf8 = Sf10
            # GG7 = (dGvLv * Mun) .* (dfvLv * Mun)
            GG1 = 2(dGvLv * Mun) .* (dfvLv * Mun)
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        else
            # L = 0
            dfvLv[1,1] = - 2.0
            dGvLv[1,1] = sqrtpi / 6 * erf(ubi) / ubi
            # L = 1
            ubi2 = ubi^2
            # L = 2
            GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
            if NM2 ≥ 3
                GvLv2[1,3] = 1/4 * ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                dGvLv[1,3] = 2 * GvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            if CH ≠ 0.0
                GG1 = (dfvL * Mun) .* (dHvL * Mun)
                Sf1 += CH * GG1
            end
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            dX01 = ((dGvLv * Mun) + (GvLv2[:,2:end] * Mun1) .* cotmu)
            GG7 = copy(dX01)
            if NM2 == 0
                # GG6 = GG7 = Sf8 = Sf10
                dX01 = (dfvLv * Mun)
                GG7 .*= dX01
                GG1 = 2GG7
            else
                GG6 = (dX01 + (GvLv2[:,3:end] * Mun2))
                dX01 = (dfvLv * Mun)
                GG7 .*= dX01
                GG6 .*= dX01
                GG1 = (GG7 + GG6)
            end
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        end
    else
        if ubi == 0
            uai2 = uai^2
            # L = 0
            dfvLv[1,1] = - 2(1.0 - 2/3 * uai2) * exp(-uai2)
            dGvLv[1,1] = 1 / 3
            # L = 1
            # L = 2
            fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
            if NM2 ≥ 3
                # fvLv[1,3] = 0.0
                fvLv2[1,3] = 4/3 * uai2 * exp(-uai2)
                dfvLv[1,3] = 2 * fvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            if CH ≠ 0.0
                Sf1 += CH * (dfvL * Mun) .* (dHvL * Mun)
            end
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            GG7 = (dGvLv * Mun)
            if NM2 == 0
                # GG6 = GG7 = Sf8 = Sf10
                GG7 .*= ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
                GG1 = 2GG7
            else
                GG6 = (dGvLv * Mun)
                dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
                GG7 .*= dX01
                GG6 .*= (dX01 + (fvLv2[:,3:end] * Mun2))
                GG1 = (GG7 + GG6)
            end
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        else
            uai2 = uai^2
            # L = 0
            dfvLv[1,1] = - 2(1.0 - 2/3 * uai2) * exp(-uai2)
            dGvLv[1,1] = sqrtpi / 6 * erf(ubi) / ubi
            # L = 1
            # dfvLv[1,2] = 0.0
            # dGvLv[1,2] = 0.0
            ubi2 = ubi^2
            if CH ≠ 0.0
                fvLv = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]
                HvLv = zeros(T,2,LM1) # = HvL[1:2,2:end] ./ va0[1:2]
                fvLv[1,2] = 2uai * exp(-uai2)
                HvLv[1,2] = 1 / 2 * ubi * (sqrtpi / 2 * erf(ubi) / ubi2 - exp(-ubi2))
            end
            # L = 2
            GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
            fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
            if NM2 ≥ 3
                # fvLv[1,3] = 0.0
                # HvLv[1,3] = 0.0
                fvLv2[1,3] = 4/3 * uai2 * exp(-uai2)
                GvLv2[1,3] = 1/4 * ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                dfvLv[1,3] = 2 * fvLv2[1,3]
                dGvLv[1,3] = 2 * GvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            if CH ≠ 0.0
                GG1 = (dfvL * Mun) .* (dHvL * Mun)
                GG1 += (fvLv[:,2:end] * Mun1) .* (HvLv[:,2:end] * Mun1)
                Sf1 += CH * GG1
            end
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = - GvLv2[:,2:end]
            # GG1 = dX01 * Mun1
            GG1 = - GvLv2[:,2:end] * Mun1
            # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = - fvLv2[:,2:end]
            # GG1 .*= 2(dX01 * Mun1)
            GG1 .*= 2(- fvLv2[:,2:end] * Mun1)
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
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        end
    end
end

# uai = ubi = 0
function dtfvLDMabv0(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},
    fvL::AbstractArray{T,N},FvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},Mun::AbstractArray{T,N},CF::T,CH::T,CG::T,LM1::Int) where{T,N}

   ############ 1, Sf1 = CF * f * F
   Sf1 = CF * (fvL * Mun) .* (FvL * Mun)
   ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
   dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
   dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
   # L = 0
   dfvLv[1,1] = - 2.0
   dGvLv[1,1] = 1 / 3
   # ############ 2,  Sf2
   # ############ 3,  Sf3
   if CH ≠ 0.0
       Sf1 += CH * (dfvL * Mun) .* (dHvL * Mun)
   end
   # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
   ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
   # ############ 5,  Sf6
   ############ (6,7), (Sf8, Sf10)
   # GG6 = GG7 = Sf8 = Sf10
   # GG7 = (dGvLv * Mun) .* (dfvLv * Mun)
   GG1 = 2(dGvLv * Mun) .* (dfvLv * Mun)
   ############ 4, Sf5
   GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
   Sf1 += CG * GG1
   return Sf1
end

##

# mM = 1 → CF = 1.0

#          CH = 0.0

##

"""
  Outputs:
    Sf1 = dtfvLDMabv0(ddfvL,fvL,FvL,ddGvL,mu,Mun,Mun1,Mun2,CG,nabi,vabthi,uabi,LM1)
    Sf1 = dtfvLDMabv0(ddfvL,fvL,FvL,ddGvL,Mun,CG,nabi,vabthi,LM1)
"""

function dtfvLDMabv0(ddfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},FvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},mu::AbstractArray{T,N},Mun::AbstractArray{T,N},
    Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},CG::T,nabi::AbstractVector{T},
    vabthi::AbstractVector{T},uabi::AbstractVector{T},LM1::Int) where{T,N,NM1,NM2}

    nai, nbi = nabi[1], nabi[2]
    vathi, vbthi = vabthi[1], vabthi[2]
    uai, ubi = uabi[1], uabi[2]
    cotmu = mu ./ (1 .-mu.^2).^0.5
    ############ 1, Sf1 = 1 * f * F
    Sf1 = (fvL * Mun) .* (FvL * Mun)
    ############ SH = S2 + S3 + (S4) = 0
    dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
    dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
    if uai == 0
        if ubi == 0
            # L = 0
            dfvLv[1,1] = - nai/vathi^3/vathi * 2
            dGvLv[1,1] = nbi/vbthi / 3
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            # GG6 = GG7 = Sf8 = Sf10
            # GG7 = (dGvLv * Mun) .* (dfvLv * Mun)
            GG1 = 2(dGvLv * Mun) .* (dfvLv * Mun)
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        else
            # L = 0
            dfvLv[1,1] = - nai/vathi^3/vathi * 2
            dGvLv[1,1] = sqrtpi / 6 * nbi/vbthi * erf(ubi) / ubi
            # L = 1
            ubi2 = ubi^2
            # L = 2
            GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
            if NM2 ≥ 3
                GvLv2[1,3] = nbi/vbthi^2 / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                dGvLv[1,3] = 2vbthi * GvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            dX01 = ((dGvLv * Mun) + (GvLv2[:,2:end] * Mun1) .* cotmu)
            GG7 = copy(dX01)
            if NM2 == 0
                # GG6 = GG7 = Sf8 = Sf10
                dX01 = (dfvLv * Mun)
                GG7 .*= dX01
                GG1 = 2GG7
            else
                GG6 = (dX01 + (GvLv2[:,3:end] * Mun2))
                dX01 = (dfvLv * Mun)
                GG7 .*= dX01
                GG6 .*= dX01
                GG1 = (GG7 + GG6)
            end
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        end
    else
        if ubi == 0
            uai2 = uai^2
            # L = 0
            dfvLv[1,1] = - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
            dGvLv[1,1] = nbi/vbthi / 3
            # L = 1
            # L = 2
            fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
            if NM2 ≥ 3
                # fvLv[1,3] = 0.0
                fvLv2[1,3] = 4/3 * nai/vathi^3 / vathi^2 * uai2 * exp(-uai2)
                dfvLv[1,3] = 2vathi * fvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            GG7 = (dGvLv * Mun)
            if NM2 == 0
                # GG6 = GG7 = Sf8 = Sf10
                GG7 .*= ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
                GG1 = 2GG7
            else
                GG6 = (dGvLv * Mun)
                dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
                GG7 .*= dX01
                GG6 .*= (dX01 + (fvLv2[:,3:end] * Mun2))
                GG1 = (GG7 + GG6)
            end
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        else
            uai2 = uai^2
            # L = 0
            dfvLv[1,1] = - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
            dGvLv[1,1] = sqrtpi / 6 * nbi/vbthi * erf(ubi) / ubi
            # L = 1
            # dfvLv[1,2] = 0.0
            # dGvLv[1,2] = 0.0
            ubi2 = ubi^2
            # L = 2
            GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
            fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
            if NM2 ≥ 3
                # fvLv[1,3] = 0.0
                # HvLv[1,3] = 0.0
                fvLv2[1,3] = 4/3 * nai/vathi^3 / vathi^2 * uai2 * exp(-uai2)
                GvLv2[1,3] = nbi/vbthi^2 / 4ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                dfvLv[1,3] = 2vathi * fvLv2[1,3]
                dGvLv[1,3] = 2vbthi * GvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = (1 - 2vbthi) × GvLv2[:,2:end]
            # GG1 = dX01 * Mun1
            GG1 = (1 - 2vbthi) * GvLv2[:,2:end] * Mun1
            # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = (1 - 2vathi) × fvLv2[:,2:end]
            # GG1 .*= 2(dX01 * Mun1)
            GG1 .*= 2((1 - 2vathi) * fvLv2[:,2:end] * Mun1)
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
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        end
    end
end

# uai = ubi = 0
function dtfvLDMabv0(ddfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},
    FvL::AbstractArray{T,N},ddGvL::AbstractArray{T,N},Mun::AbstractArray{T,N},
    CG::T,nabi::AbstractVector{T},vabthi::AbstractVector{T},LM1::Int) where{T,N}

   nai, nbi = nabi[1], nabi[2]
   vathi, vbthi = vabthi[1], vabthi[2]
   ############ 1, Sf1 = CF * f * F
   Sf1 = (fvL * Mun) .* (FvL * Mun)
   ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
   dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
   dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
   # L = 0
   dfvLv[1,1] = - nai/vathi^3/vathi * 2
   dGvLv[1,1] = nbi/vbthi / 3
   # ############ 2,  Sf2
   # ############ 3,  Sf3
   # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
   ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
   # ############ 5,  Sf6
   ############ (6,7), (Sf8, Sf10)
   # GG6 = GG7 = Sf8 = Sf10
   # GG7 = (dGvLv * Mun) .* (dfvLv * Mun)
   GG1 = 2(dGvLv * Mun) .* (dfvLv * Mun)
   ############ 4, Sf5
   GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
   Sf1 += CG * GG1
   return Sf1
end

## nai = 1, vthi = 1

"""
  Outputs:
    Sf1 = dtfvLDMabv0(ddfvL,fvL,FvL,ddGvL,mu,Mun,Mun1,Mun2,CG,uabi,LM1)
    Sf1 = dtfvLDMabv0(ddfvL,fvL,FvL,ddGvL,Mun,CG,LM1)
"""

function dtfvLDMabv0(ddfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},FvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},mu::AbstractArray{T,N},Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},
    Mun2::AbstractArray{T,NM2},CG::T,uabi::AbstractVector{T},LM1::Int) where{T,N,NM1,NM2}

    uai, ubi = uabi[1], uabi[2]
    cotmu = mu ./ (1 .-mu.^2).^0.5
    ############ 1, Sf1 = 1 * f * F
    Sf1 = (fvL * Mun) .* (FvL * Mun)
    ############ SH = S2 + S3 + (S4) = 0
    dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
    dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
    if uai == 0
        if ubi == 0
            # L = 0
            dfvLv[1,1] = - 2.0
            dGvLv[1,1] = 1 / 3
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            # GG6 = GG7 = Sf8 = Sf10
            # GG7 = (dGvLv * Mun) .* (dfvLv * Mun)
            GG1 = 2(dGvLv * Mun) .* (dfvLv * Mun)
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        else
            # L = 0
            dfvLv[1,1] = - 2.0
            dGvLv[1,1] = sqrtpi / 6 * erf(ubi) / ubi
            # L = 1
            ubi2 = ubi^2
            # L = 2
            GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
            if NM2 ≥ 3
                GvLv2[1,3] = 1/4 * ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                dGvLv[1,3] = 2 * GvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            dX01 = ((dGvLv * Mun) + (GvLv2[:,2:end] * Mun1) .* cotmu)
            GG7 = copy(dX01)
            if NM2 == 0
                # GG6 = GG7 = Sf8 = Sf10
                dX01 = (dfvLv * Mun)
                GG7 .*= dX01
                GG1 = 2GG7
            else
                GG6 = (dX01 + (GvLv2[:,3:end] * Mun2))
                dX01 = (dfvLv * Mun)
                GG7 .*= dX01
                GG6 .*= dX01
                GG1 = (GG7 + GG6)
            end
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        end
    else
        if ubi == 0
            uai2 = uai^2
            # L = 0
            dfvLv[1,1] = - 2(1.0 - 2/3 * uai2) * exp(-uai2)
            dGvLv[1,1] = 1 / 3
            # L = 1
            # L = 2
            fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
            if NM2 ≥ 3
                # fvLv[1,3] = 0.0
                fvLv2[1,3] = 4/3 * uai2 * exp(-uai2)
                dfvLv[1,3] = 2 * fvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            ############ (6,7), (Sf8, Sf10)
            GG7 = (dGvLv * Mun)
            if NM2 == 0
                # GG6 = GG7 = Sf8 = Sf10
                GG7 .*= ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
                GG1 = 2GG7
            else
                GG6 = (dGvLv * Mun)
                dX01 = ((dfvLv * Mun) + (fvLv2[:,2:end] * Mun1) .* cotmu)
                GG7 .*= dX01
                GG6 .*= (dX01 + (fvLv2[:,3:end] * Mun2))
                GG1 = (GG7 + GG6)
            end
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        else
            uai2 = uai^2
            # L = 0
            dfvLv[1,1] = - 2(1.0 - 2/3 * uai2) * exp(-uai2)
            dGvLv[1,1] = sqrtpi / 6 * erf(ubi) / ubi
            # L = 1
            # dfvLv[1,2] = 0.0
            # dGvLv[1,2] = 0.0
            ubi2 = ubi^2
            # L = 2
            GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
            fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
            if NM2 ≥ 3
                # fvLv[1,3] = 0.0
                # HvLv[1,3] = 0.0
                fvLv2[1,3] = 4/3 * uai2 * exp(-uai2)
                GvLv2[1,3] = 1/4 * ubi2 * (sqrtpi/2 * (1/ubi - 2ubi / 3) * erf(ubi) - exp(-ubi2))
                dfvLv[1,3] = 2 * fvLv2[1,3]
                dGvLv[1,3] = 2 * GvLv2[1,3]
            end
            # ############ 2,  Sf2
            # ############ 3,  Sf3
            # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
            ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
            # ############ 5,  Sf6
            # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = - GvLv2[:,2:end]
            # GG1 = dX01 * Mun1
            GG1 = - GvLv2[:,2:end] * Mun1
            # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = - fvLv2[:,2:end]
            # GG1 .*= 2(dX01 * Mun1)
            GG1 .*= 2(- fvLv2[:,2:end] * Mun1)
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
            ############ 4, Sf5
            GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
            Sf1 += CG * GG1
            return Sf1
        end
    end
end

# uai = ubi = 0
function dtfvLDMabv0(ddfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},FvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},Mun::AbstractArray{T,N},CG::T,LM1::Int) where{T,N}

   ############ 1, Sf1 = CF * f * F
   Sf1 = (fvL * Mun) .* (FvL * Mun)
   ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
   dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
   dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
   # L = 0
   dfvLv[1,1] = - 2.0
   dGvLv[1,1] = 1 / 3
   # ############ 2,  Sf2
   # ############ 3,  Sf3
   # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
   ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
   # ############ 5,  Sf6
   ############ (6,7), (Sf8, Sf10)
   # GG6 = GG7 = Sf8 = Sf10
   # GG7 = (dGvLv * Mun) .* (dfvLv * Mun)
   GG1 = 2(dGvLv * Mun) .* (dfvLv * Mun)
   ############ 4, Sf5
   GG1 += (ddfvL * Mun) .* (ddGvL * Mun)
   Sf1 += CG * GG1
   return Sf1
end
