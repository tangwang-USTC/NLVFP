

"""
  When `v̂ = 0`: boundary conditions of self-collison

  The boundary conditions of the Fokker-Planck self-collison in `collisionsLSaa.jl`.

  Warning: coefficient `na / vth^3 / π^(3/2)` of `f̂ₗ(v̂)` is not included in following codes.

  Warning: transformation of the inverse spherical harmonics is not implemented in the following codes.

  Inputs:
    uai = uai[isp] / vth[isp]
    XvL: XvL[1:2,:], X ∈ [f,F,H,G,df,ddf,dH,dG,ddG]

  Outputs:
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,mu,Mun,Mun1,Mun2,nai,vathi,uai,LM1)
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,Mun,nai,vathi,LM1)
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,mu,Mun,Mun1,Mun2,nai,uai,LM1)
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,Mun,nai,LM1)
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,mu,Mun,Mun1,Mun2,uai,LM1)
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,Mun,LM1)
"""
# mM = 1     →  CF = 1.0
#               CH = 0.0

# vabth = 1  →  CG = 1/2

# FvL = fvL
"""
  Outputs:
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,mu,Mun,Mun1,Mun2,nai,vathi,uai,LM1)
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,Mun,nai,vathi,LM1)
"""
function dtfvLDMaav0(ddfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},ddGvL::AbstractArray{T,N},
    mu::AbstractArray{T,N},Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},
    Mun2::AbstractArray{T,NM2},nai::T,vathi::T,uai::T,LM1::Int) where{T,N,NM1,NM2}

   cotmu = mu ./ (1 .-mu.^2).^0.5
   ############ 1, Sf1 = CF * f * F
   Sf1 = (fvL * Mun).^2
   ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
   dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
   dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
   uai2 = uai^2
   # L = 0
   dfvLv[1,1] = - nai/vathi^3/vathi * 2(1.0 - 2/3 * uai2) * exp(-uai2)
   dGvLv[1,1] = sqrtpi / 6 * nai/vathi * erf(uai) / uai
   # L = 1
   # dfvLv[1,2] = 0.0
   # dGvLv[1,2] = 0.0
   # L = 2
   GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
   fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
   if NM2 ≥ 3
       fvLv2[1,3] = 4/3 * nai/vathi^3 / vathi^2 * uai2 * exp(-uai2)
       GvLv2[1,3] = nai/vathi^2 / 4uai2 * (sqrtpi/2 * (1/uai - 2uai / 3) * erf(uai) - exp(-uai2))
       dfvLv[1,3] = 2vathi * fvLv2[1,3]
       dGvLv[1,3] = 2vathi * GvLv2[1,3]
   end
   # ############ 2,  Sf2
   # ############ 3,  Sf3
   # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
   ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
   # ############ 5,  Sf6
   # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = (1 - 2vbthi) × GvLv2[:,2:end]
   # GG1 = dX01 * Mun1
   GG1 = (1 - 2vathi) * GvLv2[:,2:end] * Mun1
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
   Sf1 += GG1 / 2
   return Sf1
end

# uai = 0
function dtfvLDMaav0(ddfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},Mun::AbstractArray{T,N},nai::T,vathi::T,LM1::Int) where{T,N}

   ############ 1, Sf1 = CF * f * F
   Sf1 = (fvL * Mun).^2
   ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
   dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
   dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
   # L = 0
   dfvLv[1,1] = - nai/vathi^3/vathi * 2
   dGvLv[1,1] = nai/vathi / 3
   # ############ 2,  Sf2
   # ############ 3,  Sf3
   # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
   ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
   # ############ 5,  Sf6
   ############ (6,7), (Sf8, Sf10)
   # GG6 = GG7 = Sf8 = Sf10
   # GG7 = (dGvLv * Mun) .* (dfvLv * Mun)
   Sf1 += (dGvLv * Mun) .* (dfvLv * Mun)
   ############ 4, Sf5
   Sf1 += (ddfvL * Mun) .* (ddGvL * Mun) / 2
   return Sf1
end

## vatbi = 1

"""
  Outputs:
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,mu,Mun,Mun1,Mun2,nai,uai,LM1)
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,Mun,nai,LM1)
"""

function dtfvLDMaav0(ddfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},ddGvL::AbstractArray{T,N},
    mu::AbstractArray{T,N},Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},
    Mun2::AbstractArray{T,NM2},nai::T,uai::T,LM1::Int) where{T,N,NM1,NM2}

   cotmu = mu ./ (1 .-mu.^2).^0.5
   ############ 1, Sf1 = CF * f * F
   Sf1 = (fvL * Mun).^2
   ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
   dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
   dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
   uai2 = uai^2
   # L = 0
   dfvLv[1,1] = - nai * 2(1.0 - 2/3 * uai2) * exp(-uai2)
   dGvLv[1,1] = sqrtpi / 6 * nai * erf(uai) / uai
   # L = 1
   # dfvLv[1,2] = 0.0
   # dGvLv[1,2] = 0.0
   # L = 2
   GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
   fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
   if NM2 ≥ 3
       fvLv2[1,3] = 4/3 * nai * uai2 * exp(-uai2)
       GvLv2[1,3] = nai / 4uai2 * (sqrtpi/2 * (1/uai - 2uai / 3) * erf(uai) - exp(-uai2))
       dfvLv[1,3] = 2 * fvLv2[1,3]
       dGvLv[1,3] = 2 * GvLv2[1,3]
   end
   # ############ 2,  Sf2
   # ############ 3,  Sf3
   # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
   ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
   # ############ 5,  Sf6
   # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = (1 - 2vbthi = -1) × GvLv2[:,2:end]
   # GG1 = dX01 * Mun1
   GG1 = - GvLv2[:,2:end] * Mun1
   # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = (1 - 2vathi = -1) × fvLv2[:,2:end]
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
   Sf1 += GG1 / 2
   return Sf1
end

# uai = 0
function dtfvLDMaav0(ddfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},Mun::AbstractArray{T,N},nai::T,LM1::Int) where{T,N}

   ############ 1, Sf1 = CF * f * F
   Sf1 = (fvL * Mun).^2
   ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
   dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
   dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
   # L = 0
   dfvLv[1,1] = - nai * 2
   dGvLv[1,1] = nai / 3
   # ############ 2,  Sf2
   # ############ 3,  Sf3
   # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
   ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
   # ############ 5,  Sf6
   ############ (6,7), (Sf8, Sf10)
   # GG6 = GG7 = Sf8 = Sf10
   # GG7 = (dGvLv * Mun) .* (dfvLv * Mun)
   Sf1 += (dGvLv * Mun) .* (dfvLv * Mun)
   ############ 4, Sf5
   Sf1 += (ddfvL * Mun) .* (ddGvL * Mun) / 2
   return Sf1
end

## nai = 1, vatbi = 1
"""
  Outputs:
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,mu,Mun,Mun1,Mun2,uai,LM1)
    Sf1 = dtfvLDMaav0(ddfvL,fvL,ddGvL,Mun,LM1)
"""

function dtfvLDMaav0(ddfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},ddGvL::AbstractArray{T,N},
    mu::AbstractArray{T,N},Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},
    Mun2::AbstractArray{T,NM2},uai::T,LM1::Int) where{T,N,NM1,NM2}

   cotmu = mu ./ (1 .-mu.^2).^0.5
   ############ 1, Sf1 = CF * f * F
   Sf1 = (fvL * Mun).^2
   ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
   dGvLv = zeros(T,2,LM1) # = dGvL[1:2,2:end] ./ va0[1:2]
   dfvLv = zeros(T,2,LM1) # = dfvL ./ vG0[1:2]
   uai2 = uai^2
   # L = 0
   dfvLv[1,1] = - 2(1.0 - 2/3 * uai2) * exp(-uai2)
   dGvLv[1,1] = sqrtpi / 6 * erf(uai) / uai
   # L = 1
   # dfvLv[1,2] = 0.0
   # dGvLv[1,2] = 0.0
   # L = 2
   GvLv2 = zeros(T,2,LM1) # = GvL[1:2,2:end] ./ va0[1:2]^2
   fvLv2 = zeros(T,2,LM1) # = fvL[1:2,2:end] ./ vG0[1:2]^2
   if NM2 ≥ 3
       fvLv2[1,3] = 4/3 * uai2 * exp(-uai2)
       GvLv2[1,3] = (sqrtpi/2 * (1/uai - 2uai / 3) * erf(uai) - exp(-uai2)) / 4uai2
       dfvLv[1,3] = 2 * fvLv2[1,3]
       dGvLv[1,3] = 2 * GvLv2[1,3]
   end
   # ############ 2,  Sf2
   # ############ 3,  Sf3
   # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`
   ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
   # ############ 5,  Sf6
   # dX01 = (GvLv2[:,2:end] - dGvLv[:,2:end])  # = (1 - 2vbthi = -1) × GvLv2[:,2:end]
   # GG1 = dX01 * Mun1
   GG1 = - GvLv2[:,2:end] * Mun1
   # dX01 = (fvLv2[:,2:end] - dfvLv[:,2:end])   # = (1 - 2vathi = -1) × fvLv2[:,2:end]
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
   Sf1 += GG1 / 2
   return Sf1
end

# uai = 0
function dtfvLDMaav0(ddfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},Mun::AbstractArray{T,N},LM1::Int) where{T,N}

   ############ 1, Sf1 = CF * f * F
   Sf1 = (fvL * Mun).^2
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
   Sf1 += (dGvLv * Mun) .* (dfvLv * Mun)
   ############ 4, Sf5
   Sf1 += (ddfvL * Mun) .* (ddGvL * Mun) / 2
   return Sf1
end
