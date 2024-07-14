"""
    Calculate Rosenbluth potentials `Ĥ(𝓋̂ )` and `Ĝ(𝓋̂ )` and their first two derivatives
  according to the definitions of Shkarofsky integrals
  which are calculated by Clenshaw-Curtis quadrature method.

  Ĥₗₘ(𝓋̂ ) = (ILFLm + JL1FLm) / 𝓋̂) /(2L+1)

  Ĝₗₘ(𝓋̂ ) = Igₗₘ * 𝓋̂ / (2L+1)
     Igₗₘ = (IL2FLm + JL1FLm) / (2L+3) - (ILFLm + Jn1FLm) / (2L-1), L ≥ 1;

  Ĝₗₘ(𝓋̂ ) = (Igₗₘ * 𝓋̂ + JL0FLm) / (2L+1)
     Igₗₘ = (IL2FLm + JL1FLm) / (2L+3) - ILFLm / (2L-1), L = 0;

  where

    v̂ᵦₜₕ = vᵦₜₕ / vₜₕ,

  `vₜₕ` is the effective thermal velocity of `f(𝐯)`

  and the angular coefficient of spherical coodinate, `4π`, is not included here.

  Inputs:
    LM:
    FL0: The `Lᵗʰ`-order coefficients of normalized distribution functions,
         = f̂L0(v̂,ℓ) =n0/π^1.5 * (2ℓ+1)//2 * Kₗ
    vabth = vₐₜₕ/vᵦₜₕ
    vth: [vₐₜₕ, vᵦₜₕ], the effective thermal velocity of spices `a` and `β`.

  Outputs:
  ddHvL,dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(ddHvL,dHvL,HvL,ddGvL,dGvL,GvL,
                               FvL,vG,nvlevel0,nc0,nck,ocp,vth,LM,ns)

"""

# 3D
function HGshkarofsky(ddHvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},
    HvL::AbstractArray{T,N},ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},
    GvL::AbstractArray{T,N},FvL::AbstractArray{T,N},vG::AbstractVector{Tb},
    nvlevel0::AbstractVector{Int64},nc0::Int64,nck::Int64,ocp::Int64,
    vth::Vector{Float64},LM::Vector{Int64},ns::Int64) where{T<:Real,Tb, N}

    nsp_vec = 1:ns
    FLn = FvL[:,1,1]
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec .≠ isp]
        iFv = nspF[1]
        vabth = vth[isp] / vth[iFv]
        va = vG * vabth
        for L1 = 1:LM[iFv]+1
            FLn = FvL[:,L1,isp]
            ddHvL[:,L1,isp], dHvL[:,L1,isp], HvL[:,L1,isp], ddGvL[:,L1,isp],
                   dGvL[:,L1,isp],GvL[:,L1,isp] = HGshkarofsky(ddHvL[:,L1,isp],
                   dHvL[:,L1,isp], HvL[:,L1,isp], ddGvL[:,L1,isp],dGvL[:,L1,isp],
                   GvL[:,L1,isp],FLn,va,nvlevel0,nc0,nck,ocp,L1)
        end
    end
    return ddHvL,dHvL,HvL,ddGvL,dGvL,GvL
end

"""
  limit(ILFLm,{v,0}) ∝ v^(L + 3)
  limit(IL2FLm,{v,0}) ∝ v^(L + 3)
  limit(JL1FLm,{v,0}) ∝ v^(L + 1)
  limit(JLn1FLm,{v,0}) ∝ v^(L - 1)
  limit(JL0FLm,{v,0}) ∝ v^(L)

  limit(ILFLm,{v,∞}) → √π / 4 * δₗ⁰

  Inputs:
    FvL: The `Lᵗʰ`-order coefficients of normalized distribution functions,
         = F̂ₗ(𝓋̂) = FLn(𝓋̂,ℓ)
    va: denotes `𝓋̂ = vG * vabth`

  Outputs:
            dHvL,HvL, ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                                   FvL,va,nvlevel0,nc0,nck,ocp,LM)
      ddHvL,dHvL,HvL, ddGvL,dGvL,GvL = HGshkarofsky(ddHvL,dHvL,HvL,ddGvL,dGvL,
                                   GvL,FvL,va,nvlevel0,nc0,nck,ocp,LM)

"""

# # 2D, ddGvL,dGvL,GvL
function HGshkarofsky(ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    FvL::AbstractArray{T,N},va::AbstractVector{Tb},nvlevel0::AbstractVector{Int64},
    nc0::Int64,nck::Int64,ocp::Int64,LM::Int64) where{T<:Real,Tb,N}

    for L1 in 1:LM+1
        ddGvL[:,L1],dGvL[:,L1],GvL[:,L1] = HGshkarofsky(ddGvL[:,L1],
               dGvL[:,L1],GvL[:,L1],FvL[:,L1],va,nvlevel0,nc0,nck,ocp,L1)
    end
    return ddGvL,dGvL,GvL
end

# # 2D, dHvL,HvL, ddGvL,dGvL,GvL
function HGshkarofsky(dHvL::AbstractArray{T,N},HvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    FvL::AbstractArray{T,N},va::AbstractVector{Tb},nvlevel0::AbstractVector{Int64},
    nc0::Int64,nck::Int64,ocp::Int64,LM::Int64) where{T<:Real,Tb,N}

    for L1 in 1:LM+1
        dHvL[:,L1],HvL[:,L1],ddGvL[:,L1],dGvL[:,L1],GvL[:,L1] = HGshkarofsky(dHvL[:,L1],
               HvL[:,L1],ddGvL[:,L1],dGvL[:,L1],GvL[:,L1],FvL[:,L1],
               va,nvlevel0,nc0,nck,ocp,L1)
    end
    return dHvL,HvL, ddGvL,dGvL,GvL
end

# 2D, ddHvL, dHvL,HvL, ddGvL,dGvL,GvL
function HGshkarofsky(ddHvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},HvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    FvL::AbstractArray{T,N},va::AbstractVector{Tb},nvlevel0::AbstractVector{Int64},
    nc0::Int64,nck::Int64,ocp::Int64,LM::Int64) where{T<:Real,Tb,N}

    for L1 in 1:LM+1
        ddHvL[:,L1], dHvL[:,L1], HvL[:,L1], ddGvL[:,L1],dGvL[:,L1],
               GvL[:,L1] = HGshkarofsky(ddHvL[:,L1],dHvL[:,L1],HvL[:,L1],ddGvL[:,L1],
               dGvL[:,L1],GvL[:,L1],FvL[:,L1],va,nvlevel0,nc0,nck,ocp,L1)
    end
    return ddHvL,dHvL,HvL,ddGvL,dGvL,GvL
end

"""
  Inputs:
    FvL: The `Lᵗʰ`-order coefficients of normalized distribution functions,
         = F̂ₗ(𝓋̂) = FLn(𝓋̂,ℓ)
    va: denotes `𝓋̂ = vG * vabth`

  Outputs:
    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL,va,nvlevel0,
                                          nc0,nck,ocp,LM,FvLa,vaa,nvlevel0a,ncF)

"""

# 2D, vaa
function HGshkarofsky(dHvL::AbstractArray{T,N},HvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    FvL::AbstractArray{T,N},va::AbstractVector{Tb},nvlevel0::AbstractVector{Int64},
    nc0::Int64,nck::Int64,ocp::Int64,LM::Int64,FvLa::AbstractVector,vaa::AbstractVector{T},
    nvlevel0a::AbstractVector{Int64},ncF::Int64) where{T<:Real,Tb,N}

    nva = ncF + nc0 - 1
    # nvlevela = (ncF - 1) * ocp
    nvak = nck + ((ncF - 1) * ocp - ncF + 2) - 1
    dHLn,HLn = zeros(T,nva),zeros(T,nva)
    ddGLn,dGLn,GLn = zeros(T,nva),zeros(T,nva),zeros(T,nva)
    for L1 in 1:LM+1
        dHLn,HLn,ddGLn,dGLn,GLn = HGshkarofsky(dHLn,HLn,ddGLn,dGLn,GLn,
                    [FvL[:,L1];FvLa[L1][2:end]],[va;vaa[2:end]],
                    [nvlevel0;nvlevel0a[2:end].+(nc0-1)],nva,nvak,ocp,L1)
        dHvL[:,L1],HvL[:,L1] = dHLn[1:nc0],HLn[1:nc0]
        ddGvL[:,L1],dGvL[:,L1],GvL[:,L1] = ddGLn[1:nc0],dGLn[1:nc0],GLn[1:nc0]
    end
    return dHvL,HvL,ddGvL,dGvL,GvL
end

# # 2D, vaa,ddHvL

"""
  The boundary conditions in theory:

    HLn(v=0) = c₀ δₗ⁰
    ∂ᵥHLn(v=0) = c₁ δₗ¹
    ∂ᵥ∂ᵥHLn(v=0) = c₀₂ δₗ⁰ +  c₂ δₗ²

    HLn(v → ∞) = 0, ∀ L, but very slowly and similar to the derivatives..
    ∂ᵥHLn(v → ∞) = 0, ∀ L
    ∂ᵥ∂ᵥHLn(v → ∞) = 0, ∀ L

    GLn(v=0) = C₀ δₗ⁰
    ∂ᵥGLn(v=0) = C₁ δₗ¹
    ∂ᵥ∂ᵥGLn(v=0) = C₀₂ δₗ⁰ +  C₂ δₗ²

    GLn(v → ∞)  = CC₀₀ 𝓋̂ δₗ⁰ +  CC₁₀ δₗ¹, but very slowly and similar to the derivatives..
    ∂ᵥGLn(v → ∞) = CC₀₁ δₗ⁰
    ∂ᵥ∂ᵥGLn(v → ∞) = 0, ∀ L

  limit(ILFLm,{v,∞}) → √π / 4 * δₗ⁰

  Inputs:
    FL0: The `Lᵗʰ`-order coefficients of normalized distribution functions,
         = F̂ₗ(𝓋̂) = FLn(𝓋̂,ℓ)
    va: denotes `𝓋̂ = vG * vabth`

  Outputs:
      ddHLn,dHLn,HLn, ddGLn,dGLn,GLn = HGshkarofsky(ddHLn,dHLn,HLn,ddGLn,dGLn,GLn,
                                          FLn,va,nvlevel0,nc0,nck,ocp,L1)
      dHLn,HLn, ddGLn,dGLn,GLn = HGshkarofsky(dHLn,HLn,ddGLn,dGLn,GLn,
                                          FLn,va,nvlevel0,nc0,nck,ocp,L1)

"""

# 1D, ddHLn,dHLn,HLn, ddGLn,dGLn,GLn
function HGshkarofsky(ddHLn::AbstractVector{T},dHLn::AbstractVector{T},HLn::AbstractVector{T},
    ddGLn::AbstractVector{T},dGLn::AbstractVector{T},GLn::AbstractVector{T},
    FLn::AbstractVector{T},va::AbstractVector{Tb},nvlevel0::AbstractVector{Int64},
    nc0::Int64,nck::Int64,ocp::Int64,L1::Int64) where{T<:Real,Tb}

    va0 = va[nvlevel0]
    if L1 == 1
        ILn1FL0,IL1FL0,JLFL0,JL0FL0 = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILn1FL0,IL1FL0 = shkarofskyIL0(ILn1FL0,IL1FL0,FLn,va,nvlevel0,nc0,ocp)
        JLFL0,JL0FL0 = shkarofskyJL0(JLFL0,JL0FL0,FLn,va,nc0,nck,ocp)
        HLn[:] = ILn1FL0 + JLFL0
        dHLn[:] = - ILn1FL0 ./ va0
        ddHLn[:] = - FLn[nvlevel0] + 2ILn1FL0 ./ va0.^2

        GLn[:] = ((IL1FL0 + JLFL0) / 3 + ILn1FL0) .* va0.^2 + JL0FL0
        dGLn[:] = (ILn1FL0 + (2JLFL0 - IL1FL0) / 3.0) .* va0
        ddGLn[:] = (IL1FL0 + JLFL0) * (2 / 3)
        if va0[1] == 0.0
            dHLn[1] = 0.0
            @warn("The value of `ddHLn(va[1] = 0.0)` when `L = 0` is a undermined constant. Please giving it according to `RddHL0 = 0`!")
        end
    elseif L1 == 2
        ILn2FL0,IL2FL0,JL1n2FL0,JLn1FL0 = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILn2FL0,IL2FL0 = shkarofskyIL1(ILn2FL0,IL2FL0,FLn,va,nvlevel0,nc0,ocp)
        JL1n2FL0,JLn1FL0 = shkarofskyJL1(JL1n2FL0,JLn1FL0,FLn,va,nc0,nck,ocp)
        HLn = (ILn2FL0 + JL1n2FL0) .* va0 / 3.0
        dHLn[:] = - 2.0 / 3.0 * ILn2FL0 + JL1n2FL0 / 3.0
        ddHLn[:] = - FLn[nvlevel0] + 2.0 * ILn2FL0 ./ va0

        GLn = ((IL2FL0 / 5.0 - JLn1FL0) + va0.^2 .* (JL1n2FL0 / 5.0 - ILn2FL0)) .* va0 / 3.0
        dGLn[:] = - JLn1FL0 / 3.0 - 2.0 / 15.0 * IL2FL0 + va0.^2 .* JL1n2FL0 / 5.0
        ddGLn[:] = 2 / 5 * (IL2FL0 ./ va0 + va0 .* JL1n2FL0)
        if va0[1] == 0.0
            HLn[1], ddHLn[1], ddGLn[1] = 0.0, 0.0, 0.0
        end
    elseif L1 == 3
        ILn1FL0,IL1FL0,JLFL0,JLn2FL0 = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILn1FL0,IL1FL0 = shkarofskyIL2(ILn1FL0,IL1FL0,FLn,va,nvlevel0,nc0,ocp)
        JLFL0,JLn2FL0 = shkarofskyJL2(JLFL0,JLn2FL0,FLn,va,nvlevel0,nc0,nck,ocp)
        HLn = (ILn1FL0 + JLFL0) / 5.0
        dHLn[:] = (- 3 / 5 * ILn1FL0 + 2 / 5 * JLFL0) ./ va0
        ddHLn[:] = - FLn[nvlevel0] + (12 / 5 * ILn1FL0 + 2 / 5 * JLFL0) ./ va0.^2

        GLn = ((IL1FL0 + JLFL0) / 35.0 - (ILn1FL0 + JLn2FL0) / 15.0) .* va0.^2
        dGLn[:] = ILn1FL0 / 15 - 2 / 15 * JLn2FL0 - 3 / 35 * IL1FL0 + 4 / 35 * JLFL0
        dGLn[:] .*= va0
        ddGLn[:] = 12 / 35 * (IL1FL0 + JLFL0) - 2 / 15 * (ILn1FL0 + JLn2FL0)
        if va0[1] == 0.0
            dHLn[1],ddHLn[1] = 0.0, 0.0
        end
    else
        ILFLm,IL2FLm,JL1FLm,JLn1FLm = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILFLm,IL2FLm = shkarofskyI(ILFLm,IL2FLm,FLn,va,nvlevel0,nc0,ocp,L1)
        JL1FLm,JLn1FLm = shkarofskyJ(JL1FLm,JLn1FLm,FLn,va,nvlevel0,nc0,nck,ocp,L1)
        L = L1 - 1
        L2 = 2L
        L2p1 = L2 + 1
        L2n1 = L2 - 1
        L2p3 = L2 + 3
        HLn = (ILFLm + JL1FLm) ./ va0 / L2p1
        dHLn[:] = (- L1 / L2p1 * ILFLm + L / L2p1 * JL1FLm) ./ va0.^2
        ddHLn[:] = - FLn[nvlevel0] + (L1 * (L1+1) / L2p1 * ILFLm + L * (L-1) / L2p1 * JL1FLm) ./ va0.^3

        GLn = ((IL2FLm + JL1FLm) / (L2p1 * L2p3) - (ILFLm + JLn1FLm) / (L2p1 * L2n1)) .* va0
        dGLn[:] = (L - 1) / (L2p1 * L2n1) * ILFLm - L / (L2p1 * L2n1) * JLn1FLm
        dGLn += - L1 / (L2p1 * L2p3) * IL2FLm + (L1 + 1) / (L2p1 * L2p3) * JL1FLm
        ddGLn[:] = - L * (L - 1) / (L2p1 * L2n1) * (ILFLm + JLn1FLm)
        ddGLn += L1 * (L1 + 1) / (L2p1 * L2p3) * (IL2FLm + JL1FLm)
        ddGLn ./= va0
        if va0[1] == 0.0
            HLn[1], dHLn[1], ddHLn[1], ddGLn[1] = 0.0, 0.0, 0.0, 0.0
        end
    end
    return ddHLn,dHLn,HLn, ddGLn,dGLn,GLn
end

# 1D, dHLn,HLn, ddGLn,dGLn,GLn
function HGshkarofsky(dHLn::AbstractVector{T},HLn::AbstractVector{T},
    ddGLn::AbstractVector{T},dGLn::AbstractVector{T},GLn::AbstractVector{T},
    FLn::AbstractVector{T},va::AbstractVector{Tb},nvlevel0::AbstractVector{Int64},
    nc0::Int64,nck::Int64,ocp::Int64,L1::Int64) where{T<:Real,Tb}

    va0 = va[nvlevel0]
    if L1 == 1
        ILn1FL0,IL1FL0,JLFL0,JL0FL0 = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILn1FL0,IL1FL0 = shkarofskyIL0(ILn1FL0,IL1FL0,FLn,va,nvlevel0,nc0,ocp)
        JLFL0,JL0FL0 = shkarofskyJL0(JLFL0,JL0FL0,FLn,va,nc0,nck,ocp)
        HLn[:] = ILn1FL0 + JLFL0
        dHLn[:] = - ILn1FL0 ./ va0

        GLn[:] = ((IL1FL0 + JLFL0) / 3 + ILn1FL0) .* va0.^2 + JL0FL0
        dGLn[:] = (ILn1FL0 + (2JLFL0 - IL1FL0) / 3.0) .* va0
        ddGLn[:] = (IL1FL0 + JLFL0) * (2 / 3)
        if va0[1] == 0.0
            dHLn[1] = 0.0
        end
    elseif L1 == 2
        ILn2FL0,IL2FL0,JL1n2FL0,JLn1FL0 = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILn2FL0,IL2FL0 = shkarofskyIL1(ILn2FL0,IL2FL0,FLn,va,nvlevel0,nc0,ocp)
        JL1n2FL0,JLn1FL0 = shkarofskyJL1(JL1n2FL0,JLn1FL0,FLn,va,nc0,nck,ocp)
        HLn = (ILn2FL0 + JL1n2FL0) .* va0 / 3.0
        dHLn[:] = - 2.0 / 3.0 * ILn2FL0 + JL1n2FL0 / 3.0

        GLn = ((IL2FL0 / 5.0 - JLn1FL0) + va0.^2 .* (JL1n2FL0 / 5.0 - ILn2FL0)) .* va0 / 3.0
        dGLn[:] = - JLn1FL0 / 3.0 - 2.0 / 15.0 * IL2FL0 + va0.^2 .* JL1n2FL0 / 5.0
        ddGLn[:] = 2 / 5 * (IL2FL0 ./ va0 + va0 .* JL1n2FL0)
        if va0[1] == 0.0
            HLn[1], ddGLn[1] = 0.0, 0.0
        end
    elseif L1 == 3
        ILn1FL0,IL1FL0,JLFL0,JLn2FL0 = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILn1FL0,IL1FL0 = shkarofskyIL2(ILn1FL0,IL1FL0,FLn,va,nvlevel0,nc0,ocp)
        JLFL0,JLn2FL0 = shkarofskyJL2(JLFL0,JLn2FL0,FLn,va,nvlevel0,nc0,nck,ocp)
        HLn = (ILn1FL0 + JLFL0) / 5.0
        dHLn[:] = (- 3 / 5 * ILn1FL0 + 2 / 5 * JLFL0) ./ va0

        GLn = ((IL1FL0 + JLFL0) / 35.0 - (ILn1FL0 + JLn2FL0) / 15.0) .* va0.^2
        dGLn[:] = ILn1FL0 / 15 - 2 / 15 * JLn2FL0 - 3 / 35 * IL1FL0 + 4 / 35 * JLFL0
        dGLn[:] .*= va0
        ddGLn[:] = 12 / 35 * (IL1FL0 + JLFL0) - 2 / 15 * (ILn1FL0 + JLn2FL0)
        if va0[1] == 0.0
            dHLn[1] = 0.0
        end
    else
        ILFLm,IL2FLm,JL1FLm,JLn1FLm = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILFLm,IL2FLm = shkarofskyI(ILFLm,IL2FLm,FLn,va,nvlevel0,nc0,ocp,L1)
        JL1FLm,JLn1FLm = shkarofskyJ(JL1FLm,JLn1FLm,FLn,va,nvlevel0,nc0,nck,ocp,L1)
        L = L1 - 1
        L2 = 2L
        L2p1 = L2 + 1
        L2n1 = L2 - 1
        L2p3 = L2 + 3
        HLn = (ILFLm + JL1FLm) ./ va0 / L2p1
        dHLn[:] = (- L1 / L2p1 * ILFLm + L / L2p1 * JL1FLm) ./ va0.^2

        GLn = ((IL2FLm + JL1FLm) / (L2p1 * L2p3) - (ILFLm + JLn1FLm) / (L2p1 * L2n1)) .* va0
        dGLn[:] = (L - 1) / (L2p1 * L2n1) * ILFLm - L / (L2p1 * L2n1) * JLn1FLm
        dGLn += - L1 / (L2p1 * L2p3) * IL2FLm + (L1 + 1) / (L2p1 * L2p3) * JL1FLm
        ddGLn[:] = - L * (L - 1) / (L2p1 * L2n1) * (ILFLm + JLn1FLm)
        ddGLn += L1 * (L1 + 1) / (L2p1 * L2p3) * (IL2FLm + JL1FLm)
        ddGLn ./= va0
        if va0[1] == 0.0
            HLn[1], dHLn[1], ddGLn[1] = 0.0, 0.0, 0.0
        end
    end
    return dHLn,HLn, ddGLn,dGLn,GLn
end

# 1D, ddGLn,dGLn,GLn
function HGshkarofsky(ddGLn::AbstractVector{T},dGLn::AbstractVector{T},GLn::AbstractVector{T},
    FLn::AbstractVector{T},va::AbstractVector{Tb},nvlevel0::AbstractVector{Int64},
    nc0::Int64,nck::Int64,ocp::Int64,L1::Int64) where{T<:Real,Tb}

    va0 = va[nvlevel0]
    if L1 == 1
        ILn1FL0,IL1FL0,JLFL0,JL0FL0 = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILn1FL0,IL1FL0 = shkarofskyIL0(ILn1FL0,IL1FL0,FLn,va,nvlevel0,nc0,ocp)
        JLFL0,JL0FL0 = shkarofskyJL0(JLFL0,JL0FL0,FLn,va,nc0,nck,ocp)

        GLn[:] = ((IL1FL0 + JLFL0) / 3 + ILn1FL0) .* va0.^2 + JL0FL0
        dGLn[:] = (ILn1FL0 + (2JLFL0 - IL1FL0) / 3.0) .* va0
        ddGLn[:] = (IL1FL0 + JLFL0) * (2 / 3)
    elseif L1 == 2
        ILn2FL0,IL2FL0,JL1n2FL0,JLn1FL0 = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILn2FL0,IL2FL0 = shkarofskyIL1(ILn2FL0,IL2FL0,FLn,va,nvlevel0,nc0,ocp)
        JL1n2FL0,JLn1FL0 = shkarofskyJL1(JL1n2FL0,JLn1FL0,FLn,va,nc0,nck,ocp)

        GLn = ((IL2FL0 / 5.0 - JLn1FL0) + va0.^2 .* (JL1n2FL0 / 5.0 - ILn2FL0)) .* va0 / 3.0
        dGLn[:] = - JLn1FL0 / 3.0 - 2.0 / 15.0 * IL2FL0 + va0.^2 .* JL1n2FL0 / 5.0
        ddGLn[:] = 2 / 5 * (IL2FL0 ./ va0 + va0 .* JL1n2FL0)
        if va0[1] == 0.0
            ddGLn[1] = 0.0
        end
    elseif L1 == 3
        ILn1FL0,IL1FL0,JLFL0,JLn2FL0 = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILn1FL0,IL1FL0 = shkarofskyIL2(ILn1FL0,IL1FL0,FLn,va,nvlevel0,nc0,ocp)
        JLFL0,JLn2FL0 = shkarofskyJL2(JLFL0,JLn2FL0,FLn,va,nvlevel0,nc0,nck,ocp)

        GLn = ((IL1FL0 + JLFL0) / 35.0 - (ILn1FL0 + JLn2FL0) / 15.0) .* va0.^2
        dGLn[:] = ILn1FL0 / 15 - 2 / 15 * JLn2FL0 - 3 / 35 * IL1FL0 + 4 / 35 * JLFL0
        dGLn[:] .*= va0
        ddGLn[:] = 12 / 35 * (IL1FL0 + JLFL0) - 2 / 15 * (ILn1FL0 + JLn2FL0)
    else
        ILFLm,IL2FLm,JL1FLm,JLn1FLm = zeros(T,nc0),zeros(T,nc0),zeros(T,nc0),zeros(T,nc0)
        ILFLm,IL2FLm = shkarofskyI(ILFLm,IL2FLm,FLn,va,nvlevel0,nc0,ocp,L1)
        JL1FLm,JLn1FLm = shkarofskyJ(JL1FLm,JLn1FLm,FLn,va,nvlevel0,nc0,nck,ocp,L1)
        L = L1 - 1
        L2 = 2L
        L2p1 = L2 + 1
        L2n1 = L2 - 1
        L2p3 = L2 + 3
        GLn = ((IL2FLm + JL1FLm) / (L2p1 * L2p3) - (ILFLm + JLn1FLm) / (L2p1 * L2n1)) .* va0
        dGLn[:] = (L - 1) / (L2p1 * L2n1) * ILFLm - L / (L2p1 * L2n1) * JLn1FLm
        dGLn += - L1 / (L2p1 * L2p3) * IL2FLm + (L1 + 1) / (L2p1 * L2p3) * JL1FLm
        ddGLn[:] = - L * (L - 1) / (L2p1 * L2n1) * (ILFLm + JLn1FLm)
        ddGLn += L1 * (L1 + 1) / (L2p1 * L2p3) * (IL2FLm + JL1FLm)
        ddGLn ./= va0
        if va0[1] == 0.0
            ddGLn[1] = 0.0
        end
    end
    return ddGLn,dGLn,GLn
end

"""

  Inputs:
    HLn:
    isRel: ∈ [:unit, :Max, :Maxd0, :Maxd1, :Maxd2]
              where `:Maxd2` denotes `:MaxddHLn` or `:MaxddGLn`
              and `:Maxd1` denotes `:MaxdHLn` or `:MaxdGLn`
              and `:Maxd0` denotes `:MaxHLn` or `:MaxGLn`
              and `:Max` denotes   `:MaxFLn` or `:MaxHLn`

  Outputs:
    RddH = RddHL(ddHLn,dHLn,HLn,Fln,va,L1;isRel=isRel)

"""

function RddHL(ddHLn::AbstractVector{T2},dHLn::AbstractVector{T2},HLn::AbstractVector{T},
    FLn::AbstractVector{T},va::AbstractVector{Tb},L1::Int64;isRel::Symbol=:unit) where{T,Tb,T2}

    if isRel == :unit
        if L1 == 1
            return va .* (ddHLn + FLn) +  2dHLn
        elseif L1 == 2
            return va.^2 .* (ddHLn + FLn)  +  2 * (va .* dHLn - HLn)
        else
            return va.^2 .* (ddHLn + FLn)  +  va .* 2dHLn - L1 * (L1-1) * HLn
        end
    elseif isRel == :Maxd2
        if L1 == 1
            return (va .* (ddHLn + FLn) +  2dHLn) ./maximum(abs.(va .* ddHLn))
        elseif L1 == 2
            return (va.^2 .* (ddHLn + FLn)  +  2 * (va .* dHLn - HLn)) ./maximum(abs.(va.^2 .* ddHLn))
        else
            return (va.^2 .* (ddHLn + FLn)  +  va .* 2dHLn - L1 * (L1-1) * HLn) ./maximum(abs.(va.^2 .* ddHLn))
        end
    elseif isRel == :Maxd1
        if L1 == 1
            return (va .* (ddHLn + FLn) +  2dHLn) ./2maximum(abs.(dHLn))
        elseif L1 == 2
            return (va.^2 .* (ddHLn + FLn)  +  2 * (va .* dHLn - HLn)) ./2maximum(abs.(va .* dHLn))
        else
            return (va.^2 .* (ddHLn + FLn)  +  va .* 2dHLn - L1 * (L1-1) * HLn) ./2maximum(abs.(va .* dHLn))
        end
    elseif isRel == :Maxd0
        if L1 == 1
            return (va .* (ddHLn + FLn) +  2dHLn) ./maximum(va .* FLn)
        elseif L1 == 2
            return (va.^2 .* (ddHLn + FLn)  +  2 * (va .* dHLn - HLn)) ./2maximum(abs.(HLn))
        else
            return (va.^2 .* (ddHLn + FLn)  +  va .* 2dHLn - L1 * (L1-1) * HLn) ./(L1 * (L1-1) * maximum(abs.(HLn)))
        end
    elseif isRel == :Max
        if L1 == 1
            return (va .* (ddHLn + FLn) +  2dHLn) ./maximum(va .* FLn)
        elseif L1 == 2
            return (va.^2 .* (ddHLn + FLn)  +  2 * (va .* dHLn - HLn)) ./maximum(va.^2 .* abs.(FLn))
        else
            return (va.^2 .* (ddHLn + FLn)  +  va .* 2dHLn - L1 * (L1-1) * HLn) ./maximum(va.^2 .* abs.(FLn))
        end
    end
end

function RddHLp(ddHLn::AbstractVector{T2},dHLn::AbstractVector{T2},HLn::AbstractVector{T},
    FLn::AbstractVector{T},va::AbstractVector{Tb},L1::Int64;isRel::Symbol=:unit) where{T,Tb,T2}

    if L1 == 1
        RR = va .* (ddHLn + FLn) +  2dHLn
        xlabel = string("vG,isRel=",isRel)
        label = string("H")
        pH = plot(va,HLn,label=label,legend=legendtR)
        label = string("dH")
        pdH = plot(va,dHLn,label=label,line=(3,:auto),legend=legendbR)
        label = string("FLn")
        pddH = plot(va,FLn,label=label,xlabel=xlabel,line=(1,:auto),legend=legendtR)
        label = string("ddH")
        pddH = plot!(va,ddHLn,label=label,xlabel=xlabel,line=(1,:auto),legend=legendtR)
        label = string("2dHv")
        pddH = plot!(va,2dHLn./ va,label=label,xlabel=xlabel,line=(3,:auto))
        label = string("RddH")
        pRddH = plot(va,RR*neps,label=label,xlabel=xlabel,line=(3,:auto))
        display(plot(pH,pdH,pddH,pRddH,layout=(2,2)))
    else
        RR = va.^2 .* (ddHLn + FLn)  +  va .* 2dHLn - L1 * (L1-1) * HLn
        xlabel = string("vG,isRel=",isRel)
        label = string("H")
        pH = plot(va,HLn,label=label,legend=legendtR)
        label = string("dH")
        pdH = plot!(va,dHLn,label=label,line=(3,:auto),legend=legendbR)
        label = string("FLn")
        pddH = plot(va,FLn,label=label,line=(1,:auto),legend=legendtR)
        label = string("ddH")
        pddH = plot!(va,ddHLn,label=label,line=(1,:auto),legend=legendtR)
        label = string("2dHv")
        pddH = plot!(va,2dHLn./ va,label=label,line=(3,:auto))
        label = string("Hv2")
        pddH = plot!(va,L1 * (L1-1) * HLn./ va.^2,label=label,line=(3,:auto))
        label = string("RddH")
        pRddH = plot(va,RR,label=label,xlabel=xlabel,line=(3,:auto))
        label = string("RddH/Max")
        if isRel == :Maxd2
            RRRR = RR./maximum(va.^2 .* abs.(ddHLn))
        elseif isRel == :Maxd1
            RRRR = RR./(2maximum(va .* abs.(dHLn)))
        elseif isRel == :Maxd0
            RRRR = RR./(L1 * (L1-1) * maximum(abs.(HLn)))
        else
            RRRR = RR./maximum(va.^2 .* abs.(FLn))
        end
        pRddHF = plot(va,RRRR,label=label,xlabel=xlabel,line=(3,:auto))
        display(plot(pH,pddH,pRddH,pRddHF,layout=(2,2)))
    end
    if isRel == :unit
        if L1 == 1
            return va .* (ddHLn + FLn) +  2dHLn
        elseif L1 == 2
            return va.^2 .* (ddHLn + FLn)  +  2 * (va .* dHLn - HLn)
        else
            return va.^2 .* (ddHLn + FLn)  +  va .* 2dHLn - L1 * (L1-1) * HLn
        end
    elseif isRel == :Maxd2
        if L1 == 1
            return (va .* (ddHLn + FLn) +  2dHLn) ./maximum(abs.(va .* ddHLn))
        elseif L1 == 2
            return (va.^2 .* (ddHLn + FLn)  +  2 * (va .* dHLn - HLn)) ./maximum(abs.(va.^2 .* ddHLn))
        else
            return (va.^2 .* (ddHLn + FLn)  +  va .* 2dHLn - L1 * (L1-1) * HLn) ./maximum(abs.(va.^2 .* ddHLn))
        end
    elseif isRel == :Maxd1
        if L1 == 1
            return (va .* (ddHLn + FLn) +  2dHLn) ./2maximum(abs.(dHLn))
        elseif L1 == 2
            return (va.^2 .* (ddHLn + FLn)  +  2 * (va .* dHLn - HLn)) ./2maximum(abs.(va .* dHLn))
        else
            return (va.^2 .* (ddHLn + FLn)  +  va .* 2dHLn - L1 * (L1-1) * HLn) ./2maximum(abs.(va .* dHLn))
        end
    elseif isRel == :Maxd0
        if L1 == 1
            return (va .* (ddHLn + FLn) +  2dHLn) ./maximum(va .* abs.(FLn))
        elseif L1 == 2
            return (va.^2 .* (ddHLn + FLn)  +  2 * (va .* dHLn - HLn)) ./2maximum(abs.(HLn))
        else
            return (va.^2 .* (ddHLn + FLn)  +  va .* 2dHLn - L1 * (L1-1) * HLn) ./(L1 * (L1-1) * maximum(abs.(HLn)))
        end
    elseif isRel == :Max
        if L1 == 1
            return (va .* (ddHLn + FLn) +  2dHLn) ./maximum(va .* abs.(FLn))
        elseif L1 == 2
            return (va.^2 .* (ddHLn + FLn)  +  2 * (va .* dHLn - HLn)) ./maximum(va.^2 .* abs.(FLn))
        else
            return (va.^2 .* (ddHLn + FLn)  +  va .* 2dHLn - L1 * (L1-1) * HLn) ./maximum(va.^2 .* abs.(FLn))
        end
    end
end

"""

  Inputs:
    GLn:

  Outputs:
    RddG = RddGL(ddGLn,dGLn,GLn,Fln,va,L1;isRel=isRel)

"""

function RddGL(ddGLn::AbstractVector{T2},dGLn::AbstractVector{T2},GLn::AbstractVector{T2},
    HLn::AbstractVector{T2},va::AbstractVector{Tb},L1::Int64;isRel::Symbol=:unit) where{Tb,T2}

    if isRel == :unit
        if L1 == 1
            return va .* (ddGLn - 2HLn) +  2dGLn
        elseif L1 == 2
            return va.^2 .* (ddGLn - 2HLn) +  2 * (va .* dGLn - GLn)
        else
            return va.^2 .* (ddGLn - 2HLn) +  2dGLn .* va - L1 * (L1-1) * GLn
        end
    elseif isRel == :Maxd2
        if L1 == 1
            return  (va .* (ddGLn - 2HLn) +  2dGLn) ./ maximum(abs.(va .* ddGLn))
        elseif L1 == 2
            return (va.^2 .* (ddGLn - 2HLn) +  2 * (va .* dGLn - GLn)) ./ maximum(abs.(va.^2 .* ddGLn))
        else
            return (va.^2 .* (ddGLn - 2HLn) +  2dGLn .* va - L1 * (L1-1) * GLn) ./ maximum(abs.(va.^2 .* ddGLn))
        end
    elseif isRel == :Maxd1
        if L1 == 1
            return  (va .* (ddGLn - 2HLn) +  2dGLn) ./ maximum(abs.(2dGLn))
        elseif L1 == 2
            return (va.^2 .* (ddGLn - 2HLn) +  2 * (va .* dGLn - GLn)) ./ maximum(abs.(2va .* dGLn))
        else
            return (va.^2 .* (ddGLn - 2HLn) +  2dGLn .* va - L1 * (L1-1) * GLn) ./ maximum(abs.(2va .* dGLn))
        end
    elseif isRel == :Maxd0
        if L1 == 1
            return  (va .* (ddGLn - 2HLn) +  2dGLn) ./  2maximum(va .* HLn)
        elseif L1 == 2
            return (va.^2 .* (ddGLn - 2HLn) +  2 * (va .* dGLn - GLn)) ./ (- 2minimum(abs.(GLn)))
        else
            return (va.^2 .* (ddGLn - 2HLn) +  2dGLn .* va - L1 * (L1-1) * GLn) ./ (- L1 * (L1-1) * minimum(abs.(GLn)))
        end
    elseif isRel == :Max
        if L1 == 1
            return  (va .* (ddGLn - 2HLn) +  2dGLn) ./ 2maximum(va .* HLn)
        elseif L1 == 2
            return (va.^2 .* (ddGLn - 2HLn) +  2 * (va .* dGLn - GLn)) ./ 2maximum(va.^2 .* abs.(HLn))
        else
            return (va.^2 .* (ddGLn - 2HLn) +  2dGLn .* va - L1 * (L1-1) * GLn) ./ 2maximum(va.^2 .* abs.(HLn))
        end
    end
end

function RddGLp(ddGLn::AbstractVector{T2},dGLn::AbstractVector{T2},GLn::AbstractVector{T2},
    HLn::AbstractVector{T2},va::AbstractVector{Tb},L1::Int64;isRel::Symbol=:unit) where{Tb,T2}

    if L1 == 1
        RR = va .* (ddGLn - 2HLn) +  2dGLn
        xlabel = string("vG,isRel=",isRel)
        label = string("G")
        pG = plot(va,GLn,label=label,legend=legendtR)
        label = string("dG")
        pdG = plot(va,dGLn,label=label,line=(3,:auto),legend=legendbR)
        label = string("ddG")
        pddG = plot(va,ddGLn,label=label,xlabel=xlabel,line=(1,:auto),legend=legendtR)
        label = string("dGv")
        pddG = plot!(va,dGLn./ va,label=label,xlabel=xlabel,line=(3,:auto))
        label = string("RddG")
        pRddG = plot(va,RR*neps,label=label,xlabel=xlabel,line=(3,:auto))
        display(plot(pG,pdG,pddG,pRddG,layout=(2,2)))
    else
        RR = va.^2 .* (ddGLn - 2HLn) +  2dGLn .* va - L1 * (L1-1) * GLn
        xlabel = string("vG,isRel=",isRel)
        label = string("G")
        pG = plot(va,GLn,label=label,legend=legendtR)
        label = string("dG")
        pdG = plot!(va,dGLn,label=label,ine=(3,:auto),legend=legendbR)
        label = string("ddG")
        pddG = plot(va,ddGLn,label=label,xlabel=xlabel,line=(1,:auto),legend=legendtR)
        label = string("dGv")
        pddG = plot!(va,dGLn./ va,label=label,xlabel=xlabel,line=(3,:auto))
        label = string("Gv2")
        pddG = plot!(va,GLn./ va.^2,label=label,xlabel=xlabel,line=(3,:auto))
        label = string("RddG")
        pRddG = plot(va,RR,label=label,xlabel=xlabel,line=(3,:auto))
        if isRel == :Maxd2
            RRRR = RR./maximum(va.^2 .* abs.(ddGLn))
        elseif isRel == :Maxd1
            RRRR = RR./(2maximum(va .* abs.(dGLn)))
        elseif isRel == :Maxd0
            RRRR = RR./(L1 * (L1-1) * minimum(abs.(GLn)))
        else
            RRRR = RR./2maximum(va.^2 .* abs.(HLn))
        end
        label = string("RddG/Max")
        pRddGR = plot(va,RRRR,label=label,xlabel=xlabel,line=(3,:auto))
        display(plot(pG,pddG,pRddG,pRddGR,layout=(2,2)))
    end
    if isRel == :unit
        if L1 == 1
            return va .* (ddGLn - 2HLn) +  2dGLn
        elseif L1 == 2
            return va.^2 .* (ddGLn - 2HLn) +  2 * (va .* dGLn - GLn)
        else
            return va.^2 .* (ddGLn - 2HLn) +  2dGLn .* va - L1 * (L1-1) * GLn
        end
    elseif isRel == :Maxd2
        if L1 == 1
            return  (va .* (ddGLn - 2HLn) +  2dGLn) ./ maximum(abs.(va .* ddGLn))
        elseif L1 == 2
            return (va.^2 .* (ddGLn - 2HLn) +  2 * (va .* dGLn - GLn)) ./ maximum(abs.(va.^2 .* ddGLn))
        else
            return (va.^2 .* (ddGLn - 2HLn) +  2dGLn .* va - L1 * (L1-1) * GLn) ./ maximum(abs.(va.^2 .* ddGLn))
        end
    elseif isRel == :Maxd1
        if L1 == 1
            return  (va .* (ddGLn - 2HLn) +  2dGLn) ./ maximum(abs.(2dGLn))
        elseif L1 == 2
            return (va.^2 .* (ddGLn - 2HLn) +  2 * (va .* dGLn - GLn)) ./ maximum(abs.(2va .* dGLn))
        else
            return (va.^2 .* (ddGLn - 2HLn) +  2dGLn .* va - L1 * (L1-1) * GLn) ./ maximum(abs.(2va .* dGLn))
        end
    elseif isRel == :Maxd0
        if L1 == 1
            return  (va .* (ddGLn - 2HLn) +  2dGLn) ./  2maximum(va .* HLn)
        elseif L1 == 2
            return (va.^2 .* (ddGLn - 2HLn) +  2 * (va .* dGLn - GLn)) ./ (- 2minimum(abs.(GLn)))
        else
            return (va.^2 .* (ddGLn - 2HLn) +  2dGLn .* va - L1 * (L1-1) * GLn) ./ (- L1 * (L1-1) * minimum(abs.(GLn)))
        end
    elseif isRel == :Max
        if L1 == 1
            return  (va .* (ddGLn - 2HLn) +  2dGLn) ./ 2maximum(va .* HLn)
        elseif L1 == 2
            return (va.^2 .* (ddGLn - 2HLn) +  2 * (va .* dGLn - GLn)) ./ 2maximum(va.^2 .* abs.(HLn))
        else
            return (va.^2 .* (ddGLn - 2HLn) +  2dGLn .* va - L1 * (L1-1) * GLn) ./ 2maximum(va.^2 .* abs.(HLn))
        end
    end
end
