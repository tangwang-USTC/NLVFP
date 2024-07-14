
"""
  Fokker-Planck collision
    δf̂/δt = CF * F̂(𝓿̂,μ) .* f̂(v̂,μ) + CH * ∇𝓿̂ Ĥ(𝓿̂) .* ∇v̂ f̂(v̂,μ) + CG * ∇𝓿̂∇𝓿̂ Ĝ(𝓿̂) .* ∇v̂∇v̂ f̂(v̂,μ)

          = ∑ᵢ[(SfLᵢ * Mun) .* (SFLᵢ * Mun)] * Mμ

    where
      mM = 1
      vabth = 1.0 |> T/
     CF = mM               = 1
     CH = (1 - mM) * vbath = 0
     CG = 1 // 2 * vbath^2 = 0.5
     SF = Mvn * XLm * Mun , X = F, H ,G, X = X(v̂)

  Dierckx.jl: Spline1D
              derivative
              extropolate
  DataInterpolations.jl: QuadraticInterpolation
  SmoothingSpline.jl
    spl = fit(SmoothingSpline,v,dG[:,iu],1e-3)
    dG[:,iu] = predict(spl)

  Extrapolating for f(v̂ .≪ 1)
"""

"""

  Inputs:
    δtf: [nc0,LM1,ns]
    vhk:
    ma:
    na = na / n20
    vth = vth / Mms
    fvL0 = f̂(v̂,L), the normalized distribution function by cf,
              without cf = na / π^1.5 / vₜₕ³ due to fvu(v̂,μ) = fvL0(v̂,ℓ) * Mμ
    GvL = Ĝ(𝓋̂,L) , without cF due to fvL0 without cf
    isRel: for `H, G`
    vthi: when `vthi[iFv]` denoes `v̂ᵦₜₕ = vᵦₜₕ / vₜₕ` of `FLn`.

  Outputs:
    δtf,ddfvL,dfvL,fvL0,fvL,FvL,FvLa,vaa,nvlevel0a,ncF = dtfvLSplineaa(δtf,ddfvL,
            dfvL,fvL0,FvL,
            vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
            mu,Mμ,Mun,Mun1,Mun2,LM,LM1,
            nai,uai,vthi,
            CΓ,ma,Zq,spices,na,vth,ns;nMod=nMod,
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
            is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0)

"""

# # 3.5D, [[1:3,nMod],LM1,ns], δtf, ddfvL, dfvL,fvL0,FvL, is_inner = 0

# 3.5D, [nMod,LM1,ns], δtf, ddfvL, dfvL,fvL0,FvL, is_inner = 0
function dtfvLSplineaa(δtf::AbstractVector{Matrix{T}},ddfvL::AbstractVector{Matrix{T}},
    dfvL::AbstractVector{Matrix{T}},fvL0::AbstractVector{Matrix{T}},FvL::AbstractVector{Matrix{T}},
    vhk::Vector{AbstractVector{T}},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},mu::AbstractArray{T,N2},Mμ::AbstractArray{T,N2},
    Mun::AbstractArray{T,N2},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},LM::Vector{Int64},LM1::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    is_normal::Bool=true,restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=1000,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int64=1,dnvs::Int64=1,
    is_normδtf::Bool=false,is_boundaryv0::Bool=false,is_fit_f::Bool=false,
    is_extrapolate_FLn::Bool=true) where{T,N2,NM1,NM2}
    
    fvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        fvL[isp] = zeros(T,nck[isp],LM1)
    end
    nvlevele = Vector{Vector{Int64}}(undef,ns)
    nsp_vec = 1:ns
    for isp in nsp_vec
        nvlevele[isp] = nvlevel0[isp][nvlevele0[isp]]
    end
    if is_extrapolate_FLn 
        ddfvL,dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,
                  ncF = FfvLCS(ddfvL,dfvL,fvL,FvL,fvL0,
                  vhk,nc0,ocp,nvlevele,LM,LM1,
                  nai,uai,vthi,nMod,vth,ns;
                  is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                  autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                  p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
        sum(ncF) == 0 || @show 1, ncF
    else
        ncF = zeros(Int64,ns)
        ddfvL,dfvL,fvL,FvL = FfvLCS(ddfvL,dfvL,fvL,FvL,fvL0,
                  vhk,nc0,nvlevele,LM,
                  nai,uai,vthi,nMod,vth,ns;
                  is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                  autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                  p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
    end
    for isp in nsp_vec
        GvL,dGvL = zeros(T,nc0[isp],LM1),zeros(T,nc0[isp],LM1)
        ddGvL = zeros(T,nc0[isp],LM1)
        ddGvL,dGvL,GvL = HGshkarofsky(ddGvL,dGvL,GvL,fvL[isp],vhk[isp],
                        nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[isp])
        fvL0[isp] = fvL[isp][nvlevele[isp],:]
        if is_boundaryv0
            if nMod[isp] == 1
                δtf[isp] = dtfvLSplineaa(ddfvL[isp],dfvL[isp],fvL0[isp],
                                 ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                                 vhk[isp][nvlevele[isp]],mu,Mμ,Mun,Mun1,Mun2,LM1,
                                 uai[isp][1]/vthi[isp][1];is_boundaryv0=is_boundaryv0)
            else
                δtf[isp] = dtfvLSplineaa(ddfvL[isp],dfvL[isp],fvL0[isp],ddGvL[nvlevele0[isp],:],
                                 dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                                 vhk[isp][nvlevele[isp]],mu,Mμ,Mun,Mun1,Mun2,LM1,
                                 nai[isp],uai[isp]./vthi[isp],vthi[isp],nMod[isp];is_boundaryv0=is_boundaryv0)
            end
        else
            δtf[isp] = dtfvLSplineaa(ddfvL[isp],dfvL[isp],fvL0[isp],ddGvL[nvlevele0[isp],:],
                             dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],vhk[isp][nvlevele[isp]],
                             mu,Mμ,Mun,Mun1,Mun2)
            for L1 in 1:LM1
                if L1 == 1
                    δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                else
                    δtf[isp][1,L1] = 0.0
                end
            end
        end

        # when `is_normδtf = 0`, `cf3[isp] = na[isp]/vth[isp] / π^(3/2)` is not included.
        if is_lnA_const
            lnAg = lnAgamma(εᵣ,ma[isp],Zq[isp]^2,na[isp],vth[isp];is_normδtf=is_normδtf)
        else
            lnAg = lnAgamma_fM(εᵣ,ma[isp],Zq[isp],spices[isp],na[isp],vth[isp];is_normδtf=is_normδtf)
        end
        δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
    end  # for isp
    if is_extrapolate_FLn 
        return δtf, ddfvL,dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,ncF
    else
        return δtf, ddfvL,dfvL,FvL
    end
end

"""
  Inputs:
    fvL0: = fvL[nvlevele,:,isp]
    dfvL: = dfvLe
    ddfvL: = ddfvLe
    FvL: = FvLe
    uai = uai[:,isp] ./ vthi[:,isp]


  Outputs:
    δtf = dtfvLSplineaa(ddfvL,dfvL,fvL0,ddGvL,dGvL,GvL,vhe,mu,Mμ,Mun,Mun1,Mun2,
                      nai,uai,vathi,LM1,nMod;is_boundaryv0=is_boundaryv0)
"""

# 2D, [nMod,LM1] , is_inner == 0, `4π` is not included in the following code but in `3D`.
function dtfvLSplineaa(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvL0::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    vhe::AbstractVector{T},mu::AbstractArray{T,N},Mμ::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    nai::AbstractVector{T},uai::AbstractVector{T},vathi::AbstractVector{T},
    LM1::Int64,nMod::Int64;is_boundaryv0::Bool=false) where{T,N,NM1,NM2}

    CG = 0.5
    cotmu = mu ./ (1 .-mu.^2).^0.5
    # # #################### S = Mvn * XLm * Mun , X = F, H ,G, X = X(vhe)
    fvuP1 = fvL0[:,2:end] * Mun1 ./ vhe
    GvuP1 = GvL[:,2:end] * Mun1 ./ vhe
    dfvuP0 = dfvL * Mun
    dGvuP0 = dGvL * Mun
    ############ 1, Sf1 = CF * f * F
    Sf = (fvL0 * Mun) .^2
    Sf1 = deepcopy(Sf[1:2,:])     # = Sf1[1,:]
    ############ SG= S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    dX01 = GvL[:,2:end] ./ vhe - dGvL[:,2:end]
    GG = dX01 * Mun1
    dX01 = fvL0[:,2:end] ./ vhe - dfvL[:,2:end]
    GG .*= 2(dX01 * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = dGvuP0 + GvuP1 .* cotmu
    GG7 = deepcopy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = dfvuP0 + fvuP1 .* cotmu
        GG7 .*= dX01
        GG += 2GG7
    else
        GG6 = dX01 + (GvL[:,3:end] * Mun2 ./ vhe)
        dX01 = dfvuP0 + fvuP1 .* cotmu
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvL0[:,3:end] * Mun2 ./ vhe))
        GG += (GG7 + GG6)
    end
    GG ./= vhe .^2
    ############ 4, Sf5
    GG7 = (ddfvL * Mun) .* (ddGvL * Mun)    # GG5
    Sf1[1,:] += CG * GG7[1,:]                  # = Sf1[1,:] + Sf5[1,:]
    GG += GG7
    Sf += CG * GG
    if is_boundaryv0 && vhe[1] == 0.0
        # uai = uai./vathi
        if nMod == 1
            vathi[1] || @warn("vathi should be unit when `nMod == 1`!")
            if uai[1] == 0
                Sf1 = dtfvLDMaav0(Sf1,Mun,LM1)
            else
                Sf1 = dtfvLDMaav0(Sf1,mu,Mun,Mun1,Mun2,uai[1],LM1)
            end
        else
            Sf1 = dtfvLDMaav0(Sf1,mu,Mun,Mun1,Mun2,nai,uai,vathi,LM1,nMod)
        end
    end
    Sf[1,:] = Sf1[1,:]
    return Sf * Mμ
end

"""

  Outputs:
    δtf,ddfvL,dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,
            ncF = dtfvLSplineaa(δtf,ddfvL,dfvL,fvL,FvL,fvL0,
            vhk,nc0,nck,ocp,nvlevele0,nvlevel0,mu,Mμ,Mun,Mun1,Mun2,CΓ,
            εᵣ,ma,Zq,spices,na,vth,nai,uai,vthi,LM,LM1,ns;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
            is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0)

"""

# 3D, δtf, ddfvL, dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,ncF, is_inner = 0, nMod = 1
function dtfvLSplineaa(δtf::AbstractVector{Matrix{T}},ddfvL::AbstractVector{Matrix{T}},dfvL::AbstractVector{Matrix{T}},
    fvL::AbstractVector{Matrix{T}},fvL0::AbstractVector{Matrix{T}},FvL::AbstractVector{Matrix{T}},
    vhk::Vector{AbstractVector{T}},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},mu::AbstractArray{T,N2},Mμ::AbstractArray{T,N2},
    Mun::AbstractArray{T,N2},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},LM::Vector{Int64},LM1::Int64,
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;nMod::Int64=1,
    is_normal::Bool=true,restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=1000,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,
    n10::Int64=1,dnvs::Int64=1,is_normδtf::Bool=false,
    is_boundaryv0::Bool=false,is_fit_f::Bool=false) where{T,N2,NM1,NM2}
    
    nvlevele = Vector{Vector{Int64}}(undef,ns)
    nsp_vec = 1:ns
    for isp in nsp_vec
        nvlevele[isp] = nvlevel0[isp][nvlevele0[isp]]
    end
    ddfvL,dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,
              ncF = FfvLCS(ddfvL,dfvL,fvL,FvL,fvL0,
              vhk,nvlevele,vth,LM,LM1,
              nai,uai,vthi,ns;
              is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
              autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
              p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
    for isp in nsp_vec
        GvL,dGvL = zeros(T,nc0[isp],LM1),zeros(T,nc0[isp],LM1)
        ddGvL = zeros(T,nc0[isp],LM1)
        ddGvL,dGvL,GvL = HGshkarofsky(ddGvL,dGvL,GvL,fvL[isp],vhk[isp],
                        nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[isp])
        δtf[isp] = dtfvLSplineaa(ddfvL[isp],dfvL[isp],fvL[isp][nvlevele[isp],:],
                         ddGvL,dGvL,GvL,vhk[isp][nvlevele[isp]],
                         mu,Mμ,Mun,Mun1,Mun2,LM1,
                         uai[isp]/vthi[isp];is_boundaryv0=is_boundaryv0)
        if is_boundaryv0 == false
            for L1 in 1:LM1
                if L1 == 1
                    δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                else
                    δtf[isp][1,L1] = 0.0
                end
            end
        end

        # when `is_normδtf = 0`, `cf3[isp] = na[isp]/vth[isp] / π^(3/2)` is not included.
        if is_lnA_const
            lnAg = lnAgamma(εᵣ,ma[isp],Zq[isp]^2,na[isp],vth[isp];is_normδtf=is_normδtf)
        else
            lnAg = lnAgamma_fM(εᵣ,ma[isp],Zq[isp],spices[isp],na[isp],vth[isp];is_normδtf=is_normδtf)
        end
        δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
    end  # for isp
    return δtf, ddfvL,dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,ncF
end

"""
  Inputs:
    fvL0: = fvL[nvlevel0,:,isp]
    dfvL: = dfvL[nvlevel0,:,isp]
    ddfvL: = ddfvL[nvlevel0,:,isp]
    FvL: = FvL[nvlevel0,:,isp]
    uai = uai[isp] / vthi[isp]


  Outputs:
    δtf = dtfvLSplineaa(ddfvL,dfvL,fvL0,ddGvL,dGvL,GvL,
                      vhe,mu,Mμ,Mun,Mun1,Mun2,LM1,uai)
"""

# 2D, nc0 , is_inner == 0, `4π` is not included in the following code but in `3D`.
function dtfvLSplineaa(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvL0::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    vhe::AbstractVector{T},mu::AbstractArray{T,N},Mμ::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},LM1::Int64,
    uai::T;is_boundaryv0::Bool=false) where{T,N,NM1,NM2}

    CG = 0.5
    cotmu = mu ./ (1 .-mu.^2).^0.5
    # # #################### S = Mvn * XLm * Mun , X = F, H ,G, X = X(vhe)
    fvuP1 = fvL0[:,2:end] * Mun1 ./ vhe
    GvuP1 = GvL[:,2:end] * Mun1 ./ vhe
    dfvuP0 = dfvL * Mun
    dGvuP0 = dGvL * Mun
    ############ 1, Sf1 = CF * f * F
    Sf = (fvL0 * Mun) .^2
    Sf1 = deepcopy(Sf[1:2,:])     # = Sf1[1,:]
    ############ SG= S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    dX01 = GvL[:,2:end] ./ vhe - dGvL[:,2:end]
    GG = dX01 * Mun1
    dX01 = fvL0[:,2:end] ./ vhe - dfvL[:,2:end]
    GG .*= 2(dX01 * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = dGvuP0 + GvuP1 .* cotmu
    GG7 = deepcopy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = dfvuP0 + fvuP1 .* cotmu
        GG7 .*= dX01
        GG += 2GG7
    else
        GG6 = dX01 + (GvL[:,3:end] * Mun2 ./ vhe)
        dX01 = dfvuP0 + fvuP1 .* cotmu
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvL0[:,3:end] * Mun2 ./ vhe))
        GG += (GG7 + GG6)
    end
    GG ./= vhe .^2
    ############ 4, Sf5
    GG7 = (ddfvL * Mun) .* (ddGvL * Mun)    # GG5
    Sf1[1,:] += CG * GG7[1,:]                  # = Sf1[1,:] + Sf5[1,:]
    GG += GG7
    Sf += CG * GG
    if is_boundaryv0 && vhe[1] == 0.0
        if uai[1] == 0
            Sf1 = dtfvLDMaav0(Sf1,Mun,LM1)
        else
            Sf1 = dtfvLDMaav0(Sf1,mu,Mun,Mun1,Mun2,uai,LM1)
        end
    end
    Sf[1,:] = Sf1[1,:]
    return Sf * Mμ
end

function dtfvLSplineaa(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvL0::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    vhe::AbstractVector{T},mu::AbstractArray{T,N},Mμ::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2}) where{T,N,NM1,NM2}

    CG = 0.5
    cotmu = mu ./ (1 .-mu.^2).^0.5
    # # #################### S = Mvn * XLm * Mun , X = F, H ,G, X = X(vhe)
    fvuP1 = fvL0[:,2:end] * Mun1 ./ vhe
    GvuP1 = GvL[:,2:end] * Mun1 ./ vhe
    dfvuP0 = dfvL * Mun
    dGvuP0 = dGvL * Mun
    ############ 1, Sf1 = CF * f * F
    Sf = (fvL0 * Mun) .^2
    Sf1 = deepcopy(Sf[1:2,:])     # = Sf1[1,:]
    ############ SG= S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    dX01 = GvL[:,2:end] ./ vhe - dGvL[:,2:end]
    GG = dX01 * Mun1
    dX01 = fvL0[:,2:end] ./ vhe - dfvL[:,2:end]
    GG .*= 2(dX01 * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = dGvuP0 + GvuP1 .* cotmu
    GG7 = deepcopy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = dfvuP0 + fvuP1 .* cotmu
        GG7 .*= dX01
        GG += 2GG7
    else
        GG6 = dX01 + (GvL[:,3:end] * Mun2 ./ vhe)
        dX01 = dfvuP0 + fvuP1 .* cotmu
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvL0[:,3:end] * Mun2 ./ vhe))
        GG += (GG7 + GG6)
    end
    GG ./= vhe .^2
    ############ 4, Sf5
    GG7 = (ddfvL * Mun) .* (ddGvL * Mun)    # GG5
    Sf1[1,:] += CG * GG7[1,:]                  # = Sf1[1,:] + Sf5[1,:]
    GG += GG7
    Sf += CG * GG
    Sf[1,:] = Sf1[1,:]
    return Sf * Mμ
end
