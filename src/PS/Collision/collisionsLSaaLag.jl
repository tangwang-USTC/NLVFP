
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
    δtf,ddfvL,dfvL,fvL0,FvL,FvLa,vaa,nvlevel0a,
            ncF = dtfvLSplineaaLag(δtf,ddfvL,dfvL,fvL0,FvL,
            vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
            mu,Mμ,Mun,Mun1,Mun2,LM,LM1,
            nai,uai,vthi,
            CΓ,εᵣ,ma,Zq,spices,na,vth,ns;nMod=nMod,
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
            is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0)
    δtf,ddfvL,dfvL,fvL0,FvL,FvLa,vaa,nvlevel0a,
            ncF = dtfvLSplineaaLag(δtf,ddfvL,dfvL,fvL0,FvL,
            vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
            mu,Mμ,Mun,Mun1,Mun2,LM,LM1,
            nai,uai,vthi,
            CΓ,εᵣ,ma,Zq,na,vth,ns;nMod=nMod,
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
            is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0)

"""

# 3.5D, [nMod,LM1,ns], δtf, ddfvL, dfvL,fvL0,FvL, is_inner = 0    
function dtfvLSplineaaLag(δtf::AbstractVector{Matrix{T}},ddfvL::AbstractVector{Matrix{T}},
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
    
    nvlevele = Vector{Vector{Int64}}(undef,ns)
    nsp_vec = 1:ns
    for isp in nsp_vec
        nvlevele[isp] = nvlevel0[isp][nvlevele0[isp]]
    end
    fvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        fvL[isp] = zeros(T,nck[isp],LM1)
    end
    if is_extrapolate_FLn
        ddfvL,dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,ncF = FfvLCS(ddfvL,dfvL,fvL,FvL,fvL0,
                  vhk,nc0,ocp,nvlevele,LM,LM1,
                  nai,uai,vthi,nMod,vth,ns;
                  is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                  autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                  p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
    else
        ncF = zeros(Int64,ns)
        ddfvL,dfvL,fvL,FvL = FfvLCSLag!(ddfvL,dfvL,fvL,FvL,fvL0,
                  vhk,nc0,nvlevele,LM,
                  nai,uai,vthi,nMod,vth,ns;
                  is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                  autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                  p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
    end
    sum(ncF) == 0 || @show 1, ncF
    for isp in nsp_vec
        GvL,dGvL = zeros(T,nc0[isp],LM1),zeros(T,nc0[isp],LM1)
        ddGvL = zeros(T,nc0[isp],LM1)
        ddGvL,dGvL,GvL = HGshkarofsky(ddGvL,dGvL,GvL,fvL[isp],vhk[isp],nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[isp])
        fvL0[isp] = fvL[isp][nvlevele[isp],:]
        if is_boundaryv0
            if nMod[isp] == 1
                δtf[isp] = dtfvLSplineaa(ddfvL[isp],dfvL[isp],fvL0[isp],ddGvL[nvlevele0[isp],:],
                                 dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],vhk[isp][nvlevele[isp]],
                                 mu,Mμ,Mun,Mun1,Mun2,LM1,
                                 uai[isp][1]/vthi[isp][1];is_boundaryv0=is_boundaryv0)
            else
                δtf[isp] = dtfvLSplineaa(ddfvL[isp],dfvL[isp],fvL0[isp],ddGvL[nvlevele0[isp],:],
                                 dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],vhk[isp][nvlevele[isp]],
                                 mu,Mμ,Mun,Mun1,Mun2,LM1,
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
            lnAg = lnAgamma(εᵣ,ma[isp],Zq[isp]^2,spices[isp],na[isp],vth[isp];is_normδtf=is_normδtf)
        else
            lnAg = lnAgamma_fM(εᵣ,ma[isp],Zq[isp],spices[isp],na[isp],vth[isp];is_normδtf=is_normδtf)
        end
        # @show 13,isp,(1,3),spices[isp], lnAg/
        δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
    end  # for isp
    if is_extrapolate_FLn
        return δtf, ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,ncF
    else
        return δtf, ddfvL,dfvL,FvL
    end
end

# 3.5D, [LM1,ns],nMod=1, δtf, ddfvL, dfvL,fvL0,FvL, is_inner = 0    
function dtfvLSplineaaLag(δtf::AbstractVector{Matrix{T}},ddfvL::AbstractVector{Matrix{T}},
    dfvL::AbstractVector{Matrix{T}},fvL0::AbstractVector{Matrix{T}},FvL::AbstractVector{Matrix{T}},
    vhk::Vector{AbstractVector{T}},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},mu::AbstractArray{T,N2},Mμ::AbstractArray{T,N2},
    Mun::AbstractArray{T,N2},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},LM::Vector{Int64},LM1::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    is_normal::Bool=true,restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=1000,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int64=1,dnvs::Int64=1,
    is_normδtf::Bool=false,is_boundaryv0::Bool=false,is_fit_f::Bool=false,
    is_extrapolate_FLn::Bool=true) where{T,N2,NM1,NM2}
    
    nvlevele = Vector{Vector{Int64}}(undef,ns)
    nsp_vec = 1:ns
    for isp in nsp_vec
        nvlevele[isp] = nvlevel0[isp][nvlevele0[isp]]
    end
    fvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        fvL[isp] = zeros(T,nck[isp],LM1)
    end
    if is_extrapolate_FLn
        ddfvL,dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,ncF = FfvLCS(ddfvL,dfvL,fvL,FvL,fvL0,
                  vhk,nc0,ocp,nvlevele,LM,LM1,
                  nai,uai,vthi,vth,ns;
                  is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                  autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                  p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
    else
        ncF = zeros(Int64,ns)
        ddfvL,dfvL,fvL,FvL = FfvLCSLag!(ddfvL,dfvL,fvL,FvL,fvL0,
                  vhk,nc0,nvlevele,LM,
                  nai,uai,vthi,vth,ns;
                  is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                  autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                  p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
    end
    sum(ncF) == 0 || @show 1, ncF
    for isp in nsp_vec
        GvL,dGvL = zeros(T,nc0[isp],LM1),zeros(T,nc0[isp],LM1)
        ddGvL = zeros(T,nc0[isp],LM1)
        ddGvL,dGvL,GvL = HGshkarofsky(ddGvL,dGvL,GvL,fvL[isp],vhk[isp],nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[isp])
        fvL0[isp] = fvL[isp][nvlevele[isp],:]
        if is_boundaryv0
            δtf[isp] = dtfvLSplineaa(ddfvL[isp],dfvL[isp],fvL0[isp],ddGvL[nvlevele0[isp],:],
                             dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],vhk[isp][nvlevele[isp]],
                             mu,Mμ,Mun,Mun1,Mun2,LM1,
                             uai[isp][1]/vthi[isp][1];is_boundaryv0=is_boundaryv0)
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
            lnAg = lnAgamma(εᵣ,ma[isp],Zq[isp]^2,spices[isp],na[isp],vth[isp];is_normδtf=is_normδtf)
        else
            lnAg = lnAgamma_fM(εᵣ,ma[isp],Zq[isp],spices[isp],na[isp],vth[isp];is_normδtf=is_normδtf)
        end
        δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
    end  # for isp
    if is_extrapolate_FLn
        return δtf, ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,ncF
    else
        return δtf, ddfvL,dfvL,FvL
    end
end

###########################################
# 3.5D, [nMod,LM1,ns], δtf, ddfvL, dfvL,fvL0,FvL, is_inner = 0
function dtfvLSplineaaLag(δtf::AbstractVector{Matrix{T}},ddfvL::AbstractVector{Matrix{T}},
    dfvL::AbstractVector{Matrix{T}},fvL0::AbstractVector{Matrix{T}},FvL::AbstractVector{Matrix{T}},
    vhk::AbstractVector{T},nc0::Int64,nck::Int64,ocp::Int64,
    nvlevele0::Vector{Int64},nvlevel0::Vector{Int64},mu::AbstractArray{T,N2},Mμ::AbstractArray{T,N2},
    Mun::AbstractArray{T,N2},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},LM::Vector{Int64},LM1::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    is_normal::Bool=true,restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=1000,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int64=1,dnvs::Int64=1,
    is_normδtf::Bool=false,is_boundaryv0::Bool=false,is_fit_f::Bool=false,
    is_extrapolate_FLn::Bool=true) where{T,N2,NM1,NM2}
    
    nsp_vec = 1:ns
    nvlevele = nvlevel0[nvlevele0]
    fvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        fvL[isp] = zeros(T,nck,LM1)
        FvL[isp] = zeros(T,nck,LM1)
    end
    if is_extrapolate_FLn
        ddfvL,dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,
                  ncF = FfvLCSLag!(ddfvL,dfvL,fvL,FvL,fvL0,
                  vhk,nc0,ocp,nvlevele,LM,LM1,
                  nai,uai,vthi,nMod,vth,ns;
                  is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                  autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                  p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
    else
        ncF = zeros(Int64,ns)
        ddfvL,dfvL,fvL,FvL = FfvLCSLag!(ddfvL,dfvL,fvL,FvL,fvL0,
                  vhk,nc0,nvlevele,LM,
                  nai,uai,vthi,nMod,vth,ns;
                  is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                  autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                  p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
    end
    for isp in nsp_vec
        GvL,dGvL = zeros(T,nc0,LM1),zeros(T,nc0,LM1)
        ddGvL = zeros(T,nc0,LM1)
        ddGvL,dGvL,GvL = HGshkarofsky(ddGvL,dGvL,GvL,fvL[isp],vhk,nvlevel0,nc0,nck,ocp,LM[isp])
        fvL0[isp] = fvL[isp][nvlevele,:]
        if is_boundaryv0
            if nMod[isp] == 1
                δtf[isp] = dtfvLSplineaa(ddfvL[isp],dfvL[isp],fvL0[isp],ddGvL[nvlevele0,:],
                                 dGvL[nvlevele0,:],GvL[nvlevele0,:],vhk[nvlevele],
                                 mu,Mμ,Mun,Mun1,Mun2,LM1,
                                 uai[isp][1]/vthi[isp][1];is_boundaryv0=is_boundaryv0)
            else
                δtf[isp] = dtfvLSplineaa(ddfvL[isp],dfvL[isp],fvL0[isp],ddGvL[nvlevele0,:],
                                 dGvL[nvlevele0,:],GvL[nvlevele0,:],vhk[nvlevele],
                                 mu,Mμ,Mun,Mun1,Mun2,LM1,
                                 nai[isp],uai[isp]./vthi[isp],vthi[isp],nMod[isp];is_boundaryv0=is_boundaryv0)
            end
        else
            δtf[isp] = dtfvLSplineaa(ddfvL[isp],dfvL[isp],fvL0[isp],ddGvL[nvlevele0,:],
                             dGvL[nvlevele0,:],GvL[nvlevele0,:],vhk[nvlevele],mu,Mμ,Mun,Mun1,Mun2)
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
            lnAg = lnAgamma(εᵣ,ma[isp],Zq[isp]^2,spices[isp],na[isp],vth[isp];is_normδtf=is_normδtf)
        else
            lnAg = lnAgamma_fM(εᵣ,ma[isp],Zq[isp],spices[isp],na[isp],vth[isp];is_normδtf=is_normδtf)
        end
        δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
    end  # for isp
    if is_extrapolate_FLn 
        return δtf, ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,ncF
    else
        return δtf, ddfvL,dfvL,FvL
    end
end
