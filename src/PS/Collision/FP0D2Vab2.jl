
"""
  Applying the same grids (uniform or Chebyshev) for both spices which mean `uai ≈ ubi`.

    `FvL` 

  Inputs: 
    Rc = Rc[1:njMs,:,:] 

  Outputs:
    FP0D2Vab2!(δtf,fvL0k,vhe,vhk,nvG,nc0,nck,ocp,
           nvlevele0,nvlevel0,LM,LM1k,
           naik,uaik,vthik,nModk,
           CΓ,εᵣ,ma,Zq,spices,nak,uhkL,vathk,Rc,Mc,dtk;
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
           is_extrapolate_FLn=is_extrapolate_FLn)
"""
# [nMod,nv,LM,ns=2] in `_Ms` version, for both spices which `uai ≈ ubi` 
# when `is_dtk_order_Rcaa = true`
function FP0D2Vab2!(δtf::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},
    vhe::AbstractVector{StepRangeLen},vhk::Vector{AbstractVector{T}},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},nModk::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak::AbstractVector{T},vathk::AbstractVector{T},
    Rc::AbstractArray{T,N},Mc::AbstractArray{T,N},nMjMs::Vector{Int64},dtk::T;ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,is_fit_f::Bool=false,
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_norm_error::Bool=true,dtk_order_Rc::Symbol=:mid) where{T,N}
    
    mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1k - 1)

    nsp_vec = 1:ns
    FvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    ddfvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        FvL[isp] = zeros(T,nck[isp],LM1k)
        dfvL[isp] = zeros(T,nvG[isp],LM1k)
        ddfvL[isp] = zeros(T,nvG[isp],LM1k)
    end

    ############################################################ Self-collisions.
    dtfvLaa = similar(δtf)

    # Computing the first-two order derivatives of `fvL` and it's mapping functions `FvL` when `nMod .== 1` and `uai[1] = - uai[2]`
    if is_extrapolate_FLn
        dtfvLaa,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    else
        ncF = zeros(Int64,ns)
        dtfvLaa,ddfvL,dfvL,FvL = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    end

    # Verifying the mass, momentum and total energy conservation laws during self-collision process.
    if is_dtk_order_Rcaa
        if gridv_type == :uniform
            dtMcsd2l!(Rc,dtfvLaa,vhe,nMjMs,ma.*nak,vathk,LMk,ns;is_renorm=is_renorm)
        elseif gridv_type == :chebyshev
            dtMcsd2l!(Rc,dtfvLaa,vhe,nvG,nMjMs,ma.*nak,vathk,LMk,ns;
                      is_renorm=is_renorm,is_norm_error=is_norm_error)
        end

        dtk = dt_Rc(dtk,Rc,Mc,LMk,nModk,nMjMs,ns;
                    rtol_DnIK=rtol_DnIK,dtk_order_Rc=dtk_order_Rc)
    end
    
    ######################################################## Collision between different spices 
    for isp in nsp_vec
        isp == 1 ? iFv = 2 : iFv = 1
        nvlevele = nvlevel0[isp][nvlevele0[isp]]
        mM = ma[isp] / ma[iFv]
        vabth = vathk[isp] / vathk[iFv]
        va = vhk[isp] * vabth     # 𝓋̂ = v̂a * vabth
        
        HvL,dHvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        GvL,dGvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        ddGvL = zeros(T,nc0[isp],LM1k)
        if ncF[isp] ≥ 2
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv], 
                        FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
        else
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv])
        end
        if nModk[isp] == 1 && nModk[iFv] == 1
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uaik[isp][1],uaik[iFv][1],
                    mM,vabth;is_boundaryv0=is_boundaryv0)
        else
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    naik[isp],naik[iFv],uaik[isp]./vthik[isp],
                    uaik[iFv]./vthik[iFv],vthik[isp],vthik[iFv],nModk[isp],nModk[iFv],
                    mM,vabth;is_boundaryv0=is_boundaryv0)
        end
        if is_boundaryv0 == false
            for L1 in 1:LM1k
                if L1 == 1
                    δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                else
                    δtf[isp][1,L1] = 0.0
                end
            end
        end

        if is_lnA_const
            Zqab = Zq[isp] * Zq[iFv]
            lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nak[isp],nak[iFv],vathk[isp],vathk[iFv];is_normδtf=true)
        else
            lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                            [nak[isp],nak[iFv]],[vathk[isp],vathk[iFv]],isp,iFv;is_normδtf=true)
        end
        # @show 0,isp,(2,3),spices[isp], lnAg
        δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
    end
    
    if is_δtfvLaa === 1
        for isp in nsp_vec
            δtf[isp] += dtfvLaa[isp]
        end
    end
    return dtk
end
function FP0D2Vab2!(δtf::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},
    vhe::Vector{AbstractVector{T}},vhk::Vector{AbstractVector{T}},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},nModk::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak::AbstractVector{T},vathk::AbstractVector{T},
    Rc::AbstractArray{T,N},Mc::AbstractArray{T,N},nMjMs::Vector{Int64},dtk::T;ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,is_fit_f::Bool=false,
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_norm_error::Bool=true,dtk_order_Rc::Symbol=:mid) where{T,N}
    
    mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1k - 1)

    nsp_vec = 1:ns
    FvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    ddfvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        FvL[isp] = zeros(T,nck[isp],LM1k)
        dfvL[isp] = zeros(T,nvG[isp],LM1k)
        ddfvL[isp] = zeros(T,nvG[isp],LM1k)
    end

    ############################################################ Self-collisions.
    dtfvLaa = similar(δtf)

    # Computing the first-two order derivatives of `fvL` and it's mapping functions `FvL` when `nMod .== 1` and `uai[1] = - uai[2]`
    if is_extrapolate_FLn
        dtfvLaa,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    else
        ncF = zeros(Int64,ns)
        dtfvLaa,ddfvL,dfvL,FvL = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    end

    # Verifying the mass, momentum and total energy conservation laws during self-collision process.
    if is_dtk_order_Rcaa
        if gridv_type == :uniform
            dtMcsd2l!(Rc,dtfvLaa,vhe,nMjMs,ma.*nak,vathk,LMk,ns;is_renorm=is_renorm)
        elseif gridv_type == :chebyshev
            dtMcsd2l!(Rc,dtfvLaa,vhe,nvG,nMjMs,ma.*nak,vathk,LMk,ns;
                      is_renorm=is_renorm,is_norm_error=is_norm_error)
        end

        dtk = dt_Rc(dtk,Rc,Mc,LMk,nModk,nMjMs,ns;
                    rtol_DnIK=rtol_DnIK,dtk_order_Rc=dtk_order_Rc)
    end
    
    ######################################################## Collision between different spices 
    for isp in nsp_vec
        isp == 1 ? iFv = 2 : iFv = 1
        nvlevele = nvlevel0[isp][nvlevele0[isp]]
        mM = ma[isp] / ma[iFv]
        vabth = vathk[isp] / vathk[iFv]
        va = vhk[isp] * vabth     # 𝓋̂ = v̂a * vabth
        
        HvL,dHvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        GvL,dGvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        ddGvL = zeros(T,nc0[isp],LM1k)
        if ncF[isp] ≥ 2
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv], 
                        FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
        else
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv])
        end
        if nModk[isp] == 1 && nModk[iFv] == 1
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uaik[isp][1],uaik[iFv][1],
                    mM,vabth;is_boundaryv0=is_boundaryv0)
        else
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    naik[isp],naik[iFv],uaik[isp]./vthik[isp],uaik[iFv]./vthik[iFv],
                    vthik[isp],vthik[iFv],nModk[isp],nModk[iFv],
                    mM,vabth;is_boundaryv0=is_boundaryv0)
        end
        if is_boundaryv0 == false
            for L1 in 1:LM1k
                if L1 == 1
                    δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                else
                    δtf[isp][1,L1] = 0.0
                end
            end
        end

        if is_lnA_const
            Zqab = Zq[isp] * Zq[iFv]
            lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nak[isp],nak[iFv],vathk[isp],vathk[iFv];is_normδtf=true)
        else
            lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                            [nak[isp],nak[iFv]],[vathk[isp],vathk[iFv]],isp,iFv;is_normδtf=true)
        end
        # @show 0,isp,(2,3),spices[isp], lnAg
        δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
    end
    
    if is_δtfvLaa === 1
        for isp in nsp_vec
            δtf[isp] += dtfvLaa[isp]
        end
    end
    return dtk
end

# Ccol
function FP0D2Vab2!(δtf::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},
    vhe::AbstractVector{StepRangeLen},vhk::Vector{AbstractVector{T}},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},nModk::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak::AbstractVector{T},vathk::AbstractVector{T},
    Rc::AbstractArray{T,N},Mc::AbstractArray{T,N},nMjMs::Vector{Int64},Ccol::AbstractVector{T},dtk::T;ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,is_fit_f::Bool=false,
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_norm_error::Bool=true,dtk_order_Rc::Symbol=:mid) where{T,N}
    
    mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1k - 1)

    nsp_vec = 1:ns
    FvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    ddfvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        FvL[isp] = zeros(T,nck[isp],LM1k)
        dfvL[isp] = zeros(T,nvG[isp],LM1k)
        ddfvL[isp] = zeros(T,nvG[isp],LM1k)
    end

    ############################################################ Self-collisions.
    dtfvLaa = similar(δtf)

    # Computing the first-two order derivatives of `fvL` and it's mapping functions `FvL` when `nMod .== 1` and `uai[1] = - uai[2]`
    if is_extrapolate_FLn
        dtfvLaa,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    else
        ncF = zeros(Int64,ns)
        dtfvLaa,ddfvL,dfvL,FvL = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    end

    # Verifying the mass, momentum and total energy conservation laws during self-collision process.
    if is_dtk_order_Rcaa
        if gridv_type == :uniform
            dtMcsd2l!(Rc,dtfvLaa,vhe,nMjMs,ma.*nak,vathk,LMk,ns;is_renorm=is_renorm)
        elseif gridv_type == :chebyshev
            dtMcsd2l!(Rc,dtfvLaa,vhe,nvG,nMjMs,ma.*nak,vathk,LMk,ns;
                      is_renorm=is_renorm,is_norm_error=is_norm_error)
        end

        dtk = dt_Rc(dtk,Rc,Mc,LMk,nModk,nMjMs,ns;
                    rtol_DnIK=rtol_DnIK,dtk_order_Rc=dtk_order_Rc)
    end
    
    ######################################################## Collision between different spices 
    dtCFL = [1e10,1e10]
    for isp in nsp_vec
        isp == 1 ? iFv = 2 : iFv = 1
        nvlevele = nvlevel0[isp][nvlevele0[isp]]
        mM = ma[isp] / ma[iFv]
        vabth = vathk[isp] / vathk[iFv]
        va = vhk[isp] * vabth     # 𝓋̂ = v̂a * vabth
        
        HvL,dHvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        GvL,dGvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        ddGvL = zeros(T,nc0[isp],LM1k)
        if ncF[isp] ≥ 2
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv], 
                        FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
        else
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv])
        end
        if nModk[isp] == 1 && nModk[iFv] == 1
            δtf[isp], dtCFL2 = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uaik[isp][1],uaik[iFv][1],
                    mM,vabth,dtCFL[isp];is_boundaryv0=is_boundaryv0)
        else
            δtf[isp], dtCFL2 = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    naik[isp],naik[iFv],uaik[isp]./vthik[isp],uaik[iFv]./vthik[iFv],
                    vthik[isp],vthik[iFv],nModk[isp],nModk[iFv],
                    mM,vabth,dtCFL[isp];is_boundaryv0=is_boundaryv0)
        end
        if is_boundaryv0 == false
            for L1 in 1:LM1k
                if L1 == 1
                    δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                else
                    δtf[isp][1,L1] = 0.0
                end
            end
        end

        if is_lnA_const
            Zqab = Zq[isp] * Zq[iFv]
            lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nak[isp],nak[iFv],vathk[isp],vathk[iFv];is_normδtf=true)
        else
            lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                            [nak[isp],nak[iFv]],[vathk[isp],vathk[iFv]],isp,iFv;is_normδtf=true)
        end
        Ccol[isp] = CΓ * lnAg
        δtf[isp] *= Ccol[isp]   # CΓ is owing to the dimensionless process
    end
    
    if is_δtfvLaa === 1
        for isp in nsp_vec
            δtf[isp] += dtfvLaa[isp]
        end
    end
    return dtk, min(dtCFL[1],dtCFL[2])
end

# when `is_dtk_order_Rcaa = false`
function FP0D2Vab2!(δtf::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},vhk::Vector{AbstractVector{T}},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},nModk::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak::AbstractVector{T},vathk::AbstractVector{T};ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,is_fit_f::Bool=false,
    is_extrapolate_FLn::Bool=true) where{T}
    
    mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1k - 1)

    nsp_vec = 1:ns
    FvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    ddfvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        FvL[isp] = zeros(T,nck[isp],LM1k)
        dfvL[isp] = zeros(T,nvG[isp],LM1k)
        ddfvL[isp] = zeros(T,nvG[isp],LM1k)
    end

    # Self-collisions.
    dtfvLaa = similar(δtf)

    # Computing the first-two order derivatives of `fvL` and it's mapping functions `FvL` when `nMod .== 1` and `uai[1] = - uai[2]`
    if is_extrapolate_FLn
        dtfvLaa,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    else
        ncF = zeros(Int64,ns)
        dtfvLaa,ddfvL,dfvL,FvL = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    end
    # @show is_normδtf
    
    # Collision between different spices
    for isp in nsp_vec
        isp == 1 ? iFv = 2 : iFv = 1
        nvlevele = nvlevel0[isp][nvlevele0[isp]]
        mM = ma[isp] / ma[iFv]
        vabth = vathk[isp] / vathk[iFv]
        va = vhk[isp] * vabth     # 𝓋̂ = v̂a * vabth
        
        HvL,dHvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        GvL,dGvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        ddGvL = zeros(T,nc0[isp],LM1k)
        if ncF[isp] ≥ 2
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv], 
                        FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
        else
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv])
        end
        if nModk[isp] == 1 && nModk[iFv] == 1
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uaik[isp][1],uaik[iFv][1],
                    mM,vabth;is_boundaryv0=is_boundaryv0)
        else
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    naik[isp],naik[iFv],uaik[isp]./vthik[isp],uaik[iFv]./vthik[iFv],
                    vthik[isp],vthik[iFv],nModk[isp],nModk[iFv],
                    mM,vabth;is_boundaryv0=is_boundaryv0)
        end
        if is_boundaryv0 == false
            for L1 in 1:LM1k
                if L1 == 1
                    δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                else
                    δtf[isp][1,L1] = 0.0
                end
            end
        end

        if is_lnA_const
            Zqab = Zq[isp] * Zq[iFv]
            lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nak[isp],nak[iFv],vathk[isp],vathk[iFv];is_normδtf=true)
        else
            lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                            [nak[isp],nak[iFv]],[vathk[isp],vathk[iFv]],isp,iFv;is_normδtf=true)
        end
        # @show 1,isp,(2,3),spices[isp], lnAg
        δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
    end
    
    if is_δtfvLaa === 1
        for isp in nsp_vec
            δtf[isp] += dtfvLaa[isp]
        end
    end
end

# Ccol
function FP0D2Vab2!(δtf::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},vhk::Vector{AbstractVector{T}},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},Ccol::AbstractVector{T},
    nak::AbstractVector{T},vathk::AbstractVector{T},nModk::Vector{Int64};ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,is_fit_f::Bool=false,
    is_extrapolate_FLn::Bool=true) where{T}
    
    mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1k - 1)

    nsp_vec = 1:ns
    FvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    ddfvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        FvL[isp] = zeros(T,nck[isp],LM1k)
        dfvL[isp] = zeros(T,nvG[isp],LM1k)
        ddfvL[isp] = zeros(T,nvG[isp],LM1k)
    end

    # Self-collisions.
    dtfvLaa = similar(δtf)
    dtCFL = [1e10,1e10]

    # Computing the first-two order derivatives of `fvL` and it's mapping functions `FvL` when `nMod .== 1` and `uai[1] = - uai[2]`
    if is_extrapolate_FLn
        dtfvLaa,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    else
        ncF = zeros(Int64,ns)
        dtfvLaa,ddfvL,dfvL,FvL = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    end
    # @show is_normδtf
    
    # Collision between different spices
    for isp in nsp_vec
        isp == 1 ? iFv = 2 : iFv = 1
        nvlevele = nvlevel0[isp][nvlevele0[isp]]
        mM = ma[isp] / ma[iFv]
        vabth = vathk[isp] / vathk[iFv]
        va = vhk[isp] * vabth     # 𝓋̂ = v̂a * vabth
        
        HvL,dHvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        GvL,dGvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        ddGvL = zeros(T,nc0[isp],LM1k)
        if ncF[isp] ≥ 2
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv], 
                        FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
        else
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv])
        end
        if nModk[isp] == 1 && nModk[iFv] == 1
            δtf[isp], dtCFL2 = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uaik[isp][1],uaik[iFv][1],
                    mM,vabth,dtCFL[isp];is_boundaryv0=is_boundaryv0)
        else
            δtf[isp], dtCFL2 = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    naik[isp],naik[iFv],uaik[isp]./vthik[isp],
                    uaik[iFv]./vthik[iFv],vthik[isp],vthik[iFv],nModk[isp],nModk[iFv],
                    mM,vabth,dtCFL[isp];is_boundaryv0=is_boundaryv0)
        end
        if is_boundaryv0 == false
            for L1 in 1:LM1k
                if L1 == 1
                    δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                else
                    δtf[isp][1,L1] = 0.0
                end
            end
        end

        if is_lnA_const
            Zqab = Zq[isp] * Zq[iFv]
            lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nak[isp],nak[iFv],vathk[isp],vathk[iFv];is_normδtf=true)
        else
            lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                            [nak[isp],nak[iFv]],[vathk[isp],vathk[iFv]],isp,iFv;is_normδtf=true)
        end
        Ccol[isp] = CΓ * lnAg
        δtf[isp] *= Ccol[isp]   # CΓ is owing to the dimensionless process
    end
    
    if is_δtfvLaa === 1
        for isp in nsp_vec
            δtf[isp] += dtfvLaa[isp]
        end
    end
    return min(dtCFL[1],dtCFL[2])
end

# [nv,LM,ns = 2], nMod = 1
function FP0D2Vab2!(δtf::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},vhk::Vector{AbstractVector{T}},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak::AbstractVector{T},vathk::AbstractVector{T};ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,is_fit_f::Bool=false,
    is_extrapolate_FLn::Bool=true) where{T}
    
    mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1k - 1)

    nsp_vec = 1:ns
    FvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    ddfvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        FvL[isp] = zeros(T,nck[isp],LM1k)
        dfvL[isp] = zeros(T,nvG[isp],LM1k)
        ddfvL[isp] = zeros(T,nvG[isp],LM1k)
    end

    # Self-collisions.
    dtfvLaa = similar(δtf)

    # Computing the first-two order derivatives of `fvL` and it's mapping functions `FvL` when `nMod .== 1` and `uai[1] = - uai[2]`
    if is_extrapolate_FLn
        dtfvLaa,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    else
        ncF = zeros(Int64,ns)
        dtfvLaa,ddfvL,dfvL,FvL = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LMk,LM1k,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,nak,vathk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=true,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    end
    
    # Collision between different spices
    for isp in nsp_vec
        isp == 1 ? iFv = 2 : iFv = 1
        nvlevele = nvlevel0[isp][nvlevele0[isp]]
        mM = ma[isp] / ma[iFv]
        vabth = vathk[isp] / vathk[iFv]
        va = vhk[isp] * vabth     # 𝓋̂ = v̂a * vabth
        
        HvL,dHvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        GvL,dGvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
        ddGvL = zeros(T,nc0[isp],LM1k)
        if ncF[isp] ≥ 2
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv], 
                        FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
        else
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv])
        end
        δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                FvL[isp][nvlevele,:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                vhk[isp][nvlevele],va[nvlevele],
                mu,Mμ,Mun,Mun1,Mun2,LM1k,
                uaik[isp][1],uaik[iFv][1],
                mM,vabth;is_boundaryv0=is_boundaryv0)
        if is_boundaryv0 == false
            for L1 in 1:LM1k
                if L1 == 1
                    δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                else
                    δtf[isp][1,L1] = 0.0
                end
            end
        end

        if is_lnA_const
            Zqab = Zq[isp] * Zq[iFv]
            lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nak[isp],nak[iFv],vathk[isp],vathk[iFv];is_normδtf=true)
        else
            lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                            [nak[isp],nak[iFv]],[vathk[isp],vathk[iFv]],isp,iFv;is_normδtf=true)
        end
        # @show 2,isp,(2,1),spices[isp], lnAg
        δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
    end
    wedfgbhn

    if is_δtfvLaa === 1
        for isp in nsp_vec
            δtf[isp] += dtfvLaa[isp]
        end
    end
end

"""
  Inputs:
  Outputs:
    FP0D2Vaa!
    FP0D2Vab2!(δtf,fvL0k,vhk,nvG,nc0,nck,ocp,
            nvlevele0,nvlevel0,LMk,LM1k,
            CΓ,εᵣ,ma,Zq,spices,nk,uhkL,vthk;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs, 
            is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
            is_extrapolate_FLn=is_extrapolate_FLn)
"""

# [nv,LM,ns=2], nMod = 1 in `_Ms` version, where `uai[1] = - uai[2]`, `nai = vthi = 1` in the Lagrange coordinate system with relative velocity `uCk`
function FP0D2Vab2!(δtf::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},vhk::Vector{AbstractVector{T}},
    nvG::Vector{Int64},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},LMk::Vector{Int64},LM1k::Int64,
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nk::AbstractVector{T},uhkL::AbstractVector{T},vthk::AbstractVector{T};ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    is_boundaryv0::Bool=false,is_fit_f::Bool=false,is_extrapolate_FLn::Bool=true) where{T}
    
    mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1k - 1)

    nsp_vec = 1:ns
    FvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    ddfvL = Vector{Matrix{T}}(undef,ns)
    nvlevele = Vector{Vector{Int64}}(undef,ns)
    for isp in nsp_vec
        FvL[isp] = zeros(T,nck[isp],LM1k)
        dfvL[isp] = zeros(T,nvG[isp],LM1k)
        ddfvL[isp] = zeros(T,nvG[isp],LM1k)
        nvlevele[isp] = nvlevel0[isp][nvlevele0[isp]]
    end

    # Self-collisions is zero-effects

    # Collision between different spices
    if is_extrapolate_FLn
        ncF, vaa = zeros(Int, ns), Vector{AbstractVector{T}}(undef,ns)
        nvlevel0a = Vector{Vector{Int64}}(undef, ns)
        FvLa = Array{Any}(undef, LM1k, ns)
        FfvLCSLag!(ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,ncF,fvL0k,
                vhk,nc0,ocp,nvlevele,vthk,uhkL;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
        for isp in nsp_vec
            isp == 1 ? iFv = 2 : iFv = 1
            mM = ma[isp] / ma[iFv]
            vabth = vthk[isp] / vthk[iFv]
            va = vhk[isp] * vabth     # 𝓋̂ = v̂a * vabth
            
            HvL,dHvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
            GvL,dGvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
            ddGvL = zeros(T,nc0[isp],LM1k)
            if ncF[isp] ≥ 2
                dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                            FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv],uhkL[iFv], 
                            FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
            else
                dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                            FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv],uhkL[iFv])
            end
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele[isp],:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele[isp]],va[nvlevele[isp]],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uhkL[isp][1],uhkL[iFv][1],
                    mM,vabth;is_boundaryv0=is_boundaryv0)
            if is_boundaryv0 == false
                for L1 in 1:LM1k
                    if L1 == 1
                        δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                    else
                        δtf[isp][1,L1] = 0.0
                    end
                end
            end
            Zqab = Zq[isp] * Zq[iFv]
            # dtfLn = dtfMab(εᵣ,ma[isp],mM,Zqab,spices,nk[iFv],vthk[isp],vthk[iFv],vhk[isp][nvlevele[isp]])
            # @show 1, dtfLn - δtf[isp][:,1]

            if is_lnA_const
                lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nk[isp],nk[iFv],vthk[isp],vthk[iFv];is_normδtf=true)
            else
                lnAg = lnAgamma_fM([ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                                [nk[isp],nk[iFv]],[vthk[isp],vthk[iFv]],isp,iFv;is_normδtf=true)
            end
            # @show 3,isp,(2,1),spices[isp], lnAg
            δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
        end
        sum(ncF) == 0 || @show ncF
    else
        FfvLCSLag!(ddfvL,dfvL,FvL,fvL0k,
                vhk,nc0,nvlevele,vthk,uhkL;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
        for isp in nsp_vec
            isp == 1 ? iFv = 2 : iFv = 1
            mM = ma[isp] / ma[iFv]
            vabth = vthk[isp] / vthk[iFv]
            va = vhk[isp] * vabth     # 𝓋̂ = v̂a * vabth
            
            HvL,dHvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
            GvL,dGvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
            ddGvL = zeros(T,nc0[isp],LM1k)
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv],uhkL[iFv])
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele[isp],:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele[isp]],va[nvlevele[isp]],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uhkL[isp][1],uhkL[iFv][1],
                    mM,vabth;is_boundaryv0=is_boundaryv0)
            if is_boundaryv0 == false
                for L1 in 1:LM1k
                    if L1 == 1
                        δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                    else
                        δtf[isp][1,L1] = 0.0
                    end
                end
            end
            Zqab = Zq[isp] * Zq[iFv]
            # dtfLn = dtfMab(ma[isp],mM,Zqab,nk[iFv],vthk[isp],vthk[iFv],vhk[isp][nvlevele[isp]], spices,εᵣ)
            dtfLn = dtfMab(mM,vabth,vhk[isp][nvlevele[isp]])

            if is_lnA_const
                lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nk[isp],nk[iFv],vthk[isp],vthk[iFv];is_normδtf=true)
            else
                lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                                [nk[isp],nk[iFv]],[vthk[isp],vthk[iFv]],isp,iFv;is_normδtf=true)
            end
            δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
        end
    end
end
# Ccol
function FP0D2Vab2!(δtf::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},vhk::Vector{AbstractVector{T}},
    nvG::Vector{Int64},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},LMk::Vector{Int64},LM1k::Int64,
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},Ccol::AbstractVector{T},
    nk::AbstractVector{T},uhkL::AbstractVector{T},vthk::AbstractVector{T};ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    is_boundaryv0::Bool=false,is_fit_f::Bool=false,is_extrapolate_FLn::Bool=true,
    is_plot_dfln_thoery::Bool=false) where{T}
    
    mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1k - 1)

    nsp_vec = 1:ns
    FvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    ddfvL = Vector{Matrix{T}}(undef,ns)
    nvlevele = Vector{Vector{Int64}}(undef,ns)
    for isp in nsp_vec
        FvL[isp] = zeros(T,nck[isp],LM1k)
        dfvL[isp] = zeros(T,nvG[isp],LM1k)
        ddfvL[isp] = zeros(T,nvG[isp],LM1k)
        nvlevele[isp] = nvlevel0[isp][nvlevele0[isp]]
    end

    # Self-collisions is zero-effects

    # Collision between different spices
    dtCFL = [1e10,1e10]
    Cf3 = nk ./ vthk.^3
    if is_extrapolate_FLn
        ncF, vaa = zeros(Int, ns), Vector{AbstractVector{T}}(undef,ns)
        nvlevel0a = Vector{Vector{Int64}}(undef, ns)
        FvLa = Array{Any}(undef, LM1k, ns)
        FfvLCSLag!(ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,ncF,fvL0k,
                vhk,nc0,ocp,nvlevele,vthk,uhkL;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
        for isp in nsp_vec
            isp == 1 ? iFv = 2 : iFv = 1
            mM = ma[isp] / ma[iFv]
            vabth = vthk[isp] / vthk[iFv]
            va = vhk[isp] * vabth     # 𝓋̂ = v̂a * vabth
            
            HvL,dHvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
            GvL,dGvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
            ddGvL = zeros(T,nc0[isp],LM1k)
            if ncF[isp] ≥ 2
                dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                            FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv],uhkL[iFv], 
                            FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
            else
                dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                            FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv],uhkL[iFv])
            end
            δtf[isp], dtCFL2 = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele[isp],:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele[isp]],va[nvlevele[isp]],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uhkL[isp][1],uhkL[iFv][1],
                    mM,vabth,dtCFL[isp];is_boundaryv0=is_boundaryv0)
            if is_boundaryv0 == false
                for L1 in 1:LM1k
                    if L1 == 1
                        δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                    else
                        δtf[isp][1,L1] = 0.0
                    end
                end
            end
            Zqab = Zq[isp] * Zq[iFv]
            # dtfLn = dtfMab(εᵣ,ma[isp],mM,Zqab,spices,nk[iFv],vthk[isp],vthk[iFv],vhk[isp][nvlevele[isp]])
            # @show 3, dtfLn - δtf[isp][:,1]

            if is_lnA_const
                lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nk[isp],nk[iFv],vthk[isp],vthk[iFv];is_normδtf=true)
            else
                lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                                [nk[isp],nk[iFv]],[vthk[isp],vthk[iFv]],isp,iFv;is_normδtf=true)
            end
            Ccol[isp] = CΓ * lnAg
            # @show Ccol[isp]
            dtCFL2 *= Ccol[isp] * Cf3[isp]
            dtCFL2 /=  Cf3[isp]
            # @show dtCFL2
            dtCFL[isp] = min(dtCFL[isp], dtCFL2)
            δtf[isp] *= Ccol[isp]   # CΓ is owing to the dimensionless process
        end
        sum(ncF) == 0 || @show ncF
    else
        FfvLCSLag!(ddfvL,dfvL,FvL,fvL0k,
                vhk,nc0,nvlevele,vthk,uhkL;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
        for isp in nsp_vec
            isp == 1 ? iFv = 2 : iFv = 1
            # @show isp
            mM = ma[isp] / ma[iFv]
            vabth = vthk[isp] / vthk[iFv]
            va = vhk[isp] * vabth     # 𝓋̂ = v̂a * vabth
            
            HvL,dHvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
            GvL,dGvL = zeros(T,nc0[isp],LM1k),zeros(T,nc0[isp],LM1k)
            ddGvL = zeros(T,nc0[isp],LM1k)
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LMk[iFv],uhkL[iFv])
            δtf[isp], dtCFL2 = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele[isp],:],dHvL[nvlevele0[isp],:],HvL[nvlevele0[isp],:],
                    ddGvL[nvlevele0[isp],:],dGvL[nvlevele0[isp],:],GvL[nvlevele0[isp],:],
                    vhk[isp][nvlevele[isp]],va[nvlevele[isp]],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uhkL[isp][1],uhkL[iFv][1],
                    mM,vabth,dtCFL[isp];is_boundaryv0=is_boundaryv0)
            if is_boundaryv0 == false
                for L1 in 1:LM1k
                    if L1 == 1
                        δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                    else
                        δtf[isp][1,L1] = 0.0
                    end
                end
            end
            Zqab = Zq[isp] * Zq[iFv]
            # dtfLn = dtfMab(εᵣ,ma,mM,Zqab,spices,nb,vath,vbth,vhk[isp][nvlevele[isp]])
            # dtfLn = dtfMab(εᵣ,ma[isp],mM,Zqab,spices,nk[iFv],vthk[isp],vthk[iFv],vhk[isp][nvlevele[isp]],)

            # dtfLn = dtfMab(mM,vabth,vhk[isp][nvlevele[isp]])
            # ivhn = 57
            # @show isp, dtfLn[1:3]
            # @show δtf[isp][1:3,1]
            # a = (4 / pi^2 * nk[iFv] / vthk[iFv]^3) * dtfLn[1:ivhn] ./ δtf[isp][1:ivhn,1]
            # @show 41, a[2:6]
            # rtghn
            # if is_plot_dfln_thoery
            #     if isp == 1
            #         label = string("a,nnv=",nnv[isp])
            #         display(plot(a[2:end] .- a[8],label=label))
            #     else
            #         label = string("b,nnv=",nnv[isp])
            #         display(plot(a[2:end] .- a[8],label=label))
            #     end
            # end

            if is_lnA_const
                lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nk[isp],nk[iFv],vthk[isp],vthk[iFv];is_normδtf=true)
            else
                lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                                [nk[isp],nk[iFv]],[vthk[isp],vthk[iFv]],isp,iFv;is_normδtf=true)
            end
            Ccol[isp] = CΓ * lnAg
            # @show Ccol[isp]
            dtCFL2 *= Ccol[isp] * Cf3[isp]
            dtCFL2 /=  Cf3[isp]
            # @show dtCFL2
            dtCFL[isp] = min(dtCFL[isp], dtCFL2)
            δtf[isp] *= Ccol[isp]   # CΓ is owing to the dimensionless process
        end
        # @show dtCFL
    end
    return min(dtCFL[1],dtCFL[2])
end

# [ns=2], nMod = 1 in `_IK` version 
function FP0D2Vab2!(δtf::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},
    vhk::AbstractVector{T},nvG::Int64,nc0::Int64,nck::Int64,ocp::Int64,
    nvlevele0::Vector{Int64},nvlevel0::Vector{Int64},LMk::Vector{Int64},LM1k::Int64,
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nk::AbstractVector{T},uhkL::AbstractVector{T},vthk::AbstractVector{T};ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    is_boundaryv0::Bool=false,is_fit_f::Bool=false,is_extrapolate_FLn::Bool=true) where{T}
    
    mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1k - 1)
    nvlevele = nvlevel0[nvlevele0]

    nsp_vec = 1:ns
    FvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    ddfvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        FvL[isp] = zeros(T,nck,LM1k)
        dfvL[isp] = zeros(T,nvG,LM1k)
        ddfvL[isp] = zeros(T,nvG,LM1k)
    end

    # Self-collisions is zero-effects

    # Collision between different spices
    if is_extrapolate_FLn
        ncF, vaa = zeros(Int, ns), Vector{AbstractVector{T}}(undef,ns)
        nvlevel0a = Vector{Vector{Int64}}(undef, ns)
        FvLa = Array{Any}(undef, LM1k, ns)
        FfvLCSLag!(ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,ncF,fvL0k,
                vhk,nc0,ocp,nvlevele,vthk,uhkL;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
        for isp in nsp_vec
            isp == 1 ? iFv = 2 : iFv = 1
            mM = ma[isp] / ma[iFv]
            vabth = vthk[isp] / vthk[iFv]
            va = vhk * vabth     # 𝓋̂ = v̂a * vabth
            
            HvL,dHvL = zeros(T,nc0,LM1k),zeros(T,nc0,LM1k)
            GvL,dGvL = zeros(T,nc0,LM1k),zeros(T,nc0,LM1k)
            ddGvL = zeros(T,nc0,LM1k)
            if ncF[isp] ≥ 2
                dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                            FvL[isp],va,nvlevel0,nc0,nck,ocp,LMk[iFv],uhkL[iFv], 
                            FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
            else
                dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                            FvL[isp],va,nvlevel0,nc0,nck,ocp,LMk[iFv],uhkL[iFv])
            end
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0,:],HvL[nvlevele0,:],
                    ddGvL[nvlevele0,:],dGvL[nvlevele0,:],GvL[nvlevele0,:],
                    vhk[nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uhkL[isp][1],uhkL[iFv][1],
                    mM,vabth;is_boundaryv0=is_boundaryv0)
            if is_boundaryv0 == false
                for L1 in 1:LM1k
                    if L1 == 1
                        δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                    else
                        δtf[isp][1,L1] = 0.0
                    end
                end
            end

            if is_lnA_const
                Zqab = Zq[isp] * Zq[iFv]
                lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nk[isp],nk[iFv],vthk[isp],vthk[iFv];is_normδtf=true)
            else
                lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                                [nk[isp],nk[iFv]],[vthk[isp],vthk[iFv]],isp,iFv;is_normδtf=true)
            end
            δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
        end
        sum(ncF) == 0 || @show ncF
    else
        FfvLCSLag!(ddfvL,dfvL,FvL,fvL0k,
                vhk,nc0,nvlevele,vthk,uhkL;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
        for isp in nsp_vec
            isp == 1 ? iFv = 2 : iFv = 1
            mM = ma[isp] / ma[iFv]
            vabth = vthk[isp] / vthk[iFv]
            va = vhk * vabth     # 𝓋̂ = v̂a * vabth
            
            HvL,dHvL = zeros(T,nc0,LM1k),zeros(T,nc0,LM1k)
            GvL,dGvL = zeros(T,nc0,LM1k),zeros(T,nc0,LM1k)
            ddGvL = zeros(T,nc0,LM1k)
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0,nc0,nck,ocp,LMk[iFv],uhkL[iFv])
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0,:],HvL[nvlevele0,:],
                    ddGvL[nvlevele0,:],dGvL[nvlevele0,:],GvL[nvlevele0,:],
                    vhk[nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uhkL[isp][1],uhkL[iFv][1],
                    mM,vabth;is_boundaryv0=is_boundaryv0)
            if is_boundaryv0 == false
                for L1 in 1:LM1k
                    if L1 == 1
                        δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                    else
                        δtf[isp][1,L1] = 0.0
                    end
                end
            end

            if is_lnA_const
                Zqab = Zq[isp] * Zq[iFv]
                lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],nk[isp],nk[iFv],vthk[isp],vthk[iFv];is_normδtf=true)
            else
                lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                                [nk[isp],nk[iFv]],[vthk[isp],vthk[iFv]],isp,iFv;is_normδtf=true)
            end
            δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
        end
    end
end

# [nMod=2], ns = 1 in `_IK` version, For self-collision process when `ma[1] = ma[2]` and `Zq[1] = Zq[2]` but different `uai`
function FP0D2Vaa2!(δtf::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},
    vhk::AbstractVector{T},nvG::Int64,nc0::Int64,nck::Int64,ocp::Int64,
    nvlevele0::Vector{Int64},nvlevel0::Vector{Int64},LMk::Vector{Int64},LM1k::Int64,
    CΓ::T,εᵣ::T,ma::T,Zq::Int64,spices::Symbol,
    nk::AbstractVector{T},uhkL::AbstractVector{T},vthk::AbstractVector{T};ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    is_boundaryv0::Bool=false,is_fit_f::Bool=false,
    is_extrapolate_FLn::Bool=true) where{T}
    
    mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1k - 1)
    nvlevele = nvlevel0[nvlevele0]

    nsp_vec = 1:ns
    FvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    ddfvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        FvL[isp] = zeros(T,nck,LM1k)
        dfvL[isp] = zeros(T,nvG,LM1k)
        ddfvL[isp] = zeros(T,nvG,LM1k)
    end

    # Self-collisions is zero-effects

    # Collision between different spices
    if is_extrapolate_FLn
        ncF, vaa = zeros(Int, ns), Vector{AbstractVector{T}}(undef,ns)
        nvlevel0a = Vector{Vector{Int64}}(undef, ns)
        FvLa = Array{Any}(undef, LM1k, ns)
        FfvLCSLag!(ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,ncF,fvL0k,
                vhk,nc0,ocp,nvlevele,vthk,uhkL;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
        for isp in nsp_vec
            isp == 1 ? iFv = 2 : iFv = 1
            vabth = vthk[isp] / vthk[iFv]
            va = vhk * vabth     # 𝓋̂ = v̂a * vabth

            HvL,dHvL = zeros(T,nc0,LM1k),zeros(T,nc0,LM1k)
            GvL,dGvL = zeros(T,nc0,LM1k),zeros(T,nc0,LM1k)
            ddGvL = zeros(T,nc0,LM1k)
            if ncF[isp] ≥ 2
                dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                            FvL[isp],va,nvlevel0,nc0,nck,ocp,LMk[iFv],uhkL[iFv], 
                            FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
            else
                dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                            FvL[isp],va,nvlevel0,nc0,nck,ocp,LMk[iFv],uhkL[iFv])
            end
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0,:],HvL[nvlevele0,:],
                    ddGvL[nvlevele0,:],dGvL[nvlevele0,:],GvL[nvlevele0,:],
                    vhk[nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uhkL[isp][1],uhkL[iFv][1],
                    1.0,vabth;is_boundaryv0=is_boundaryv0)
            if is_boundaryv0 == false
                for L1 in 1:LM1k
                    if L1 == 1
                        δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                    else
                        δtf[isp][1,L1] = 0.0
                    end
                end
            end

            if is_lnA_const
                lnAg = lnAgamma(εᵣ,ma,Zq^2,[spices[isp],spices[iFv]],nk[isp],nk[iFv],vthk[isp],vthk[iFv];is_normδtf=true)
            else
                lnAg = lnAgamma_fM(ma,Zq,nk,vthk,spices,εᵣ,isp,iFv;is_normδtf=true)
                # lnAg = lnAgamma_fM(εᵣ,[ma,ma],[Zq,Zq],[spices,spices],
                #                   [nk[isp],nk[iFv]],[vthk[isp],vthk[iFv]],isp,iFv;is_normδtf=true)
            end
            δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
        end
        sum(ncF) == 0 || @show ncF
    else
        FfvLCSLag!(ddfvL,dfvL,FvL,fvL0k,
                vhk,nc0,nvlevele,vthk,uhkL;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_fit_f=is_fit_f)
        for isp in nsp_vec
            isp == 1 ? iFv = 2 : iFv = 1
            vabth = vthk[isp] / vthk[iFv]
            va = vhk * vabth     # 𝓋̂ = v̂a * vabth

            HvL,dHvL = zeros(T,nc0,LM1k),zeros(T,nc0,LM1k)
            GvL,dGvL = zeros(T,nc0,LM1k),zeros(T,nc0,LM1k)
            ddGvL = zeros(T,nc0,LM1k)
            dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                        FvL[isp],va,nvlevel0,nc0,nck,ocp,LMk[iFv],uhkL[iFv])
            δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                    FvL[isp][nvlevele,:],dHvL[nvlevele0,:],HvL[nvlevele0,:],
                    ddGvL[nvlevele0,:],dGvL[nvlevele0,:],GvL[nvlevele0,:],
                    vhk[nvlevele],va[nvlevele],
                    mu,Mμ,Mun,Mun1,Mun2,LM1k,
                    uhkL[isp][1],uhkL[iFv][1],
                    1.0,vabth;is_boundaryv0=is_boundaryv0)
            if is_boundaryv0 == false
                for L1 in 1:LM1k
                    if L1 == 1
                        δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                    else
                        δtf[isp][1,L1] = 0.0
                    end
                end
            end

            if is_lnA_const
                lnAg = lnAgamma(εᵣ,ma,Zq^2,[spices[isp],spices[iFv]],nk[isp],nk[iFv],vthk[isp],vthk[iFv];is_normδtf=true)
            else
                lnAg = lnAgamma_fM(ma,Zq,nk,vthk,spices,εᵣ,isp,iFv;is_normδtf=true)
                # lnAg = lnAgamma_fM(εᵣ,[ma,ma],[Zq,Zq],[spices,spices],
                #                   [nk[isp],nk[iFv]],[vthk[isp],vthk[iFv]],isp,iFv;is_normδtf=true)
            end
            δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
        end
    end
end

