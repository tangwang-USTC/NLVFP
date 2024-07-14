"""
  Inputs:
    dtnIKs = zeros(4,2)
    Rdtsaa: To decide whether reaches a new bifurcation point according to the 第二类独立特征可分辨判据。

  Outputs:
    dtk = dtIKab!(dtIKs,Rdtsaa,sk1,dtnIKs,edtnIKs,
           nk2,uk2,vthk2,spices2,nvG,ocp,vGdom,LMk,
           CΓ,εᵣ,ma,Zq,spices,nk,uk,vthk,nMod,DThk,dtk;
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
           is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
           eps_fup=eps_fup,eps_flow=eps_flow,
           maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
           abstol=abstol,reltol=reltol,
           vadaptlevels=vadaptlevels,gridv_type=gridv_type,
           is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
           
           is_check_conservation_dtM=is_check_conservation_dtM,
           is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
           is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
"""

# [prod(nMod)≥2,ns≥3], [nk, uk, vthk]

function dtIKab!(dtIKs::AbstractArray{T,N},Rdtsaa::Vector{TA2},sk::AbstractArray{T,N2},
    dtnIKs::AbstractArray{T,N2},edtnIKs::AbstractArray{T,N2},
    nk2::Vector{AbstractVector{T}},uk2::Vector{AbstractVector{T}},vthk2::Vector{AbstractVector{T}},spices2::Vector{Symbol},
    nvG::Int64,ocp::Int64,vGdom::AbstractVector{T},LMk::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nk::Vector{AbstractVector{T}},uk::Vector{AbstractVector{T}},vthk::Vector{AbstractVector{T}},
    ns::Int64,nMod::Vector{Int64},DThk::AbstractVector{T},dtk::T;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,N,TA2,N2}
    
    
    vec = [1,2]
    for isp in 1:ns-1
        vec[1] = isp
        nk2[1], uk2[1], vthk2[1], spices2[1] = nk[isp], uk[isp], vthk[isp], spices[isp]
        for iFv in isp+1:ns
            vec[2] = iFv
            nk2[2], uk2[2], vthk2[2], spices2[2] = nk[iFv], uk[iFv], vthk[iFv], spices[iFv]

            dtk = dtIKab!(dtIKs,Rdtsaa,sk,dtnIKs,edtnIKs,nvG,ocp,vGdom,LMk,
                   CΓ,εᵣ,ma,Zq,spices2,nk2,uk2,vthk2,nMod,DThk,dtk;
                   is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                   autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                   p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                   rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                   is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                   eps_fup=eps_fup,eps_flow=eps_flow,
                   maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                   abstol=abstol,reltol=reltol,
                   vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                   is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                   
                   is_check_conservation_dtM=is_check_conservation_dtM,
                   is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                   is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                   
            Rc[:,:,isp] += Rc2[:,:,1]
            edtnIKTs[:,isp] += edtnIKTs2[:,1]
            DThk[isp] += DThk2[1]
            Rc[:,:,iFv] += Rc2[:,:,2]
            edtnIKTs[:,iFv] += edtnIKTs2[:,2]
            DThk[iFv] += DThk2[2]
        end
    end

    return dtk
end

# [prod(nMod)≥2,ns=2], [nk, uk, vthk]
function dtIKab!(dtIKs::AbstractArray{T,N},Rdtsaa::Vector{TA2},sk::AbstractArray{T,N2},
    dtnIKs::AbstractArray{T,N2},edtnIKs::AbstractArray{T,N2},
    nvG::Int64,ocp::Int64,vGdom::AbstractVector{T},LMk::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nk::Vector{AbstractVector{T}},uk::Vector{AbstractVector{T}},vthk::Vector{AbstractVector{T}},
    nMod::Vector{Int64},DThk::AbstractVector{T},dtk::T;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    is_self_add::Bool=true) where{T,N,TA2,N2}
    
    isp, iFv = 1, 2
    IkL, KkL = zeros(T,2), zeros(T,2)
    if prod(nMod) == 1
        ka = 1
        kb = 1
        nk2 = [nk[isp][ka], nk[iFv][kb]]
        uk2 = [uk[isp][ka], uk[iFv][kb]]
        vthk2 = [vthk[isp][ka], vthk[iFv][kb]]
        dtk = dtIKab!(dtnIKs,edtnIKs,nvG,ocp,vGdom,LMk,
               CΓ,εᵣ,ma,Zq,spices,nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
               is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
               eps_fup=eps_fup,eps_flow=eps_flow,
               maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
               abstol=abstol,reltol=reltol,
               vadaptlevels=vadaptlevels,gridv_type=gridv_type,
               is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
               
               is_check_conservation_dtM=is_check_conservation_dtM,
               is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
               is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
        dtIKs[ka,1,isp] = deepcopy(dtnIKs[3,isp])        # ∂ₜKa
        dtIKs[ka,2,isp] = deepcopy(dtnIKs[2,isp])        # ∂ₜIa
        dtIKs[kb,1,iFv] = deepcopy(dtnIKs[3,iFv])        # ∂ₜKb
        dtIKs[kb,2,iFv] = deepcopy(dtnIKs[2,iFv])        # ∂ₜIb
    elseif nMod[isp] == 1
        # Computing the `a-b` collision process
        ka = 1
        kb = 1
        nk2 = [nk[isp][ka], nk[iFv][kb]]
        uk2 = [uk[isp][ka], uk[iFv][kb]]
        vthk2 = [vthk[isp][ka], vthk[iFv][kb]]
        dtk = dtIKab!(dtnIKs,edtnIKs,nvG,ocp,vGdom,LMk,
               CΓ,εᵣ,ma,Zq,spices,nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
               is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
               eps_fup=eps_fup,eps_flow=eps_flow,
               maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
               abstol=abstol,reltol=reltol,
               vadaptlevels=vadaptlevels,gridv_type=gridv_type,
               is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
               
               is_check_conservation_dtM=is_check_conservation_dtM,
               is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
               is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
        dtIKs[ka,1,isp] = deepcopy(dtnIKs[3,isp])        # ∂ₜKa
        dtIKs[ka,2,isp] = deepcopy(dtnIKs[2,isp])        # ∂ₜIa
        dtIKs[kb,1,iFv] = deepcopy(dtnIKs[3,iFv])        # ∂ₜKb
        dtIKs[kb,2,iFv] = deepcopy(dtnIKs[2,iFv])        # ∂ₜIb

        edtnIKs2 = deepcopy(edtnIKs)
        for kb in 2:nMod[iFv]
            nk2[2] = deepcopy(nk[iFv][kb])
            uk2[2] = deepcopy(uk[iFv][kb])
            vthk2[2] = deepcopy(vthk[iFv][kb])
            dtk = dtIKab!(dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                   CΓ,εᵣ,ma,Zq,spices,nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                   is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                   autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                   p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                   rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                   is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                   eps_fup=eps_fup,eps_flow=eps_flow,
                   maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                   abstol=abstol,reltol=reltol,
                   vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                   is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                   
                   is_check_conservation_dtM=is_check_conservation_dtM,
                   is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                   is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
            dtIKs[ka,1,isp] += dtnIKs[3,isp]
            dtIKs[ka,2,isp] += dtnIKs[2,isp]
            dtIKs[kb,1,iFv] = deepcopy(dtnIKs[3,iFv])
            dtIKs[kb,2,iFv] = deepcopy(dtnIKs[2,iFv])
            edtnIKs += edtnIKs2
        end
        edtnIKs /= nMod[iFv]

        # Computing the self-collision process for `iFv` spice
        if is_self_add
            if nMod[iFv] == 2
                ka = 1
                kb = 2
                nk2 = [nk[iFv][ka], nk[iFv][kb]]
                uk2 = [uk[iFv][ka], uk[iFv][kb]]
                vthk2 = [vthk[iFv][ka], vthk[iFv][kb]]
                sk2 = sk[ka,iFv] + sk[kb,iFv]
                dtk, Rdtsaa[iFv][1] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[iFv],Zq[iFv],spices[iFv],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,iFv] += dtnIKs[3,isp]        # ∂ₜKb1
                dtIKs[ka,2,iFv] += dtnIKs[2,isp]        # ∂ₜIb1
                dtIKs[kb,1,iFv] += dtnIKs[3,iFv]        # ∂ₜKb2
                dtIKs[kb,2,iFv] += dtnIKs[2,iFv]        # ∂ₜIb2
                edtnIKs += edtnIKs2
                edtnIKs /= nMod[iFv]
            elseif nMod[iFv] == 3
                ka = 1
                kb = 2
                nk2 = [nk[iFv][ka], nk[iFv][kb]]
                uk2 = [uk[iFv][ka], uk[iFv][kb]]
                vthk2 = [vthk[iFv][ka], vthk[iFv][kb]]
                sk2 = sk[ka,iFv] + sk[kb,iFv]
                dtk, Rdtsaa[iFv][1] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[iFv],Zq[iFv],spices[iFv],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,iFv] += dtnIKs[3,isp]        # ∂ₜKb1
                dtIKs[ka,2,iFv] += dtnIKs[2,isp]        # ∂ₜIb1
                dtIKs[kb,1,iFv] += dtnIKs[3,iFv]        # ∂ₜKb2
                dtIKs[kb,2,iFv] += dtnIKs[2,iFv]        # ∂ₜIb2
                edtnIKs += edtnIKs2

                ka = 1
                kb = 3
                nk2 = [nk[iFv][ka], nk[iFv][kb]]
                uk2 = [uk[iFv][ka], uk[iFv][kb]]
                vthk2 = [vthk[iFv][ka], vthk[iFv][kb]]
                sk2 = sk[ka,iFv] + sk[kb,iFv]
                dtk, Rdtsaa[iFv][2] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[iFv],Zq[iFv],spices[iFv],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,iFv] += dtnIKs[3,isp]        # ∂ₜKb1
                dtIKs[ka,2,iFv] += dtnIKs[2,isp]        # ∂ₜIb1
                dtIKs[kb,1,iFv] += dtnIKs[3,iFv]        # ∂ₜKb2
                dtIKs[kb,2,iFv] += dtnIKs[2,iFv]        # ∂ₜIb2
                edtnIKs += edtnIKs2

                ka = 2
                kb = 3
                nk2 = [nk[iFv][ka], nk[iFv][kb]]
                uk2 = [uk[iFv][ka], uk[iFv][kb]]
                vthk2 = [vthk[iFv][ka], vthk[iFv][kb]]
                sk2 = sk[ka,iFv] + sk[kb,iFv]
                dtk, Rdtsaa[iFv][3] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[iFv],Zq[iFv],spices[iFv],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,iFv] += dtnIKs[3,isp]        # ∂ₜKb1
                dtIKs[ka,2,iFv] += dtnIKs[2,isp]        # ∂ₜIb1
                dtIKs[kb,1,iFv] += dtnIKs[3,iFv]        # ∂ₜKb2
                dtIKs[kb,2,iFv] += dtnIKs[2,iFv]        # ∂ₜIb2
                edtnIKs += edtnIKs2
                edtnIKs /= nMod[iFv]
            else
                fdsdfg
            end
        end
    elseif nMod[iFv] == 1
        # Computing the `a-b` collision process
        ka = 1
        kb = 1
        nk2 = [nk[isp][ka], nk[iFv][kb]]
        uk2 = [uk[isp][ka], uk[iFv][kb]]
        vthk2 = [vthk[isp][ka], vthk[iFv][kb]]
        dtk = dtIKab!(dtnIKs,edtnIKs,nvG,ocp,vGdom,LMk,
               CΓ,εᵣ,ma,Zq,spices,nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
               is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
               eps_fup=eps_fup,eps_flow=eps_flow,
               maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
               abstol=abstol,reltol=reltol,
               vadaptlevels=vadaptlevels,gridv_type=gridv_type,
               is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
               
               is_check_conservation_dtM=is_check_conservation_dtM,
               is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
               is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
        dtIKs[ka,1,isp] = deepcopy(dtnIKs[3,isp])        # ∂ₜKa
        dtIKs[ka,2,isp] = deepcopy(dtnIKs[2,isp])        # ∂ₜIa
        dtIKs[kb,1,iFv] = deepcopy(dtnIKs[3,iFv])        # ∂ₜKb
        dtIKs[kb,2,iFv] = deepcopy(dtnIKs[2,iFv])        # ∂ₜIb
        edtnIKs2 = deepcopy(edtnIKs)

        for ka in 2:nMod[isp]
            nk2[1] = deepcopy(nk[isp][ka])
            uk2[1] = deepcopy(uk[isp][ka])
            vthk2[1] = deepcopy(vthk[isp][ka])
            dtk = dtIKab!(dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                   CΓ,εᵣ,ma,Zq,spices,nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                   is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                   autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                   p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                   rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                   is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                   eps_fup=eps_fup,eps_flow=eps_flow,
                   maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                   abstol=abstol,reltol=reltol,
                   vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                   is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                   
                   is_check_conservation_dtM=is_check_conservation_dtM,
                   is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                   is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
            dtIKs[ka,1,isp] = deepcopy(dtnIKs[3,isp])
            dtIKs[ka,2,isp] = deepcopy(dtnIKs[2,isp])
            dtIKs[kb,1,iFv] += dtnIKs[3,iFv]
            dtIKs[kb,2,iFv] += dtnIKs[2,iFv]
            edtnIKs += edtnIKs2
        end
        edtnIKs /= nMod[isp]

        # Computing the self-collision process for `isp` spice
        if is_self_add
            if nMod[isp] == 2
                ka = 1
                kb = 2
                nk2 = [nk[isp][ka], nk[isp][kb]]
                uk2 = [uk[isp][ka], uk[isp][kb]]
                vthk2 = [vthk[isp][ka], vthk[isp][kb]]
                sk2 = sk[ka,isp] + sk[kb,isp]
                dtk, Rdtsaa[isp][1] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[isp],Zq[isp],spices[isp],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,isp] += dtnIKs[3,1]        # ∂ₜKa1
                dtIKs[ka,2,isp] += dtnIKs[2,1]        # ∂ₜIa1
                dtIKs[kb,1,isp] += dtnIKs[3,2]        # ∂ₜKa2
                dtIKs[kb,2,isp] += dtnIKs[2,2]        # ∂ₜIa2
                edtnIKs += edtnIKs2
                edtnIKs /= nMod[isp]
            elseif nMod[isp] == 3
                ka = 1
                kb = 2
                nk2 = [nk[isp][ka], nk[isp][kb]]
                uk2 = [uk[isp][ka], uk[isp][kb]]
                vthk2 = [vthk[isp][ka], vthk[isp][kb]]
                sk2 = sk[ka,isp] + sk[kb,isp]
                dtk, Rdtsaa[isp][1] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[isp],Zq[isp],spices[isp],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,isp] += dtnIKs[3,isp]        # ∂ₜKa1
                dtIKs[ka,2,isp] += dtnIKs[2,isp]        # ∂ₜIa1
                dtIKs[kb,1,isp] += dtnIKs[3,iFv]        # ∂ₜKa2
                dtIKs[kb,2,isp] += dtnIKs[2,iFv]        # ∂ₜIa2
                edtnIKs += edtnIKs2

                ka = 1
                kb = 3
                nk2 = [nk[isp][ka], nk[isp][kb]]
                uk2 = [uk[isp][ka], uk[isp][kb]]
                vthk2 = [vthk[isp][ka], vthk[isp][kb]]
                sk2 = sk[ka,isp] + sk[kb,isp]
                dtk, Rdtsaa[isp][2] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[isp],Zq[isp],spices[isp],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,isp] += dtnIKs[3,isp]        # ∂ₜKa1
                dtIKs[ka,2,isp] += dtnIKs[2,isp]        # ∂ₜIa1
                dtIKs[kb,1,isp] += dtnIKs[3,iFv]        # ∂ₜKa2
                dtIKs[kb,2,isp] += dtnIKs[2,iFv]        # ∂ₜIa2
                edtnIKs += edtnIKs2

                ka = 2
                kb = 3
                nk2 = [nk[isp][ka], nk[isp][kb]]
                uk2 = [uk[isp][ka], uk[isp][kb]]
                vthk2 = [vthk[isp][ka], vthk[isp][kb]]
                sk2 = sk[ka,isp] + sk[kb,isp]
                dtk, Rdtsaa[isp][3] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[isp],Zq[isp],spices[isp],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,isp] += dtnIKs[3,isp]        # ∂ₜKa1
                dtIKs[ka,2,isp] += dtnIKs[2,isp]        # ∂ₜIa1
                dtIKs[kb,1,isp] += dtnIKs[3,iFv]        # ∂ₜKa2
                dtIKs[kb,2,isp] += dtnIKs[2,iFv]        # ∂ₜIa2
                edtnIKs += edtnIKs2
                edtnIKs /= nMod[isp]
            else
                fdsdfgssss
            end
        end
    else
        # Computing the `a-b` collision process
        ka = 1
        kb = 1
        nk2 = [nk[isp][ka], nk[iFv][kb]]
        uk2 = [uk[isp][ka], uk[iFv][kb]]
        vthk2 = [vthk[isp][ka], vthk[iFv][kb]]
        dtk = dtIKab!(dtnIKs,edtnIKs,nvG,ocp,vGdom,LMk,
               CΓ,εᵣ,ma,Zq,spices,nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
               is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
               eps_fup=eps_fup,eps_flow=eps_flow,
               maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
               abstol=abstol,reltol=reltol,
               vadaptlevels=vadaptlevels,gridv_type=gridv_type,
               is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
               
               is_check_conservation_dtM=is_check_conservation_dtM,
               is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
               is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
        dtIKs[ka,1,isp] = deepcopy(dtnIKs[3,isp])        # ∂ₜKa
        dtIKs[ka,2,isp] = deepcopy(dtnIKs[2,isp])        # ∂ₜIa
        dtIKs[kb,1,iFv] = deepcopy(dtnIKs[3,iFv])        # ∂ₜKb
        dtIKs[kb,2,iFv] = deepcopy(dtnIKs[2,iFv])        # ∂ₜIb
        edtnIKs2 = deepcopy(edtnIKs)

        for kb in 2:nMod[iFv]
            nk2[2] = deepcopy(nk[iFv][kb])
            uk2[2] = deepcopy(uk[iFv][kb])
            vthk2[2] = deepcopy(vthk[iFv][kb])
            dtk = dtIKab!(dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                   CΓ,εᵣ,ma,Zq,spices,nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                   is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                   autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                   p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                   rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                   is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                   eps_fup=eps_fup,eps_flow=eps_flow,
                   maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                   abstol=abstol,reltol=reltol,
                   vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                   is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                   
                   is_check_conservation_dtM=is_check_conservation_dtM,
                   is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                   is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
            dtIKs[ka,1,isp] += dtnIKs[3,isp]
            dtIKs[ka,2,isp] += dtnIKs[2,isp]
            dtIKs[kb,1,iFv] = deepcopy(dtnIKs[3,iFv])
            dtIKs[kb,2,iFv] = deepcopy(dtnIKs[2,iFv])
            edtnIKs += edtnIKs2
        end
        edtnIKs /= nMod[iFv]

        for ka in 2:nMod[isp]
            kb = 1
            nk2 = [nk[isp][ka], nk[iFv][kb]]
            uk2 = [uk[isp][ka], uk[iFv][kb]]
            vthk2 = [vthk[isp][ka], vthk[iFv][kb]]
            dtk = dtIKab!(dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                   CΓ,εᵣ,ma,Zq,spices,nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                   is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                   autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                   p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                   rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                   is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                   eps_fup=eps_fup,eps_flow=eps_flow,
                   maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                   abstol=abstol,reltol=reltol,
                   vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                   is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                   
                   is_check_conservation_dtM=is_check_conservation_dtM,
                   is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                   is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
            dtIKs[ka,1,isp] = deepcopy(dtnIKs[3,isp])        # ∂ₜKa
            dtIKs[ka,2,isp] = deepcopy(dtnIKs[2,isp])        # ∂ₜIa
            dtIKs[kb,1,iFv] += dtnIKs[3,iFv]                 # ∂ₜKb
            dtIKs[kb,2,iFv] += dtnIKs[2,iFv]                 # ∂ₜIb
            edtnIKs += edtnIKs2
    
            for kb in 2:nMod[iFv]
                nk2 = [nk[isp][ka], nk[iFv][kb]]
                uk2 = [uk[isp][ka], uk[iFv][kb]]
                vthk2 = [vthk[isp][ka], vthk[iFv][kb]]
                dtk = dtIKab!(dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma,Zq,spices,nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,isp] += dtnIKs[3,isp]
                dtIKs[ka,2,isp] += dtnIKs[2,isp]
                dtIKs[kb,1,iFv] += dtnIKs[3,iFv]
                dtIKs[kb,2,iFv] += dtnIKs[2,iFv]
                edtnIKs += edtnIKs2
            end
        end
        edtnIKs /= ((nMod[iFv] - 1) * (nMod[isp] - 1) + 1)

        
        # Computing the self-collision process for `isp` spice and `iFv` spice
        if is_self_add
            if nMod[isp] == 2
                ka = 1
                kb = 2
                nk2 = [nk[isp][ka], nk[isp][kb]]
                uk2 = [uk[isp][ka], uk[isp][kb]]
                vthk2 = [vthk[isp][ka], vthk[isp][kb]]
                sk2 = sk[ka,isp] + sk[kb,isp]
                dtk, Rdtsaa[isp][1] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[isp],Zq[isp],spices[isp],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,isp] += dtnIKs[3,1]        # ∂ₜKa1
                dtIKs[ka,2,isp] += dtnIKs[2,1]        # ∂ₜIa1
                dtIKs[kb,1,isp] += dtnIKs[3,2]        # ∂ₜKa2
                dtIKs[kb,2,isp] += dtnIKs[2,2]        # ∂ₜIa2
                edtnIKs += edtnIKs2
                edtnIKs /= nMod[isp]
            elseif nMod[isp] == 3
                ka = 1
                kb = 2
                nk2 = [nk[isp][ka], nk[isp][kb]]
                uk2 = [uk[isp][ka], uk[isp][kb]]
                vthk2 = [vthk[isp][ka], vthk[isp][kb]]
                sk2 = sk[ka,isp] + sk[kb,isp]
                dtk, Rdtsaa[isp][1] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[isp],Zq[isp],spices[isp],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,isp] += dtnIKs[3,isp]        # ∂ₜKa1
                dtIKs[ka,2,isp] += dtnIKs[2,isp]        # ∂ₜIa1
                dtIKs[kb,1,isp] += dtnIKs[3,iFv]        # ∂ₜKa2
                dtIKs[kb,2,isp] += dtnIKs[2,iFv]        # ∂ₜIa2
                edtnIKs += edtnIKs2

                ka = 1
                kb = 3
                nk2 = [nk[isp][ka], nk[isp][kb]]
                uk2 = [uk[isp][ka], uk[isp][kb]]
                vthk2 = [vthk[isp][ka], vthk[isp][kb]]
                sk2 = sk[ka,isp] + sk[kb,isp]
                dtk, Rdtsaa[isp][2] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[isp],Zq[isp],spices[isp],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,isp] += dtnIKs[3,isp]        # ∂ₜKa1
                dtIKs[ka,2,isp] += dtnIKs[2,isp]        # ∂ₜIa1
                dtIKs[kb,1,isp] += dtnIKs[3,iFv]        # ∂ₜKa2
                dtIKs[kb,2,isp] += dtnIKs[2,iFv]        # ∂ₜIa2
                edtnIKs += edtnIKs2

                ka = 2
                kb = 3
                nk2 = [nk[isp][ka], nk[isp][kb]]
                uk2 = [uk[isp][ka], uk[isp][kb]]
                vthk2 = [vthk[isp][ka], vthk[isp][kb]]
                sk2 = sk[ka,isp] + sk[kb,isp]
                dtk, Rdtsaa[isp][3] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[isp],Zq[isp],spices[isp],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,isp] += dtnIKs[3,isp]        # ∂ₜKa1
                dtIKs[ka,2,isp] += dtnIKs[2,isp]        # ∂ₜIa1
                dtIKs[kb,1,isp] += dtnIKs[3,iFv]        # ∂ₜKa2
                dtIKs[kb,2,isp] += dtnIKs[2,iFv]        # ∂ₜIa2
                edtnIKs += edtnIKs2
                edtnIKs /= nMod[isp]
            else
                fdsdfgssss
            end
            
            if nMod[iFv] == 2
                ka = 1
                kb = 2
                nk2 = [nk[iFv][ka], nk[iFv][kb]]
                uk2 = [uk[iFv][ka], uk[iFv][kb]]
                vthk2 = [vthk[iFv][ka], vthk[iFv][kb]]
                sk2 = sk[ka,iFv] + sk[kb,iFv]
                dtk, Rdtsaa[iFv][1] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[iFv],Zq[iFv],spices[iFv],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,iFv] += dtnIKs[3,isp]        # ∂ₜKb1
                dtIKs[ka,2,iFv] += dtnIKs[2,isp]        # ∂ₜIb1
                dtIKs[kb,1,iFv] += dtnIKs[3,iFv]        # ∂ₜKb2
                dtIKs[kb,2,iFv] += dtnIKs[2,iFv]        # ∂ₜIb2
                edtnIKs += edtnIKs2
                edtnIKs /= nMod[iFv]
            elseif nMod[iFv] == 3
                ka = 1
                kb = 2
                nk2 = [nk[iFv][ka], nk[iFv][kb]]
                uk2 = [uk[iFv][ka], uk[iFv][kb]]
                vthk2 = [vthk[iFv][ka], vthk[iFv][kb]]
                sk2 = sk[ka,iFv] + sk[kb,iFv]
                dtk, Rdtsaa[iFv][1] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[iFv],Zq[iFv],spices[iFv],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,iFv] += dtnIKs[3,isp]        # ∂ₜKb1
                dtIKs[ka,2,iFv] += dtnIKs[2,isp]        # ∂ₜIb1
                dtIKs[kb,1,iFv] += dtnIKs[3,iFv]        # ∂ₜKb2
                dtIKs[kb,2,iFv] += dtnIKs[2,iFv]        # ∂ₜIb2
                edtnIKs += edtnIKs2

                ka = 1
                kb = 3
                nk2 = [nk[iFv][ka], nk[iFv][kb]]
                uk2 = [uk[iFv][ka], uk[iFv][kb]]
                vthk2 = [vthk[iFv][ka], vthk[iFv][kb]]
                sk2 = sk[ka,iFv] + sk[kb,iFv]
                dtk, Rdtsaa[iFv][2] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[iFv],Zq[iFv],spices[iFv],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,iFv] += dtnIKs[3,isp]        # ∂ₜKb1
                dtIKs[ka,2,iFv] += dtnIKs[2,isp]        # ∂ₜIb1
                dtIKs[kb,1,iFv] += dtnIKs[3,iFv]        # ∂ₜKb2
                dtIKs[kb,2,iFv] += dtnIKs[2,iFv]        # ∂ₜIb2
                edtnIKs += edtnIKs2

                ka = 2
                kb = 3
                nk2 = [nk[iFv][ka], nk[iFv][kb]]
                uk2 = [uk[iFv][ka], uk[iFv][kb]]
                vthk2 = [vthk[iFv][ka], vthk[iFv][kb]]
                sk2 = sk[ka,iFv] + sk[kb,iFv]
                dtk, Rdtsaa[iFv][3] = dtIKaa!(sk2,dtnIKs,edtnIKs2,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,ma[iFv],Zq[iFv],spices[iFv],nk2,uk2,vthk2,DThk,IkL,KkL,dtk;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                       
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                dtIKs[ka,1,iFv] += dtnIKs[3,isp]        # ∂ₜKb1
                dtIKs[ka,2,iFv] += dtnIKs[2,isp]        # ∂ₜIb1
                dtIKs[kb,1,iFv] += dtnIKs[3,iFv]        # ∂ₜKb2
                dtIKs[kb,2,iFv] += dtnIKs[2,iFv]        # ∂ₜIb2
                edtnIKs += edtnIKs2
                edtnIKs /= nMod[iFv]
            else
                fdsdfg
            end
        end
    end
    return dtk
end


"""
  Inputs:
    dtnIKs = zeros(4,2)

  Outputs:
    dtk = dtIKab!(dtnIKs,edtnIKs,nvG,ocp,vGdom,LM,
           CΓ,εᵣ,ma,Zq,spices,nk,uk,vthk,DThk,IkL,KkL,dtk;
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
           is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
           eps_fup=eps_fup,eps_flow=eps_flow,
           maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
           abstol=abstol,reltol=reltol,
           vadaptlevels=vadaptlevels,gridv_type=gridv_type,
           is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
           
           is_check_conservation_dtM=is_check_conservation_dtM,
           is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
           is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
"""
# dtIK[nki, uki, vthki]

# [ns=2], nMod = 1 where `uai[1] = - uai[2]`, `nai = vthi = 1` in the Lagrange coordinate system with relative velocity `uCk`
function dtIKab!(dtnIKs::AbstractArray{T,2},edtnIKs::AbstractArray{T,N2},
    nvG::Int64,ocp::Int64,vGdom::AbstractVector{T},LMk::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nk::AbstractVector{T},uk::AbstractVector{T},vthk::AbstractVector{T},
    DThk::AbstractVector{T},IkL::AbstractVector{T}, KkL::AbstractVector{T},dtk::T;ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,N2}
    
    # Computing the relative velocity `uCk` which satisfing `uai = - ubi` in the Lagrange coordinate system with `uCk`
    uCk = uCab(uk, vthk)
    ukL = uk .- uCk
    uhkL = ukL ./ vthk
    sum(uhkL) ≤ epsT1000 || @warn("ab: The relative velocity of the Lagrange coordinate system `uCk` is not optimezed,",sum(uhkL)) 
    # @show 2, fmtf4.([sum_kbn(uhkL) / (uhkL[1]), uhkL[1]])

    # if abs(uhkL[1]) ≤ epsn8
    #     # @warn("`FM` model maybe a better approximation when,",uhkL[1]) 
    #     # if abs(uhkL[1]) ≤ epsn10
    #     #     @warn("`FM` model is be proposed") 
    #     # end
    # end

    # Updating the meshgrids on the velocity axis when `ns == 2, nMod .== 1` which means `nai = vthi .== 1`
    nvG, nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0 = vHadapt1D(
        nvG, ocp, vGdom, abs(uhkL[1]);
        eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
        abstol=abstol,reltol=reltol,
        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
        
    # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
    nsp_vec = 1:ns
    # `L_limit` denotes an extra row is given which may be used.
    fvL0k = Vector{Matrix{T}}(undef,ns) 
    for isp in nsp_vec
        fvL0k[isp] = zeros(nvG,L_limit+1)
    end

    LM1k = fvLDMz!(fvL0k, vhe, LMk, uhkL; 
        L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)

    # Checking the conservation laws of the renormalized distribution function `fvL0k1`
    nIKTh = zeros(4, ns)
    if gridv_type == :uniform
        nIKThs!(nIKTh, fvL0k, vhe, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
    elseif gridv_type == :chebyshev
        nIKThs!(nIKTh, fvL0k, vhe, nvG, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
    else
        sdfgh
    end
    DThk[:] = nIKTh[4, :]
    if maximum(DThk) > atol_IKTh
        @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a = 3/2 * T̂a + ûa²`", DThk)
        if maximum(DThk) > rtol_IKTh 
            printstyled("`errTh < rtol_IKTh` which means the convergence of the algorithm is falure!",color=:red,"\n")
        end
    end
  
    # # Updating the FP collision terms according to the `FPS` operators.
    δtf = Vector{Matrix{T}}(undef,ns)     # δtfvLa
    for isp in nsp_vec
        δtf[isp] = zeros(T,nvG,LM1k)
    end

    # Self-collisions: zero-effects
    
    # Computing the first-two order derivatives of `fvL` and it's mapping functions `FvL` when `nMod .== 1` and `uai[1] = - uai[2]`
    FP0D2Vab2!(δtf,fvL0k,vhk,nvG,nc0,nck,ocp,
            nvlevele0,nvlevel0,LMk,LM1k,
            CΓ,εᵣ,ma,Zq,spices,nk,uhkL,vthk;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs, 
            is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
            is_extrapolate_FLn=is_extrapolate_FLn)
            
    # Verifying the mass, momentum and total energy conservation laws of `δtfa` in Lagrange coordinate system.
    if gridv_type == :uniform
        RdtnIKTs!(dtnIKs,edtnIKs,δtf,vhe,uhkL,ma,nk,vthk,ns;
                atol_nIK=atol_nIK,is_out_errdt=true)   # is_norm_error = true
    elseif gridv_type == :chebyshev
        RdtnIKTs!(dtnIKs,edtnIKs,δtf,vhe,nvG,uhkL,ma,nk,vthk,ns;
                atol_nIK=atol_nIK,is_out_errdt=true)   # is_norm_error = true
    end
    # @show dtnIKs[3,:]

    # Conservation in discrete
    if is_enforce_errdtnIKab
        dtnIKposteriorC!(dtnIKs,edtnIKs)
    else
        if is_check_conservation_dtM 
            if norm(dtnIKs[1,:])  ≥ epsTe6
                @warn("δₜn̂a: The mass conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",dtnIKs[1,:])
            end
       
            # RD_I
            if abs(sum(dtnIKs[2,:])) > epsTe6
                RDIab = abs(dtnIKs[2,1] - dtnIKs[2,2])
                if RDIab ≠ 0.0
                    err_RdtI = sum(dtnIKs[2,:]) / RDIab
                    if err_RdtI > epsTe6
                        @warn("δₜÎa: The momentum conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_RdtI)
                    end
                end
            end
        
            # RD_K
            if abs(sum(dtnIKs[3,:])) > epsTe6
                RDKab = abs(dtnIKs[3,1] - dtnIKs[3,2])
                if RDKab ≠ 0.0
                    err_RdtK = sum(dtnIKs[3,:]) / RDKab
                    if err_RdtK > epsTe6
                        @warn("δₜnK̂a: The the total energy conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_RdtK)
                    end
                end
            end
        end
    end

    # Computing the moments in the Lagrange coordinate system.
    IK2_lagrange!(IkL, KkL, ma .* nk, ukL, vthk)

    # Evoluating the times step
    dtk = dt_RdtnIK2(dtk,dtnIKs[2:3,:],uk,IkL,KkL; rtol_DnIK=rtol_DnIK)

    return dtk
end

"""
  Inputs:
    dtnIKs = zeros(4,2)

  Outputs:
    dtk, Rdtsaa[isp][k] = dtIKaa!(sk,dtIKs,nvG,ocp,vGdom,LM,
           CΓ,εᵣ,ma,Zq,spices,nk,uk,vthk,DThk,IkL,KkL,dtk;
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
           is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            
           eps_fup=eps_fup,eps_flow=eps_flow,
           maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
           abstol=abstol,reltol=reltol,
           vadaptlevels=vadaptlevels,gridv_type=gridv_type,
           is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
           
           is_check_conservation_dtM=is_check_conservation_dtM,
           is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
           is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
"""
# [nMod=2], ns = 1, For self-collision process when `ma[1] = ma[2]` and `Zq[1] = Zq[2]` but different `uai`
function dtIKaa!(skaa::T,dtnIKs::AbstractArray{T,2},edtnIKs::AbstractArray{T,N2},
    nvG::Int64,ocp::Int64,vGdom::AbstractVector{T},LMk::Vector{Int64},
    CΓ::T,εᵣ::T,ma::T,Zq::Int64,spices::Symbol,
    nk::AbstractVector{T},uk::AbstractVector{T},vthk::AbstractVector{T},DThk::AbstractVector{T},
    IkL::AbstractVector{T}, KkL::AbstractVector{T},dtk::T;ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,N2}
    
    # Computing the relative velocity `uCk` which satisfing `uai = - ubi` in the Lagrange coordinate system with `uCk`
    uCk = uCab(uk, vthk)
    ukL = uk .- uCk
    uhkL = ukL ./ vthk
    sum(uhkL) ≤ epsT1000 || @warn("aa: The relative velocity of the Lagrange coordinate system `uCk` is not optimezed,",sum(uhkL)) 
    # @show 1, fmtf4.([sum_kbn(uhkL) / (uhkL[1]), uhkL[1]])

    # Updating the meshgrids on the velocity axis when `ns == 2, nMod .== 1` which means `nai = vthi .== 1`
    nvG, nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0 = vHadapt1D(
        nvG, ocp, vGdom, abs(uhkL[1]);
        eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
        abstol=abstol,reltol=reltol,
        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)

    # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
    nsp_vec = 1:ns
    # `L_limit` denotes an extra row is given which may be used.
    fvL0k = Vector{Matrix{T}}(undef,ns) 
    for isp in nsp_vec
        fvL0k[isp] = zeros(nvG,L_limit+1)
    end

    LM1k, ~ = fvLDMz!(fvL0k, vhe, LMk, ns, uhkL; 
            L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)
    
    # Checking the conservation laws of the renormalized distribution function `fvL0k1`
    nIKTh = zeros(4, ns)
    if gridv_type == :uniform
        nIKThs!(nIKTh, fvL0k, vhe, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
    elseif gridv_type == :chebyshev
        nIKThs!(nIKTh, fvL0k, vhe, nvG, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
    else
        sdfgh
    end
    DThk[:] = nIKTh[4, :]
    if maximum(DThk) > atol_IKTh
        @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a = 3/2 * T̂a + ûa²`", DThk)
        if maximum(DThk) > rtol_IKTh 
            printstyled("`errTh < rtol_IKTh` which means the convergence of the algorithm is falure!",color=:red,"\n")
        end
    end
  
    # # Updating the FP collision terms according to the `FPS` operators.
    δtf = Vector{Matrix{T}}(undef,ns)     # δtfvLa
    for isp in nsp_vec
        δtf[isp] = zeros(T,nvG,LM1k)
    end

    # Computing the first-two order derivatives of `fvL0` and it's mapping functions `FvL` when `nMod .== 1` and `uai[1] = - uai[2]`
    FP0D2Vaa2!(δtf,fvL0k,vhk,nvG,nc0,nck,ocp,
            nvlevele0,nvlevel0,LMk,LM1k,
            CΓ,εᵣ,ma,Zq,spices,nk,uhkL,vthk;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs, 
            is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
            is_extrapolate_FLn=is_extrapolate_FLn)
    

    # Verifying the mass, momentum and total energy conservation laws of `δtfa`.
    if gridv_type == :uniform
        RdtnIKTs!(dtnIKs,edtnIKs,δtf,vhe,uhkL,[ma,ma],nk,vthk,ns;
                atol_nIK=atol_nIK,is_out_errdt=true)   # is_norm_error = true
    elseif gridv_type == :chebyshev
        RdtnIKTs!(dtnIKs,edtnIKs,δtf,vhe,nvG,uhkL,[ma,ma],nk,vthk,ns;
                atol_nIK=atol_nIK,is_out_errdt=true)   # is_norm_error = true
    end

    # Conservation in discrete
    if is_enforce_errdtnIKab
        dtnIKposteriorC!(dtnIKs,edtnIKs)
    else
        if is_check_conservation_dtM 
            if norm(dtnIKs[1,:])  ≥ epsTe6
                @warn("δₜn̂a: The mass conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",dtnIKs[1,:])
            end
      
            if abs(sum(dtnIKs[2,:])) > epsTe6
                RDIab = abs(dtnIKs[2,1] - dtnIKs[2,2])
                if RDIab ≠ 0.0
                    err_RdtI = sum_kbn(dtnIKs[2,:]) / RDIab
                    if err_RdtI > epsTe6
                        @warn("δₜÎa: The momentum conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_RdtI)
                    end
                end
            end
        
            if abs(sum(dtnIKs[3,:])) > epsTe6
                RDKab = abs(dtnIKs[3,1] - dtnIKs[3,2])
                if RDKab ≠ 0.0
                    err_RdtK = sum_kbn(dtnIKs[3,:]) / RDKab
                    if err_RdtK > epsTe6
                        @warn("δₜnK̂a: The the total energy conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_RdtK)
                    end
                end
            end
        end
    end

    # Computing the moments in the Lagrange coordinate system.
    IK2_lagrange!(IkL, KkL, ma .* nk, ukL, vthk)

    # Evoluating the times step
    dtk = dt_RdtnIK2(dtk,dtnIKs[2:3,:],uk,IkL,KkL; rtol_DnIK=rtol_DnIK)

    return dtk, entropy_rate_fDM(uk./vthk,vthk,dtnIKs[2,:],dtnIKs[3,:]) / skaa
    # dtsaa = entropy_rate_fDM(uhak,vthk,dtIa,dtKa)
    # Rdtsaa = entropyN_rate_fDM(nk,uk./vthk,vthk,dtnIKs[2,:],dtnIKs[3,:])   # dtsaa / sum(na)
    # return dtsaa / skaa
end

