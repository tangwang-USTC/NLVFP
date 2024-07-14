

"""
  Checking difference owing to the King smoothing function

    `Dfln = flnk1 - Klnk1`

  The change rate of moments of distribution function owing to the Coulomb collision process 
  which is discribed by the Fokker-Planck collision equations.

  A2. dtfa = dtfab + dtfac

  An. dtfa = dtfa(fa, (fb, fc))

  Inputs:
    Variables:

      Mhck, nvG, ocp, ns:
      nak,uak,vathk: Fitst three moments.

    Parameters: predictive values of parameters which could accelerate the convergence of the algorithm.

      LM1: To be the parameter `L_limit` at next time step.
      naik, uaik, vthik, nModk: To be the predictive values of characteristic parameters at next time step.
      dtk: To be the predictive timestep at next time step.
    
    Storages:

      Rck1[njMs+1,1,:]                    # `w3k = Rdtvth = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`

  Outputs:
    dtk = dtMcab!(Mck,Mhck1,Rc,edtnIKTs,err_Rc2,Mhck,nvG,ocp,vGdom,LMk,LM1,
           nai,uai,vthi,CΓ,εᵣ,ma,Zq,spices,na,uak,vth,ns,nMod,nMjMs,DThk,
           Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
           NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
           restartfit=restartfit,maxIterTR=maxIterTR,maxIterKing=maxIterKing,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
           optimizer=optimizer,factor=factor,is_Jacobian=is_Jacobian,
           is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,
           is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
           eps_fup=eps_fup,eps_flow=eps_flow,
           maxiter_vGm=maxiter_vGm, vGm_limit=vGm_limit,
           abstol=abstol, reltol=reltol, 
           vadaptlevels=vadaptlevels,gridv_type=gridv_type,
           is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
           is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
           is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
    dtk = dtMcab2!(Mck,Mhck1,Rc,nvG,ocp,vGdom,LMk,LM1,naik,uaik,vthik,
           CΓ,εᵣ,ma,Zq,spices,nak,uak,vthk,nMod,nMjMs,DThk,
           Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,ns=2,
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
           is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,
           is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
           eps_fup=eps_fup,eps_flow=eps_flow,
           maxiter_vGm=maxiter_vGm, vGm_limit=vGm_limit, 
           abstol=abstol, reltol=reltol,
           vadaptlevels=vadaptlevels,gridv_type=gridv_type,
           is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
           is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
           is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
"""
 
# Rc[Mhck], 

# [nMod,ns≥3], `dtk_order_Rc ∈ [:mid, :max]`, same
#              `dtk_order_Rc = :min` and `is_dtk_order_Rcaa = true`
function dtMcab!(Mck::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    Rc::AbstractArray{T,N},edtnIKTs::AbstractArray{T,N2},
    Rc2::AbstractArray{T,N},edtnIKTs2::AbstractArray{T,N2},CRDn::AbstractVector{T},
    err_Rc2::AbstractArray{T,N},DThk2::AbstractVector{T},
    naik2::Vector{AbstractVector{T}},uaik2::Vector{AbstractVector{T}},vthik2::Vector{AbstractVector{T}},Mhck::Vector{Matrix{T}},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2}, LM::Vector{Int64}, LM1::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    Mc::AbstractArray{T,N},Mc2::AbstractArray{T,N},
    nak::AbstractVector{T},uak::AbstractVector{T},vathk::AbstractVector{T},
    ns::Int64,nModk::Vector{Int64},nMjMs::Vector{Int64},DThk::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},dtk::T;Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false) where{T,N,N2}
    
    # # Calculate the parameters `nai, uai, vthi` from the re-normalized moments `ℳ̂ⱼ,ₗ⁰`
    submoment!(naik, uaik, vthik, nModk, ns, Mhck,
            Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)

    # Updating the rate of change of the kinetic moments according to parameters `nai, uai, vthi`
    dtk = dtMcabn!(Mck,Mhck1,Rc,edtnIKTs,Rc2,edtnIKTs2,CRDn,err_Rc2,DThk2,naik2,uaik2,vthik2,
           nvG,ocp,vGdom,LM,LM1,naik,uaik,vthik,
           CΓ,εᵣ,ma,Zq,spices,Mc,Mc2,nak,uak,vathk,ns,nModk,nMjMs,DThk,
           Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
           is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,
           is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
           eps_fup=eps_fup,eps_flow=eps_flow,
           maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
           abstol=abstol,reltol=reltol,
           vadaptlevels=vadaptlevels,gridv_type=gridv_type,
           is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
           is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
           is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
           dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
    return dtk
end
 
# [nMod,ns≥3], `dtk_order_Rc = :min` and `is_dtk_order_Rcaa = false`, same
function dtMcab!(Mck::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    Rc::AbstractArray{T,N},edtnIKTs::AbstractArray{T,N2},
    Rc2::AbstractArray{T,N},edtnIKTs2::AbstractArray{T,N2},CRDn::AbstractVector{T},
    err_Rc2::AbstractArray{T,N},DThk2::AbstractVector{T},
    naik2::Vector{AbstractVector{T}},uaik2::Vector{AbstractVector{T}},vthik2::Vector{AbstractVector{T}},Mhck::Vector{Matrix{T}},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2}, LM::Vector{Int64}, LM1::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak::AbstractVector{T},uak::AbstractVector{T},vathk::AbstractVector{T},
    ns::Int64,nModk::Vector{Int64},nMjMs::Vector{Int64},DThk::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},dtk::T;Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,N,N2}
    
    # # Calculate the parameters `nai, uai, vthi` from the re-normalized moments `ℳ̂ⱼ,ₗ⁰`
    submoment!(naik, uaik, vthik, nModk, ns, Mhck,
            Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)

    # Updating the rate of change of the kinetic moments according to parameters `nai, uai, vthi`
    dtk = dtMcabn!(Mck,Mhck1,Rc,edtnIKTs,Rc2,edtnIKTs2,CRDn,err_Rc2,DThk2,naik2,uaik2,vthik2,
           nvG,ocp,vGdom,LM,LM1,naik,uaik,vthik,
           CΓ,εᵣ,ma,Zq,spices,nak,uak,vathk,ns,nModk,nMjMs,DThk,
           Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
           is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,
           is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
           eps_fup=eps_fup,eps_flow=eps_flow,
           maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
           abstol=abstol,reltol=reltol,
           vadaptlevels=vadaptlevels,gridv_type=gridv_type,
           is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
           is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
           is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
    return dtk
end

# [ns≥3], nMod .= 1
# dtMcabn!

"""
"""

# [nMod,ns=2], `dtk_order_Rc ∈ [:mid, :max]`, same
#              `dtk_order_Rc = :min` and `is_dtk_order_Rcaa = true`
function dtMcab!(Mck::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    Rc::AbstractArray{T,N},edtnIKTs::AbstractArray{T,N2},err_Rc::AbstractArray{T,N},Mhck::Vector{Matrix{T}},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2}, LM::Vector{Int64}, LM1::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},Mc::AbstractArray{T,N},
    nak::AbstractVector{T},uak::AbstractVector{T},vathk::AbstractVector{T},
    nModk::Vector{Int64},nMjMs::Vector{Int64},DThk::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},dtk::T;Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],ns::Int64=2,
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,
    
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,

    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false) where{T,N,N2}
    
    # # Calculate the parameters `nai, uai, vthi` from the re-normalized moments `ℳ̂ⱼ,ₗ⁰`
    submoment!(naik, uaik, vthik, nModk, ns, Mhck,
            Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)
    # @show 790, sum(naik[2] .* uaik[2]) - Mhck[2][1,2]
    # @show 791, sum(naik[1] .* uaik[1]) - Mhck[1][1,2]

    # Updating the rate of change of the kinetic moments according to parameters `nai, uai, vthi`
    dtk = dtMcab2!(Mck,Mhck1,Rc,edtnIKTs,err_Rc,
           nvG,ocp,vGdom,LM,LM1,naik,uaik,vthik,
           CΓ,εᵣ,ma,Zq,spices,Mc,nak,uak,vathk,nModk,nMjMs,DThk,
           Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
           is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,
           is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
           eps_fup=eps_fup,eps_flow=eps_flow,
           maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
           abstol=abstol,reltol=reltol,
           vadaptlevels=vadaptlevels,gridv_type=gridv_type,
           is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
           is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
           is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
           dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
    
    # @show 792, sum(naik[2] .* uaik[2]) - Mhck[2][1,2]
    # @show 793, sum(naik[1] .* uaik[1]) - Mhck[1][1,2]
    return dtk
end

# [nMod,ns=2], `dtk_order_Rc = :min` and `is_dtk_order_Rcaa = false`, same
function dtMcab!(Mck::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    Rc::AbstractArray{T,N},edtnIKTs::AbstractArray{T,N2},err_Rc::AbstractArray{T,N},Mhck::Vector{Matrix{T}},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2}, LM::Vector{Int64}, LM1::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak::AbstractVector{T},uak::AbstractVector{T},vathk::AbstractVector{T},
    nModk::Vector{Int64},nMjMs::Vector{Int64},DThk::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},dtk::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],ns::Int64=2,
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,N,N2}
    
    # # Calculate the parameters `nai, uai, vthi` from the re-normalized moments `ℳ̂ⱼ,ₗ⁰`
    submoment!(naik, uaik, vthik, nModk, ns, Mhck,
            Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)

    # @show 790, sum(naik[2] .* uaik[2]) - Mhck[2][1,2]
    # @show 791, sum(naik[1] .* uaik[1]) - Mhck[1][1,2]

    # Updating the rate of change of the kinetic moments according to parameters `nai, uai, vthi`
    dtk = dtMcab2!(Mck,Mhck1,Rc,edtnIKTs,err_Rc,
           nvG,ocp,vGdom,LM,LM1,naik,uaik,vthik,
           CΓ,εᵣ,ma,Zq,spices,nak,uak,vathk,nModk,nMjMs,
           DThk,Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
           is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,
           is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
           eps_fup=eps_fup,eps_flow=eps_flow,
           maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
           abstol=abstol,reltol=reltol,
           vadaptlevels=vadaptlevels,gridv_type=gridv_type,
           is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
           is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
           is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
    
    # @show 792, sum(naik[2] .* uaik[2]) - Mhck[2][1,2]
    # @show 793, sum(naik[1] .* uaik[1]) - Mhck[1][1,2]
    return dtk
end

# [ns=2], `nMod[:] .= 1` 
# dtMcab2!

"""
"""
# Rc[nai, uai, vthi] in Lagrange coordinate system 

# [nMod,ns≥3], `dtk_order_Rc ∈ [:mid, :max]`
#              `dtk_order_Rc = :min` and `is_dtk_order_Rcaa = true`, same
function dtMcabn!(Mck::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    Rc::AbstractArray{T,N},edtnIKTs::AbstractArray{T,N2},
    Rc2::AbstractArray{T,N},edtnIKTs2::AbstractArray{T,N2},CRDn::AbstractVector{T},
    err_Rc2::AbstractArray{T,N},DThk2::AbstractVector{T},
    naik2::Vector{AbstractVector{T}},uaik2::Vector{AbstractVector{T}},vthik2::Vector{AbstractVector{T}},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2}, LMk::Vector{Int64}, LM1::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    Mc::AbstractArray{T,N},Mc2::AbstractArray{T,N},
    nak::AbstractVector{T},uak::AbstractVector{T},vathk::AbstractVector{T},
    ns::Int64,nModk::Vector{Int64},nMjMs::Vector{Int64},DThk::AbstractVector{T},Nspan_optim_nuTi::AbstractVector{T},dtk::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false) where{T,N,N2}

    Rc .= 0.0
    edtnIKTs .= 0.0
    DThk .= 0.0
    
    vec = [1,2]
    for isp in 1:ns-1
        vec[1] = isp
        naik2[1], uaik2[1], vthik2[1] = naik[isp], uaik[isp], vthik[isp]
        Mc2[:,:,1] = Mc[:,:,isp]
        for iFv in isp+1:ns
            vec[2] = iFv
            LM1 = maximum(LMk[vec]) + 1

            if prod(nModk[vec]) == 1
                Mck2 = deepcopy(Rc2)
                Mck2[1] == deepcopy(Mck[isp])
                Mck2[2] == deepcopy(Mck[iFv])
                dtk = dtMcab2!(Mck2,Mhck1,Rc2,edtnIKTs2,err_Rc2,
                        nvG[vec],ocp[vec],vGdom[:,vec],LMk[vec],LM1,
                        CΓ,εᵣ,ma[vec],Zq[vec],spices[vec],nak[vec],uak[vec],vathk[vec],
                        nMjMs[vec],DThk2,dtk;
                        is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                        autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM,is_boundaryv0=is_boundaryv0,
                        is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
                        eps_fup=eps_fup,eps_flow=eps_flow,
                        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                        abstol=abstol,reltol=reltol,
                        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                        is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
            else
                naik2[2], uaik2[2], vthik2[2] = naik[iFv], uaik[iFv], vthik[iFv]
                Mc2[:,:,2] = Mc[:,:,iFv]

                Mck2 = deepcopy(Rc2)
                Mck2[1] == deepcopy(Mck[isp])
                Mck2[2] == deepcopy(Mck[iFv])
                dtk = dtMcab2!(Mck2,Mhck1,Rc2,edtnIKTs2,err_Rc2,
                        nvG[vec],ocp[vec],vGdom[:,vec],LMk[vec],LM1,naik2,uaik2,vthik2,
                        CΓ,εᵣ,ma[vec],Zq[vec],spices[vec],Mc2,nak[vec],uak[vec],vathk[vec],
                        nModk[vec],nMjMs[vec],DThk2,Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
                        is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                        autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                        is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,
                        is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
                        eps_fup=eps_fup,eps_flow=eps_flow,
                        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                        abstol=abstol,reltol=reltol,
                        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                        is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                        dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
            end
            Rc[:,:,isp] += Rc2[:,:,1]
            edtnIKTs[:,isp] += edtnIKTs2[:,1]
            DThk[isp] += DThk2[1]
            Rc[:,:,iFv] += Rc2[:,:,2]
            edtnIKTs[:,iFv] += edtnIKTs2[:,2]
            CRDn[1] += min(abs(edtnIKTs2[1,1]),abs(edtnIKTs2[1,2]))
            DThk[iFv] += DThk2[2]
        end
    end
    ns1 = ns - 1
    edtnIKTs /= ns1
    DThk /= ns1
    CRDn[1] /= ns1

    return dtk
end

#            , `dtk_order_Rc = :min` and `is_dtk_order_Rcaa = false`, same
function dtMcabn!(Mck::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    Rc::AbstractArray{T,N},edtnIKTs::AbstractArray{T,N2},
    Rc2::AbstractArray{T,N},edtnIKTs2::AbstractArray{T,N2},CRDn::AbstractVector{T},
    err_Rc2::AbstractArray{T,N},DThk2::AbstractVector{T},
    naik2::Vector{AbstractVector{T}},uaik2::Vector{AbstractVector{T}},vthik2::Vector{AbstractVector{T}},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2}, LMk::Vector{Int64}, LM1::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak::AbstractVector{T},uak::AbstractVector{T},vathk::AbstractVector{T},
    ns::Int64,nModk::Vector{Int64},nMjMs::Vector{Int64},DThk::AbstractVector{T},Nspan_optim_nuTi::AbstractVector{T},dtk::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,N,N2}

    Rc .= 0.0
    edtnIKTs .= 0.0
    DThk .= 0.0
    CRDn[1] = 0.0
    
    vec = [1,2]
    for isp in 1:ns-1
        vec[1] = isp
        naik2[1], uaik2[1], vthik2[1] = naik[isp], uaik[isp], vthik[isp]
        for iFv in isp+1:ns
            vec[2] = iFv
            LM1 = maximum(LMk[vec]) + 1

            if prod(nModk[vec]) == 1

                Mck2 = deepcopy(Rc2)
                Mck2[1] == deepcopy(Mck[isp])
                Mck2[2] == deepcopy(Mck[iFv])
                dtk = dtMcab2!(Mck2,Mhck1,Rc2,edtnIKTs2,err_Rc2,
                        nvG[vec],ocp[vec],vGdom[:,vec],LMk[vec],LM1,
                        CΓ,εᵣ,ma[vec],Zq[vec],spices[vec],nak[vec],uak[vec],vathk[vec],
                        nMjMs[vec],DThk2,dtk;
                        is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                        autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM,is_boundaryv0=is_boundaryv0,
                        is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
                        eps_fup=eps_fup,eps_flow=eps_flow,
                        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                        abstol=abstol,reltol=reltol,
                        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                        is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
            else
                naik2[2], uaik2[2], vthik2[2] = naik[iFv], uaik[iFv], vthik[iFv]

                Mck2 = deepcopy(Rc2)
                Mck2[1] == deepcopy(Mck[isp])
                Mck2[2] == deepcopy(Mck[iFv])
                dtk = dtMcab2!(Mck2,Mhck1,Rc2,edtnIKTs2,err_Rc2,
                        nvG[vec],ocp[vec],vGdom[:,vec],LMk[vec],LM1,naik2,uaik2,vthik2,
                        CΓ,εᵣ,ma[vec],Zq[vec],spices[vec],nak[vec],uak[vec],vathk[vec],
                        nModk[vec],nMjMs[vec],DThk2,Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
                        is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                        autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                        is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,
                        is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
                        eps_fup=eps_fup,eps_flow=eps_flow,
                        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                        abstol=abstol,reltol=reltol,
                        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                        is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
            end
            Rc[:,:,isp] += Rc2[:,:,1]
            edtnIKTs[:,isp] += edtnIKTs2[:,1]
            DThk[isp] += DThk2[1]
            Rc[:,:,iFv] += Rc2[:,:,2]
            edtnIKTs[:,iFv] += edtnIKTs2[:,2]
            CRDn[1] += min(abs(edtnIKTs2[1,1]),abs(edtnIKTs2[1,2]))
            DThk[iFv] += DThk2[2]
        end
    end
    ns1 = ns - 1
    edtnIKTs /= ns1
    DThk /= ns1
    CRDn[1] /= ns1

    return dtk
end

# [ns≥3], nMod .= 1, same
function dtMcabn!(Mck::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    Rc::AbstractArray{T,N},edtnIKTs::AbstractArray{T,N2},
    Rc2::AbstractArray{T,N},edtnIKTs2::AbstractArray{T,N2},CRDn::AbstractVector{T},
    err_Rc2::AbstractArray{T,N},DThk2::AbstractVector{T},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2}, LMk::Vector{Int64}, LM1::Int64,
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak::AbstractVector{T},uak::AbstractVector{T},vathk::AbstractVector{T},
    ns::Int64,nMjMs::Vector{Int64},DThk::AbstractVector{T},dtk::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],
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
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,N,N2}

    Rc .= 0.0
    edtnIKTs .= 0.0
    DThk .= 0.0
    
    vec = [1,2]
    for isp in 1:ns-1
        vec[1] = isp
        for iFv in isp+1:ns
            vec[2] = iFv
            LM1 = maximum(LMk[vec]) + 1

            Mck2 = deepcopy(Rc2)
            Mck2[1] == deepcopy(Mck[isp])
            Mck2[2] == deepcopy(Mck[iFv])
            dtk = dtMcab2!(Mck2,Mhck1,Rc2,edtnIKTs2,err_Rc2,
                    nvG[vec],ocp[vec],vGdom[:,vec],LMk[vec],LM1,
                    CΓ,εᵣ,ma[vec],Zq[vec],spices[vec],nak[vec],uak[vec],vathk[vec],
                    nMjMs[vec],DThk2,dtk;
                    is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                    rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM,is_boundaryv0=is_boundaryv0,
                    is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
                    eps_fup=eps_fup,eps_flow=eps_flow,
                    maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                    abstol=abstol,reltol=reltol,
                    vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                    is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                    is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                    is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                    
            Rc[:,:,isp] += Rc2[:,:,1]
            edtnIKTs[:,isp] += edtnIKTs2[:,1]
            DThk[isp] += DThk2[1]
            Rc[:,:,iFv] += Rc2[:,:,2]
            edtnIKTs[:,iFv] += edtnIKTs2[:,2]
            CRDn[1] += min(abs(edtnIKTs2[1,1]),abs(edtnIKTs2[1,2]))
            DThk[iFv] += DThk2[2]
        end
    end
    ns1 = ns - 1
    edtnIKTs /= ns1
    DThk /= ns1
    CRDn[1] /= ns1

    return dtk
end

"""
"""

# [nMod,ns=2], `dtk_order_Rc ∈ [:mid, :max]`
#              `dtk_order_Rc = :min` and `is_dtk_order_Rcaa = true`
function dtMcab2!(Mck::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    Rc::AbstractArray{T,N},edtnIKTs::AbstractArray{T,N2},err_Rc::AbstractArray{T,N},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2}, LMk::Vector{Int64}, LM1::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},Mc::AbstractArray{T,N},
    nak::AbstractVector{T},uak::AbstractVector{T},vathk::AbstractVector{T},
    nModk::Vector{Int64},nMjMs::Vector{Int64},DThk::AbstractVector{T},Nspan_optim_nuTi::AbstractVector{T},dtk::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false) where{T,N,N2}

    is_dtf_MS_IK = false
    rfghjnkm
    # Computing the relative velocity `uCk` which satisfing `ua = - ub` in the Lagrange coordinate system with `uCk`
    uCk = uCab(uak, vathk)
    # @show 1, uCk
    uakL = uak .- uCk
    uhkL = uakL ./ vathk
    sum(uhkL) ≤ epsT1000 || @warn("ab: The relative velocity of the Lagrange coordinate system `uCk` is not optimezed,",sum(uhkL)) 
    # @show 2, fmtf4.([sum_kbn(uhkL) / (uhkL[1]), uhkL[1]])

    # if abs(uhkL[1]) ≤ epsn8
    #     # @warn("`FM` model maybe a better approximation when,",uhkL[1]) 
    #     # if abs(uhkL[1]) ≤ epsn10
    #     #     @warn("`FM` model is be proposed") 
    #     # end
    # end

    # Updating the meshgrids on the velocity axis in Lagrange coordinate system.
    nc0, nck = zeros(Int64,ns), zeros(Int64,ns)
    if gridv_type == :uniform
        vhe = Vector{StepRangeLen}(undef, ns)
    elseif gridv_type == :chebyshev
        vhe = Vector{AbstractVector{T}}(undef, ns)
    else
        sdfgh
    end
    vhk = Vector{AbstractVector{T}}(undef, ns)
    nvlevele0 = Vector{Vector{Int64}}(undef, ns)
    nvlevel0 = Vector{Vector{Int64}}(undef, ns)

    nsp_vec = 1:ns
    # `L_limit + 1 = LM1 + 1` denotes an extra row is given which may be used.
    fvL0k = Vector{Matrix{T}}(undef,ns) 
    for isp in nsp_vec
        fvL0k[isp] = zeros(nvG[isp],LM1+1)
    end

    δtfh = Vector{Matrix{T}}(undef,ns)     # δtfvLa

    # Transformating the characteristic parameters `uai` to be the ones in the Lagrange coordinate system
    uaikL = Vector{AbstractVector{T}}(undef,ns)
    for isp in nsp_vec
        uaikL[isp] = uaik[isp] .- uCk / vathk[isp]
    end
    vHadapt1D!(vhe,vhk, vGdom, nvG, nc0, nck, ocp, 
          nvlevele0, nvlevel0, naik,uaikL,vthik, nModk, ns;
          eps_fup=eps_fup,eps_flow=eps_flow,
          maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
          abstol=abstol,reltol=reltol,
          vadaptlevels=vadaptlevels,gridv_type=gridv_type,
          is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
        
    @show nvG, nc0, nck, vGdom[2,:]
    @show vhe[1][nc0[1]], vhe[2][nc0[2]]
    @show vhk[1][nck[1]], vhk[2][nck[2]]

    # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
    LM1k, fvL0k = fvLDMz!(fvL0k, vhe, nvG, LMk, ns,naik,uaikL,vthik, nModk; 
        L_limit=LM1, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full)
    
    for isp in nsp_vec
        δtfh[isp] = zeros(T,nvG[isp],LM1k)
    end

    # # Updating the FP collision terms according to the `FPS` operators.
    if is_dtk_order_Rcaa
        dtk = FP0D2Vab2!(δtfh,fvL0k,vhe,vhk,nvG,nc0,nck,ocp,
               nvlevele0,nvlevel0,LMk,LM1k,naik,uaikL,vthik,
               CΓ,εᵣ,ma,Zq,spices,Rc,Mc,nak,vathk,nModk,nMjMs,dtk;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
               is_norm_error=is_norm_error,dtk_order_Rc=dtk_order_Rc)
    else
        FP0D2Vab2!(δtfh,fvL0k,vhk,nvG,nc0,nck,ocp,
               nvlevele0,nvlevel0,LMk,LM1k,naik,uaikL,vthik,
               CΓ,εᵣ,ma,Zq,spices,nak,vathk,nModk;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               is_extrapolate_FLn=is_extrapolate_FLn)
    end

    # Checking the change rate of harmonic of distribution function
    Rdtfln3 = zeros(T,ns)
    for isp in nsp_vec
        Rdtfln3[isp] = δtfh[isp][3,1] ./ fvL0k[isp][3,1]
    end
    # @show Rdtfln3

    ylabel = string("1,Rdtfln")

    isp = 1
    Rdtfln = δtfh[isp][:,1] ./ fvL0k[isp][:,1]
    label = string("a")
    pRdtflna = plot(vhe[isp],Rdtfln,label=label,ylabel=ylabel)

    isp = 2
    Rdtfln = δtfh[isp][:,1] ./ fvL0k[isp][:,1]
    label = string("b")
    pRdtflnb = plot(vhe[isp],Rdtfln,label=label)
    display(plot(pRdtflna,pRdtflnb,layout=(2,1)))

    # Checking the conservation laws of the renormalized distribution function `fvL0k1`
    nIKTh = zeros(4, ns)
    if gridv_type == :uniform
        nIKThs!(nIKTh, fvL0k, vhe, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
    elseif gridv_type == :chebyshev
        if is_dtf_MS_IK
            nIKThs!(nIKTh, fvL0k, vhe, nvG[1], ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
        else
            nIKThs!(nIKTh, fvL0k, vhe, nvG, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
        end
    end

    ρk1 = ma .* nak

    # Calculating the rate of change of the kinetic moments in the Lagrange coordinate system.
    aa = Rc[1:njMs,:,:]
    if gridv_type == :uniform
        dtMcsd2l!(aa,err_Rc,δtfh,vhe,nMjMs,ρk1,vathk,LMk,ns;is_renorm=is_renorm)
    elseif gridv_type == :chebyshev
        if is_dtf_MS_IK
            dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG[1],nMjMs,ρk1,vathk,LMk,ns;
                      is_renorm=is_renorm,is_norm_error=is_norm_error)
        else
            dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG,nMjMs,ρk1,vathk,LMk,ns;
                      is_renorm=is_renorm,is_norm_error=is_norm_error)
        end
    end
    Rc[1:njMs,:,:] = aa

    DThk[:] = nIKTh[4, :]
    if maximum(DThk) > atol_IKTh
        @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a = 3/2 * T̂a + ûa²`", DThk)
        if maximum(DThk) > rtol_IKTh 
            printstyled("`errTh < rtol_IKTh` which means the convergence of the algorithm is falure!",color=:red,"\n")
        end
    end
    
    # Computing the relative change rate of thermal velocity in the Lagrange coordinate system 
    # according to   # `Rdtvth = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`, `𝒲 = RdtK - 2 * IhL .* RdtI`
    Rdtvth = Rc[njMs+1,1,:]
    Rc[njMs+1,1,:] = RdtvthRc!(Rdtvth,uhkL,Rc[1,2,:],Rc[2,1,:],ns)

    # Verifying the mass, momentum and total energy conservation laws of `δtfa` in Lagrange coordinate system.
    edtnIKTs[1,:] = deepcopy(err_Rc[1,1,:])           # edtn
    edtnIKTs[2,:] = deepcopy(err_Rc[1,2,:])           # edtI
    edtnIKTs[3,:] = deepcopy(err_Rc[2,1,:])           # edtK
    # Checking the constraint according to (eRdtKI): `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
    edtnIKTs[4,:] = eRdtKIRc!(edtnIKTs[4,:],uhkL,Rc[1,2,:],Rc[2,1,:],Rdtvth,ns)
    
    # Conservation in discrete
    if is_enforce_errdtnIKab
        dtnIKposteriorC!(Rc,err_Rc,nMjMs)
    else
        sdfgbhnm
        dtnIKTs[4,:] = Rc[njMs+1,1,:]
        if is_check_conservation_dtM 
            if norm(dtnIKTs[1,:])  ≥ epsTe6
                @warn("δₜn̂a: The mass conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",dtnIKTs[1,:])
            end
      
            # RD_I
            if abs(sum(dtnIKTs[2,:])) > epsTe6
                RDIab = abs(dtnIKTs[2,1] - dtnIKTs[2,2])
                if RDIab ≠ 0.0
                    err_RdtI = sum(dtnIKTs[2,:]) / RDIab
                    if err_RdtI > epsTe6
                        @warn("δₜÎa: The momentum conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_RdtI)
                    end
                end
            end
        
            # RD_K
            if abs(sum(dtnIKTs[3,:])) > epsTe6
                RDKab = abs(dtnIKTs[3,1] - dtnIKTs[3,2])
                if RDKab ≠ 0.0
                    err_RdtK = sum(dtnIKTs[3,:]) / RDKab
                    if err_RdtK > epsTe6
                        @warn("δₜnK̂a: The the total energy conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_RdtK)
                    end
                end
            end
        end
    end

    # Computing the moments in the Lagrange coordinate system. 
    if dtk_order_Rc == :min
        if 1 == 2
            IkL, KkL = zeros(T,ns), zeros(T,ns)
            IK2_lagrange!(IkL, KkL, ma .* nak, uakL, vathk)
        
            # Evoluating the timestep owing to the total momentums and total energy in the Lagrange coordinate system.
            # if is_renorm
            #     dtI = deepcopy(Rc[1,2,:])              # dtI
            #     dtK = Rc[2,1,:] * CMcKa      # dtK
            # end
        
            dtk = dt_RdtnIK2(dtk,Rc[1,2,:],Rc[2,1,:] * CMcKa,uakL,IkL,KkL; rtol_DnIK=rtol_DnIK)
        end
    
        if ratio_Rdtfln > 0.0
            # @show 11, dtk
            dtk = dt_dtfln(dtk,Rdtfln3;ratio_Rdtfln=ratio_Rdtfln)
            # @show 22, dtk
        end
    else
        dtk = dt_Rc(dtk,Rc,Mc,LMk,nModk,nMjMs,ns;
                    rtol_DnIK=rtol_DnIK,dtk_order_Rc=dtk_order_Rc)
    end
    if is_plot_DflKing
        dtkk1 = dtk / 100

        Mck1 = deepcopy(Mck)
        Mck1integral0!(Mck1,Rc,ns,dtkk1)

        # Calculate the parameters `nak1,vathk1,Iak1,Kak1` from `Mck1`
        nak1,vathk1,Iak1,Kak1,δvathk1 = zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns)
        nIKT_update!(nak1,vathk1,Iak1,Kak1,δvathk1,ma,ns,Mck1;
                    is_vth_ode=is_vth_ode,is_corrections_na=is_corrections[1])
        
        # # Computing the re-normalized moments 
        if is_Ms_nuT == 4                                # for `nModk1 = 1`
            rtghj
            # Renew the values of `Mhck1` according to the quantities of `Mck1`
            MhcknuT!(Mhck1,Mck1,ρk1,vathk1,LMk,ns,nMjMs,zeros(Int64,ns);is_renorm=is_renorm)
        else
            MhckMck!(Mhck1,Mck1[1:njMs,:,:],ρk1,LMk,ns,nMjMs,vathk1)
        end
        
        # # Calculate the parameters `uai`
        uaik1L = deepcopy(uhkL)
        submoment!(uaik1L, Mhck1, ns)

        Msnnt = deepcopy(Mhck1)
        Msnnt = MsnntL2fL0(Msnnt,nMjMs,LMk,ns,uaik1L;is_renorm=is_renorm)
        # @show Msnnt - Mhck1

        # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        for isp in nsp_vec
            uaik1L[isp] = uaik1L[isp] .- uCk / vathk1[isp]
        end

        vhek1 = deepcopy(vhe)
            
        # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        KvL0k1 = deepcopy(fvL0k)
        LM1k1, KvL0k1 = fvLDMz!(KvL0k1, vhek1, LMk, ns, uaik1L; 
            L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)

        # @show norm(KvL0k1 -(fvL0k))
        # MsnnEvens!(Mhck0, errMhc, fvL0k, vhe, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
        # Mck0 = deepcopy(Mck)
        # MckMhck!(Mck0,Mhck0,ρk1,vathk,LMk,ns,nMjMs)
        
        Mhc2k11 = deepcopy(Mhck1)
        MsnnEvens!(Mhc2k11, errMhc, KvL0k1, vhe, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
        Mck11 = deepcopy(Mck)
        MckMhck!(Mck11,Mhc2k11,ρk1,vathk,LMk,ns,nMjMs)
        # @show Mhc2k11[2] - Mhck1[2]
        # @show Mhc2k11[1] - Mhck1[1]
        # @show Mck11[:,:,2] - Mck1[:,:,2]
        # @show Mck11[:,:,1] - Mck1[:,:,1]
        # wesdfvgb333
        # # 
        # @show KvL0k1[1][1,1]
        # @show vathk1 ./ vathk .- 1

        #####################3 double
        if 1 == 2
            # Calculate the parameters `nak12,vathk12,Iak12,Kak12` from `Mck12`
            nak12,vathk12,Iak12,Kak12,δvathk12 = zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns)
            nIKT_update!(nak12,vathk12,Iak12,Kak12,δvathk12,ma,ns,Mck11;
                        is_vth_ode=is_vth_ode,is_corrections_na=is_corrections[1])
            
            # # Computing the re-normalized moments 
            Mhck12 = deepcopy(Mhck1)
            if is_Ms_nuT == 4                                # for `nModk1 = 1`
                rtghj
                # Renew the values of `Mhck1` according to the quantities of `Mck1`
                MhcknuT!(Mhck12,Mck11,ρk1,vathk12,LMk,ns,nMjMs,zeros(Int64,ns);is_renorm=is_renorm)
            else
                MhckMck!(Mhck12,Mck11[1:njMs,:,:],ρk1,LMk,ns,nMjMs,vathk12)
            end
        
            # # Calculate the parameters `uai`
            uaik12L = deepcopy(uhkL)
            submoment!(uaik12L, Mhck12, ns)
    
            # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
            for isp in nsp_vec
                uaik12L[isp] = uaik12L[isp] .- uCk / vathk12[isp]
            end
    
            vhek12 = deepcopy(vhe)
            # for isp in nsp_vec
            #     vhek12[isp] = vhe[isp] * (vathk[isp] / vathk[isp])
            # end
                
            # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
            KvL0k12 = deepcopy(fvL0k)
            LM1k12, KvL0k12 = fvLDMz!(KvL0k12, vhek12, LMk, ns, uaik12L; 
                L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)
    
            Mhc2k121 = deepcopy(Mhck12)
            MsnnEvens!(Mhc2k121, errMhc, KvL0k12, vhek12, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
            Mck121 = deepcopy(Mck)
            MckMhck!(Mck121,Mhc2k121,ρk1,vathk12,LMk,ns,nMjMs)
            # @show Mhc2k121[2] - Mhck12[2]
            # @show Mhc2k121[1] - Mhck12[1]
            @show Mck121[:,:,2] - Mck11[:,:,2]
            @show Mck121[:,:,1] - Mck11[:,:,1]
            # 
            @show KvL0k12[1][1,1]
            @show vathk12 ./ vathk .- 1
            
            dfgbn
        end

        if 2 == 32
            Mhc2k101 = deepcopy(Mhck1)
            MsnnEvens!(Mhc2k101, errMhc, fvL0k, vhe, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
            Mck101 = deepcopy(Mck)
            MckMhck!(Mck101,Mhc2k101,ρk1,vathk,LMk,ns,nMjMs)
            @show Mck101 - Mck0

            # Calculating the rate of change of the kinetic moments in the Lagrange coordinate system.
            Rck101 = deepcopy(Rc)
            aa = Rck101[1:njMs,:,:]
            if gridv_type == :uniform
                dtMcsd2l!(aa,err_Rc,δtfh,vhe,nMjMs,ρk1,vathk,LMk,ns;is_renorm=is_renorm)
            elseif gridv_type == :chebyshev
                if is_dtf_MS_IK
                    dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG[1],nMjMs,ρk1,vathk,LMk,ns;
                              is_renorm=is_renorm,is_norm_error=is_norm_error)
                else
                    dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG,nMjMs,ρk1,vathk,LMk,ns;
                              is_renorm=is_renorm,is_norm_error=is_norm_error)
                end
            end
            Rck101[1:njMs,:,:] = aa
            
            Mck100 = Mck101 +  dtkk1 * Rck101
            # @show Mck100[:,:,2] - Mck1[:,:,2]
            # @show Mck100[:,:,1] - Mck1[:,:,1]
            # retghn
        end

        ########################################################################
        fvL0k1 = fvL0k + dtkk1 * δtfh          # `δₜn̂ = R̂(0,0,0) ~ 0`, but `δₜK̂` is wrong
        for isp in nsp_vec
            fvL0k1[isp] *= (vathk1[isp] / vathk[isp])^3
        end
        
        # @show fvL0k[1][1,1]
        # @show fvL0k1[1][1,1]
        Mhc2k10 = deepcopy(Mhck1)
        for isp in nsp_vec
            vhek1[isp] = vhe[isp] * (vathk[isp] / vathk1[isp])
        end
        MsnnEvens!(Mhc2k10, errMhc, fvL0k1, vhek1, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
        Mck10 = deepcopy(Mck)
        MckMhck!(Mck10,Mhc2k10,ρk1,vathk1,LMk,ns,nMjMs)
        @show Mhc2k10[2] - Mhck1[2]
        @show Mhc2k10[1] - Mhck1[1]
        @show Mck10[:,:,2] - Mck1[:,:,2]
        @show Mck10[:,:,1] - Mck1[:,:,1]

        # 
        DKing = (KvL0k1 - fvL0k1)
        # ylabel = L"\Delta \hat{f}_l^0"
        ylabel = "Δf̂ₗ⁰"
        isp = 1
        label = string("a")
        pDKinga = plot(vhek1[isp], DKing[isp],label=label,ylabel=ylabel,line=(3,:auto))
        isp = 2
        # label = string(spices[isp])
        label = string("b")
        pDKingb = plot(vhek1[isp], DKing[isp],label=label,line=(3,:auto))
        # display(plot(pDKinga,pDKingb,layout=(2,1)))

        # RDKing = (KvL0k1 - fvL0k1) ./ fvL0k1
        ylabel = "RΔf̂ₗ⁰"
        isp = 1
        DKing[isp] ./= fvL0k1[isp]
        # label = string(spices[isp])
        label = string("a")
        pRDKinga = plot(vhek1[isp], DKing[isp],label=label,ylabel=ylabel,line=(3,:auto))
        isp = 2
        DKing[isp] ./= fvL0k1[isp]
        # label = string(spices[isp])
        label = string("b")
        pRDKingb = plot(vhek1[isp], DKing[isp],label=label,line=(3,:auto))
        # display(plot(pRDKinga,pRDKingb,layout=(2,1)))

        display(plot(pDKinga,pDKingb,pRDKinga,pRDKingb,layout=(2,2)))
        wedrfgnh111
    end
    
    return dtk
end

#              `dtk_order_Rc = :min` and `is_dtk_order_Rcaa = false`
function dtMcab2!(Mck::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    Rc::AbstractArray{T,N},edtnIKTs::AbstractArray{T,N2},err_Rc::AbstractArray{T,N},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2}, LMk::Vector{Int64}, LM1::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak::AbstractVector{T},uak::AbstractVector{T},vathk::AbstractVector{T},
    nModk::Vector{Int64},nMjMs::Vector{Int64},DThk::AbstractVector{T},Nspan_optim_nuTi::AbstractVector{T},dtk::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],ns::Int64=2,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,N,N2}

    is_dtf_MS_IK = false
    
    # Computing the relative velocity `uCk` which satisfing `ua = - ub` in the Lagrange coordinate system with `uCk`
    uCk = uCab(uak, vathk)
    # uCk = 0.0
    # @show 2, uCk
    uakL = uak .- uCk
    uhkL = uakL ./ vathk
    sum(uhkL) ≤ epsT1000 || @warn("ab: The relative velocity of the Lagrange coordinate system `uCk` is not optimezed,",sum(uhkL)) 
    # @show 2, fmtf4.([sum_kbn(uhkL) / (uhkL[1]), uhkL[1]])

    # if abs(uhkL[1]) ≤ epsn8
    #     # @warn("`FM` model maybe a better approximation when,",uhkL[1]) 
    #     # if abs(uhkL[1]) ≤ epsn10
    #     #     @warn("`FM` model is be proposed") 
    #     # end
    # end

    # Updating the meshgrids on the velocity axis in Lagrange coordinate system.
    nc0, nck = zeros(Int64,ns), zeros(Int64,ns)
    if gridv_type == :uniform
        vhe = Vector{StepRangeLen}(undef, ns)
    elseif gridv_type == :chebyshev
        vhe = Vector{AbstractVector{T}}(undef, ns)
    else
        sdfgh
    end
    vhk = Vector{AbstractVector{T}}(undef, ns)
    nvlevele0 = Vector{Vector{Int64}}(undef, ns)
    nvlevel0 = Vector{Vector{Int64}}(undef, ns)

    nsp_vec = 1:ns
    # `L_limit + 1 = LM1 + 1` denotes an extra row is given which may be used.
    fvL0k = Vector{Matrix{T}}(undef,ns) 
    for isp in nsp_vec
        fvL0k[isp] = zeros(nvG[isp],LM1+1)
    end

    δtfh = Vector{Matrix{T}}(undef,ns)     # δtfvLa

    # Transformating the characteristic parameters `uai` to be the ones in the Lagrange coordinate system
    uaikL = Vector{AbstractVector{T}}(undef,ns)
    for isp in nsp_vec
        uaikL[isp] = uaik[isp] .- uCk / vathk[isp]
    end
    vHadapt1D!(vhe,vhk, vGdom, nvG, nc0, nck, ocp, 
          nvlevele0, nvlevel0, naik,uaikL,vthik, nModk, ns;
          eps_fup=eps_fup,eps_flow=eps_flow,
          maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
          abstol=abstol,reltol=reltol,
          vadaptlevels=vadaptlevels,gridv_type=gridv_type,
          is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
        
    @show nvG, nc0, nck, vGdom[2,:]
    @show vhe[1][nvG[1]], vhe[2][nvG[2]]
    @show vhk[1][nck[1]], vhk[2][nck[2]]
    # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
    LM1k, fvL0k = fvLDMz!(fvL0k, vhe, nvG, LMk, ns,naik,uaikL,vthik, nModk; 
        L_limit=LM1, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full)
    
    for isp in nsp_vec
        δtfh[isp] = zeros(T,nvG[isp],LM1k)
    end

    # # Updating the FP collision terms according to the `FPS` operators.
    FP0D2Vab2!(δtfh,fvL0k,vhk,nvG,nc0,nck,ocp,
           nvlevele0,nvlevel0,LMk,LM1k,naik,uaikL,vthik,
           CΓ,εᵣ,ma,Zq,spices,nak,vathk,nModk;
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
           is_extrapolate_FLn=is_extrapolate_FLn)

    # Checking the change rate of harmonic of distribution function
    Rdtfln3 = zeros(T,ns)
    for isp in nsp_vec
        Rdtfln3[isp] = δtfh[isp][3,1] ./ fvL0k[isp][3,1]
    end
    # @show Rdtfln3

    ylabel = string("2,Rdtfln")

    isp = 1
    Rdtfln = δtfh[isp][:,1] ./ fvL0k[isp][:,1]
    label = string("a")
    pRdtflna = plot(vhe[isp],Rdtfln,label=label,ylabel=ylabel)

    isp = 2
    Rdtfln = δtfh[isp][:,1] ./ fvL0k[isp][:,1]
    label = string("b")
    pRdtflnb = plot(vhe[isp],Rdtfln,label=label)
    display(plot(pRdtflna,pRdtflnb,layout=(2,1)))


    # Checking the conservation laws of the renormalized distribution function `fvL0k1`
    nIKTh = zeros(4, ns)
    if gridv_type == :uniform
        nIKThs!(nIKTh, fvL0k, vhe, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
    elseif gridv_type == :chebyshev
        if is_dtf_MS_IK
            nIKThs!(nIKTh, fvL0k, vhe, nvG[1], ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
        else
            nIKThs!(nIKTh, fvL0k, vhe, nvG, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
        end
    end

    ρk1 = ma .* nak

    # Calculating the rate of change of the kinetic moments in the Lagrange coordinate system.
    aa = Rc[1:njMs,:,:]
    if gridv_type == :uniform
        dtMcsd2l!(aa,err_Rc,δtfh,vhe,nMjMs,ρk1,vathk,LMk,ns;is_renorm=is_renorm)
    elseif gridv_type == :chebyshev
        if is_dtf_MS_IK
            dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG[1],nMjMs,ρk1,vathk,LMk,ns;
                      is_renorm=is_renorm,is_norm_error=is_norm_error)
        else
            dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG,nMjMs,ρk1,vathk,LMk,ns;
                      is_renorm=is_renorm,is_norm_error=is_norm_error)
        end
    end
    Rc[1:njMs,:,:] = aa

    DThk[:] = nIKTh[4, :]
    if maximum(DThk) > atol_IKTh
        @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a = 3/2 * T̂a + ûa²`", DThk)
        if maximum(DThk) > rtol_IKTh 
            printstyled("`errTh < rtol_IKTh` which means the convergence of the algorithm is falure!",color=:red,"\n")
        end
    end
    
    # Computing the relative change rate of thermal velocity in the Lagrange coordinate system 
    # according to   # `Rdtvth = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`, `𝒲 = RdtK - 2 * IhL .* RdtI`
    Rdtvth = Rc[njMs+1,1,:]
    Rc[njMs+1,1,:] = RdtvthRc!(Rdtvth,uhkL,Rc[1,2,:],Rc[2,1,:],ns)

    # Verifying the mass, momentum and total energy conservation laws of `δtfa` in Lagrange coordinate system.
    edtnIKTs[1,:] = deepcopy(err_Rc[1,1,:])           # edtn
    edtnIKTs[2,:] = deepcopy(err_Rc[1,2,:])           # edtI
    edtnIKTs[3,:] = deepcopy(err_Rc[2,1,:])           # edtK
    # Checking the constraint according to (eRdtKI): `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
    edtnIKTs[4,:] = eRdtKIRc!(edtnIKTs[4,:],uhkL,Rc[1,2,:],Rc[2,1,:],Rdtvth,ns)
    
    # Conservation in discrete
    if is_enforce_errdtnIKab
        dtnIKposteriorC!(Rc,err_Rc,nMjMs)
    else
        dtnIKTs = zeros(4,ns)
        dtnIKTs[4,:] = Rc[njMs+1,1,:]
        if is_check_conservation_dtM 
            if norm(dtnIKTs[1,:])  ≥ epsTe6
                @warn("δₜn̂a: The mass conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",dtnIKTs[1,:])
            end
      
            # RD_I
            if abs(sum(dtnIKTs[2,:])) > epsTe6
                RDIab = abs(dtnIKTs[2,1] - dtnIKTs[2,2])
                if RDIab ≠ 0.0
                    err_RdtI = sum(dtnIKTs[2,:]) / RDIab
                    if err_RdtI > epsTe6
                        @warn("δₜÎa: The momentum conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_RdtI)
                    end
                end
            end
        
            # RD_K
            if abs(sum(dtnIKTs[3,:])) > epsTe6
                RDKab = abs(dtnIKTs[3,1] - dtnIKTs[3,2])
                if RDKab ≠ 0.0
                    err_RdtK = sum(dtnIKTs[3,:]) / RDKab
                    if err_RdtK > epsTe6
                        @warn("δₜnK̂a: The the total energy conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_RdtK)
                    end
                end
            end
        end
    end

    # Computing the moments in the Lagrange coordinate system.
    IkL, KkL = zeros(T,ns), zeros(T,ns)
    IK2_lagrange!(IkL, KkL, ma .* nak, uakL, vathk)

    # Evoluating the timestep owing to the total momentums and total energy in the Lagrange coordinate system.
    # if is_renorm
    #     dtI = deepcopy(Rc[1,2,:])              # dtI
    #     dtK = Rc[2,1,:] * CMcKa                # dtK
    # end
    
    dtk = dt_RdtnIK2(dtk,Rc[1,2,:],Rc[2,1,:] * CMcKa,uakL,IkL,KkL; rtol_DnIK=rtol_DnIK)

    if is_plot_DflKing
        dtkk1 = dtk / 100

        Mck1 = deepcopy(Mck)
        Mck1integral0!(Mck1,Rc,ns,dtkk1)

        # Calculate the parameters `nak1,vathk1,Iak1,Kak1` from `Mck1`
        nak1,vathk1,Iak1,Kak1,δvathk1 = zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns)
        nIKT_update!(nak1,vathk1,Iak1,Kak1,δvathk1,ma,ns,Mck1;
                    is_vth_ode=is_vth_ode,is_corrections_na=is_corrections[1])
        
        # # Computing the re-normalized moments 
        if is_Ms_nuT == 4                                # for `nModk1 = 1`
            rtghj
            # Renew the values of `Mhck1` according to the quantities of `Mck1`
            MhcknuT!(Mhck1,Mck1,ρk1,vathk1,LMk,ns,nMjMs,zeros(Int64,ns);is_renorm=is_renorm)
        else
            MhckMck!(Mhck1,Mck1[1:njMs,:,:],ρk1,LMk,ns,nMjMs,vathk1)
        end
        
        # # Calculate the parameters `uai`
        uaik1L = deepcopy(uhkL)
        submoment!(uaik1L, Mhck1, ns)

        Msnnt = deepcopy(Mhck1)
        Msnnt = MsnntL2fL0(Msnnt,nMjMs,LMk,ns,uaik1L;is_renorm=is_renorm)
        # @show Msnnt - Mhck1

        # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        for isp in nsp_vec
            uaik1L[isp] = uaik1L[isp] .- uCk / vathk1[isp]
        end

        vhek1 = deepcopy(vhe)
            
        # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        KvL0k1 = deepcopy(fvL0k)
        LM1k1, KvL0k1 = fvLDMz!(KvL0k1, vhek1, LMk, ns, uaik1L; 
            L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)

        # @show norm(KvL0k1 -(fvL0k))
        # MsnnEvens!(Mhck0, errMhc, fvL0k, vhe, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
        # Mck0 = deepcopy(Mck)
        # MckMhck!(Mck0,Mhck0,ρk1,vathk,LMk,ns,nMjMs)
        
        Mhc2k11 = deepcopy(Mhck1)
        MsnnEvens!(Mhc2k11, errMhc, KvL0k1, vhe, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
        Mck11 = deepcopy(Mck)
        MckMhck!(Mck11,Mhc2k11,ρk1,vathk,LMk,ns,nMjMs)
        # @show Mhc2k11[2] - Mhck1[2]
        # @show Mhc2k11[1] - Mhck1[1]
        # @show Mck11[:,:,2] - Mck1[:,:,2]
        # @show Mck11[:,:,1] - Mck1[:,:,1]
        # wesdfvgb333
        # # 
        # @show KvL0k1[1][1,1]
        # @show vathk1 ./ vathk .- 1

        #####################3 double
        if 1 == 2
            # Calculate the parameters `nak12,vathk12,Iak12,Kak12` from `Mck12`
            nak12,vathk12,Iak12,Kak12,δvathk12 = zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns)
            nIKT_update!(nak12,vathk12,Iak12,Kak12,δvathk12,ma,ns,Mck11;
                        is_vth_ode=is_vth_ode,is_corrections_na=is_corrections[1])
            
            # # Computing the re-normalized moments 
            Mhck12 = deepcopy(Mhck1)
            if is_Ms_nuT == 4                                # for `nModk1 = 1`
                rtghj
                # Renew the values of `Mhck1` according to the quantities of `Mck1`
                MhcknuT!(Mhck12,Mck11,ρk1,vathk12,LMk,ns,nMjMs,zeros(Int64,ns);is_renorm=is_renorm)
            else
                MhckMck!(Mhck12,Mck11[1:njMs,:,:],ρk1,LMk,ns,nMjMs,vathk12)
            end
        
            # # Calculate the parameters `uai`
            uaik12L = deepcopy(uhkL)
            submoment!(uaik12L, Mhck12, ns)
    
            # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
            for isp in nsp_vec
                uaik12L[isp] = uaik12L[isp] .- uCk / vathk12[isp]
            end
    
            vhek12 = deepcopy(vhe)
            # for isp in nsp_vec
            #     vhek12[isp] = vhe[isp] * (vathk[isp] / vathk[isp])
            # end
                
            # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
            KvL0k12 = deepcopy(fvL0k)
            LM1k12, KvL0k12 = fvLDMz!(KvL0k12, vhek12, LMk, ns, uaik12L; 
                L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)
    
            Mhc2k121 = deepcopy(Mhck12)
            MsnnEvens!(Mhc2k121, errMhc, KvL0k12, vhek12, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
            Mck121 = deepcopy(Mck)
            MckMhck!(Mck121,Mhc2k121,ρk1,vathk12,LMk,ns,nMjMs)
            # @show Mhc2k121[2] - Mhck12[2]
            # @show Mhc2k121[1] - Mhck12[1]
            @show Mck121[:,:,2] - Mck11[:,:,2]
            @show Mck121[:,:,1] - Mck11[:,:,1]
            # 
            @show KvL0k12[1][1,1]
            @show vathk12 ./ vathk .- 1
            
            dfgbn
        end

        if 2 == 32
            Mhc2k101 = deepcopy(Mhck1)
            MsnnEvens!(Mhc2k101, errMhc, fvL0k, vhe, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
            Mck101 = deepcopy(Mck)
            MckMhck!(Mck101,Mhc2k101,ρk1,vathk,LMk,ns,nMjMs)
            @show Mck101 - Mck0

            # Calculating the rate of change of the kinetic moments in the Lagrange coordinate system.
            Rck101 = deepcopy(Rc)
            aa = Rck101[1:njMs,:,:]
            if gridv_type == :uniform
                dtMcsd2l!(aa,err_Rc,δtfh,vhe,nMjMs,ρk1,vathk,LMk,ns;is_renorm=is_renorm)
            elseif gridv_type == :chebyshev
                if is_dtf_MS_IK
                    dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG[1],nMjMs,ρk1,vathk,LMk,ns;
                              is_renorm=is_renorm,is_norm_error=is_norm_error)
                else
                    dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG,nMjMs,ρk1,vathk,LMk,ns;
                              is_renorm=is_renorm,is_norm_error=is_norm_error)
                end
            end
            Rck101[1:njMs,:,:] = aa
            
            Mck100 = Mck101 +  dtkk1 * Rck101
            # @show Mck100[:,:,2] - Mck1[:,:,2]
            # @show Mck100[:,:,1] - Mck1[:,:,1]
            # retghn
        end

        ########################################################################
        fvL0k1 = fvL0k + dtkk1 * δtfh          # `δₜn̂ = R̂(0,0,0) ~ 0`, but `δₜK̂` is wrong
        for isp in nsp_vec
            fvL0k1[isp] *= (vathk1[isp] / vathk[isp])^3
        end
        
        # @show fvL0k[1][1,1]
        # @show fvL0k1[1][1,1]
        Mhc2k10 = deepcopy(Mhck1)
        for isp in nsp_vec
            vhek1[isp] = vhe[isp] * (vathk[isp] / vathk1[isp])
        end
        MsnnEvens!(Mhc2k10, errMhc, fvL0k1, vhek1, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
        Mck10 = deepcopy(Mck)
        MckMhck!(Mck10,Mhc2k10,ρk1,vathk1,LMk,ns,nMjMs)
        @show Mhc2k10[2] - Mhck1[2]
        @show Mhc2k10[1] - Mhck1[1]
        @show Mck10[:,:,2] - Mck1[:,:,2]
        @show Mck10[:,:,1] - Mck1[:,:,1]

        # 
        DKing = (KvL0k1 - fvL0k1)
        # ylabel = L"\Delta \hat{f}_l^0"
        ylabel = "Δf̂ₗ⁰"
        isp = 1
        label = string("a")
        pDKinga = plot(vhek1[isp], DKing[isp],label=label,ylabel=ylabel,line=(3,:auto))
        isp = 2
        # label = string(spices[isp])
        label = string("b")
        pDKingb = plot(vhek1[isp], DKing[isp],label=label,line=(3,:auto))
        # display(plot(pDKinga,pDKingb,layout=(2,1)))

        # RDKing = (KvL0k1 - fvL0k1) ./ fvL0k1
        ylabel = "RΔf̂ₗ⁰"
        isp = 1
        DKing[isp] ./= fvL0k1[isp]
        # label = string(spices[isp])
        label = string("a")
        pRDKinga = plot(vhek1[isp], DKing[isp],label=label,ylabel=ylabel,line=(3,:auto))
        isp = 2
        DKing[isp] ./= fvL0k1[isp]
        # label = string(spices[isp])
        label = string("b")
        pRDKingb = plot(vhek1[isp], DKing[isp],label=label,line=(3,:auto))
        # display(plot(pRDKinga,pRDKingb,layout=(2,1)))

        display(plot(pDKinga,pDKingb,pRDKinga,pRDKingb,layout=(2,2)))
        # wedrfgnh222
    end
    
    if ratio_Rdtfln > 0.0
        # @show 11, dtk
        dtk = dt_dtfln(dtk,Rdtfln3;ratio_Rdtfln=ratio_Rdtfln)
        # @show 22, dtk
    end

    return dtk
end

# [ns=2], `nMod[:] .= 1`
function dtMcab2!(Mck::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    Rc::AbstractArray{T,N},edtnIKTs::AbstractArray{T,N2},err_Rc::AbstractArray{T,N},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2}, LMk::Vector{Int64}, LM1::Int64,
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak::AbstractVector{T},uak::AbstractVector{T},vathk::AbstractVector{T},
    nMjMs::Vector{Int64},DThk::AbstractVector{T},dtk::T;ns::Int64=2,
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
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,N,N2}
    
    # Computing the relative velocity `uCk` which satisfing `ua = - ub` in the Lagrange coordinate system with `uCk`
    uCk = uCab(uak, vathk)
    # @show 3, uCk
    uakL = uak .- uCk
    uhkL = uakL ./ vathk
    sum(uhkL) ≤ epsT1000 || @warn("ab: The relative velocity of the Lagrange coordinate system `uCk` is not optimezed,",sum(uhkL)) 
    # @show uhkL
    
    # Updating the meshgrids on the velocity axis in Lagrange coordinate system.
    nc0, nck = zeros(Int64,ns), zeros(Int64,ns)
    if gridv_type == :uniform
        vhe = Vector{StepRangeLen}(undef, ns)
    elseif gridv_type == :chebyshev
        vhe = Vector{AbstractVector{T}}(undef, ns)
    else
        sdfgh
    end
    vhk = Vector{AbstractVector{T}}(undef, ns)
    nvlevele0 = Vector{Vector{Int64}}(undef, ns)
    nvlevel0 = Vector{Vector{Int64}}(undef, ns)

    nsp_vec = 1:ns
    # `L_limit + 1 = LM1 + 1` denotes an extra row is given which may be used.
    fvL0k = Vector{Matrix{T}}(undef,ns) 
    for isp in nsp_vec
        fvL0k[isp] = zeros(nvG[isp],LM1+1)
    end

    δtfh = Vector{Matrix{T}}(undef,ns)     # δtfvLa

    is_dtf_MS_IK = false
    if is_dtf_MS_IK
        # Updating the meshgrids on the velocity axis when `ns == 2, nMod .== 1` which means `nai = vthi .== 1`
        nvG, nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0 = vHadapt1D(
            nvG[1], ocp[1], vGdom[:,1], abs(uhkL[2]);
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
            
        # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        LM1k = fvLDMz!(fvL0k, vhe, LMk, uhkL; 
            L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)

        for isp in nsp_vec
            δtfh[isp] = zeros(T,nvG[isp],LM1k)
        end
    
        # # Updating the FP collision terms according to the `FPS` operators.
        FP0D2Vab2!(δtfh,fvL0k,vhk,nvG[1],nc0,nck,ocp[1],
                nvlevele0,nvlevel0,LMk,LM1k,
                CΓ,εᵣ,ma,Zq,spices,nak,uhkL,vathk;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs, 
                is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    else 
        vHadapt1D!(vhe,vhk, vGdom, nvG, nc0, nck, ocp, 
              nvlevele0, nvlevel0, uhkL, ns;
              eps_fup=eps_fup,eps_flow=eps_flow,
              maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
              abstol=abstol,reltol=reltol,
              vadaptlevels=vadaptlevels,gridv_type=gridv_type,
              is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
            
        # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        LM1k, fvL0k = fvLDMz!(fvL0k, vhe, LMk, ns, uhkL; 
            L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)
        
        for isp in nsp_vec
            δtfh[isp] = zeros(T,nvG[isp],LM1k)
        end
        
        # # Updating the FP collision terms according to the `FPS` operators.
        FP0D2Vab2!(δtfh,fvL0k,vhk,nvG,nc0,nck,ocp,
                nvlevele0,nvlevel0,LMk,LM1k,
                CΓ,εᵣ,ma,Zq,spices,nak,uhkL,vathk;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs, 
                is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    end

    # Checking the change rate of harmonic of distribution function
    Rdtfln3 = zeros(T,ns)
    for isp in nsp_vec
        Rdtfln3[isp] = δtfh[isp][3,1] ./ fvL0k[isp][3,1]
    end
    # @show Rdtfln3

    ylabel = string("3,Rdtfln")

    isp = 1
    Rdtfln = δtfh[isp][:,1] ./ fvL0k[isp][:,1]
    label = string("a")
    pRdtflna = plot(vhe[isp],Rdtfln,label=label,ylabel=ylabel)

    isp = 2
    Rdtfln = δtfh[isp][:,1] ./ fvL0k[isp][:,1]
    label = string("b")
    pRdtflnb = plot(vhe[isp],Rdtfln,label=label)
    display(plot(pRdtflna,pRdtflnb,layout=(2,1)))


    # Checking the conservation laws of the renormalized distribution function `fvL0k1`
    nIKTh = zeros(4, ns)
    if gridv_type == :uniform
        nIKThs!(nIKTh, fvL0k, vhe, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
    elseif gridv_type == :chebyshev
        if is_dtf_MS_IK
            nIKThs!(nIKTh, fvL0k, vhe, nvG[1], ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
        else
            nIKThs!(nIKTh, fvL0k, vhe, nvG, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
        end
    end

    ρk1 = ma .* nak

    # if is_plot_DflKing
    #     Mhck0 = deepcopy(Mhck1)
    #     MsnnEvens!(Mhck0, errMhc, fvL0k, vhe, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
    #     Mck0 = deepcopy(Mck)
    #     MckMhck!(Mck0,Mhck0,ρk1,vathk,LMk,ns,nMjMs)
    #     # @show Mhck0[2] - Mhck1[2]
    #     # @show Mhck0[1] - Mhck1[1]
    #     # @show Mck0[:,:,2] - Mck[:,:,2]
    #     # @show Mck0[:,:,1] - Mck[:,:,1]
    # end

    # Calculating the rate of change of the kinetic moments in the Lagrange coordinate system.
    aa = Rc[1:njMs,:,:]
    if gridv_type == :uniform
        dtMcsd2l!(aa,err_Rc,δtfh,vhe,nMjMs,ρk1,vathk,LMk,ns;is_renorm=is_renorm)
    elseif gridv_type == :chebyshev
        if is_dtf_MS_IK
            dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG[1],nMjMs,ρk1,vathk,LMk,ns;
                      is_renorm=is_renorm,is_norm_error=is_norm_error)
        else
            dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG,nMjMs,ρk1,vathk,LMk,ns;
                      is_renorm=is_renorm,is_norm_error=is_norm_error)
        end
    end
    Rc[1:njMs,:,:] = aa

    DThk[:] = nIKTh[4, :]
    if maximum(DThk) > atol_IKTh
        @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a = 3/2 * T̂a + ûa²`", DThk)
        if maximum(DThk) > rtol_IKTh 
            printstyled("`errTh < rtol_IKTh` which means the convergence of the algorithm is falure!",color=:red,"\n")
        end
    end
    
    # Computing the relative change rate of thermal velocity in the Lagrange coordinate system 
    # according to   # `Rdtvth = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`, `𝒲 = RdtK - 2 * IhL .* RdtI`
    Rdtvth = Rc[njMs+1,1,:]
    Rc[njMs+1,1,:] = RdtvthRc!(Rdtvth,uhkL,Rc[1,2,:],Rc[2,1,:],ns)

    # Verifying the mass, momentum and total energy conservation laws of `δtfa` in Lagrange coordinate system.
    edtnIKTs[1,:] = deepcopy(err_Rc[1,1,:])           # edtn
    edtnIKTs[2,:] = deepcopy(err_Rc[1,2,:])           # edtI
    edtnIKTs[3,:] = deepcopy(err_Rc[2,1,:])           # edtK
    # Checking the constraint according to (eRdtKI): `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
    edtnIKTs[4,:] = eRdtKIRc!(edtnIKTs[4,:],uhkL,Rc[1,2,:],Rc[2,1,:],Rdtvth,ns)
    
    # Conservation in discrete
    if is_enforce_errdtnIKab
        dtnIKposteriorC!(Rc,err_Rc,nMjMs)
    else
        dtnIKTs = zeros(4,ns)
        dtnIKTs[4,:] = Rc[njMs+1,1,:]
        if is_check_conservation_dtM 
            if norm(dtnIKTs[1,:])  ≥ epsTe6
                @warn("δₜn̂a: The mass conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",dtnIKTs[1,:])
            end
      
            # RD_I
            if abs(sum(dtnIKTs[2,:])) > epsTe6
                RDIab = abs(dtnIKTs[2,1] - dtnIKTs[2,2])
                if RDIab ≠ 0.0
                    err_RdtI = sum(dtnIKTs[2,:]) / RDIab
                    if err_RdtI > epsTe6
                        @warn("δₜÎa: The momentum conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_RdtI)
                    end
                end
            end
        
            # RD_K
            if abs(sum(dtnIKTs[3,:])) > epsTe6
                RDKab = abs(dtnIKTs[3,1] - dtnIKTs[3,2])
                if RDKab ≠ 0.0
                    err_RdtK = sum(dtnIKTs[3,:]) / RDKab
                    if err_RdtK > epsTe6
                        @warn("δₜnK̂a: The the total energy conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_RdtK)
                    end
                end
            end
        end
    end

    # Computing the moments in the Lagrange coordinate system.
    IkL, KkL = zeros(T,ns), zeros(T,ns)
    IK2_lagrange!(IkL, KkL, ρk1, uakL, vathk)

    # Evoluating the timestep owing to the total momentums and total energy in the Lagrange coordinate system.
    # if is_renorm
        dtI = deepcopy(Rc[1,2,:])              # dtI
        dtK = Rc[2,1,:] * CMcKa                # dtK
    # end
    dtk = dt_RdtnIK2(dtk,Rc[1,2,:],Rc[2,1,:] * CMcKa,uakL,IkL,KkL; rtol_DnIK=rtol_DnIK)
    @show dtK

    if is_plot_DflKing
        dtkk1 = dtk / 100

        Mck1 = deepcopy(Mck)
        Mck1integral0!(Mck1,Rc,ns,dtkk1)

        # Calculate the parameters `nak1,vathk1,Iak1,Kak1` from `Mck1`
        nak1,vathk1,Iak1,Kak1,δvathk1 = zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns)
        nIKT_update!(nak1,vathk1,Iak1,Kak1,δvathk1,ma,ns,Mck1;
                    is_vth_ode=is_vth_ode,is_corrections_na=is_corrections[1])
        
        # # Computing the re-normalized moments 
        if is_Ms_nuT == 4                                # for `nModk1 = 1`
            rtghj
            # Renew the values of `Mhck1` according to the quantities of `Mck1`
            MhcknuT!(Mhck1,Mck1,ρk1,vathk1,LMk,ns,nMjMs,zeros(Int64,ns);is_renorm=is_renorm)
        else
            MhckMck!(Mhck1,Mck1[1:njMs,:,:],ρk1,LMk,ns,nMjMs,vathk1)
        end
        
        # # Calculate the parameters `uai`
        uaik1L = deepcopy(uhkL)
        submoment!(uaik1L, Mhck1, ns)

        Msnnt = deepcopy(Mhck1)
        Msnnt = MsnntL2fL0(Msnnt,nMjMs,LMk,ns,uaik1L;is_renorm=is_renorm)
        # @show Msnnt - Mhck1

        # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        for isp in nsp_vec
            uaik1L[isp] = uaik1L[isp] .- uCk / vathk1[isp]
        end

        vhek1 = deepcopy(vhe)
            
        # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        KvL0k1 = deepcopy(fvL0k)
        LM1k1, KvL0k1 = fvLDMz!(KvL0k1, vhek1, LMk, ns, uaik1L; 
            L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)

        # @show norm(KvL0k1 -(fvL0k))
        # MsnnEvens!(Mhck0, errMhc, fvL0k, vhe, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
        # Mck0 = deepcopy(Mck)
        # MckMhck!(Mck0,Mhck0,ρk1,vathk,LMk,ns,nMjMs)
        
        Mhc2k11 = deepcopy(Mhck1)
        MsnnEvens!(Mhc2k11, errMhc, KvL0k1, vhe, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
        Mck11 = deepcopy(Mck)
        MckMhck!(Mck11,Mhc2k11,ρk1,vathk,LMk,ns,nMjMs)
        # @show Mhc2k11[2] - Mhck1[2]
        # @show Mhc2k11[1] - Mhck1[1]
        # @show Mck11[:,:,2] - Mck1[:,:,2]
        # @show Mck11[:,:,1] - Mck1[:,:,1]
        # wesdfvgb333
        # # 
        # @show KvL0k1[1][1,1]
        # @show vathk1 ./ vathk .- 1

        #####################3 double
        if 1 == 2
            # Calculate the parameters `nak12,vathk12,Iak12,Kak12` from `Mck12`
            nak12,vathk12,Iak12,Kak12,δvathk12 = zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns), zeros(T,ns)
            nIKT_update!(nak12,vathk12,Iak12,Kak12,δvathk12,ma,ns,Mck11;
                        is_vth_ode=is_vth_ode,is_corrections_na=is_corrections[1])
            
            # # Computing the re-normalized moments 
            Mhck12 = deepcopy(Mhck1)
            if is_Ms_nuT == 4                                # for `nModk1 = 1`
                rtghj
                # Renew the values of `Mhck1` according to the quantities of `Mck1`
                MhcknuT!(Mhck12,Mck11,ρk1,vathk12,LMk,ns,nMjMs,zeros(Int64,ns);is_renorm=is_renorm)
            else
                MhckMck!(Mhck12,Mck11[1:njMs,:,:],ρk1,LMk,ns,nMjMs,vathk12)
            end
        
            # # Calculate the parameters `uai`
            uaik12L = deepcopy(uhkL)
            submoment!(uaik12L, Mhck12, ns)
    
            # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
            for isp in nsp_vec
                uaik12L[isp] = uaik12L[isp] .- uCk / vathk12[isp]
            end
    
            vhek12 = deepcopy(vhe)
            # for isp in nsp_vec
            #     vhek12[isp] = vhe[isp] * (vathk[isp] / vathk[isp])
            # end
                
            # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
            KvL0k12 = deepcopy(fvL0k)
            LM1k12, KvL0k12 = fvLDMz!(KvL0k12, vhek12, LMk, ns, uaik12L; 
                L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)
    
            Mhc2k121 = deepcopy(Mhck12)
            MsnnEvens!(Mhc2k121, errMhc, KvL0k12, vhek12, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
            Mck121 = deepcopy(Mck)
            MckMhck!(Mck121,Mhc2k121,ρk1,vathk12,LMk,ns,nMjMs)
            # @show Mhc2k121[2] - Mhck12[2]
            # @show Mhc2k121[1] - Mhck12[1]
            @show Mck121[:,:,2] - Mck11[:,:,2]
            @show Mck121[:,:,1] - Mck11[:,:,1]
            # 
            @show KvL0k12[1][1,1]
            @show vathk12 ./ vathk .- 1
            
            dfgbn
        end

        if 2 == 32
            Mhc2k101 = deepcopy(Mhck1)
            MsnnEvens!(Mhc2k101, errMhc, fvL0k, vhe, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
            Mck101 = deepcopy(Mck)
            MckMhck!(Mck101,Mhc2k101,ρk1,vathk,LMk,ns,nMjMs)
            @show Mck101 - Mck0

            # Calculating the rate of change of the kinetic moments in the Lagrange coordinate system.
            Rck101 = deepcopy(Rc)
            aa = Rck101[1:njMs,:,:]
            if gridv_type == :uniform
                dtMcsd2l!(aa,err_Rc,δtfh,vhe,nMjMs,ρk1,vathk,LMk,ns;is_renorm=is_renorm)
            elseif gridv_type == :chebyshev
                if is_dtf_MS_IK
                    dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG[1],nMjMs,ρk1,vathk,LMk,ns;
                              is_renorm=is_renorm,is_norm_error=is_norm_error)
                else
                    dtMcsd2l!(aa,err_Rc,δtfh,vhe,nvG,nMjMs,ρk1,vathk,LMk,ns;
                              is_renorm=is_renorm,is_norm_error=is_norm_error)
                end
            end
            Rck101[1:njMs,:,:] = aa
            
            Mck100 = Mck101 +  dtkk1 * Rck101
            # @show Mck100[:,:,2] - Mck1[:,:,2]
            # @show Mck100[:,:,1] - Mck1[:,:,1]
            # retghn
        end

        ########################################################################
        fvL0k1 = fvL0k + dtkk1 * δtfh          # `δₜn̂ = R̂(0,0,0) ~ 0`, but `δₜK̂` is wrong
        for isp in nsp_vec
            fvL0k1[isp] *= (vathk1[isp] / vathk[isp])^3
        end
        
        # @show fvL0k[1][1,1]
        # @show fvL0k1[1][1,1]
        Mhc2k10 = deepcopy(Mhck1)
        for isp in nsp_vec
            vhek1[isp] = vhe[isp] * (vathk[isp] / vathk1[isp])
        end
        MsnnEvens!(Mhc2k10, errMhc, fvL0k1, vhek1, nMjMs, LMk, ns; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
        Mck10 = deepcopy(Mck)
        MckMhck!(Mck10,Mhc2k10,ρk1,vathk1,LMk,ns,nMjMs)
        @show Mhc2k10[2] - Mhck1[2]
        @show Mhc2k10[1] - Mhck1[1]
        @show Mck10[:,:,2] - Mck1[:,:,2]
        @show Mck10[:,:,1] - Mck1[:,:,1]

        # 
        DKing = (KvL0k1 - fvL0k1)
        # ylabel = L"\Delta \hat{f}_l^0"
        ylabel = "Δf̂ₗ⁰"
        isp = 1
        label = string("a")
        pDKinga = plot(vhek1[isp], DKing[isp],label=label,ylabel=ylabel,line=(3,:auto))
        isp = 2
        # label = string(spices[isp])
        label = string("b")
        pDKingb = plot(vhek1[isp], DKing[isp],label=label,line=(3,:auto))
        # display(plot(pDKinga,pDKingb,layout=(2,1)))

        # RDKing = (KvL0k1 - fvL0k1) ./ fvL0k1
        ylabel = "RΔf̂ₗ⁰"
        isp = 1
        DKing[isp] ./= fvL0k1[isp]
        # label = string(spices[isp])
        label = string("a")
        pRDKinga = plot(vhek1[isp], DKing[isp],label=label,ylabel=ylabel,line=(3,:auto))
        isp = 2
        DKing[isp] ./= fvL0k1[isp]
        # label = string(spices[isp])
        label = string("b")
        pRDKingb = plot(vhek1[isp], DKing[isp],label=label,line=(3,:auto))
        # display(plot(pRDKinga,pRDKingb,layout=(2,1)))

        display(plot(pDKinga,pDKingb,pRDKinga,pRDKingb,layout=(2,2)))
        wedrfgnh
    end
    
    if ratio_Rdtfln > 0.0
        # @show 11, dtk
        dtk = dt_dtfln(dtk,Rdtfln3;ratio_Rdtfln=ratio_Rdtfln)
        # @show 22, dtk
    end

    return dtk
end

