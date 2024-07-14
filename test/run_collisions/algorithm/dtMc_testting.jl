

dtMc2 = zeros(njMs+1,LM1,ns)           # dtMc2[njMs+1,1,:] = w3k = Rdtvth = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3
err_dtnIKs = zeros(4,ns)
DThk1 = zeros(ns)
err_dtMck12 = zeros(njMs,LM1,2)
err_dtnIKs2 = zeros(4,2)
if is_fixed_timestep
    dtk0 = 10.0
else
    dtk0 = tau[1] / nτ
end

@show nModcc
@show 700, sum(naicc[1] .* vthicc[1].^2) - 1
@show 700, sum(naicc[2] .* vthicc[2].^2) - 1
if ns == 2
    if dtk_order_Rc == :min && is_dtk_order_Rcaa == false
        dtk0 = dtMcab2!(dtMc2,err_dtnIKs2,err_dtMck12,nMjMs,
               nvG,ocp,vGdom,LM,LM1,naicc,uaicc,vthicc,nModcc,
               CΓ,εᵣ,ma,Zq,spices0,na,Ia ./ ρa,1vth,DThk1,dtk0;
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
               is_extrapolate_FLn=is_extrapolate_FLn_initial,rtol_DnIK=rtol_DnIK,
               is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
    else
        Mc2cc = deepcopy(Mc2)
        dtk0 = dtMcab2!(dtMc2,err_dtnIKs2,err_dtMck12,Mc2cc,nMjMs,
               nvG,ocp,vGdom,LM,LM1,naicc,uaicc,vthicc,nModcc,
               CΓ,εᵣ,ma,Zq,spices0,na,Ia ./ ρa,1vth,DThk1,dtk0;
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
               is_extrapolate_FLn=is_extrapolate_FLn_initial,rtol_DnIK=rtol_DnIK,
               is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
    end
else
    dtMc22 = zero.(dtMc2[:,:,1:2])
    DThk12 = zeros(2)
    nai2 = Vector{AbstractVector{datatype}}(undef,2)
    uai2 = Vector{AbstractVector{datatype}}(undef,2)
    vthi2 = Vector{AbstractVector{datatype}}(undef,2)
    dtk0 = dtMcab!(dtMc2, err_dtnIKs, 
        dtMc22, err_dtnIKs2, [0.0], err_dtMck12, DThk12, nai2, uai2, vthi2, Mhc2, 
        nvG, ocp, vGdom, LM, LM1, nai, uai, vthi, 
        CΓ, εᵣ, ma, Zq, spices0, na, ua, vth, ns, nMod, nMjMs, DThk1,
        Nspan_optim_nuTi_initial,dtk0;Nspan_nuTi_max=Nspan_nuTi_maxmin,
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
        is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
            
        eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
        abstol=abstol,reltol=reltol,
        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,

        is_extrapolate_FLn=is_extrapolate_FLn_initial,rtol_DnIK=rtol_DnIK,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
end
w3k1 = copy(dtMc2[njMs+1,1,:])
dtMc2[njMs+1,1,:] .*= vth 

@show nModcc
if norm(u0) ≤ epsT1000
    DKha00 = abs(sum(naicc[1][1:nModcc[1]] .* vthicc[1][1:nModcc[1]].^2) - 1)
    DKhb00 = abs(sum(naicc[2][1:nModcc[2]] .* vthicc[2][1:nModcc[2]].^2) - 1)
else
    sgfhnm
end
DKha00 + DKha00 ≤ epsT1000 || @error("characteristic parameters are not converged! `DKh=`",(DKha00, DKha00))

Rc2 = deepcopy(dtMc2[1:njMs,:,:])
errRc2 = deepcopy(Rc2)
Rcsd2l!(Rc2,errRc2,dtfvL0,vhe,nMjMs,ma.*na,vth,LM,ns;
        is_renorm=is_renorm, is_norm_error=is_norm_error)

Rhc2 = deepcopy(Mhc2)
errRhc = deepcopy(Rhc2)
RhnnEvens!(Rhc2, errRhc, dtfvL0, vhe, nMjMs, LM, ns; 
        is_renorm=is_renorm, is_err_renorm=is_err_renorm, L_Mh_limit=L_Mh_limit)

# Conservative and non-conservative moments
RdtMcs = deepcopy(dtMc2[1:njMs,:,1])
RRcsd2l!(RdtMcs, dtMc2[1:njMs,:,:], njMs, ρa, Ia, Ka)

if 1 == 2
     
    dtMhc2 = deepcopy(Mhc2)
    dtMhcsd2l!(dtMhc2,Mhc2,Rhc2,w3k1,nMjMs,LM,ns)
    DdtMhc2 = dtMhc2[1] + dtMhc2[2]
    DdtMc2 = dtMc2[1:njMs,:,1] + dtMc2[1:njMs,:,2]
end

# H-theorem
dtKIak0 = zeros(2,ns)               #  [[K, I], ns]
dtKIak0[1,:] = Rc2[2,1,:] * CMcKa     # dtKa
dtKIak0[2,:] = Rc2[1,2,:]             # dtIa

# [ns], Updating the entropy and its change rate with assumpation of dauble-Maxwellian distribution
sak0 = zeros(ns)
entropy_fDM!(sak0,ma,na,vth,Ia2,Ka2,ns)                        

# dtsabk0 = dtsak0 + dtsbk1
# [ns = 2] Iahk = uak1 ./ vth
dtsabk0 = entropy_rate_fDM(ma,vth,Ia2 ./ (ρa .* vth),dtKIak0[2,:],dtKIak0[1,:],ns)

Rdtsabk0 = dtsabk0 / sum(sak0)
@show dtsabk0, sak0
# Rdtsabk0 = entropyN_rate_fDM(ma,na,vth,Ia2,Ka2,dtKIak0[2,:],dtKIak0[1,:],ns)
@show Rdtsabk0
