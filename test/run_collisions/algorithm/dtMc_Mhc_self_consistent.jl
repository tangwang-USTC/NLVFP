dtk00 = 1e-8
tk00 = 0.0
k00 = 1
Mck10 = deepcopy(Mc2)
Rck10 = deepcopy(dtMc2)
RDMck10 = zero.(Mck10)
edtnIKTs0 = deepcopy(edtnIKTscc)
errRhck120 = zero.(Rck10[1:njMs, :, 1:2])

Mhck10 = deepcopy(Mhc2)
Mhck0 = deepcopy(Mhck10)
Rhck1 = deepcopy(Mhck10)
Rhck = deepcopy(Mhck10)
dtMhck1 = deepcopy(Mhck10)
dtMhck = deepcopy(Mhck0)
errMhc = deepcopy(Mhck0)
RDMhck1max = ones(ns)

naik10, uaik10, vthik10, nModk10 = deepcopy(nai), deepcopy(uai), deepcopy(vthi), deepcopy(nMod)
naik0, uaik0, vthik0, nModk0 = deepcopy(nai), deepcopy(uai), deepcopy(vthi), deepcopy(nMod)
nak10, vathk10 = deepcopy(na), deepcopy(vth)
Rdtsabk10 = dtsabk0 ./ sum(sak0)
DThk10 = zeros(ns)
Iak10, Kak10, vathk1i0 = deepcopy(Ia), deepcopy(Ka), deepcopy(vth)
Nspan_optim_nuTi0 = Nspan_nuTi_maxmin[2] * ones(3) 

Mck0, Rck0 = deepcopy(Mck10), deepcopy(Rck10)
if ns == 2
    Mck1integrali_rs2!(Mck10, Rck10, RDMck10, edtnIKTs0, errRhck120, 
        Mhck10, Mhck0, Rhck1, Rhck, dtMhck1, dtMhck, 
        errMhc, errMhcop2, RDMhck1max, nMjMs,
        nvG, ocp, vGdom, LM, LM1, 
        naik10, uaik10, vthik10, nModk10, naik0, uaik0, vthik0, nModk0, 
        CΓ, εᵣ, ma, Zq, spices0, nak10, vathk10, 
        Rdtsabk10, DThk10, Iak10, Kak10, vathk1i0, 
        Mck0, Rck0, na, vth, Rdtsabk10, Nspan_optim_nuTi0, 
        NK0, NKmax0, k00, tk00, dtk00;
    
        orderEmbeded=orderRK,iterEmbeded=iterRK,
        Nspan_nuTi_max=Nspan_nuTi_maxmin,
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
        
        is_vth_ode=is_vth_ode, is_corrections=is_corrections,
        is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
        dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
        is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
        is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
        is_nuTi_initial=true,is_fixed_NK=is_fixed_NK,
        is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=false)
    1
else
end