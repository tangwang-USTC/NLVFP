
    # IKk = [[[K], [I]], nMod, ns] 
    IKk1 = deepcopy(IKk0)
    DThk = zeros(2)
    nk11 = deepcopy(nai)
    uk11 = deepcopy(uai)
    vthk11 = deepcopy(vthi)
    nuTk1_sub_initial!(nk11,uk11,vthk11,na,vth,nai0,uai0,vthi0,nMod0,ns)

    nak11,Iak11,Kak11,vathk11 = zeros(ns), zeros(ns), zeros(ns), zeros(ns)
    submomentN!(nak11,Iak11,Kak11,vathk11,nai,uai,vthi,ma,nk11,uk11,vthk11,nMod0,ns)

    dtIKk1 = zero.(IKk0)
    dtsaak1 = dtsaa_initial(nMod,ns)
    saak1 = zeros(maximum(nMod),ns)
    dtnIKs0 = zeros(4,2)
    edtnIKs0 = zeros(4,2)

    is_dtIKabplot_testing = false
    if is_dtIKabplot_testing
        dt_inital_0 = 1 * pst0["dt"]
        dt_inital_0, dHvL = dtIKabplot!(dtnIKs0,
               vhk[1],nvG[1],ocp[1],vGdom[:,1],nvlevele0[1],nvlevel0[1],LM,
               CΓ,εᵣ,ma,Zq,nak11,ua,vathk11,DThk,zeros(2),zeros(2),dt_inital_0;
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
               is_extrapolate_FLn=is_extrapolate_FLn)
        pstk["dt"] > dt_inital_0 || @warn("0: The initial time step is decided by `dtIK/Ik` in Lagrange coordinate instead of `tauk`!",dt_inital_0)
        pstk["dt"] = min(pstk["dt"], dt_inital_0)
    end
    
    dt_inital = 1 * pst0["dt"]
    dt_inital, dtIKk_update!(dtIKk1,IKk1,dtsaak1,saak1,
                dtnIKs0,edtnIKs0,nvG[1],ocp[1],vGdom[:,1],LM,
                CΓ,εᵣ,ma,Zq,nk11,uk11,vthk11,spices0,nMod0,DThk,dt_inital;
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
    pstk["dt"] > dt_inital || @warn("1: The initial time step is decided by `dtIK/Ik` in Lagrange coordinate instead of `tauk`!",dt_inital)
    pstk["dt"] = min(pstk["dt"], dt_inital)
    if is_dtIKabplot_testing == 3
        @show pst0["dt"], dt_inital_0, dt_inital
    else
        @show pst0["dt"], dt_inital
    end