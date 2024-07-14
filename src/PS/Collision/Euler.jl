
i_iter += 1
println("....................")
@show i_iter
nak1[:] = deepcopy(nak)
vathk1[:] = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
# Rck1i[:,:,:] = (Rck + Rck1i) / 2        # Rck1k = Rc_(k+1/2)
dtk1 = 1dtk

dtk1 = Mck1integral!(Mck1, Rck1i, Mck, Rck, edtnIKTs, err_Rck12, 
    Mhck1, errMhc, errMhcop, nMjMs,
    nvG, ocp, vGdom, LMk, LM1k, 
    naik1, uaik1, vthik1, nModk1, naik, uaik, vthik, nModk, 
    CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, Rdtsabk1, Rdtsabk, 
    DThk1, Iak1, Kak1, δvathk1, Nspan_optim_nuTi, kt, tk, dtk1;
    Nspan_nuTi_max=Nspan_nuTi_max,
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
    is_vth_ode=is_vth_ode,
    is_corrections=is_corrections,is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
    is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
    dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
    is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt)
δvathi = vathk1i ./ vathk1 .- 1                # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
ratio_vathi = δvathi - δvathi_up

# # Rck1 = Rck1i
if norm(ratio_vathi) ≤ rtol_vthi || norm(δvathi) ≤ atol_vthi
    break
end
# @show 72, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
# @show 72, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
@show 72, nModk1[1], length(naik1[1]), sum(naik1[1][1:nModk1[1]] .* vthik1[1][1:nModk1[1]].^2) - 1
@show 72, nModk1[2], length(naik1[2]), sum(naik1[2][1:nModk1[2]] .* vthik1[2][1:nModk1[2]].^2) - 1
Kak1 = Mck1[2,1,:] * CMcKa 
Iak1 = Mck1[1,2,:]
RDKab = sum(Kak1) / Kab0 - 1
@show 7772,nModk1, RDKab

vathk1i[:] = deepcopy(vathk1)
@show 1,i_iter, δvathk1, δvathi