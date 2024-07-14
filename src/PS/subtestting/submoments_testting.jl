 
nMod = deepcopy(nMod0)
nai = deepcopy(nai0)     # `n̂a = naᵢ / na`
uai = deepcopy(uai0)     # `ûa = uaᵢ / vth`
vthi = deepcopy(vthi0)    # `v̂th = vathᵢ / vth`
# nha,vhth,uha = zeros(ns), zeros(ns), zeros(ns)

Nspan_optim_nuTi_initial = Nspan_nuTi_maxmin[2] * ones(3)    # for `dtMc_testting.jl`
dtk0 = 10.0
naicc,uaicc,vthicc,nModcc = deepcopy(nai),deepcopy(uai),deepcopy(vthi),1nMod
Rdtsabk1cc,Rdtsabkcc = 0.0, 0.0
edtnIKTscc = zeros(4,ns)
errMhcop2 = zero.(Mhc2)
dtk0 = submoment!(naicc,uaicc,vthicc,nModcc,
        1LM,nai,uai,vthi,nMod,Mhc2,errMhcop2,nMjMs,
        edtnIKTscc,Rdtsabk1cc,Rdtsabkcc,ns,
        Nspan_optim_nuTi_initial,3,3,1,0.0,dtk0;
        Nspan_nuTi_max=Nspan_nuTi_maxmin,
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
        optimizer=optimizer,factor=factor,autodiff=autodiff,
        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
        is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
        is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
        is_nuTi_initial=true,is_fixed_NK=is_fixed_NK,is_nMod_update_back=is_nMod_update_back)
uhsum, vhthsum = zeros(ns), zeros(ns)
uhsum, vhthsum = nuTsNorm!(uhsum, vhthsum, nai,uai,vthi)
nhsum = [sum(nai[k]) for k in 1:ns]
errnh = nhsum - nha
erruh = uhsum - uha
errvhth = vhthsum - vhth
println()
@show errnh
@show erruh
@show errvhth
println()
@show nai - nai0
@show uai - uai0
@show vthi - vthi0;
if is_fvL_CP
    if norm(vthi - vthi0) ≥ epsT1000
        @error("`vthi` must be approximately equal to `vthi0!!!")
        for isp30 in 1:ns
            uai[isp30] = deepcopy(uai0[isp30])
            vthi[isp30] = deepcopy(vthi0[isp30])
        end
    end
else
    if norm(vthi - vthi0) ≥ epsT1000
        @error("`vthi` must be approximately equal to `vthi0!!!",vthi ./ vthi0 )
    end
end
