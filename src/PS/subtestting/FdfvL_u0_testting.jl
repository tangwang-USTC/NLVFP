# When `u0[isp3] = 0`
 
### Computing `dfvL0` and `ddfvL0` and `FvL`
vabth = vth[isp3] / vth[iFv3]
mM = ma[isp3] / ma[iFv3]
va = vGk * vabth                       # vabth * vGk
vb = vGk / vabth                       # vGk / vabth
va0 = va[nvlevel0]
vb0 = vb[nvlevel0]
nvaplot = va .< 30
### Computing `dfvL0` and `ddfvL0` and `FvL`
if nMod == 21
    # 1D
    # Notes: In `fvLmodel`, `u` are reference datas to optimize the convergence of the weight function by
    # giving a better guess parameters of the `neurons` in the model.
    # The model follows the principle of simplicity with no need to be a accurate model,
    # which just acts as a weight function to scaling the original function to be a low-order polynomial function.
    ddfLn0, dfLn0, fLnnew, FLnk = zeros(nc0), zeros(nc0), zeros(nck), zeros(nck)
    ys = copy(fLn1[nvlevel0])
    ddfLn0, dfLn0, fLnnew, FLnk = FfvLCS(ddfLn0, dfLn0, fLnnew, FLnk, ys,
        vGk, va, nvlevel0, vabth, nai[isp3],uai[isp3],vthi[isp3],ℓ;
        isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
    fLn0 = fLnnew[nvlevel0]
    # 2D
    ddfvL20, dfvL20, fvL2new, FvL2 = zeros(nc0, LM1), zeros(nc0, LM1), zeros(nck, LM1), zeros(nck, LM1)
    ddfvL20, dfvL20, fvL2new, FvL2 = FfvLCS(ddfvL20, dfvL20, fvL2new, FvL2,
        fvL2[nvlevel0, :], vGk, va, nvlevel0, vabth, nai[isp3],uai[isp3],vthi[isp3],LM1;
        isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
    # 3D
    ddfvL30, dfvL30, fvL3new, FvL3 = zeros(nc0, LM1, ns), zeros(nc0, LM1, ns), zeros(nck, LM1, ns), zeros(nck, LM1, ns)
    ddfvL30, dfvL30, fvL3new, FvL3 = FfvLCS(ddfvL30, dfvL30, fvL3new, FvL3,
        fvL[nvlevel0, :, :], vGk, nvlevel0, vth, nai,uai,vthi,LM, LM1, ns;
        isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
    # fvL300 = copy(fvL[nvlevel0,:,:])
else
    # 3D
    ddfvL30,dfvL30,fvL3new,FvL3 = zeros(nc0,LM1,ns),zeros(nc0,LM1,ns),zeros(nck,LM1,ns),zeros(nck,LM1,ns)
    ddfvL30,dfvL30,fvL3new,FvL3 = FfvLCS(ddfvL30,dfvL30,fvL3new,FvL3,
          fvL[nvlevel0,:,:],vGk,nc0,ocp,nvlevel0,vth,nai,uai,vthi,LM,LM1,ns,nMod;
          isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
          autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
          p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    # 1D
    ddfLn0, dfLn0, fLnnew, FLnk = ddfvL30[:,L1,isp3],dfvL30[:,L1,isp3],fvL3new[:,L1,isp3],FvL3[:,L1,isp3]
    fLn0 = fLnnew[nvlevel0]
end
nvplot = vG0 .> 0
if is_plotdf == 1
    label = string("f,L=", L1 - 1)
    pf = plot(vG0[nvplot], (fLn0)[nvplot], label=label, line=(3, :auto))
    label = string("F,L=", L1 - 1)
    pf = plot!(vG0[nvplot], (FvL3[nvlevel0, L1, isp3])[nvplot], label=label, line=(3, :auto))

    label = string("df,L=", L1 - 1)
    pdf = plot(vG0[nvplot], (dfLn0)[nvplot], label=label, line=(3, :auto))
    # label = string("df2")
    # pdf = plot!(vG0[nvplot],neps*(dfvL20[nvplot,L1]),label=label,line=(3,:auto))
    label = string("df3")
    pdf = plot!(vG0[nvplot], (dfvL30[nvplot, L1, isp3]), label=label, line=(3, :auto))

    label = string("ddf")
    pddf = plot(vG0[nvplot], (ddfLn0)[nvplot], label=label, line=(3, :auto))
    # label = string("ddf2")
    # pddf = plot!(vG0[nvplot],neps*(ddfvL20[nvplot,L1]),label=label,line=(3,:auto))
    label = string("ddf3")
    pddf = plot!(vG0[nvplot], (ddfvL30[nvplot, L1, isp3]), label=label, line=(3, :auto))
    display(plot(pf, pdf, pddf, layout=(3, 1)))
end
FvLb2 = FvL3[:, :, iFv3]
FLn = FvL3[:, L1, isp3]
FLn0 = FLn[nvlevel0]
FLnb0 = FvL3[nvlevel0, L1, iFv3]
