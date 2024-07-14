### Computing `dfvL0` and `ddfvL0` and `FvL`
vabth = vth[isp3] / vth[iFv3]
mM = ma[isp3] / ma[iFv3]
va = vGk * vabth                       # vabth * vGk
vb = vGk / vabth                       # vGk / vabth
va0 = va[nvlevel0]
vb0 = vb[nvlevel0]
nvaplot = va .< 30
## ## Computing F(𝓋̂) in theory
if 1 === 1
    # `FLnt = F̂ₗ(v̂ᵦ*vabth, ûᵦ)`
    if prod(nMod) === 1
        FLnbDMt, FLnbDMv0t = FL0DM(uai[isp3][1]; L1=L1)
        FLnDMt, FLnDMv0t = FL0DM(uai[iFv3][1]; L1=L1)
    else
        FLnbDMt, FLnbDMv0t = FL0DM(uai[isp3]; L1=L1)
        FLnDMt, FLnDMv0t = FL0DM(uai[iFv3]; L1=L1)
    end
    if vGk[1] ≠ 0.0
        FLnt = FLnDMt(vb)
        FLnbt = FLnbDMt(va)
    else
        FLnt = zero.(va)
        FLnt[2:end] = FLnDMt(vb[2:end])
        FLnt[1] = FLnDMv0t(vb[1])
        FLnbt = zero.(vb)
        FLnbt[2:end] = FLnbDMt(va[2:end])
        FLnbt[1] = FLnbDMv0t(va[1])
    end
    FLn0t = FLnt[nvlevel0]
    FLnb0t = FLnbt[nvlevel0]
end

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
    label = string("err_f,L=", L1 - 1)
    pf = plot(vG0[nvplot], neps * (fLn0-fLn0t)[nvplot], label=label, line=(3, :auto))
    label = string("err_F,L=", L1 - 1)
    pf = plot!(vG0[nvplot], neps * (FvL3[nvlevel0, L1, iFv3]-FLn0t)[nvplot], label=label, line=(3, :auto))

    label = string("err_df,L=", L1 - 1)
    pdf = plot(vG0[nvplot], neps * (dfLn0-dfLn0t)[nvplot], label=label, line=(3, :auto))
    # label = string("err_df2")
    # pdf = plot!(vG0[nvplot],neps*(dfvL20[nvplot,L1] - dfLn0t),label=label,line=(3,:auto))
    label = string("err_df3")
    pdf = plot!(vG0[nvplot], neps * (dfvL30[nvplot, L1, isp3] - dfLn0t[nvplot]), label=label, line=(3, :auto))

    label = string("err_ddf")
    pddf = plot(vG0[nvplot], neps * (ddfLn0-ddfLn0t)[nvplot], label=label, line=(3, :auto))
    # label = string("err_ddf2")
    # pddf = plot!(vG0[nvplot],neps*(ddfvL20[nvplot,L1] - ddfLn0t[nvplot]),label=label,line=(3,:auto))
    label = string("err_ddf3")
    pddf = plot!(vG0[nvplot], neps * (ddfvL30[nvplot, L1, isp3] - ddfLn0t[nvplot]), label=label, line=(3, :auto))
    display(plot(pf, pdf, pddf, layout=(3, 1)))
end
FvLb2 = FvL3[:, :, iFv3]
FLn = FvL3[:, L1, isp3]
FLn0 = FLn[nvlevel0]
FLnb0 = FvL3[nvlevel0, L1, isp3]
