### Computing `FvL`,  F(𝓋̂) in theory
# `FLnt = F̂ₗ(v̂ᵦ*vabth, ûᵦ)`
FLnbDMt, FLnbDMv0t = FL0DM(uai[isp3];L1=L1)
FLnDMt, FLnDMv0t = FL0DM(uai[iFv3];L1=L1)
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
# ## Interpolations for F(𝓋̂) applying the renormalization models.
if 1 == 1
    vs = vG0
    # 3D
    ddfvL30F,dfvL30F = zeros(nc0,LM1,ns),zeros(nc0,LM1,ns)
    FvL3 = zeros(nck,LM1,ns)
    ddfvL30F,dfvL30F,FvL3 = FfvLCS(ddfvL30F,dfvL30F,fvL,FvL3,
            vGk,nc0,nck,nc0,ocp,nvlevel0,bc3,uai,vthi,LM,ns;
            method3M=method3M,isrenormalization=isrenormalization,
            isnormal=isnormal,restartfit=restartfit,
            maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    # 2D
    points = zeros(Int,nck)
    points = Splinemapping(points,vG0,va,nc0,nck,vabth)
    ddfvL02, dfvL02 = zeros(nc0,LM1),zeros(nc0,LM1)
    FvLb2 = zeros(nck,LM1)
    ddfvL02, dfvL02, FvLb2 = FfvLCS(ddfvL02,dfvL02,fvL2,FvLb2,points,
            vs,va,nc0,nck,nc0,ocp,nvlevel0,bc2,uai[isp3],LM1,vabth;
            method3M=method3M,isrenormalization=isrenormalization,
            isnormal=isnormal,restartfit=restartfit,
            maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    # 1D
    # Notes: In `fvLmodel`, `u` are reference datas to optimize the convergence of the weight function by
    # giving a better guess parameters of the `neurons` in the model.
    # The model follows the principle of simplicity with no need to be a accurate model,
    # which just acts as a weight function to scaling the original function to be a low-order polynomial function.
    ddfvL01,dfvL01 = zeros(nc0),zeros(nc0)
    FvLb1 = zeros(nck)
    ddfvL01,dfvL01,FvLb1 = FfvLCS(ddfvL01,dfvL01,fLn1,FvLb1,points,
            vs,va,nc0,nck,nc0,ocp,nvlevel0,bc,uai[isp3],ℓ,vabth;
            method3M=method3M,isrenormalization=isrenormalization,
            isnormal=isnormal,restartfit=restartfit,
            maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
end
@show norm(FvLb1 - FvL3[:,L1,iFv3])
if is_plotF == 1
    label = string("err_df")
    pdf = plot(vG0,neps*(dfvL01 - dfLn0t),label=label,line=(3,:auto))
    label = string("err_df2")
    pdf = plot!(vG0,neps*(dfvL02[:,L1] - dfLn0t),label=label,line=(3,:auto))

    label = string("err_ddf")
    pddf = plot(vG0,neps*(ddfvL01 - ddfLn0t),label=label,line=(3,:auto))
    label = string("err_ddf2")
    pddf = plot!(vG0,neps*(ddfvL02[:,L1] - ddfLn0t),label=label,line=(3,:auto))

    label = string("err_Fb1")
    pFb = plot(vb,neps*(FvLb1 - FLnbt),label=label,line=(3,:auto))
    label = string("err_Fb2")
    pFb = plot!(vb,neps*(FvLb2[:,L1] - FLnbt),label=label,line=(3,:auto))

    label = string("Rerr_Fb1")
    pRFb = plot(vb,neps*(FvLb1 ./ FLnbt .-1),label=label,line=(3,:auto))
    label = string("Rerr_Fb2")
    pRF = plot!(vb,neps*(FvLb2[:,L1] ./ FLnbt .-1),label=label,line=(3,:auto))
    display(plot(pdf,pddf,pFb,pRFb,layout=(2,2)))
    @show norm(dfvL01 - dfLn0t) *neps
    @show norm(ddfvL01 - ddfLn0t) *neps
    @show norm(FvLb1 - FLnbt) *neps
end
FLnb0 = FvLb1[nvlevel0]
FvL2b = FvL3[:,:,iFv3]
FLn = FvL3[:,L1,isp3]
FLn0 = FLn[nvlevel0]

if is_plotdf == 1
    xlabel = string("log10(vGk),nc0=",nc0)
    ylabel = string("Errors[eps]")
    label3 = string("f")
    pef = plot(vG0,[fLn0 fLn0t],label=label3,ylabel=ylabel,line=(3,:auto))
    label3 = string("δf")
    pedf = plot(vG0,(fLn0 - fLn0t)*neps,label=label3,ylabel=ylabel,line=(3,:auto))
    label3 = string("δdf")
    pedf = plot!(vG0,(dfLn0 - dfLn0t)*neps,label=label3,legend=legendtR,line=(3,:auto))
    label3 = string("δddf")
    peddf = plot(vG0,(ddfLn0 - ddfLn0t)*neps,label=label3,xlabel=xlabel,legend=legendbR,line=(3,:auto))
    label3 = string("δF")
    peddF = plot(vG0,(FLn0 - FLn0t)*neps,label=label3,legend=legendtR,line=(3,:auto))
    display(plot(pef,pedf,peddf,peddF,layout=(2,2)))
end
