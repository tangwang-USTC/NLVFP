### Computing `dfvL0` and `ddfvL0` with cubic Spline method
np = nc0
pDM = p0DM_guess(uai[isp3],3)
# pDM = p0DM_guess(uai[isp3],3;pDM_noise_ratio=p_noise_ratio,pDM_noise_abs=p_noise_abs)
if 1 == 1
    # 1D
    # Notes: In `fvLmodel`, `u` are reference datas to optimize the convergence of the weight function by
    # giving a better guess parameters of the `neurons` in the model.
    # The model follows the principle of simplicity with no need to be a accurate model,
    # which just acts as a weight function to scaling the original function to be a low-order polynomial function.
    fLn0, dfLn0, ddfLn0 = zeros(np),zeros(np),zeros(np)
    xs = deepcopy(vG0)
    ys = deepcopy(fLn1[nvlevel0])
    # ddfLn0, dfLn0 = FfvLCS(ddfLn0,dfLn0,ys,xs,nc0,np,ocp,nvlevel0,bc,uai[isp3],ℓ;
    #                     method3M=method3M,isrenormalization=isrenormalization,
    #                     isnormal=isnormal,p=pDM,restartfit=restartfit,
    #                     maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
    #                     p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    # fLn0 = deepcopy(ys)
    ddfLn0, dfLn0,fLn0 = FfvLCS(ddfLn0,dfLn0,fLn0,ys,xs,nc0,np,ocp,nvlevel0,bc,uai[isp3],ℓ;
                        method3M=method3M,isrenormalization=isrenormalization,
                        isnormal=isnormal,p=pDM,restartfit=restartfit,
                        maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    # 2D
    ddfvL20,dfvL20,fvL20 = zeros(np,LM1),zeros(np,LM1),zeros(np,LM1)
    fvL200 = deepcopy(fvL[nvlevel0,:,isp3])
    # ddfvL20, dfvL20 = FfvLCS(ddfvL20,dfvL20,fvL200,xs,nc0,np,ocp,nvlevel0,bc2,uai[isp3],LM1;
    #                     method3M=method3M,isrenormalization=isrenormalization,
    #                     isnormal=isnormal,restartfit=restartfit,
    #                     maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
    #                     p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    ddfvL20, dfvL20,fvL20 = FfvLCS(ddfvL20,dfvL20,fvL20,fvL200,xs,nc0,np,ocp,nvlevel0,bc2,uai[isp3],LM1;
                        method3M=method3M,isrenormalization=isrenormalization,
                        isnormal=isnormal,restartfit=restartfit,
                        maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    # 3D
    ddfvL30,dfvL30,fvL30 = zeros(np,LM1,ns),zeros(np,LM1,ns),zeros(np,LM1,ns)
    fvL300 = deepcopy(fvL[nvlevel0,:,:])
    # ddfvL30, dfvL30 = FfvLCS(ddfvL30,dfvL30,fvL300,xs,nc0,np,ocp,nvlevel0,bc3,uai,LM,ns;
    #                     method3M=method3M,isrenormalization=isrenormalization,
    #                     isnormal=isnormal,restartfit=restartfit,
    #                     maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
    #                     p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    ddfvL30, dfvL30,fvL30 = FfvLCS(ddfvL30,dfvL30,fvL30,fvL300,xs,nc0,np,ocp,nvlevel0,bc3,uai,LM,ns;
                        method3M=method3M,isrenormalization=isrenormalization,
                        isnormal=isnormal,restartfit=restartfit,
                        maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

    if np == nc0 && is_plotdf == 1
        label = string("err_f,L=",L1-1)
        pf = plot(vG0,neps*(fLn0 - fLn0t),label=label,line=(3,:auto))

        label = string("err_df,L=",L1-1)
        pdf = plot(vG0,neps*(dfLn0 - dfLn0t),label=label,line=(3,:auto))
        label = string("err_df2")
        pdf = plot!(vG0,neps*(dfvL20[:,L1] - dfLn0t),label=label,line=(3,:auto))
        label = string("err_df3")
        pdf = plot!(vG0,neps*(dfvL30[:,L1,isp3] - dfLn0t),label=label,line=(3,:auto))

        label = string("err_ddf")
        pddf = plot(vG0,neps*(ddfLn0 - ddfLn0t),label=label,line=(3,:auto))
        label = string("err_ddf2")
        pddf = plot!(vG0,neps*(ddfvL20[:,L1] - ddfLn0t),label=label,line=(3,:auto))
        label = string("err_ddf3")
        pddf = plot!(vG0,neps*(ddfvL30[:,L1,isp3] - ddfLn0t),label=label,line=(3,:auto))
        display(plot(pf,pdf,pddf,layout=(3,1)))
        @show norm(fLn0 - fLn0t) *neps
        @show norm(dfLn0 - dfLn0t) *neps
        @show norm(ddfLn0 - ddfLn0t) *neps
    end
end
