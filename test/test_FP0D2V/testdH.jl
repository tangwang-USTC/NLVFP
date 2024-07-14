
function optimRH(HvLn,FvLn,v1,vth,LM1;ns=2)

    atol = 1e-3    # for `Rn`
    rtol = 1e-3    # for `(Rn - Rn_up) / (Rn + Rn_up)`
    rtolddH = 1e-3 # for `(Rn - Rn_up) / (Rn + Rn_up)`
    abstol = 1e-6  # for gmres
    reltol = 1e-5  # for gmres
    ϵ = 1e-4       # for Maxtrix-Vector products, Jv
    k = 3
    optimH = 0     # (=0/1 or false/trues), for optims of ∂ⁿHvL
    maxiterN = 100  # (=100, default), maximum iteration number of Newton process
    maxiterA = 20  # (=10, defau), "alp step": to find `alp` with for the optimized `x0` after the gmres! step
    maxiterR = 20  # (=10, default), re-optimming `x0` after "alp step"
    L1m = 1        # (=0, default), L1m ∈ [1:LM1]
    nsm = 1        # (=1, default), L1m ∈ [1:ns]
    ##
    L1min = 2
    L1max = LM1
    nsmin = 1
    nsmax = ns
    if L1m > 0
        L1v = L1m:L1m
    else
        L1v = L1min:L1max
    end
    if nsm > 0
        nsv = nsm:nsm
    else
        nsv = nsmin:nsmax
    end
    nsp_vec = 1:ns
    if methodNK == 1
        HvLn = optimRdH(ns,L1v,nsv,HvLn,FvLn,vth,v1;atol=atol,rtol=rtol,reltol=reltol,abstol=abstol,
                          k=k,ϵ=ϵ,maxiterR=maxiterR,maxiterA=maxiterA,maxiterN=maxiterN)
        HvLn = optimRddH(ns,L1v,nsv,HvLn,FvLn,vth,v1;atol=atol,rtol=rtolddH,reltol=reltol,abstol=abstol,
                         k=k,ϵ=ϵ,maxiterR=maxiterR,maxiterA=maxiterA,maxiterN=maxiterN)
        ###########################################
    else
        for isp in nsv
            for L1 in L1v
                x0 = HvLn[:,L1,isp]
                if norm(x0) > 1e-10
                    FLn = FvLn[:,L1,isp]
                    nspF = nsp_vec[nsp_vec .≠ isp]
                    iFv = nspF[1]
                    vabth = vth[isp] / vth[iFv]
                    va = v1 * vabth
                    # RddH =  RddHLp(x0,FLn,va,L1)
                    Hv012!(x0,va,L1)
                    dHvL, ddHvL = dvdH012L(x0,va,L1)
                    # RddH =  RddHLp(x0,FLn,va,L1)
                    x0 = optimRdH!(x0,FLn,va,L1;atol=atol,rtol=rtol,reltol=reltol,abstol=abstol,
                                      k=k,ϵ=ϵ,maxiterR=maxiterR,maxiterA=maxiterA,maxiterN=maxiterN)
                    RddH =  RddHLp(x0,FLn,va,L1)
                    x0 = optimRddH!(x0,FLn,va,L1;atol=atol,rtol=rtol,reltol=reltol,abstol=abstol,
                              k=k,ϵ=ϵ,maxiterR=maxiterR,maxiterA=maxiterA,maxiterN=maxiterN)
                    HvLn[:,L1,isp] = deepcopy(x0)
                    RddH =  RddHLp(x0,FLn,va,L1)
                end
            end
        end
    end
    HvLn
end
