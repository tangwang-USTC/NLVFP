
include(joinpath(path,"test\\run_collisions\\testdH.jl"))

# plotly()
gr()

function RH111()
    atol = 1e-4
    abstol = 1e-9
    reltol = 1e-4
    maxiterN = 2
    maxiteriN = 5
    maxiterR = 5   # (=10, default)
    ϵ = 1e-4       # for Maxtrix-Vector products, Jv
    k = 3
    optimH = 0     # (=0/1 or false/trues), for optims of ∂ⁿHvL
    is_mpoint = -1
    L1m = 1
    nsm = 2
    ##
    L1min = 1
    L1max = 3
    nsmin = 1
    nsmax = 2
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
    HvL00 = deepcopy(HvL0)

    optimRH(L1v,nsv;maxiterR=maxiterR, atol=atol)
end

function optimRH(L1v,nsv;maxiterR=10, atol=1e-3,maxiteriN=maxiteriN)
    for isp3 in nsv
        for L1 in L1v
            nspF = nsp_vec[nsp_vec .≠ isp3]
            iFv3 = nspF[1]
            vabth = vth[isp3] / vth[iFv3]
            va = vremesh * vabth
            iR = 0
            Rn = 1.0
            Rn_up = 0.0
            x0 = HvL00[:,L1,isp3]
            while Rn > atol
                if iR == 1
                    Hv0!(x0,va,L1)
                end
                # xx = testdH(L1,isp3,x0,va,vabth,FvL[:,L1,isp3])
                x0 = testdHa(L1,isp3,x0,va,vabth,FvL[:,L1,isp3];maxiteriN=maxiteriN,atol=atol)
                Rn = norm(RdHL(x0,va,FvL[:,L1,isp3],L1))
                iR += 1
                # println("iR22=",iR,",Rn=",Rn,",err=",Rn-Rn_up)s
                if iR ≥ maxiterR
                    break
                else
                    if abs(Rn - Rn_up) < 1e-6
                        break
                    else
                        Rn_up = 1Rn
                    end
                end
            end
            HvL00[:,L1,isp3] = deepcopy(x0)
        end
    end
end
