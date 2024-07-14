
function dfdtTaTb(moments,para,t)
    m0 = para[1:2]
    Zq = para[3:4]
    n0 = para[5:6]
    T0 = moments[1:2]
    dT = zeros(2)
    isp = 1
    iFv = 2
    dT[isp] = FPTaTb(isp,iFv,m0,Zq,n0,T0) * (T0[iFv] - T0[isp])
    isp = 2
    iFv = 1
    dT[isp] =  FPTaTb(isp,iFv,m0,Zq,n0,T0) * (T0[iFv] - T0[isp])
    return [dT[1] dT[2]]
end

"""
   νT = FPTaTb(isp,iFv,m0,Zq,n0,T0)

"""

# νT   [1/s]

function FPTaTb(isp::Int64,iFv::Int64,m0::AbstractVector,Zq::AbstractVector,
                 n0::AbstractVector,T0::AbstractVector)
    mDa = m0 / Dₐ    # for normalization
    cT = (mDa[isp] * mDa[iFv])^0.5 / (mDa[isp] * T0[iFv] + mDa[iFv] * T0[isp])^1.5
    ## lnA
    # lnAab = lnA(isp,iFv, m0, n20 * n0[iFv], Tk * T0[iFv])
    lnAab = lnA()
    return 4.41720911682e2 * cT * (Zq[isp]* Zq[iFv])^2 * n0[iFv] * lnAab
end  # νT
# println("t=",1/νT)
