
"""
  Inputs:
    ma: [mp]
    Zq: [e]
    na: [m⁻³]
    Ta: [eV]
    t:  [s]

  Outputs:
    dtT = dtTaTb4_SI(Ta,moments,t)

"""

# [ns]
function dtTaTb4_SI(Ta::AbstractVector{T},moments::AbstractArray{T,2},t::T) where{T}

    ma = moments[1,:]
    Zq = moments[2,:] |> Vector{Int64}
    na = moments[3,:]
    dT = moments[4,:]

    isp = 1
    iFv = 2
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] = FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    iFv = 3
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] += FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    iFv = 4
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] += FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])

    isp = 2
    iFv = 1
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] = FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    iFv = 3
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    dT[isp] += FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    iFv = 4
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] += FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])

    isp = 3
    iFv = 1
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] = FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    iFv = 2
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] += FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    iFv = 4
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] += FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])

    isp = 4
    iFv = 1
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] = FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    iFv = 2
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] += FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    iFv = 3
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] += FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    return dT
end

function dtTaTb3_SI(Ta::AbstractVector{T},moments::AbstractArray{T,2},t::T) where{T}
    
    ma = moments[1,:]
    Zq = moments[2,:] |> Vector{Int64}
    na = moments[3,:]
    dT = moments[4,:]

    isp = 1
    iFv = 2
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] = FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    iFv = 3
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] += FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])

    isp = 2
    iFv = 1
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] = FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    iFv = 3
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] += FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])

    isp = 3
    iFv = 1
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] = FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    iFv = 2
    m2 = [ma[isp], ma[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [na[isp], na[iFv]]
    T2 = [Ta[isp], Ta[iFv]]
    spices2 = [spices0[isp], spices0[iFv]]
    dT[isp] += FPTaTb_fM_SI(1,2,m2,Zq2,spices2,n2,T2) * (T2[2] - T2[1])
    return dT
end

function dtTaTb2_SI(Ta::AbstractVector{T},moments::AbstractArray{T,2},t::T) where{T}

    ma = moments[1,:]
    Zq = moments[2,:] |> Vector{Int64}
    na = moments[3,:]
    dT = moments[4,:]
fjnhhgk
    isp = 1
    iFv = 2
    dT[isp] = FPTaTb_fM_SI(isp,iFv,ma,Zq,spices0,na,Ta) * (Ta[iFv] - Ta[isp])
    isp = 2
    iFv = 1
    dT[isp] = FPTaTb_fM_SI(isp,iFv,ma,Zq,spices0,na,Ta) * (Ta[iFv] - Ta[isp])
    return dT
end
