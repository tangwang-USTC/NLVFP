"""

  Entropy of `fM` when `[nMod, ns]`

  Inputs:

  Outputs:
    entropy_fM!(sa,ma,na,vth,Ka,ns)
    sa = entropy_fM(ma,na,vth,Ka)
"""
# [ns]
function entropy_fM!(sa::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},
    vth::AbstractVector{T},Ka::AbstractVector{T},ns::Int64) where {T<:Real}

    for isp in 1:ns
        sa[isp] = entropy_fM(ma[isp],na[isp],vth[isp],Ka[isp])
    end
end


# []
function entropy_fM(ma::T,na::T,vth::T,Ka::T) where {T<:Real}

    return Ka / (ma * vth^2) - na * (log(na / vth) - lnsqrtpi3)
end

"""

  Entropy change rate of `fM` when `[nMod, ns]`

  Inputs:

  Outputs:
    entropy_rate_fM!(dtsa,ma,vth,dtKa,ns)
    dtsa = entropy_rate_fM(ma,vth,dtKa)
"""
# [ns]
function entropy_rate_fM!(dtsa::AbstractVector{T},ma::AbstractVector{T},
    vth::AbstractVector{T},dtKa::AbstractVector{T},ns::Int64) where {T<:Real}

    for isp in 1:ns
        dtsa[isp] = entropy_rate_fM(ma[isp],vth[isp],dtKa[isp])
    end
end


# []
function entropy_rate_fM(ma::T,vth::T,dtKa::T) where {T<:Real}
    
    return 2 * dtKa / (ma * vth^2)
end


"""

  normalzied Entropy change rate of `fM` when `[nMod=2]`
    
    sha = sa / (ma * na)
    Rdtsa = dtsa / (ma * na) 

  Inputs:

  Outputs:
    Rdtsa = entropyN_rate_fM(nk,vthk,dtKa)
    Rdtsa = entropyN_rate_fM(vthk,dtKa)
"""
# [nMod=2]  when `ma = mb, Za = Zb`

function entropyN_rate_fM(nk::AbstractVector{T},vthk::AbstractVector{T},dtKa::AbstractVector{T}) where {T<:Real}
    
    k = 1
    Rdtsa = entropyN_rate_fM(vthk[k],dtKa[k])
    for k in 1:2
        Rdtsa += entropyN_rate_fM(vthk[k],dtKa[k])
    end
    Rdtsa /= sum(nk)
    return Rdtsa
end

function entropyN_rate_fM(vthk::T,dtKa::T) where {T<:Real}
    
    return 2 * dtKa / vthk^2
end
