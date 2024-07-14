
"""
  Inputs:

  Outputs:
    Rdtsab = entropyN_rate_fDM(ma,na,vth,Ia,Ka,dtIa,dtKa,ns)
"""

function entropyN_rate_fDM(ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},
    Ia::AbstractVector{T},Ka::AbstractVector{T},dtIa::AbstractVector{T},dtKa::AbstractVector{T},ns::Int64) where {T<:Real}
    
    isp = 1
    if abs(Ia[isp]) ≤ epsT10
        sab = entropy_fM(ma[isp],na[isp],vth[isp],Ka[isp])
    else
        sab = entropy_fDM(ma[isp],na[isp],vth[isp],Ia[isp],Ka[isp])
    end
    for isp in 2:ns
        if abs(Ia[isp]) ≤ epsT10
            sab += entropy_fM(ma[isp],na[isp],vth[isp],Ka[isp])
        else
            sab += entropy_fDM(ma[isp],na[isp],vth[isp],Ia[isp],Ka[isp])
        end
    end

    isp = 1
    uhak = Ia[isp] ./ (ma[isp] .* na[isp] .* vth[isp])
    dtsab = entropy_rate_fDM(ma[isp],vth[isp],uhak,dtIa[isp],dtKa[isp])
    for isp in 2:ns
        dtsab += entropy_rate_fDM(ma[isp],vth[isp],uhak,dtIa[isp],dtKa[isp])
    end
    return dtsab / sab
end

"""

  Entropy of `fDM` when `[nMod, ns]`

  Inputs:

  Outputs:
    entropy_fDM!(sk,ma,nk,vthk,Ik,Kk,nModk,ns)
    entropy_fDM!(sa,ma,na,vth,Ia,Ka,ns)
    sa = entropy_fDM(ma,na,vth,Ia,Ka)
"""

# [nMod,ns]
function entropy_fDM!(sk::AbstractArray{T,N},ma::AbstractVector{T},nk::Vector{TA},
    vthk::Vector{TA},Ik::AbstractArray{T,N},Kk::AbstractArray{T,N},
    nModk::Vector{Int64},ns::Int64) where {T<:Real,N,TA}

    for isp in 1:ns
        for k in 1:nModk[isp]
            if abs(Ik[k,isp]) ≤ epsT10
                sk[k,isp] = entropy_fM(ma[isp],nk[isp][k],vthk[isp][k],Kk[k,isp])
            else
                sk[k,isp] = entropy_fDM(ma[isp],nk[isp][k],vthk[isp][k],Ik[k,isp],Kk[k,isp])
            end
        end
    end
end

# [ns]
function entropy_fDM!(sa::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},
    vth::AbstractVector{T},Ia::AbstractVector{T},Ka::AbstractVector{T},ns::Int64) where {T<:Real}

    for isp in 1:ns
        if abs(Ia[isp]) ≤ epsT10
            sa[isp] = entropy_fM(ma[isp],na[isp],vth[isp],Ka[isp])
        else
            sa[isp] = entropy_fDM(ma[isp],na[isp],vth[isp],Ia[isp],Ka[isp])
        end
    end
end


# []
function entropy_fDM(ma::T,na::T,vth::T,Ia::T,Ka::T) where {T<:Real}

    return (Ka - Ia^2 / (ma * na)) / (ma * vth^2) - na * (log(na / vth) - lnsqrtpi3)
end

"""

  Entropy change rate of `fDM` when `[nMod, ns]`

  Inputs:

  Outputs:
    Rdtsab = entropyN_rate_fDM(ma,na,vth,uha,dtIa,dtKa,ns)
"""
function entropyN_rate_fDM(ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},
    uha::AbstractVector{T},dtIa::AbstractVector{T},dtKa::AbstractVector{T},ns::Int64) where {T<:Real}
    
    isp = 1
    Rdtsab = entropy_rate_fDM(ma[isp],vth[isp],uha[isp],dtIa[isp],dtKa[isp])
    for isp in 2:ns
        Rdtsab += entropy_rate_fDM(ma[isp],vth[isp],uha[isp],dtIa[isp],dtKa[isp])
    end
    return (Rdtsab / sum(na))
end

"""

  Entropy change rate of `fDM` when `[nMod, ns]`

  Inputs:
    uha = ua / vath

  Outputs:
    entropy_rate_fDM(dtsa,ma,vth,uha,dtIa,dtKa,ns)
    dtsa = entropy_rate_fDM(ma,vth,uha,dtIa,dtKa)
"""
# [ns]
function entropy_rate_fDM(ma::AbstractVector{T},vth::AbstractVector{T},
    uha::AbstractVector{T},dtIa::AbstractVector{T},dtKa::AbstractVector{T},ns::Int64) where {T<:Real}
    
    isp = 1
    dtsab = entropy_rate_fDM(ma[isp],vth[isp],uha[isp],dtIa[isp],dtKa[isp])
    for isp in 2:ns
        dtsab += entropy_rate_fDM(ma[isp],vth[isp],uha[isp],dtIa[isp],dtKa[isp])
    end
    return dtsab
end

function entropy_rate_fDM!(dtsa::AbstractVector{T},ma::AbstractVector{T},vth::AbstractVector{T},
    uha::AbstractVector{T},dtIa::AbstractVector{T},dtKa::AbstractVector{T},ns::Int64) where {T<:Real}

    for isp in 1:ns
        dtsa[isp] = entropy_rate_fDM(ma[isp],vth[isp],uha[isp],dtIa[isp],dtKa[isp])
    end
end

# []
function entropy_rate_fDM(ma::T,vth::T,uha::T,dtIa::T,dtKa::T) where {T<:Real}
    
    return (1 + uha^2 / 3) * (2 * dtKa / (ma * vth^2)) + (1 - uha^2 * (2/3)) * (uha * dtIa / (ma * vth))
    # sak = (1 + uha^2 / 3) * (2 * dtKa / (ma * vth^2))
    # sak += (1 - uha^2 * (2/3)) * (uha * dtIa / (ma * vth))
end


"""

  normalzied Entropy change rate of `fDM` when `[nMod=2]`
    
    sha = sa / (ma * na)
    Rdtsa = dtsa / (ma * na) 

  Inputs:
    uhak = uaik ./ vthik

  Outputs:
    Rdtsa = entropyN_rate_fDM(nk,uhak,vthk,dtIa,dtKa)
    dtsa = entropyN_rate_fDM(uha,vthk,dtIa,dtKa)
"""
# [nMod=2]  when `ma = mb, Za = Zb`
function entropy_rate_fDM(uhak::AbstractVector{T},vthk::AbstractVector{T},
    dtIa::AbstractVector{T},dtKa::AbstractVector{T}) where {T<:Real}
    
    k = 1
    dtsaa = entropyN_rate_fDM(uhak[k],vthk[k],dtIa[k],dtKa[k])
    k = 2
    dtsaa += entropyN_rate_fDM(uhak[k],vthk[k],dtIa[k],dtKa[k])
    return dtsaa
end

function entropyN_rate_fDM(nk::AbstractVector{T},uhak::AbstractVector{T},vthk::AbstractVector{T},
    dtIa::AbstractVector{T},dtKa::AbstractVector{T}) where {T<:Real}
    
    k = 1
    Rdtsa = entropyN_rate_fDM(uhak[k],vthk[k],dtIa[k],dtKa[k])
    k = 2
    Rdtsa += entropyN_rate_fDM(uhak[k],vthk[k],dtIa[k],dtKa[k])
    # Rdtsa /= sum(nk)
    return (Rdtsa / sum(nk))
end

function entropyN_rate_fDM(uha::T,vthk::T,dtIa::T,dtKa::T) where {T<:Real}
    
    return (1 + uha^2 / 3) * (2 * dtKa / vthk^2) + (1 - uha^2 * (2/3)) * (uha * (dtIa / vthk))
    # sak = (1 + uha^2 / 3) * (2 * dtKa / vthk^2)
    # sak += (1 - uha^2 * (2/3)) * (uha * dtIa / vthk)
end

"""

  Inputs:
  Outputs:
    dtsaa = dtsaa_initial(nMod,ns)
"""

function dtsaa_initial(nMod::Vector{Int64},ns::Int64) 

    
    dtsaak = Vector{Any}(undef, ns)
    for isp in 1:ns
        if nMod[isp] ≥ 2
            dtsaak[isp] = zeros(binomial(nMod[isp],2))
        end
    end
    return dtsaak
end