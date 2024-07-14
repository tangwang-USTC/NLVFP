
"""
  Outputs:
    nuT_sub_update!(uk1,vthk1,mak1,nk1,IKk1,nModk1)
    nIKk_update!(IKk1,mak1,nk1,uk1,vthk1,nModk1)
"""

# [nMod,ns]
function nuT_sub_update!(uk1::Vector{TA},vthk1::Vector{TA},mak1::AbstractVector{T},
    nk1::Vector{TA},IKk1::AbstractArray{T,N},nModk1::Vector{Int64}) where{T,N,TA}

    for isp in 1:2
        nuT_sub_update!(uk1[isp],vthk1[isp],mak1[isp],nk1[isp],IKk1[:,:,isp],nModk1[isp])
    end
end

# [nMod]
function nuT_sub_update!(uk1::AbstractVector{T},vthk1::AbstractVector{T},
    mak1::T,nk1::AbstractVector{T},IKk1::AbstractArray{T,N},nMod::Int64) where{T,N}

    if nMod == 1
        k = 1
        uk1[k], vthk1[k] = nuT_sub_update(mak1 * nk1[k],IKk1[k,:])
    elseif nMod == 2
        k = 1
        uk1[k], vthk1[k] = nuT_sub_update(mak1 * nk1[k],IKk1[k,:])
        k = 2
        uk1[k], vthk1[k] = nuT_sub_update(mak1 * nk1[k],IKk1[k,:])
    else
        k = 1
        uk1[k], vthk1[k] = nuT_sub_update(mak1 * nk1[k],IKk1[k,:])
        for k in 2:nMod
            uk1[k], vthk1[k] = nuT_sub_update(mak1 * nk1[k],IKk1[k,:])
        end
    end
end

# [], ns = nMod = 1
function nuT_sub_update(rho::T,IKk1::AbstractVector{T}) where{T}
    
    if abs(IKk1[2]) ≤ epsT10
        return 0.0, (4/3 * IKk1[1] / rho) ^0.5
    else
        return IKk1[2] / rho, (2/3 * (2IKk1[1] / rho - (IKk1[2] / rho)^2)) ^0.5
    end
    # Ikk1 = IKk1[2]
    # Kkk1 = IKk1[1]
    # uk1 = Ikk1 / rho
    # vthk1 = (2/3 * (2Kk1 / rho - (Ikk1 / rho)^2)) ^0.5
    # return uk1, vthk1
end


# [nMod,ns], IKk1
function nIKk_update!(IKk1::AbstractArray{T,N},mak1::AbstractVector{T},nk1::Vector{TA},
    uk1::Vector{TA},vthk1::Vector{TA},nModk1::Vector{Int64};
    is_nMod_renew::Vector{Bool}=ones(Bool,ns)) where{T,N,TA}

    for isp in 1:2
        if is_nMod_renew[isp]
            a = IKk1[:,:,isp]
            nIKk_update!(a,mak1[isp],nk1[isp],uk1[isp],vthk1[isp],nModk1[isp])
            IKk1[:,:,isp] = a
        end            
    end
end

# [nMod]
function nIKk_update!(IKk1::AbstractArray{T,N},mak1::T,nk1::AbstractVector{T},
    uk1::AbstractVector{T},vthk1::AbstractVector{T},nMod::Int64) where{T,N}

    if nMod == 1
        k = 1
        IKk1[k,:] = nIKk_update!(IKk1[k,:],mak1 * nk1[k],uk1[k], vthk1[k])
        IKk1[nMod+1:end,:] .= 0.0
    elseif nMod == 2
        k = 1
        IKk1[k,:] = nIKk_update!(IKk1[k,:],mak1 * nk1[k],uk1[k], vthk1[k])
        k = 2
        IKk1[k,:] = nIKk_update!(IKk1[k,:],mak1 * nk1[k],uk1[k], vthk1[k])
        IKk1[nMod+1:end,:] .= 0.0
    else
        k = 1
        IKk1[k,:] = nIKk_update!(IKk1[k,:],mak1 * nk1[k],uk1[k], vthk1[k])
        for k in 2:nMod
            IKk1[k,:] = nIKk_update!(IKk1[k,:],mak1 * nk1[k],uk1[k], vthk1[k])
        end
        IKk1[nMod+1:end,:] .= 0.0
    end
end

# [], ns = nMod = 1
function nIKk_update!(IKk1::AbstractVector{T},rho::T,uk1::T,vthk1::T) where{T}

    if abs(uk1) ≤ epsT10
        IKk1[1] = 0.75 * rho * vthk1^2
        IKk1[2] = 0.0
    else
        IKk1[1] = 0.5 * rho * (1.5 * vthk1^2 + uk1^2)
        IKk1[2] = rho * uk1
    end
    return IKk1
end

"""
  Outputs:
    nuTk1_sub_initial!(nk1,uk1,vthk1.na,vth,naik,uaik,vthik,nMod,ns)
    nuTk1_sub_update!(nk1,uk1,vthk1.na,vth,naik,uaik,vthik,nMod,ns,is_nMod_renew)
"""

function nuTk1_sub_initial!(nk1::Vector{TA}, uk1::Vector{TA}, vthk1::Vector{TA}, 
    na::AbstractVector{T}, vth::AbstractVector{T}, 
    naik::Vector{TA}, uaik::Vector{TA}, vthik::Vector{TA}, 
    nMod::Vector{Int64},ns::Int64) where{T,TA}

    for isp in 1:ns
        for k in 1:nMod[isp]
            nk1[isp][k] = na[isp] * naik[isp][k]
            uk1[isp][k] = vth[isp] * uaik[isp][k]
            vthk1[isp][k] = vth[isp] * vthik[isp][k]
        end
        nk1[isp][nMod[isp]+1:end] .= 0.0
    end
end

function nuTk1_sub_update!(nk1::Vector{TA}, uk1::Vector{TA}, vthk1::Vector{TA}, 
    na::AbstractVector{T}, vth::AbstractVector{T}, 
    naik::Vector{TA}, uaik::Vector{TA}, vthik::Vector{TA}, 
    nMod::Vector{Int64},ns::Int64,is_nMod_renew::Vector{Bool}) where{T,TA}
    
    for isp in 1:ns
        if is_nMod_renew[isp]
            for k in 1:nMod[isp]
                nk1[isp][k] = na[isp] * naik[isp][k]
                uk1[isp][k] = vth[isp] * uaik[isp][k]
                vthk1[isp][k] = vth[isp] * vthik[isp][k]
            end
            nk1[isp][nMod[isp]+1:end] .= 0.0
        end
    end
end

"""
  Outputs:
    vthk = vthk_update(rho,Ia,Ka)
"""

# [], ns = nMod = 1
function vthk_update(rho::T,Ia::T,Ka::T) where{T}

    if Ia == 0.0
        return (Ka / CMcKa / rho[isp])^0.5
    else
        return ((Ka / CMcKa / rho[isp] - 2/3 * (Ia / rho[isp])^2))^0.5
    end
end
