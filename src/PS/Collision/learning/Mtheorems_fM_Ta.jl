"""
  M-theorems are the constraints inequations of the VFP equation.
"""

"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma = mb`, `na = ρa = ma * na`

  Outputs:
    Mhcsjt = Mhcsj_fM(na,Ta,j)
    Mhcsj_fM!(Mhcsjt,na,Ta,njMs)
    Mhcsj_fM!(Mhcsjt,nat,Tat,Nt,j)
    Mhcsj_fM!(Mhcsjt,nat,Tat,Nt,njMs)
"""
# When `ns ≥ 2` and `nMod = 1` which means that `M̂*[isp,j,ℓ=0] ≡ 1`
# M̂a[1], the M-function of the single spice at time step `tₖ`
function Mhcsj_fM(na::AbstractVector{T},Ta::AbstractVector{T},j::Int64) where{T}

    nas = sum(na)
    Tas = sum(na .* Ta) / nas
    return sum((na / nas) .* (Ta / Tas) .^(j/2))
end

# M̂a[j], 
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},
    Ta::AbstractVector{T},njMs::Int64) where{T}

    nas = sum(na)
    nha = na / nas
    # Tas = sum(na .* Ta) / nas
    Tha = Ta ./ (sum(na .* Ta) / nas)
    kj = 1
    Mhcsj[kj] = sum(nha)
    for kj in 2:njMs
        # j = 2(kj - 1)
        Mhcsj[kj] = sum(nha .* Tha .^(kj - 1))
    end
end

# M̂a[t],
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},
    Ta::AbstractArray{T,N},Nt::Int64,j::Int64) where{T,N}

    nas = sum(na)
    nhas = na / nas
    for k in 1:Nt
        Tas = sum(nhas .* Ta[k,:])
        Mhcsj[k] = sum(nhas .* (Ta[k,:] / Tas) .^(j/2))
    end
end

# M̂a[t,j],
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},na::AbstractVector{T},
    Ta::AbstractArray{T,N},Nt::Int64,njMs::Int64) where{T,N}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,na,Ta[k,:],njMs)
        Mhcsjt[k,:] = a
    end
end

# M̂a[t], na[t,:]
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractArray{T,N},
    Ta::AbstractArray{T,N},Nt::Int64,j::Int64) where{T,N}

    for k in 1:Nt
        nas = sum(na[k,:])
        Tas = sum(na[k,:] .* Ta[k,:]) / nas
        Mhcsj[k] = sum((na[k,:] / nas) .* (Ta[k,:] / Tas) .^(j/2))
    end
end

# M̂a[t,j], na[t,:]
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},na::AbstractArray{T,N},
    Ta::AbstractArray{T,N},Nt::Int64,njMs::Int64) where{T,N}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,na[k,:],Ta[k,:],njMs)
        Mhcsjt[k,:] = a
    end
end

"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma ≠ mb`, `na = ρa = ma * na`

  Outputs:
    Mhcsjt = Mhcsj_fM(ma,na,Ta,j)
    Mhcsj_fM!(Mhcsj,ma,na,Ta,njMs)
    Mhcsj_fM!(Mhcsjt,ma,nat,Tat,Nt,j)
    Mhcsj_fM!(Mhcsjt,ma,nat,Tat,Nt,njMs)
"""

# M̂[1], 
function Mhcsj_fM(ma::AbstractVector{T},na::AbstractVector{T},
    Ta::AbstractVector{T},j::Int64) where{T}

    nas = sum(na)
    mas = sum(ma .* na) / nas
    nhas = na / nas
    mhas = ma / mas
    Tas = sum(nhas .* Ta)
    # Tas = sum(na .* Ta) / nas
    vhaths2 = (Ta / Tas) ./ mhas
    return sum(nhas .* mhas .* vhaths2 .^(j/2))
end

# M̂[j],
function Mhcsj_fM!(Mhcsj::AbstractVector{T},ma::AbstractVector{T},
    na::AbstractVector{T},Ta::AbstractVector{T},njMs::Int64) where{T}

    nas = sum(na)
    mas = sum(ma .* na) / nas
    nhas = na / nas
    mhas = ma / mas
    Tas = sum(nhas .* Ta)
    # Tas = sum(na .* Ta) / nas
    vhaths2 = (Ta / Tas) ./ mhas
    kj = 1
    # j = 2(kj - 1) = 0
    Mhcsj[kj] = sum(nhas .* mhas)
    for kj in 2:njMs
        Mhcsj[kj] = sum(nhas .* mhas .* vhaths2 .^(kj - 1))
    end
end

# M̂[t], 
function Mhcsj_fM!(Mhcsjt::AbstractVector{T},ma::AbstractVector{T},
    na::AbstractVector{T},Ta::AbstractArray{T,N},Nt::Int64,j::Int64) where{T,N}

    for k in 1:Nt
        Mhcsjt[k] = Mhcsj_fM(ma,na,Ta[k,:],j)
    end
end

# M̂[t,j],
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},ma::AbstractVector{T},
    na::AbstractVector{T},Ta::AbstractArray{T,N},Nt::Int64,njMs::Int64) where{T,N}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,ma,na,Ta[k,:],njMs)
        Mhcsjt[k,:] = a
    end
end

# M̂[t], na[t,:]
function Mhcsj_fM!(Mhcsjt::AbstractVector{T},ma::AbstractVector{T},
    na::AbstractArray{T,N},Ta::AbstractArray{T,N},Nt::Int64,j::Int64) where{T,N}

    for k in 1:Nt
        Mhcsjt[k] = Mhcsj_fM(ma,na[k,:],Ta[k,:],j)
    end
end

# M̂[t,j], na[t,:]
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},ma::AbstractVector{T},
    na::AbstractArray{T,N},Ta::AbstractArray{T,N},Nt::Int64,njMs::Int64) where{T,N}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,ma,na[k,:],Ta[k,:],njMs)
        Mhcsjt[k,:] = a
    end
end


"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma = mb`, `na = ρa = ma * na`

  Outputs:
    Mhcsjt = Mhcsj_fM(na,Ta,j,Mhcj)
    Mhcsj_fM!(Mhcsjt,na,Ta,njMs,Mhcj)
    Mhcsj_fM!(Mhcsjt,nat,Tat,Nt,j,Mhcj)
    Mhcsj_fM!(Mhcsjt,nat,Tat,Nt,njMs,Mhcj)
"""
# When `ns ≥ 2` and `nMod ≥ 2`
# M̂a[1], the M-function of the single spice at time step `tₖ`
function Mhcsj_fM(na::AbstractVector{T},Ta::AbstractVector{T},j::Int64,Mhcj::AbstractVector{T}) where{T}

    nas = sum(na)
    Tas = sum(na .* Ta) / nas
    return sum((na / nas) .* (Ta / Tas) .^(j/2) .* Mhcj)
end

# M̂a[j], 
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},
    Ta::AbstractVector{T},njMs::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

    nas = sum(na)
    nha = na / nas
    # Tas = sum(na .* Ta) / nas
    Tha = Ta ./ (sum(na .* Ta) / nas)
    kj = 1
    Mhcsj[kj] = sum(nha .* Mhcj[kj,:])
    for kj in 2:njMs
        # j = 2(kj - 1)
        Mhcsj[kj] = sum(nha .* Tha .^(kj - 1) .* Mhcj[kj,:])
    end
end

# M̂a[t],
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},
    Ta::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N}) where{T,N}
    
    nas = sum(na)
    nhas = na / nas
    for k in 1:Nt
        Tas = sum(na .* Ta[k,:]) / nas
        Mhcsj[k] = sum(nhas .* (Ta[k,:] / Tas) .^(j/2) .* Mhcj[k,:])
    end
end

# M̂a[t,j], Mhcjt[t,j,isp]
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},na::AbstractVector{T},
    Ta::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcjt::AbstractArray{T,N3}) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,na,Ta[k,:],njMs,Mhcjt[k,:,:])
        Mhcsjt[k,:] = a
    end
end

# M̂a[t], na[t,:]
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractArray{T,N},
    Ta::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

    for k in 1:Nt
        nas = sum(na[k,:])
        Tas = sum(na[k,:] .* Ta[k,:]) / nas
        Mhcsj[k] = sum((na[k,:] / nas) .* (Ta[k,:] / Tas) .^(j/2) .* Mhcj[k,:])
    end
end

# M̂a[t,j], na[t,:], Mhcjt[t,j,isp]
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},na::AbstractArray{T,N},
    Ta::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcjt::AbstractArray{T,N3}) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,na[k,:],Ta[k,:],njMs,Mhcjt[k,:,:])
        Mhcsjt[k,:] = a
    end
end

"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma ≠ mb`, `na = ρa = ma * na`

  Outputs:
    Mhcsjt = Mhcsj_fM(ma,na,Ta,j,Mhcj)
    Mhcsj_fM!(Mhcsj,ma,na,Ta,njMs,Mhcj)
    Mhcsj_fM!(Mhcsjt,ma,nat,Tat,Nt,j,Mhcjt)
    Mhcsj_fM!(Mhcsjt,ma,nat,Tat,Nt,njMs,Mhcjt)
"""

# M̂[1], 
function Mhcsj_fM(ma::AbstractVector{T},na::AbstractVector{T},
    Ta::AbstractVector{T},j::Int64,Mhcj::AbstractVector{T}) where{T}

    nas = sum(na)
    mas = sum(ma .* na) / nas
    nhas = na / nas
    mhas = ma / mas
    Tas = sum(nhas .* Ta)
    # Tas = sum(na .* Ta) / nas
    vhaths2 = (Ta / Tas) ./ mhas
    return sum(nhas .* mhas .* vhaths2 .^(j/2) .* Mhcj)
end

# M̂[j],
function Mhcsj_fM!(Mhcsj::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},
    Ta::AbstractVector{T},njMs::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

    nas = sum(na)
    mas = sum(ma .* na) / nas
    nhas = na / nas
    mhas = ma / mas
    Tas = sum(nhas .* Ta)
    # Tas = sum(na .* Ta) / nas
    vhaths2 = (Ta / Tas) ./ mhas
    kj = 1
    # j = 2(kj - 1) = 0
    Mhcsj[kj] = sum(nhas .* mhas .* Mhcj[kj,:])
    for kj in 2:njMs
        Mhcsj[kj] = sum(nhas .* mhas .* vhaths2 .^(kj - 1) .* Mhcj[kj,:])
    end
end

# M̂[t], 
function Mhcsj_fM!(Mhcsjt::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},
    Ta::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

    for k in 1:Nt
        Mhcsjt[k] = Mhcsj_fM(ma,na,Ta[k,:],j,Mhcj[k,:])
    end
end

# M̂[t,j],
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},ma::AbstractVector{T},na::AbstractVector{T},
    Ta::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcj::AbstractArray{T,N3}) where{T,N,N3}

    for k in 1:Nt
        Mhcsjt[k,:] = a
        Mhcsj_fM!(a,ma,na,Ta[k,:],njMs,Mhcj[k,:,:])
        Mhcsjt[k,:] = a
    end
end

# M̂[t], na[t,:]
function Mhcsj_fM!(Mhcsjt::AbstractVector{T},ma::AbstractVector{T},na::AbstractArray{T,N},
    Ta::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

    for k in 1:Nt
        Mhcsjt[k] = Mhcsj_fM(ma,na[k,:],Ta[k,:],j,Mhcj[k,:])
    end
end

# M̂[t,j], na[t,:]
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},ma::AbstractVector{T},na::AbstractArray{T,N},
    Ta::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcj::AbstractArray{T,N3}) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,ma,na[k,:],Ta[k,:],njMs,Mhcj[k,:,:])
        Mhcsjt[k,:] = a
    end
end