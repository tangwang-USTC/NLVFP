"""
  M-theorems are the constraints inequations of the VFP equation.
"""

"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma = mb`, `na = ρa = ma * na`

  Outputs:
    Mhcsjt = Mhcsj_fM(na,vth,j)
    Mhcsj_fM!(Mhcsjt,na,vth,njMs)
    Mhcsj_fM!(Mhcsjt,nat,vtht,Nt,j)
    Mhcsj_fM!(Mhcsjt,nat,vtht,Nt,njMs)
"""

# When `ns ≥ 2` and `nMod = 1` which means that `M̂*[isp,j,ℓ=0] ≡ 1`
# M̂a[1], the M-function of the multi-spice at time step `tₖ`
function Mhcsj_fM(na::AbstractVector{T},vth::AbstractVector{T},j::Int64) where{T}

    nhas = na / sum(na)
    vsth = sum(nhas .* vth.^2) .^ 0.5
    return sum(nhas .* (vth / vsth) .^j)
end

# M̂a[j], 
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},
    vth::AbstractVector{T},njMs::Int64) where{T}

    nhas = na / sum(na)
    vsth = sum(nhas .* vth.^2) .^ 0.5
    kj = 1
    Mhcsj[kj] = sum(nha)
    for kj in 2:njMs
        # j = 2(kj - 1)
        Mhcsj[kj] = sum(nhas .* (vth / vsth) .^(2(kj - 1)))
    end
end

# M̂a[t],
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},
    vth::AbstractArray{T,N},Nt::Int64,j::Int64) where{T,N}

    nhas = na / sum(na)
    for k in 1:Nt
        vsth = sum(nhas .* vth[k,:].^2) .^ 0.5
        Mhcsj[k] = sum(nhas .* (vth[k,:] / vsth) .^j)
    end
end

# M̂a[t,j],
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},na::AbstractVector{T},
    vth::AbstractArray{T,N},Nt::Int64,njMs::Int64) where{T,N}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,na,vth[k,:],njMs)
        Mhcsjt[k,:] = a
    end
end

# M̂a[t], na[t,:] 
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractArray{T,N},
    vth::AbstractArray{T,N},Nt::Int64,j::Int64) where{T,N}

    for k in 1:Nt
        nhas = na[k,:] ./ sum(na[k,:])
        vsth = sum(nhas .* vth[k,:].^2) .^ 0.5
        Mhcsj[k] = sum(nhas .* (vth[k,:] / vsth) .^j)
    end
end

# M̂a[t,j], na[t,:]
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},na::AbstractArray{T,N},
    vth::AbstractArray{T,N},Nt::Int64,njMs::Int64) where{T,N}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,na[k,:],vth[k,:],njMs)
        Mhcsjt[k,:] = a
    end
end

"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma ≠ mb`, `na = ρa = ma * na`

  Outputs:
    Mhcsjt = Mhcsj_fM(ma,na,vth,j)
    Mhcsj_fM!(Mhcsj,ma,na,vth,njMs)
    Mhcsj_fM!(Mhcsjt,ma,nat,vtht,Nt,j)
    Mhcsj_fM!(Mhcsjt,ma,nat,vtht,Nt,njMs)
"""

# M̂[1], 
function Mhcsj_fM(ma::AbstractVector{T},na::AbstractVector{T},
    vth::AbstractVector{T},j::Int64) where{T}

    nas = sum(na)
    # mas = sum(ma .* na) / nas
    nhas = na / nas
    mhas = ma / (sum(ma .* na) / nas)
    vhaths2 = vth.^2 ./ sum(nhas .* mhas .* vth.^2)
    return sum(nhas .* mhas .* vhaths2 .^(j/2))
end

# M̂[j],
function Mhcsj_fM!(Mhcsj::AbstractVector{T},ma::AbstractVector{T},
    na::AbstractVector{T},vth::AbstractVector{T},njMs::Int64) where{T}

    nas = sum(na)
    # mas = sum(ma .* na) / nas
    nhas = na / nas
    mhas = ma / (sum(ma .* na) / nas)
    vhaths2 = vth.^2 ./ sum(nhas .* mhas .* vth.^2)
    kj = 1
    # j = 2(kj - 1) = 0
    Mhcsj[kj] = sum(nhas .* mhas)
    for kj in 2:njMs
        Mhcsj[kj] = sum(nhas .* mhas .* vhaths2 .^(kj - 1))
    end
end

# M̂[t], 
function Mhcsj_fM!(Mhcsjt::AbstractVector{T},ma::AbstractVector{T},
    na::AbstractVector{T},vth::AbstractArray{T,N},Nt::Int64,j::Int64) where{T,N}

    for k in 1:Nt
        Mhcsjt[k] = Mhcsj_fM(ma,na,vth[k,:],j)
    end
end

# M̂[t,j],
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},ma::AbstractVector{T},
    na::AbstractVector{T},vth::AbstractArray{T,N},Nt::Int64,njMs::Int64) where{T,N}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,ma,na,vth[k,:],njMs)
        Mhcsjt[k,:] = a
    end
end

# M̂[t], na[t,:]
function Mhcsj_fM!(Mhcsjt::AbstractVector{T},ma::AbstractVector{T},
    na::AbstractArray{T,N},vth::AbstractArray{T,N},Nt::Int64,j::Int64) where{T,N}

    for k in 1:Nt
        Mhcsjt[k] = Mhcsj_fM(ma,na[k,:],vth[k,:],j)
    end
end

# M̂[t,j], na[t,:]
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},ma::AbstractVector{T},
    na::AbstractArray{T,N},vth::AbstractArray{T,N},Nt::Int64,njMs::Int64) where{T,N}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,ma,na[k,:],vth[k,:],njMs)
        Mhcsjt[k,:] = a
    end
end


"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma = mb`, `na = ρa = ma * na`

  Outputs:
    Mhcsjt = Mhcsj_fM(na,vath,j,Mhcj)
    Mhcsj_fM!(Mhcsjt,na,vath,njMs,Mhcj)
    Mhcsj_fM!(Mhcsjt,nat,vatht,Nt,j,Mhcj)
    Mhcsj_fM!(Mhcsjt,nat,vatht,Nt,njMs,Mhcj)
"""
# When `ns ≥ 2` and `nMod ≥ 2`
# M̂a[1]

function Mhcsj_fM(na::AbstractVector{T},vth::AbstractVector{T},j::Int64,Mhcj::AbstractVector{T}) where{T}

    nhas = na / sum(na)
    vsth = sum(nhas .* vth.^2) .^ 0.5k0
    return sum(nhas .* (vth / vsth) .^j .* Mhcj)
end

# M̂a[j], 
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},
    vth::AbstractVector{T},njMs::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

    nhas = na / sum(na)
    vsth = sum(nhas .* vth.^2) .^ 0.5
    kj = 1
    Mhcsj[kj] = sum(nha .* Mhcj[kj,:])
    for kj in 2:njMs
        # j = 2(kj - 1)
        Mhcsj[kj] = sum(nhas .* (vth / vsth) .^(2(kj - 1)) .* Mhcj[kj,:])
    end
end

# M̂a[t],
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},
    vth::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

    nhas = na / sum(na)
    for k in 1:Nt
        vsth = sum(nhas .* vth[k,:].^2) .^ 0.5
        Mhcsj[k] = sum(nhas .* (vth[k,:] / vsth) .^j .* Mhcj[k,:])
    end
end

# M̂a[t,j],
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},na::AbstractVector{T},
    vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcjt::AbstractArray{T,N3}) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,na,vth[k,:],njMs,Mhcjt[k,:,:])
        Mhcsjt[k,:] = a
    end
end

# M̂a[t], na[t,:]
function Mhcsj_fM!(Mhcsj::AbstractVector{T},na::AbstractArray{T,N},
    vth::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

    for k in 1:Nt
        nhas = na[k,:] ./ sum(na[k,:])
        vsth = sum(nhas .* vth[k,:].^2) .^ 0.5
        Mhcsj[k] = sum(nhas .* (vth[k,:] / vsth) .^j .* Mhcj[k,:])
    end
end

# M̂a[t,j], na[t,:]
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},na::AbstractArray{T,N},
    vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcjt::AbstractArray{T,N3}) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,na[k,:],vth[k,:],njMs,Mhcjt[k,:,:])
        Mhcsjt[k,:] = a
    end
end

"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma ≠ mb`, `na = ρa = ma * na`

  Outputs:
    Mhcsjt = Mhcsj_fM(ma,na,vath,j,Mhcj)
    Mhcsj_fM!(Mhcsj,ma,na,vath,njMs,Mhcj)
    Mhcsj_fM!(Mhcsjt,ma,nat,vatht,Nt,j,Mhcjt)
    Mhcsj_fM!(Mhcsjt,ma,nat,vatht,Nt,njMs,Mhcjt)
"""

# M̂[1], 
function Mhcsj_fM(ma::AbstractVector{T},na::AbstractVector{T},
    vth::AbstractVector{T},j::Int64,Mhcj::AbstractVector{T}) where{T}

    nas = sum(na)
    # mas = sum(ma .* na) / nas
    nhas = na / nas
    mhas = ma / (sum(ma .* na) / nas)
    vhaths2 = vth.^2 ./ sum(nhas .* mhas .* vth.^2)
    return sum(nhas .* mhas .* vhaths2 .^(j/2) .* Mhcj)
end

# M̂[j],
function Mhcsj_fM!(Mhcsj::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},
    vth::AbstractVector{T},njMs::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

    nas = sum(na)
    # mas = sum(ma .* na) / nas
    nhas = na / nas
    mhas = ma / (sum(ma .* na) / nas)
    vhaths2 = vth.^2 ./ sum(nhas .* mhas .* vth.^2)
    kj = 1
    # j = 2(kj - 1) = 0                                # na
    Mhcsj[kj] = sum(nhas .* mhas .* Mhcj[kj,:])
    for kj in 2:njMs
        Mhcsj[kj] = sum(nhas .* mhas .* vhaths2 .^(kj - 1) .* Mhcj[kj,:])
    end
end

# M̂[t], 
function Mhcsj_fM!(Mhcsjt::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},
    vth::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

    for k in 1:Nt
        Mhcsjt[k] = Mhcsj_fM(ma,na,vth[k,:],j,Mhcj[k,:])
    end
end

# M̂[t,j],
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},ma::AbstractVector{T},na::AbstractVector{T},
    vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcj::AbstractArray{T,N3}) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,ma,na,vth[k,:],njMs,Mhcj[k,:,:])
        Mhcsjt[k,:] = a
    end
end

# M̂[t], na[t,:]
function Mhcsj_fM!(Mhcsjt::AbstractVector{T},ma::AbstractVector{T},na::AbstractArray{T,N},
    vth::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

    for k in 1:Nt
        Mhcsjt[k] = Mhcsj_fM(ma,na[k,:],vth[k,:],j,Mhcj[k,:])
    end
end

# M̂[t,j], na[t,:]
function Mhcsj_fM!(Mhcsjt::AbstractArray{T,N},ma::AbstractVector{T},na::AbstractArray{T,N},
    vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcj::AbstractArray{T,N3}) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fM!(a,ma,na[k,:],vth[k,:],njMs,Mhcj[k,:,:])
        Mhcsjt[k,:] = a
    end
end
