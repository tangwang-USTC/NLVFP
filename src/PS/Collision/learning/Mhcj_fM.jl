"""
  M-theorems are the constraints inequations of the VFP equation.
"""

"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma ≠ mb`, `na = ρa = ma * na`

  Outputs:
    Mhcjt = Mhcj_fM(ma,na,vth,j)
    Mhcj_fM!(Mhcj,ma,na,vth,njMs)
    Mhcj_fM!(Mhcjt,ma,nat,vtht,Nt,j)
    Mhcj_fM!(Mhcjt,ma,nat,vtht,Nt,njMs)
"""

# When `ns = 1` and `nMod ≥ 2` and `j ∈ ℓ:2:∞` 
# M̂a[1], the M-function of the single spice at time step `tₖ`
















function Mhcj_fM(nah::AbstractVector{T},
    vth::AbstractVector{T},j::Int64) where{T}

    return sum(nhas .* mhas .* vhaths2 .^(j/2))
end

# M̂[j],
function Mhcj_fM!(Mhcj::AbstractVector{T},ma::AbstractVector{T},
    na::AbstractVector{T},vth::AbstractVector{T},njMs::Int64) where{T}

    nas = sum(na)
    # mas = sum(ma .* na) / nas
    nhas = na / nas
    mhas = ma / (sum(ma .* na) / nas)
    vhaths2 = vth.^2 ./ sum(nhas .* mhas .* vth.^2)
    kj = 1
    # j = 2(kj - 1) = 0
    Mhcj[kj] = sum(nhas .* mhas)
    for kj in 2:njMs
        Mhcj[kj] = sum(nhas .* mhas .* vhaths2 .^(kj - 1))
    end
end

# M̂[t], 
function Mhcj_fM!(Mhcjt::AbstractVector{T},ma::AbstractVector{T},
    na::AbstractVector{T},vth::AbstractArray{T,N},Nt::Int64,j::Int64) where{T,N}

    for k in 1:Nt
        Mhcjt[k] = Mhcj_fM(ma,na,vth[k,:],j)
    end
end

# M̂[t,j],
function Mhcj_fM!(Mhcjt::AbstractArray{T,N},ma::AbstractVector{T},
    na::AbstractVector{T},vth::AbstractArray{T,N},Nt::Int64,njMs::Int64) where{T,N}

    for k in 1:Nt
        a = Mhcjt[k,:]
        Mhcj_fM!(a,ma,na,vth[k,:],njMs)
        Mhcjt[k,:] = a
    end
end

# M̂[t], na[t,:]
function Mhcj_fM!(Mhcjt::AbstractVector{T},ma::AbstractVector{T},
    na::AbstractArray{T,N},vth::AbstractArray{T,N},Nt::Int64,j::Int64) where{T,N}

    for k in 1:Nt
        Mhcjt[k] = Mhcj_fM(ma,na[k,:],vth[k,:],j)
    end
end

# M̂[t,j], na[t,:]
function Mhcj_fM!(Mhcjt::AbstractArray{T,N},ma::AbstractVector{T},
    na::AbstractArray{T,N},vth::AbstractArray{T,N},Nt::Int64,njMs::Int64) where{T,N}

    for k in 1:Nt
        a = Mhcjt[k,:]
        Mhcj_fM!(a,ma,na[k,:],vth[k,:],njMs)
        Mhcjt[k,:] = a
    end
end

