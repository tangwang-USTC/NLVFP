"""
  M-theorems are the constraints inequations of the VFP equation.
"""

"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma = mb`, `na = ρa = ma * na`
    Mcjt: [t,nj], where `[Ia, M(2,0), M(3,1), M(4,0), ⋯]`
          j ∈ 1:1:N⁺, ℓ = 0 and 1

  Outputs:
    Mhcsj_fDM!(Mhcsjt,na,ua,vth,njMs,Mhcj10)
"""

# M̂[t,j],
function Mhcsj_fDM!(Mhcsjt::AbstractArray{T,N},ma::AbstractVector{T},na::AbstractVector{T},
    ua::AbstractArray{T,N},vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcj10::AbstractArray{T,N3}) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fDM!(a,ma,na,ua[k,:],vth[k,:],njMs,Mhcj10[k,:,:])
        Mhcsjt[k,:] = a
    end
end

# M̂[j],
function Mhcsj_fDM!(Mhcsj::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},
    ua::AbstractVector{T},vth::AbstractVector{T},njMs::Int64,Mhcj10::AbstractArray{T,N}) where{T,N}

    # nas = sum(na)
    nhas = na / sum(na)
    # mas = sum(ma .* nhas)
    mhas = ma / sum(ma .* nhas)
    Kρs32 = sum(mhas .* nhas .* (vth.^2 + 2/3 * ua.^2))

    vsth = (Kρs32 - 2/3 * (sum(mhas .* nhas .* ua))^2)^0.5
    vhaths = vth / vsth

    # ℓ = 1                       when `isodd(kj)`
    kj = 1
    Mhcsj[kj] = sum(nhas .* mhas .* (ua ./ vsth) .* Mhcj10[kj,:])
    for kj in 3:2:njMs
        Mhcsj[kj] = sum(nhas .* mhas .* (ua ./ vsth) .* vhaths .^(kj - 1) .* Mhcj10[kj,:])
    end

    # ℓ = 0                       when `iseven(kj)`
    for kj in 2:2:njMs
        Mhcsj[kj] = sum(nhas .* mhas .* vhaths .^kj .* Mhcj10[kj,:])
    end
end

# M̂a[t,j], 
function Mhcsj_fDM!(Mhcsjt::AbstractArray{T,N},na::AbstractVector{T},ua::AbstractVector{T},
    vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcj10::AbstractArray{T,N3}) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fDM!(a,na,ua[k,:],vth[k,:],njMs,Mhcj10[k,:,:])
        Mhcsjt[k,:] = a
    end
end

# M̂a[j],  ma[1] = ma[2]
function Mhcsj_fDM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},ua::AbstractVector{T},
    vth::AbstractVector{T},njMs::Int64,Mhcj10::AbstractArray{T,N}) where{T,N}

    sddd
    nhas = na / sum(na)
    # uas = sum(nhas .* ua)
    Kρs32 = sum(nhas .* (vth.^2 + 2/3 * ua.^2))
    # vsth = (Kρs32 - 2/3 * uas^2) .^ 0.5
    vsth = (Kρs32 - 2/3 * (sum(nhas .* ua))^2) .^ 0.5
    vhaths = vth / vsth

    # ℓ = 1                       when `isodd(kj)`
    kj = 1                     
    Mhcsj[kj] = sum(nha .* (ua / vsth) .* Mhcj10[kj,:])
    for kj in 3:2:njMs
        Mhcsj[kj] = sum(nhas .* (ua / vsth) .* vhaths .^(kj - 1) .* Mhcj10[kj,:])
    end
    
    # ℓ = 0                       when `iseven(kj)`
    for kj in 2:2:njMs
        Mhcsj[kj] = sum(nhas .* vhaths .^kj .* Mhcj10[kj,:])
    end
end
