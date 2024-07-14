"""
  M-theorems are the constraints inequations of the VFP equation.
"""

"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma = mb`, `na = ρa = ma * na`
    Mcjt: [t,nj], where `[na, Ia, M(2,0), M(3,1), M(4,0), ⋯]`
          j ∈ 0:2:N⁺, ℓ = 0
          j ∈ ℓ:2:N⁺, ℓ ≥ 1

  Outputs:
    Mhcsjt = Mhcsj_fDM(na,Ta,j,Mhcj;ℓ=ℓ)
    Mhcsj_fDM!(Mhcsjt,na,Ta,njMs,Mhcj;ℓ=ℓ)
    Mhcsj_fDM!(Mhcsjt,nat,Tat,Nt,j,Mhcj;ℓ=ℓ)
    Mhcsj_fDM!(Mhcsjt,nat,Tat,Nt,njMs,Mhcj;ℓ=ℓ)
"""
# When `ns ≥ 2` and `nMod ≥ 2`
# M̂a[1], the M-function of the multi-spice at time step `tₖ`

function Mhcsj_fDM(na::AbstractVector{T},ua::AbstractVector{T},vth::AbstractVector{T},j::Int64,Mhcj::AbstractVector{T};ℓ::Int64=0) where{T}
 
    nhas = na / sum(na)
    # uas = sum(nhas .* ua)
    Kρs32 = sum(nhas .* (vth.^2 + 1.5 * ua.^2))
    # vsth = (Kρs32 - 2/3 * uas^2) .^ 0.5
    vsth = (Kρs32 - 2/3 * (sum(nhas .* ua))^2) .^ 0.5
    if ℓ == 0
        return sum(nhas .* (vth / vsth) .^j .* Mhcj)
    elseif ℓ == 1
        if j == 1
            return sum(nhas .* (ua / vsth) .* Mhcj)
        else
            return sum(nhas .* (ua / vsth) .* (vth / vsth) .^(j-1) .* Mhcj)
        end
    else
        dfghbn
    end
end

# M̂a[j], 
function Mhcsj_fDM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},ua::AbstractVector{T},
    vth::AbstractVector{T},njMs::Int64,Mhcj::AbstractArray{T,N};ℓ::Int64=0) where{T,N}

    nhas = na / sum(na)
    # uas = sum(nhas .* ua)
    Kρs32 = sum(nhas .* (vth.^2 + 1.5 * ua.^2))
    # vsth = (Kρs32 - 2/3 * uas^2) .^ 0.5
    vsth = (Kρs32 - 2/3 * (sum(nhas .* ua))^2) .^ 0.5
    if ℓ == 0
        kj = 1
        # j = 2(kj - 1)
        Mhcsj[kj] = sum(nha .* Mhcj[kj,:])
        for kj in 2:njMs
            Mhcsj[kj] = sum(nhas .* (vth / vsth) .^(2(kj - 1)) .* Mhcj[kj,:])
        end
    elseif ℓ == 1
        kj = 1
        # j = 2(kj - 1) + 1
        Mhcsj[kj] = sum(nha .* (ua / vsth) .* Mhcj[kj,:])
        for kj in 2:njMs
            Mhcsj[kj] = sum(nhas .* (ua / vsth) .* (vth / vsth) .^(2(kj - 1)) .* Mhcj[kj,:])
        end
    else
        dfghbn
    end
end

# M̂a[t],
function Mhcsj_fDM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},ua::AbstractArray{T,N},
    vth::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N};ℓ::Int64=0) where{T,N}

    nhas = na / sum(na)
    if ℓ == 0
        for k in 1:Nt
            Kρs32 = sum(nhas .* (vth[k,:].^2 + 1.5 * ua[k,:].^2))
            vsth = (Kρs32 - 2/3 * (sum(nhas .* ua[k,:]))^2) .^ 0.5
            Mhcsj[k] = sum(nhas .* (vth[k,:] / vsth) .^j .* Mhcj[k,:])
        end
    elseif ℓ == 1
        for k in 1:Nt
            Kρs32 = sum(nhas .* (vth[k,:].^2 + 1.5 * ua[k,:].^2))
            vsth = (Kρs32 - 2/3 * (sum(nhas .* ua[k,:]))^2) .^ 0.5
            Mhcsj[k] = sum(nhas .* (ua[k,:] / vsth) .* (vth[k,:] / vsth) .^(j-1) .* Mhcj[k,:])
        end
    else
        dfghbn
    end
end

# M̂a[t,j],
function Mhcsj_fDM!(Mhcsjt::AbstractArray{T,N},na::AbstractVector{T},ua::AbstractArray{T,N},
    vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcjt::AbstractArray{T,N3};ℓ::Int64=0) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fDM!(a,na,ua[k,:],vth[k,:],njMs,Mhcjt[k,:,:];ℓ=ℓ)
        Mhcsjt[k,:] = a
    end
end

# M̂a[t], na[t,:]
function Mhcsj_fDM!(Mhcsj::AbstractVector{T},na::AbstractArray{T,N},ua::AbstractArray{T,N},
    vth::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N};ℓ::Int64=0) where{T,N}

    for k in 1:Nt
        a = Mhcsj[k]
        Mhcsj_fDM!(a,na[k,:],ua[k,:],vth[k,:],j,Mhcj[k,:];ℓ=ℓ)
        Mhcsj[k] = a

        # nhas = na[k,:] ./ sum(na[k,:])
        # Kρs32 = sum(nhas .* (vth[k,:].^2 + 1.5 * ua[k,:].^2))
        # vsth = (Kρs32 - 2/3 * (sum(nhas .* ua[k,:]))^2) .^ 0.5
        # Mhcsj[k] = sum(nhas .* (vth[k,:] / vsth) .^j .* Mhcj[k,:])
    end
end

# M̂a[t,j], na[t,:]
function Mhcsj_fDM!(Mhcsjt::AbstractArray{T,N},na::AbstractArray{T,N},ua::AbstractArray{T,N},
    vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcjt::AbstractArray{T,N3};ℓ::Int64=0) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fDM!(a,na[k,:],ua[k,:],vth[k,:],njMs,Mhcjt[k,:,:];ℓ=ℓ)
        Mhcsjt[k,:] = a
    end
end

"""
  Inputs:
    j::Int64, the order of the M function.
    na: when `ma ≠ mb`, `na = ρa = ma * na`

  Outputs:
    Mhcsjt = Mhcsj_fDM(ma,na,ua,Ta,j,Mhcj;ℓ=ℓ)
    Mhcsj_fDM!(Mhcsj,ma,na,ua,Ta,njMs,Mhcj;ℓ=ℓ)
    Mhcsj_fDM!(Mhcsjt,ma,nat,uat,Tat,Nt,j,Mhcjt;ℓ=ℓ)
    Mhcsj_fDM!(Mhcsjt,ma,nat,uat,Tat,Nt,njMs,Mhcjt;ℓ=ℓ)
"""

# M̂[1], 
function Mhcsj_fDM(ma::AbstractVector{T},na::AbstractVector{T},ua::AbstractVector{T},
    vth::AbstractVector{T},j::Int64,Mhcj::AbstractVector{T};ℓ::Int64=0) where{T}

    # nas = sum(na)
    nhas = na / sum(na)
    # mas = sum(ma .* nhas)
    mhas = ma / sum(ma .* nhas)
    # uas = sum(mhas .* nhas .* ua)
    Kρs32 = sum(mhas .* nhas .* (vth.^2 + 1.5 * ua.^2))
    # vsth2 = Kρs32 - 2/3 * uas^2
    vsth2 = Kρs32 - 2/3 * (sum(mhas .* nhas .* ua))^2
    if ℓ == 0
        return sum(nhas .* mhas .* (vth.^2 ./ vsth2) .^(j/2) .* Mhcj)
    elseif ℓ == 1
        if j == 1
            return sum(nhas .* mhas .* (ua ./ vsth2^0.5) .* Mhcj)
        else
            return sum(nhas .* mhas .* (ua ./ vsth2^0.5) .* (vth.^2 ./ vsth2) .^((j-1)/2) .* Mhcj)
        end
    else
        dfghbn
    end
end

# M̂[j],
function Mhcsj_fDM!(Mhcsj::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},
    ua::AbstractVector{T},vth::AbstractVector{T},njMs::Int64,Mhcj::AbstractArray{T,N};ℓ::Int64=0) where{T,N}

    # nas = sum(na)
    nhas = na / sum(na)
    # mas = sum(ma .* nhas)
    mhas = ma / sum(ma .* nhas)
    Kρs32 = sum(mhas .* nhas .* (vth.^2 + 1.5 * ua.^2))
    if ℓ == 0
        # vsth2 = Kρs32 - 2/3 * (sum(mhas .* nhas .* ua))^2
        vhaths2 = vth.^2 / (Kρs32 - 2/3 * (sum(mhas .* nhas .* ua))^2)
        kj = 1
        # j = 2(kj - 1) = 0
        Mhcsj[kj] = sum(nhas .* mhas .* Mhcj[kj,:])
        for kj in 2:njMs
            Mhcsj[kj] = sum(nhas .* mhas .* vhaths2 .^(kj - 1) .* Mhcj[kj,:])
        end
    elseif ℓ == 1
        vsth2 = Kρs32 - 2/3 * (sum(mhas .* nhas .* ua))^2
        vhaths2 = vth.^2 / vsth2
        kj = 1
        # j = 2(kj - 1) + 1 = 1
        # Mhcsj[kj] = sum(nhas .* mhas .* (ua ./ vsth2^0.5) .* vhaths2 .^((j-1)/2) .* Mhcj[kj,:])
        Mhcsj[kj] = sum(nhas .* mhas .* (ua ./ vsth2^0.5) .* Mhcj[kj,:])
        for kj in 2:njMs
            Mhcsj[kj] = sum(nhas .* mhas .* (ua ./ vsth2^0.5) .* vhaths2 .^(kj - 1) .* Mhcj[kj,:])
        end
    else
        dfghbn
    end
end

# M̂[t], 
function Mhcsj_fDM!(Mhcsjt::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},
    ua::AbstractArray{T,N},vth::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N};ℓ::Int64=0) where{T,N}

    for k in 1:Nt
        Mhcsjt[k] = Mhcsj_fDM(ma,na,ua[k,:],vth[k,:],j,Mhcj[k,:];ℓ=ℓ)
    end
end

# M̂[t,j],
function Mhcsj_fDM!(Mhcsjt::AbstractArray{T,N},ma::AbstractVector{T},na::AbstractVector{T},
    ua::AbstractArray{T,N},vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcj::AbstractArray{T,N3};ℓ::Int64=0) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fDM!(a,ma,na,ua[k,:],vth[k,:],njMs,Mhcj[k,:,:];ℓ=ℓ)
        Mhcsjt[k,:] = a
    end
end

# M̂[t], na[t,:]
function Mhcsj_fDM!(Mhcsjt::AbstractVector{T},ma::AbstractVector{T},na::AbstractArray{T,N},
    ua::AbstractArray{T,N},vth::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N};ℓ::Int64=0) where{T,N}

    for k in 1:Nt
        Mhcsjt[k] = Mhcsj_fDM(ma,na[k,:],ua[k,:],vth[k,:],j,Mhcj[k,:];ℓ=ℓ)
    end
end

# M̂[t,j], na[t,:]
function Mhcsj_fDM!(Mhcsjt::AbstractArray{T,N},ma::AbstractVector{T},na::AbstractArray{T,N},
    ua::AbstractArray{T,N},vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcj::AbstractArray{T,N3};ℓ::Int64=0) where{T,N,N3}

    for k in 1:Nt
        a = Mhcsjt[k,:]
        Mhcsj_fDM!(a,ma,na[k,:],ua[k,:],vth[k,:],njMs,Mhcj[k,:,:];ℓ=ℓ)
        Mhcsjt[k,:] = a
    end
end
