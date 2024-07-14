
"""
  The `jᵗʰ`-order moment of the zeroth-order coefficient of the normalized distribution function `f̂₀(v̂)`:
  
  when `is_renorm == true`,

    Msnt(j) = 4π * ∫₀^∞(v̂ʲ⁺² * f̂₀) dv̂
            = 2 / √π * Γ((j+3)/2) , j ≥ 2.

  That is:

    𝓜ⱼ(f̂₀) = 2^(j/2) * (j+1)!!,  j ∈ -2:2:N⁺
            = 2 / √π * ((j+1)/2)!,  j ∈ -1:2:N⁺
  where
    
    CjL(j,0) = (j+1)!! / 2^(j/2)

  Inputs:
    jvec:

  Outputs
    Msnt = MsntL2fM(Msnt,jvec,na,vth;is_renorm=is_renorm)
    Msnt = MsntL2fM(Msnt,jvec;is_renorm=is_renorm)

"""

Msntt(j) = 2 / sqrtpi * gamma((3+j)/2)       # When `j > 7`, the errors may be so large that cannot be ignored.
dMsntt(j) = - 2 / sqrtpi * 2 / (2+j) * gamma((4+j)/2)
ddMsntt(j) = 2 / sqrtpi * 2 / (1+j) * gamma((3+j)/2)

# 0.5D, [njMs], nai = 1, vthi = 1
function MsntL2fM(Msnt::AbstractVector{T},jvec::Vector{Int};is_renorm::Bool=true) where{T}

    k = 0
    if is_renorm
        Msnt .= 1.0
        return Msnt
    else
        for j in jvec
            k += 1
            if j == -2
                Msnt[k] = 2
            elseif j == -1
                Msnt[k] = 2.0 / sqrtpi
            elseif j == 0
                Msnt[k] = 1
            elseif j == 1
                Msnt[k] = 2.0 / sqrtpi
            elseif j == 2
                Msnt[k] = 1.5
            elseif j == 3
                Msnt[k] = 4.0 / sqrtpi
            else
                if iseven(j)
                    Msnt[k] = prod(3:2:j+1) / 2^(j/2)
                else
                    Msnt[k] = 2.0 / sqrtpi * prod(2:1:(j+1)/2)
                end
            end
        end
        return Msnt
    end
end

# 2.5D, [nMod,njMs,ns]
function MsntL2fM(Msnt::AbstractArray{T},jvec::Vector{Int},
    na::AbstractArray{T},vth::AbstractArray{T},ns::Int64;is_renorm::Bool=true) where{T}
    
    hjjgfdfg
    for isp in 1:ns
        Msnt[:,isp] = MsntL2fM(Msnt[:,isp],jvec;is_renorm=is_renorm)
        if prod(isone.(vth[:,isp]))
            Msnt[:,isp] *= sum(na[:,isp])
        else
            k = 0
            for j in jvec
                k += 1
                Msnt[k,isp] *= sum(na[:,isp] .* vth[:,isp].^j)
            end
        end
    end
    return Msnt
end

"""
"""

# 1.5D, [nMod,njMs]
function MsntL2fM(Msnt::AbstractVector{T},jvec::Vector{Int},
    na::AbstractVector{T},vth::AbstractVector{T},nMod::Int64;is_renorm::Bool=true) where{T}
    
    rtyhjkm
    Msnt = MsntL2fM(Msnt,jvec;is_renorm=is_renorm)
    if prod(isone.(vth))
        return Msnt * sum(na)
    else
        k = 0
        for j in jvec
            k += 1
            Msnt[k] *= sum(na .* vth.^j)
        end
    end
    return Msnt
end

# 1.5D, [njMs,ns]
function MsntL2fM(Msnt::AbstractArray{T},jvec::Vector{Int},ns::Int64,
    na::AbstractVector{T},vth::AbstractVector{T};is_renorm::Bool=true) where{T}

    for isp in 1:ns
        Msnt[:,isp] = MsntL2fM(Msnt[:,isp],jvec;is_renorm=is_renorm)
        if isone(vth[isp])
            Msnt[:,isp] *= na[isp]
        else
            k = 0
            for j in jvec
                k += 1
                Msnt[k,isp] *= (na[isp] * vth[isp].^j)
            end
        end
    end
    return Msnt
end

# 1.5D, [njMs,ns], nai = 1, vthi = 1
function MsntL2fM(Msnt::AbstractArray{T},jvec::Vector{Int},ns::Int64;is_renorm::Bool=true) where{T}

    for isp in 1:ns
        Msnt[:,isp] = MsntL2fM(Msnt[:,isp],jvec;is_renorm=is_renorm)
    end
    return Msnt
end

"""
"""

# 0.5D, [njMs]
function MsntL2fM(Msnt::AbstractVector{T},jvec::Vector{Int},
    na::Float64,vth::Float64;is_renorm::Bool=true) where{T}

    if vth == 1.0 && na == 1.0
        return MsntL2fM(Msnt,jvec;is_renorm=is_renorm)
    else
        Msnt[:] = MsntL2fM(Msnt,jvec;is_renorm=is_renorm)
        Msnt *= (na * vth)
        return Msnt
    end
end
