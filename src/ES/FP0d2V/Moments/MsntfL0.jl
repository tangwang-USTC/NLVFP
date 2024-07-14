
"""
  In theory, the first `jᵗʰ`-order re-normalzied moment of the first `ℓᵗʰ`-order
  coefficient of the normal distribution function `f̂ₗ(v̂)` is:

    Msnnt(j) = 𝓜ⱼ(f̂ₗ(v̂)) = 4π * ∫₀^∞(v̂ʲ⁺² * f̂ₗ(v̂)) dv̂
    Msnnt(2) = 𝓜₂(f̂₀(v̂)) = 2π * ∫₀^∞(v̂ʲ⁺² * f̂ₗ(v̂)) dv̂

  Where

    f̂ₗ(v̂) = vₜₕ³/nₐ * f(v).

  The `jᵗʰ`-order re-normalzied moments of the `ℓᵗʰ`-order
  coefficient of the normalized distribution function `f̂ₗ(v̂)`.

    𝓜̂̂ⱼ*(f̂ₗ(v̂)) = 𝓜̂ⱼ(f̂ₗ) / CjL(j,L)
               = 4π / CjL(j,L) * ∫₀⁹(v̂²⁺ʲf̂ₗ(v̂))dv̂
               = cm(j,L) * (1 + ∑ₖ(cₖ(j,L) (2ûa²)ᵏ)), k = 1:1:(j-L)/2, j ∈ L:2:N⁺

  where
     ûa = û / v̂th = uai / vthiᵢ
     CjL(j,L) = (j+L+1)!! / (2L-1)!! * 2^((j-L)/2)
     cm(j,L) = n̂a * v̂th^j * ûaᴸ
     cₖ(j,L) = (2L+1)!! / (2(L+k)+1)!! * C((j-L)/2,k)

  Here, `C((j-L)/2,k)` is the binomial coefficient `Cₙᵏ` when `n=(j-L)/2`.

  Especially, when `L = 0` leads to:

     CjL(j,L=0) = (j+1)!! * 2^(j/2),      j ∈ L:2:N⁺
     cm(j,L=0) = n̂a * v̂thʲ
     cₖ(j,L=0) = 1 / (2k+1)!! * C(j/2,k)

  Especially, when `j = L` leads to:

     CjL(L,L) = (2L+1),          j ∈ L:2:N⁺
     cm(L,L) = n̂a * ûaᴸ
     cₖ(L,L) = 0, ∀ k.

"""


"""
  When `j ∈ L:2:N⁺`
  
    Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

  Inputs:
    uai: = ûᵢ / v̂thᵢ; but in the inner procedure, we applying `uaa`

  Outputs:
    Msnnt = MsnntL2fL0(j,L,nai,uai,vthi;is_renorm=is_renorm)
    Msnnt = MsnntL2fL0(j,L,uai;is_renorm=is_renorm)
"""

# 0D, the `jᵗʰ`-order re-normalized moment of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function MsnntL2fL0(j::Int64,L::Int64,
    nai::T,uai::T,vthi::T;is_renorm::Bool=true) where{T}

    if nai == 0.0
        return 0.0 |> T
    else
        j ≥ L || @error("j ≥ L is needed in the polynomial moments here!")
        if iseven(L)
            if L == 0
                if j == 0
                    Msnnt = nai |> T
                else
                    uaa = uai / vthi
                    if uaa == 0
                        Msnnt = nai * vthi^j |> T
                    else
                        jL2 = j / 2 |> Int
                        Msnnt = 1 |> T
                        for k in 1:jL2
                            Msnnt += binomial(jL2,k) / prod(3:2:(2k+1)) * (2 * uai ^2)^k
                        end
                        Msnnt *= (nai * vthi^j)
                    end
                end
                if is_renorm
                    return Msnnt
                else
                    return Msnnt * CjLL2(j)
                end
            else
                uaa = uai / vthi
                if uaa == 0.0
                    Msnnt = 0 |> T
                else
                    if j == L
                        Msnnt = nai * vthi^j * uaa^L |> T
                    else
                        jL2 = (j - L) / 2 |> Int
                        Msnnt = 1 |> T
                        for k in 1:jL2
                            Msnnt += binomial(jL2,k) / prod((2L+3):2:(2(L+k)+1)) * (2 * uai ^2)^k
                        end
                        Msnnt *= (nai * vthi^j * uaa^L)
                    end
                end
                if is_renorm
                    return Msnnt
                else
                    return Msnnt * CjLL2(j,L)
                end
            end
        else
            uaa = uai / vthi
            if L == 1
                if j == 1
                    Msnnt = nai * vthi * uaa |> T
                else
                    if uaa == 0
                        Msnnt = 0 |> T
                    else
                        jL2 = (j - L) / 2 |> Int
                        Msnnt = 1 |> T
                        for k in 1:jL2
                            Msnnt += binomial(jL2,k) / prod((2L+3):2:(2(L+k)+1)) * (2 * uai ^2)^k
                        end
                        Msnnt *= (nai * vthi^j * uaa^L)
                    end
                end
                if is_renorm
                    return Msnnt
                else
                    return Msnnt * CjLL2(j,L)
                end
            else
                if uaa == 0.0
                    Msnnt = 0 |> T
                else
                    if j == L
                        Msnnt = nai * vthi^j * uaa^L |> T
                    else
                        jL2 = (j - L) / 2 |> Int
                        Msnnt = 1 |> T
                        for k in 1:jL2
                            Msnnt += binomial(jL2,k) / prod((2L+3):2:(2(L+k)+1)) * (2 * uai ^2)^k
                        end
                        Msnnt *= (nai * vthi^j * uaa^L)
                    end
                end
                if is_renorm
                    return Msnnt
                else
                    return Msnnt * CjLL2(j,L)
                end
            end
        end
    end
end

# 0D, j, nai = 1, vthi = 1
function MsnntL2fL0(j::Int64,L::Int64,uai::T;is_renorm::Bool=true) where{T}

    j ≥ L || @error("j ≥ L in the polynomial moments!")
    if iseven(L)
        if L == 0
            if j == 0
                Msnnt = 1 |> T
            else
                if uai == 0
                    Msnnt = 1 |> T
                else
                    jL2 = j / 2 |> Int
                    Msnnt = 1 |> T
                    for k in 1:jL2
                        Msnnt += binomial(jL2,k) / prod(3:2:(2k+1)) * (2 * uai ^2)^k
                    end
                    # Msnnt *= 1 |> T    # (uai^L)
                end
            end
            if is_renorm
                return Msnnt
            else
                return Msnnt * CjLL2even(j)
            end
        else
            if uai == 0.0
                Msnnt = 0 |> T
            else
                if j == L
                    Msnnt = uai^L |> T
                else
                    jL2 = (j - L) / 2 |> Int
                    Msnnt = 1 |> T
                    for k in 1:jL2
                        Msnnt += binomial(jL2,k) / prod((2L+3):2:(2(L+k)+1)) * (2 * uai ^2)^k
                    end
                    Msnnt *= (uai^L)
                end
            end
            if is_renorm
                return Msnnt
            else
                return Msnnt * CjLL2(j,L)
            end
        end
    else
        if L == 1
            if j == 1
                Msnnt = uai |> T
            else
                if uai == 0
                    Msnnt = 0 |> T
                else
                    jL2 = (j - L) / 2 |> Int
                    Msnnt = 1 |> T
                    for k in 1:jL2
                        Msnnt += binomial(jL2,k) / prod((2L+3):2:(2(L+k)+1)) * (2 * uai ^2)^k
                    end
                    Msnnt *= (uai^L)
                end
            end
            if is_renorm
                return Msnnt
            else
                return Msnnt * CjLL2(j,L)
            end
        else
            if uai == 0.0
                Msnnt = 0 |> T
            else
                jL2 = (j - L) / 2
                if j == L
                    Msnnt = uai^L |> T
                else
                    jL2 = (j - L) / 2 |> Int
                    Msnnt = 1 |> T
                    for k in 1:jL2
                        Msnnt += binomial(jL2,k) / prod((2L+3):2:(2(L+k)+1)) * (2 * uai ^2)^k
                    end
                    Msnnt *= (uai^L)
                end
            end
            if is_renorm
                return Msnnt
            else
                return Msnnt * CjLL2(j,L)
            end
        end
    end
end

"""
  When `j = L:2:N⁺`

  Inputs:
    uai: = û

  Outputs:
    Msnnt = MsnntL2fL0(Msnnt,njMs,L,nai,uai,vthi;is_renorm=is_renorm)
    Msnnt = MsnntL2fL0(Msnnt,njMs,L,uai;is_renorm=is_renorm)
"""

# 0.5D, [njMs]
function MsnntL2fL0(Msnnt::AbstractVector{T},njMs::Int64,L::Int64,
    nai::T,uai::T,vthi::T;is_renorm::Bool=true) where{T}

    for k in 1:njMs
        j = L + 2(k-1)
        Msnnt[k] = MsnntL2fL0(j,L,nai,uai,vthi;is_renorm=is_renorm)
    end
    return Msnnt
end

# 0.5D, [njMs], nai = 1, vthi = 1
function MsnntL2fL0(Msnnt::AbstractVector{T},njMs::Int64,L::Int64,uai::T;is_renorm::Bool=true) where{T}

    for k in 1:njMs
        j = L + 2(k-1)
        Msnnt[k] = MsnntL2fL0(j,L,uai;is_renorm=is_renorm)
    end
    return Msnnt
end

"""
  Outputs:
    Msnnt = MsnntL2fL0(Msnnt,njMs,LM,nai,uai,vthi;is_renorm=is_renorm)
    Msnnt = MsnntL2fL0(Msnnt,njMs,LM,uai;is_renorm=is_renorm)
"""

# 1.5D, [njMs,LM1] where `j(L) ∈ L:2:N⁺`
function MsnntL2fL0(Msnnt::AbstractArray{T,N},njMs::Int64,LM::Int64,
    nai::T,uai::T,vthi::T;is_renorm::Bool=true) where{T,N}

    if isone(vthi)
        for L1 in 1:LM+1
            Msnnt[:,L1] = MsnntL2fL0(Msnnt[:,L1],njMs,L1-1,uai;is_renorm=is_renorm)
            Msnnt[:,L1] *= nai
        end
    else
        uai = uai / vthi
        for L1 in 1:LM+1
            Msnnt[:,L1] = MsnntL2fL0(Msnnt[:,L1],njMs,L1-1,uai;is_renorm=is_renorm)
            # j = L1 - 1 + 2(k-1)
            for k in 1:njMs
                j = L1 + 2k - 3
                Msnnt[k,L1] *= (nai * vthi.^j)
            end
        end
    end
    return Msnnt
end

# 1.5D, [njMs,LM1] where `j(L) ∈ L:2:N⁺`, nai = 1, vthi = 1
function MsnntL2fL0(Msnnt::AbstractArray{T,N},njMs::Int64,LM::Int64,uai::T;is_renorm::Bool=true) where{T,N}

    for L1 in 1:LM+1
        Msnnt[:,L1] = MsnntL2fL0(Msnnt[:,L1],njMs,L1-1,uai;is_renorm=is_renorm)
    end
    return Msnnt
end

"""
  Outputs:
    Msnnt = MsnntL2fL0(Msnnt,njMs,L,nai,uai,vthi,ns;is_renorm=is_renorm)
    Msnnt = MsnntL2fL0(Msnnt,njMs,L,uai,ns;is_renorm=is_renorm)
"""

# 1.5D, [njMs,ns]
function MsnntL2fL0(Msnnt::AbstractArray{T,N},njMs::Int64,L::Int64,
    nai::AbstractVector{T},vthi::AbstractVector{T},uai::AbstractVector{T},
    ns::Int64;is_renorm::Bool=true) where{T,N}

    for isp in 1:ns
        Msnnt[:,isp] = MsnntL2fL0(Msnnt[:,isp],njMs,L,nai[isp],uai[isp],vthi[isp];is_renorm=is_renorm)
    end
    return Msnnt
end

# 1.5D, [njMs,ns], nai = 1, vthi = 1
function MsnntL2fL0(Msnnt::AbstractArray{T,N},njMs::Int64,L::Int64,
    uai::AbstractVector{T},ns::Int64;is_renorm::Bool=true) where{T,N}

    for isp in 1:ns
        Msnnt[:,isp] = MsnntL2fL0(Msnnt[:,isp],njMs,L,uai[isp];is_renorm=is_renorm)
    end
    return Msnnt
end

"""
  Computing re-normalzied moments when `j = L:2:N⁺`.

  Inputs:
    uai: = û

  Outputs:
    Msnnt = MsnntL2fL0(Msnnt,njMs,LM,nai,uai,vthi,ns;is_renorm=is_renorm)
    Msnnt = MsnntL2fL0(Msnnt,njMs,LM,uai,ns;is_renorm=is_renorm)
"""

# 2.5D, [njMs,LM1,ns] where `j(L) ∈ L:2:N⁺`
function MsnntL2fL0(Msnnt::AbstractArray{T,N},njMs::Int64,LM::Vector{Int64},
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},
    ns::Int64;is_renorm::Bool=true) where{T,N}

    for isp in 1:ns
        for L1 in 1:LM[isp]+1
            Msnnt[:,L1,isp] = MsnntL2fL0(Msnnt[:,L1,isp],njMs,L1-1,nai[isp],uai[isp],vthi[isp];is_renorm=is_renorm)
        end
    end
    return Msnnt
end

# 2.5D, [njMs,LM1,ns] where `j(L) ∈ L:2:N⁺`, nai = 1, vthi = 1
function MsnntL2fL0(Msnnt::AbstractArray{T,N},njMs::Int64,LM::Vector{Int64},
    ns::Int64,uai::AbstractVector{T};is_renorm::Bool=true) where{T,N}

    for isp in 1:ns
        for L1 in 1:LM[isp]+1
            Msnnt[:,L1,isp] = MsnntL2fL0(Msnnt[:,L1,isp],njMs,L1-1,uai[isp];is_renorm=is_renorm)
        end
    end
    return Msnnt
end

# 2.5D, [njMs,LM1,ns] where `j(L) ∈ L:2:N⁺`
function MsnntL2fL0(Msnnt::Vector{Matrix{T}},njMs::Int64,LM::Vector{Int64},
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},
    ns::Int64;is_renorm::Bool=true) where{T}

    for isp in 1:ns
        for L1 in 1:LM[isp]+1
            Msnnt[isp][:,L1] = MsnntL2fL0(Msnnt[isp][:,L1],njMs,L1-1,nai[isp],uai[isp],vthi[isp];is_renorm=is_renorm)
        end
    end
    return Msnnt
end

# 2.5D, [njMs,LM1,ns] where `j(L) ∈ L:2:N⁺`, nai = 1, vthi = 1
function MsnntL2fL0(Msnnt::Vector{Matrix{T}},njMs::Int64,LM::Vector{Int64},
    uai::AbstractVector{T},ns::Int64;is_renorm::Bool=true) where{T}

    for isp in 1:ns
        for L1 in 1:LM[isp]+1
            Msnnt[isp][:,L1] = MsnntL2fL0(Msnnt[isp][:,L1],njMs,L1-1,uai[isp];is_renorm=is_renorm)
        end
    end
    return Msnnt
end
function MsnntL2fL0(Msnnt::Vector{Matrix{T}},njMs::Vector{Int64},LM::Vector{Int64},
    uai::AbstractVector{T},ns::Int64;is_renorm::Bool=true) where{T}

    for isp in 1:ns
        for L1 in 1:LM[isp]+1
            Msnnt[isp][:,L1] = MsnntL2fL0(Msnnt[isp][:,L1],njMs[isp],L1-1,uai[isp];is_renorm=is_renorm)
        end
    end
    return Msnnt
end
function MsnntL2fL0(Msnnt::Vector{Matrix{T}},njMs::Vector{Int64},LM::Vector{Int64},
    uai::Vector{AbstractVector{T}},ns::Int64;is_renorm::Bool=true) where{T}

    for isp in 1:ns
        for L1 in 1:LM[isp]+1
            Msnnt[isp][:,L1] = MsnntL2fL0(Msnnt[isp][:,L1],njMs[isp],L1-1,uai[isp][1];is_renorm=is_renorm)
        end
    end
    return Msnnt
end

"""
  Outputs:
    Msnnt = MsnntL2fL0(j,L,nai,uai,vthi,nMod;is_renorm=is_renorm)
    Msnnt = MsnntL2fL0(Msnnt,njMs,L,nai,uai,vthi,nMod;is_renorm=is_renorm)
    Msnnt = MsnntL2fL0(Msnnt,njMs,L,nai,uai,vthi,nMod,ns;is_renorm=is_renorm)
    Msnnt = MsnntL2fL0(Msnnt,njMs,LM,nai,uai,vthi,nMod;is_renorm=is_renorm)
    Msnnt = MsnntL2fL0(Msnnt,njMs,LM,nai,uai,vthi,nMod,ns;is_renorm=is_renorm)
    Msnnt = MsnntL2fL0(Msnnt,njMs,LM,nai,uai,vthi,nMod,ns,jtype,dj;is_renorm=is_renorm)
"""

"""
  Outputs:
    Msnnt = MsnntL2fL0(j,L,nai,uai,vthi,nMod;is_renorm=is_renorm)
"""

# 0.5D, [nMod]
function MsnntL2fL0(j::Int64,L::Int64,
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64;
    is_renorm::Bool=true) where{T}

    i = 1
    if nai[i] > 0
        if isone(vthi[i])
            Msnnt = nai[i] * MsnntL2fL0(j,L,uai[i];is_renorm=is_renorm)
        else
            uaii = uai[i] / vthi[i]
            Msnnt = (nai[i] .* vthi[i].^j) * MsnntL2fL0(j,L,uaii;is_renorm=is_renorm)
        end
    else
        Msnnt = 0.0
    end
    for i in 2:nMod
        if nai[i] > 0
            if isone(vthi[i])
                Msnnt += nai[i] * MsnntL2fL0(j,L,uai[i];is_renorm=is_renorm)
            else
                uaii = uai[i] / vthi[i]
                Msnnt += (nai[i] .* vthi[i].^j) * MsnntL2fL0(j,L,uaii;is_renorm=is_renorm)
            end
        end
    end
    return Msnnt
end

"""
  Outputs:
    Msnnt = MsnntL2fL0(Msnnt,njMs,L,nai,uai,vthi,nMod;is_renorm=is_renorm)
"""

# 1.5D, [njMs,nMod]
function MsnntL2fL0(Msnnt::AbstractVector{T},njMs::Int64,L::Int64,
    nai::AbstractVector{T},vthi::AbstractVector{T},uai::AbstractVector{T},nMod::Int64;
    is_renorm::Bool=true) where{T}

    for k in 1:njMs
        j = L + 2(k-1)
        Msnnt[k] = MsnntL2fL0(j,L,nai,uai,vthi,nMod;is_renorm=is_renorm)
    end
    return Msnnt
end

"""
  Outputs:
    Msnnt = MsnntL2fL0(Msnnt,njMs,L,nai,uai,vthi,nMod,ns;is_renorm=is_renorm)
    Msnnt = MsnntL2fL0(Msnnt,njMs,LM,nai,uai,vthi,nMod;is_renorm=is_renorm)
"""

# 2.0D, [njMs,ns,nMod]
function MsnntL2fL0(Msnnt::AbstractArray{T,N},njMs::Int64,L::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64},
    ns::Int64;is_renorm::Bool=true) where{T,N}

    for isp in 1:ns
        Msnnt[:,isp] = MsnntL2fL0(Msnnt[:,isp],njMs,L,nai[isp],uai[isp],vthi[isp],nMod[isp];is_renorm=is_renorm)
    end
    return Msnnt
end

# 2.0D, [njMs,LM,nMod]
function MsnntL2fL0(Msnnt::AbstractArray{T,N},njMs::Int64,LM::Int64,
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64;
    is_renorm::Bool=true) where{T,N}

    for L in 0:LM
        L1 = L + 1
        # j = L + 2(k-1)
        for k in 1:njMs
            j = L1 + 2k - 3
            Msnnt[k,L1] = MsnntL2fL0(j,L,nai,uai,vthi,nMod;is_renorm=is_renorm)
        end
    end
    return Msnnt
end

"""
  Computing re-normalzied moments when `j = L:2:N⁺`.

  Inputs:
    uai: = û

  Outputs:
    Msnnt = MsnntL2fL0(Msnnt,njMs,LM,nai,uai,vthi,nMod,ns;is_renorm=is_renorm)
"""
# 2.5D, [njMs,LM1,ns,nMod] where `j(L) ∈ L:2:N⁺`
function MsnntL2fL0(Msnnt::AbstractArray{T,N},njMs::Int64,LM::Vector{Int64},
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64},
    ns::Int64;is_renorm::Bool=true) where{T,N}

    for isp in 1:ns
        Msnnt[:,:,isp] = MsnntL2fL0(Msnnt[:,:,isp],njMs,LM[isp],
            nai[isp],uai[isp],vthi[isp],nMod[isp];is_renorm=is_renorm)
    end
    return Msnnt
end

function MsnntL2fL0(Msnnt::AbstractArray{T,N},njMs::Vector{Int64},LM::Vector{Int64},
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64},
    ns::Int64;is_renorm::Bool=true) where{T,N}

    for isp in 1:ns
        Msnnt[:,:,isp] = MsnntL2fL0(Msnnt[:,:,isp],njMs[isp],LM[isp],
            nai[isp],uai[isp],vthi[isp],nMod[isp];is_renorm=is_renorm)
    end
    return Msnnt
end


# 2.5D, [njMs,LM1,ns,nMod] where `j(L) ∈ L:2:N⁺`
function MsnntL2fL0(Msnnt::Vector{Matrix{T}},njMs::Int64,LM::Vector{Int64},
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64},
    ns::Int64;is_renorm::Bool=true) where{T}

    for isp in 1:ns
        Msnnt[isp] = MsnntL2fL0(Msnnt[isp],njMs,LM[isp],
            nai[isp],uai[isp],vthi[isp],nMod[isp];is_renorm=is_renorm)
    end
    return Msnnt
end

function MsnntL2fL0(Msnnt::Vector{Matrix{T}},njMs::Vector{Int64},LM::Vector{Int64},
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64},
    ns::Int64;is_renorm::Bool=true) where{T}

    for isp in 1:ns
        Msnnt[isp] = MsnntL2fL0(Msnnt[isp],njMs[isp],LM[isp],
            nai[isp],uai[isp],vthi[isp],nMod[isp];is_renorm=is_renorm)
    end
    return Msnnt
end

"""
  Computing re-normalzied moments for general `j`.

  Inputs:
    uai: = û

  Outputs:
    Msnnt = MsnntL2fL0(Msnnt,njMs,LM,nai,uai,vthi,nMod,ns,jtype,dj;is_renorm=is_renorm)
"""
# 2.5D, [njMs,LM1,ns,nMod], jtype, dj
function MsnntL2fL0(Ms::AbstractArray{T,N2},njMs::Int64,LM::Vector{Int64},
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64},
    ns::Int64,jtype::Symbol,dj::Int64=1;is_renorm::Bool=true) where{T<:Real,N2}

    if jtype == :L
        for isp in 1:ns
            Ms[:,:,isp] = MsnntL2fL0(Ms[:,:,isp],njMs,LM[isp],
                 nai[isp],uai[isp],vthi[isp],nMod[isp];is_renorm=is_renorm)
        end
        return Ms
    elseif jtype == :n2
        for isp in 1:ns
            for L in 0:LM[isp]
                L1 = L + 1
                for k in 1:njMs
                    j = dj * (k - 1) - 2
                    Ms[k,L1,isp] = MsnntL2fL0(Ms[k,L1,isp],j,L,nMod,
                        nai[isp],uai[isp],vthi[isp];is_renorm=is_renorm)
                end
            end
        end
        return Ms
    else # jtype == :nL2
        for isp in 1:ns
            for L in 0:LM[isp]
                L1 = L + 1
                for k in 1:njMs
                    j = dj * (k - 1) - (L + 2)
                    Ms[k,L1,isp] = MsnntL2fL0(Ms[k,L1,isp],j,L,nMod,
                        nai[isp],uai[isp],vthi[isp];is_renorm=is_renorm)
                end
            end
        end
        return Ms
    end
end
