
"""
  The `jᵗʰ`-order global-normalized moments of the `ℓᵗʰ`-order coefficient of
  normalized distribution function on the entire velocity axis domian.
  The cumulative integral of `v̂²⁺ʲf̂ₗ(v̂)` can be expressed as:

    Msk[k] = int(v̂²⁺ʲ * f̂ₗ(v̂), {v̂, v̂ₖ, v̂ₖ₊₁}), k ≥ 1.

  where
    jvec = Ms[end,:] = [j1, j2, ⋯]

  Inputs:
    Ms: = [Msj;j  ⋯] where Msj = Ms[1:end-1,j], j ∈ jvec

  Outputs:
    jvec = [-2,0]
    nj = length(jvec)
    Msk = zeros(nc0,nj)
    Msk[end,:] = jvec
    Msk = Msknorm(Msk,fLnt,vGk,nvlevel,nc0,nck,ocp;isnorm=isnorm)
"""

# 1.5D,
function Msknorm(Ms::AbstractArray{T,N},fLn::AbstractVector{T},vGk::AbstractVector{Tb},
    nvlevel::AbstractVector{Int},nc0::Int,nck::Int,ocp::Int;isnorm::Bool=false) where{T<:Real,Tb,N}

    if norm(fLn) > epsT5
        nj = 0
        for j in Ms[end,:]
            nj += 1
            Ms[:,nj] = Msknorm(Ms[:,nj],fLn,vGk,nvlevel,nc0,nck,ocp,Int(j);isnorm=isnorm)
        end
    else
        Ms[1:end-1,:] .=  0.0
    end
    return Ms
end

"""

  Inputs:
    Msk: = zeros(nc0)
    ocp: OrderShapes

  Outputs:
    Msk = zeros(nc0)
    Msk[end] = j
    Msk = Msknorm(Msk,fLnt,vGk,nvlevel,nc0,nck,ocp,j;isnorm=isnorm)

"""

# 1D, the `jᵗʰ`-order moment of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function Msknorm(Msk::AbstractVector{T},fLn::AbstractVector{T},vGk::AbstractVector{Tb},
    nvlevel::AbstractVector{Int},nc0::Int,nck::Int,ocp::Int,j::Int) where{T<:Real,Tb}

    # μk = chebyshevmoments1(T, ocp)
    wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
    k = 1
    nk = nvlevel[k]
    nk1 = 1
    nk9 = ocp
    vk = vGk[nk1:nk9]
    Ixvi = - (vk[1] - vk[end]) / 2
    if j == -2
        Msk[k] = Ixvi * dot(wcck, fLn[nk1:nk9])
    elseif j == -1
        Msk[k] = Ixvi * dot(wcck, (vk .* fLn[nk1:nk9]))
    elseif j == 0
        Msk[k] = Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9]))
    elseif j == 1
        Msk[k] = Ixvi * dot(wcck, (vk.^3 .* fLn[nk1:nk9]))
    else
        Msk[k] = Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9] .* vk.^j))
    end
    for k in 2:nc0-1
        nk = nvlevel[k]
        nk1 = nk9
        nk9 = nk1 + ocp - 1
        vk = vGk[nk1:nk9]
        Ixvi = - (vk[1] - vk[end]) / 2
        if j == -2
            Msk[k] = Ixvi * dot(wcck, fLn[nk1:nk9])
        elseif j == -1
            Msk[k] = Ixvi * dot(wcck, (vk.* fLn[nk1:nk9]))
        elseif j == 0
            Msk[k] = Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9]))
        elseif j == 1
            Msk[k] = Ixvi * dot(wcck, (vk.^3 .* fLn[nk1:nk9]))
        else
            Msk[k] = Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9] .* vk.^j))
        end
    end
    return Msk
end
