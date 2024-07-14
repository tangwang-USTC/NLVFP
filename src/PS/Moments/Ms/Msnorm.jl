
"""
  Clenshaw-Curtis quadrature, normalized moments

  # Notes: `j` may be not the correct form of your need here, make the version of your own!

  Generally for single sublevel when 'nck = ∑ₖ nk', the `jᵗʰ`-order normalized moments of
  the `ℓᵗʰ`-order normalized coefficient of distribution function, f̂ₗ(v̂), will be:

    𝓜ⱼ(fₗ) = 4π * ∫₀⁹(v̂²⁺ʲf̂ₗ(v̂))dv̂

  Especially, the first few orders of normalized moments such as n̂, Î and K̂ are:

    na: = 4π * ∫₀⁹(v̂²f₀(v))dv̂        , L=0, j = 0
    Iₐ: = 4π / 3 ∫₀⁹(v̂³f₁(v))dv̂      , L=1, j = 1
    Ka: = 4π / ∫₀⁹(v̂⁴f₀(v))dv̂        , L=0, j = 2

  Here, `ma` is not included in the procedure.

  Applying the Clenshaw-Curtis quadrature, the integral coefficient will be:

    Ixv = - 2 / sqrtpi * (vGk[1] - vGk[end])

        = (- (a - b)  / 2) * cf0

    where

        cf0 = 4π / π^(3/2),

    The factor `4π` is come from the shperical surface integration of velocity space and
    `π^(-3/2)` is the coefficient of the distribution function.

    Ta: = 2/3 (Ka/na - 0.5 mₐ uₐ²)
"""

"""
  Inputs:
    nvlevel:
    nc0:
    nck:
    j:

    Ms: = [j,Msa, Msb], where 'Ms[j,1] = [Mⱼ(fₗ(v̂))]'

  Outputs:
    Ms = Msnorm(Msk,fLn,vGk,nvlevel,nc0,nck,j)

"""

# 0D, the `jᵗʰ`-order moment of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domian.
function Msnorm(Msk::T,fLn::AbstractVector{T},vGk::AbstractVector{Tb},
    nvlevel::AbstractVector{Int},nc0::Int,nck::Int,j::Int) where{T<:Real,Tb}

    if nck == nc0
        Ixv = - 2 / sqrtpi * (vGk[1] - vGk[end])
        # μk = chebyshevmoments1(T, nc0)
        wcck = clenshawcurtisweights(chebyshevmoments1(T, nc0))
        if j == -2
            Msk = Ixv * dot(wcck, fLn)
        elseif j == -1
            Msk = Ixv * dot(wcck, (vGk .* fLn))
        elseif j == 0
            Msk = Ixv * dot(wcck, (vGk.^2 .* fLn))
        elseif j == 1
            Msk = Ixv * dot(wcck, (vGk.^3 .* fLn))
        else
            Msk = Ixv * dot(wcck, (vGk.^2 .* fLn .* vGk.^j))
        end
        return Msk
    else
        k = 1
        nk = nvlevel[k]
        # μk = chebyshevmoments1(T, nk)
        wcck = clenshawcurtisweights(chebyshevmoments1(T, nk))
        nk1 = 1
        nk9 = 0 + nk
        vk = vGk[nk1:nk9]
        Ixvi = - 2. / sqrtpi * (vk[1] - vk[end])
        if j == -2
            Msk = Ixvi * dot(wcck, fLn[nk1:nk9])
        elseif j == -1
            Msk = Ixvi * dot(wcck, (vk .* fLn[nk1:nk9]))
        elseif j == 0
            Msk = Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9]))
        elseif j == 1
            Msk = Ixvi * dot(wcck, (vk.^3 .* fLn[nk1:nk9]))
        else
            Msk = Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9] .* vk.^j))
        end
        for k in 2:nc0-1
            nk = nvlevel[k]
            # @show j,k,nk,Msk
            wcck = clenshawcurtisweights(chebyshevmoments1(T, nk))
            nk1 = nk9
            nk9 = nk1 + nk - 1
            vk = vGk[nk1:nk9]
            Ixvi = - 2/sqrtpi * (vk[1] - vk[end])
            if j == -2
                Msk += Ixvi * dot(wcck, fLn[nk1:nk9])
            elseif j == -1
                Msk += Ixvi * dot(wcck, (vk.* fLn[nk1:nk9]))
            elseif j == 0
                Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9]))
            elseif j == 1
                Msk += Ixvi * dot(wcck, (vk.^3 .* fLn[nk1:nk9]))
            else
                Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9] .* vk.^j))
            end
        end
        return Msk
    end
end

"""
  The `jᵗʰ`-order moments (Ms[j,1]) of the `ℓᵗʰ`-order coefficient of distribution function
  on the entire velocity axis domian.

  where

    jvec = Ms[:,end].
    Ms[j,1] = Mⱼ(fₗ(v̂))

  Inputs:
    nvlevel:
    nc0:
    nck:
    j:

    Ms: = [j,Msa, Msb], where

  Outputs:
    Ms = Msnorm(Ms,fLn,vGk,nvlevel,nc0,nck)
"""

# 0.5D, [njMs]
function Msnorm(Ms::AbstractArray{T,N2},fLn::AbstractVector{T},vGk::AbstractVector{Tb},
    nvlevel::AbstractVector{Int},nc0::Int,nck::Int) where{T<:Real,Tb,N2}

    if norm(fLn) > epsT5
        nj = 0
        for j in Ms[:,2]
            nj += 1
            dsf
            Ms[nj,1] = Msnorm(Ms[nj,1],fLn,vGk,nvlevel,nc0,nck,Int(j))
        end
    else
        Ms[:,1] .=  0.0
    end
    return Ms
end

"""
  Inputs:
    fvL: fvL[:,:,isp]
    Ms: [MsL1, MsL2,..., jvec]

  Outputs:
    Ms = Msnorm(Ms,fvL,vGk,nvlevel,nc0,nck,LM)
"""

# 1.5D, [njMs,LM1], the first `jᵗʰ`-order moments of the first `ℓᵗʰ`-order coefficient of
# distribution function on the entire velocity axis domian.
function Msnorm(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},vGk::AbstractVector{Tb},
    nvlevel::AbstractVector{Int},nc0::Int,nck::Int,LM::Int) where{T<:Real,Tb,N2}

    for L1 in 1:LM
        if norm(fvL[:,L1]) > epsT5
            tgfhj
            nj = 0
            for j in Ms[:,end]
                nj += 1
                Ms[nj,L1] = Msnorm(Ms[nj,L1],fvL[:,L1],vGk,nvlevel,nc0,nck,Int(j))
            end
        else
            Ms[:,L1] .=  0.0
        end
    end
    return Ms
end
"""
  Inputs:
    fvL: fvL[:,L1,:]
    Ms: [Ms1, Ms2, jvec]

  Outputs:
    Ms = Msnorm(Ms,fvL,vGk,nvlevel,nc0,nck,ns)
"""

# 1.5D,[njMs,ns], the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domian.
function Msnorm(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vGk::AbstractVector{Tb},nvlevel::AbstractVector{Int},
    nc0::Int,nck::Int,ns::Int) where{T<:Real,Tb,N2}
rhj
    for isp in 1:ns
        if norm(fvL[:,isp]) > epsT5
            nj = 0
            for j in Ms[:,end]
                nj += 1
                Ms[nj,isp] = Msnorm(Ms[nj,isp],fvL[:,isp],vGk,nvlevel,nc0,nck,Int(j))
            end
        else
            Ms[:,isp] .=  0.0
        end
    end
    return Ms
end

# 1.5D, [njMs,nMod]





"""
  Computing re-normalzied moments when `j = L:2:N⁺`.

  Inputs:
    ua: = û / v̂thᵢ

  Outputs:
    Ms = Msnorm(Ms,fvL,vGk,nvlevel,nc0,nck,LM,ns)
"""
# 2.5D, [njMs,LM1,ns], the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domian.
function Msnorm(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vGk::AbstractVector{Tb},nvlevel::AbstractVector{Int},
    nc0::Int,nck::Int,LM::Vector{Int},ns::Int) where{T<:Real,Tb,N2}

    for isp in 1:ns
        for L1 in 1:LM[isp]
            if norm(fvL[:,L1,isp]) > epsT5
                L = L1 - 1
                for k in 1:njMs
                    j = L + 2(k-1)
                    Ms[k,L1,isp] = Msnorm(Ms[k,L1,isp],fvL[:,L1,isp],vGk,nvlevel,nc0,nck,Int(j))
                end
            else
                Ms[:,L1,isp] .=  0.0
            end
        end
    end
    return Ms
end
