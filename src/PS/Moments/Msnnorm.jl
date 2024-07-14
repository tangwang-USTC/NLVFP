
"""
  Clenshaw-Curtis quadrature for general moments

  Generally for single sublevel when 'nck = ∑ₖ nk', the `jᵗʰ`-order moments of
  the `ℓᵗʰ`-order coefficient of distribution function will be:

    𝓜̂ⱼ(fₗ) = 𝓜ⱼ(f̂ₗ) / CjL(j,L)
           = 4π / CjL(j,L) * ∫₀⁹(v̂²⁺ʲf̂ₗ(v̂))dv̂

  where

     CjL(j,L) = (j+L+1)!! / (2L-1)!! / 2^((j-L)/2)

  Especially, the first few orders of normalized moments such as n̂, Î and K̂ are:

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
    L:

    Ms: = [j,Msa, Msb], where 'Ms[j,1] = [Mⱼ(fₗ(v̂))]'

  Outputs:
    Ms = Msnnorm(Msk,fLn,vGk,nvlevel,nc0,nck,njMs,j,L;is_renorm=is_renorm)

"""

# 0D, the `jᵗʰ`-order moment of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function Msnnorm(Msk::T,fLn::AbstractVector{T},vGk::AbstractVector{Tb},
    nvlevel::AbstractVector{Int},nc0::Int64,nck::Int64,j::Int64,L::Int64;is_renorm::Bool=true) where{T<:Real,Tb}

    # Check whether `vGk[end]^j * fLn[end] ≪ eps(T)`
    if j ≥ 0 && (fLn[end] * vGk[end]^(j+2) ≥ epsT / 10)
        @warn("Error: `vGk[end]^j * fLn[end] > epsT/10` which may cause errors of the higher-order moments when `j=",j)
        @show fLn[end] * vGk[end]^(j+2)
    end
    if nck == nc0
        if is_renorm
            Ixv = - 2 / sqrtpi * (vGk[1] - vGk[end]) / CjLL2(j,L)
        else
            Ixv = - 2 / sqrtpi * (vGk[1] - vGk[end])
        end
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
        if is_renorm
            return Msk / CjLL2(j,L)
        else
            return Msk
        end
    end
end


"""
  The `jᵗʰ`-order moments (Ms[j,1]) of the `ℓᵗʰ`-order coefficient of distribution function
  on the entire velocity axis domain.

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
    Ms = Msnnorm(Ms,fLn,vGk,nvlevel,nc0,nck,njMs,L;is_renorm=is_renorm)
"""

# 0.5D, [njMs]
function Msnnorm(Ms::AbstractVector{T},fLn::AbstractVector{T},vGk::AbstractVector{Tb},
    nvlevel::AbstractVector{Int},nc0::Int64,nck::Int64,njMs::Int64,L::Int64;is_renorm::Bool=true) where{T<:Real,Tb}

    for k in 1:njMs
        j = L + 2(k-1)
        Ms[k] = Msnnorm(Ms[k],fLn,vGk,nvlevel,nc0,nck,Int(j),L;is_renorm=is_renorm)
    end
    return Ms
end

"""
  Inputs:
    fvL: fvL[:,:,isp]
    Ms: [MsL1, MsL2,..., jvec]

  Outputs:
    Ms = Msnnorm(Ms,fvL,vGk,nvlevel,nc0,nck,njMs,LM,LM1;is_renorm=is_renorm)
"""

# 1.5D, [njMs,LM], the first `jᵗʰ`-order moments of the first `ℓᵗʰ`-order coefficient of
# distribution function on the entire velocity axis domain.
function Msnnorm(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},vGk::AbstractVector{Tb},nvlevel::AbstractVector{Int},
    nc0::Int64,nck::Int64,njMs::Int64,LM::Int64,LM1::Int64;is_renorm::Bool=true) where{T<:Real,Tb,N2}

    for L in 0:LM
        L1 = L + 1
        for k in 1:njMs
            j = L + 2(k-1)
            Ms[k,L1] = Msnnorm(Ms[k,L1],fvL[:,L1],vGk,nvlevel,nc0,nck,Int(j),L;is_renorm=is_renorm)
        end
    end
    if LM + 1 ≠ LM1
        Ms[:,LM+2:LM1] .=  0.0
    end
    return Ms
end

"""
  Inputs:
    fvL: fvL[:,L1,:]
    Ms: [Ms1, Ms2, jvec]

  Outputs:
    Ms = Msnnorm(Ms,fvL,vGk,nvlevel,nc0,nck,njMs,L,ns;is_renorm=is_renorm)
"""

# 1.5D,[njMs,ns],
function Msnnorm(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vGk::AbstractVector{Tb},nvlevel::AbstractVector{Int},
    nc0::Int64,nck::Int64,njMs::Int64,L::Int64,ns::Int64;is_renorm::Bool=true) where{T<:Real,Tb,N2}

    for isp in 1:ns
        for k in 1:njMs
            j = L + 2(k-1)
            Ms[k,isp] = Msnnorm(Ms[k,isp],fvL[:,isp],vGk,nvlevel,nc0,nck,Int(j),L;is_renorm=is_renorm)
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
    Ms = Msnnorm(Ms,fvL,vGk,nvlevel,nc0,nck,njMs,LM,LM1,ns;is_renorm=is_renorm)
"""
# 2.5D, [njMs,LM1,ns],
function Msnnorm(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vGk::AbstractVector{Tb},nvlevel::AbstractVector{Int},nc0::Int64,nck::Int64,
    njMs::Int64,LM::Vector{Int},LM1::Int64,ns::Int64;is_renorm::Bool=true) where{T<:Real,Tb,N2}

    for isp in 1:ns
        for L in 0:LM[isp]
            L1 = L + 1
            for k in 1:njMs
                j = L + 2(k-1)
                Ms[k,L1,isp] = Msnnorm(Ms[k,L1,isp],fvL[:,L1,isp],vGk,nvlevel,
                                      nc0,nck,Int(j),L;is_renorm=is_renorm)
            end
        end
        if LM[isp] + 1 ≠ LM1
            Ms[:,LM[isp]+2:LM1,isp] .=  0.0
        end
    end
    return Ms
end

"""
  Computing re-normalzied moments for general `j`.

  Inputs:

  Outputs:
    Msnnt = Msnnorm(Msnnt,fvL,vGe,njMs,LM,ns,jtype,dj;is_renorm=is_renorm)
"""
# 3.0D, [njMs,LM1,ns,nMod], jtype, dj
function Msnnorm(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},vGk::AbstractVector{Tb},
    nvlevel::AbstractVector{Int},nc0::Int64,nck::Int64,njMs::Int64,LM::Vector{Int},LM1::Int64,
    ns::Int64,jtype::Symbol,dj::Int64=1;is_renorm::Bool=true) where{T<:Real,Tb,N2}
    
    rtgjh
    if jtype == :L
        for isp in 1:ns
            Ms[:,:,isp] = Msnnorm(Ms[:,:,isp],fvL[:,:,isp],vGk,nvlevel,nc0,nck,njMs,LM[isp];is_renorm=is_renorm)
        end
        return Ms
    elseif jtype == :n2
        for isp in 1:ns
            for L1 in 1:LM[isp]
                L = L1 - 1
                for k in 1:njMs
                    j = dj * (k - 1) - 2
                    Ms[k,L1,isp] = Msnnorm(Ms[k,L1,isp],fvL[:,L1,isp],vGk,nvlevel,nc0,nck,Int(j),L;is_renorm=is_renorm)
                end
            end
        end
        return Ms
    else # jtype == :nL2
        for isp in 1:ns
            for L1 in 1:LM[isp]
                L = L1 - 1
                for k in 1:njMs
                    j = dj * (k - 1) - (L + 2)
                    Ms[k,L1,isp] = Msnnorm(Ms[k,L1,isp],fvL[:,L1,isp],vGk,nvlevel,nc0,nck,Int(j),L;is_renorm=is_renorm)
                end
            end
        end
        return Ms
    end
end
