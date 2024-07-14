
"""

  Generally, the `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function will be:

    𝓜ⱼ(fₗ) = mₐ vₜₕʲ * vₜₕ³ * ∫₀⁹(4πv̂⁴f₀(v))dv̂

  Especially, the first few orders of moments such as n, u, K, T and vthi are:

    na: = vₜₕ³ * ∫₀⁹(4πv̂²f₀(v))dv̂        , L=0, j = 0
    Iₐ: = ma / 3 vₜₕ⁴ ∫₀⁹(4πv̂³f₁(v))dv̂   , L=1, j = 1
    Ka: = 1/2 mₐ vₜₕ⁵ ∫₀⁹(4πv̂⁴f₀(v))dv̂   , L=0, j = 2

  Here, `ma` is not included in the procedure.

  Applying the Clenshaw-Curtis quadrature, the integral coefficient will be:

    Ixv = - 2 / sqrtpi * (vGk[1] - vGk[end])

        = (- (a - b)  / 2) * cf0

    where

        cf0 = 4π / π^(3/2),

    The factor `4π` is come from the shperical surface integration of velocity space and
    `π^(-3/2)` is the coefficient of the distribution function.

    Ta: = 2/3 (Ka/na - 0.5 mₐ uₐ²)

  Inputs:
    ocp: OrderShapes
    nai:
    vthi:
    Ms: = [j,Msa, Msb], where 'Ms[j,1] = [Mⱼ(fₗ(v̂))]'

  Outputs:
    Ms = Mslevels(Msk,fLn,vGk,nvlevel,nc0,nck,ocp,j,nai,vthi)
    Ms = Mslevels(Msk,fLn,vGk,nvlevel,nc0,nck,ocp,j,nai)
    Ms = Mslevels(Msk,fLn,vGk,nvlevel,nc0,nck,ocp,j)

"""

# 1D, the `jᵗʰ`-order moment of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function Mslevels(Msk::T,fLn::AbstractVector{T},vGk::AbstractVector{Tb},nvlevel::AbstractVector{Int},
    nc0::Int,nck::Int,ocp::Int,j::Int,nai::T,vthi::T) where{T<:Real,Tb}

    if vthi == 1.0
        if nai == 1.0
            return Mslevels(Msk,fLn,vGk,nvlevel,nc0,nck,ocp,j)
        else
            return Mslevels(Msk,fLn,vGk,nvlevel,nc0,nck,ocp,j,nai)
        end
    else
        if nck == nc0
            Ixv = - 2 / sqrtpi * (vGk[1] - vGk[end])
            if j == -2
                Ixv *= (nai / vthi^2)
            elseif j == -1
                Ixv *= (nai / vthi)
            elseif j == 0
                Ixv *= nai
            elseif j == 1
                Ixv *= (nai * vthi)
            else
                Ixv *= (nai * vthi^j)
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
            # μk = chebyshevmoments1(T, ocp)
            wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
            k = 1
            nk = nvlevel[k]
            nk1 = 1
            nk9 = ocp
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
            if nk > ocp
                nks = ocp
                while nk - nks > 0
                    nks += (ocp - 1)
                    nk1 = nk9
                    nk9 = nk1 + ocp - 1
                    vk = vGk[nk1:nk9]
                    Ixvi = - 2/sqrtpi * (vk[1] - vk[end])
                    if j == -2
                        Msk += Ixvi * dot(wcck, fLn[nk1:nk9])
                    elseif j == -1
                        Msk += Ixvi * dot(wcck, (vk .* fLn[nk1:nk9]))
                    elseif j == 0
                        Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9]))
                    elseif j == 1
                        Msk += Ixvi * dot(wcck, (vk.^3 .* fLn[nk1:nk9]))
                    else
                        Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9] .* vk.^j))
                    end
                end
            end
            for k in 2:nc0-1
                nk = nvlevel[k]
                nk1 = nk9
                nk9 = nk1 + ocp - 1
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
                if nk > ocp
                    nks = ocp
                    while nk - nks > 0
                        nks += (ocp - 1)
                        nk1 = nk9
                        nk9 = nk1 + ocp - 1
                        vk = vGk[nk1:nk9]
                        Ixvi = - 2/sqrtpi * (vk[1] - vk[end])
                        if j == -2
                            Msk += Ixvi * dot(wcck, fLn[nk1:nk9])
                        elseif j == -1
                            Msk += Ixvi * dot(wcck, (vk .* fLn[nk1:nk9]))
                        elseif j == 0
                            Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9]))
                        elseif j == 1
                            Msk += Ixvi * dot(wcck, (vk.^3 .* fLn[nk1:nk9]))
                        else
                            Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9] .* vk.^j))
                        end
                    end
                end
            end
            if j == -2
                Msk *= (nai / vthi^2)
            elseif j == -1
                Msk *= (nai / vthi)
            elseif j == 0
                Msk *= nai
            elseif j == 1
                Msk *= (nai * vthi)
            else
                Msk *= (nai * vthi^j)
            end
            return Msk
        end
    end
end

# `vthi = 1`
function Mslevels(Msk::T,fLn::AbstractVector{T},vGk::AbstractVector{Tb},nvlevel::AbstractVector{Int},
    nc0::Int,nck::Int,ocp::Int,j::Int,nai::T) where{T<:Real,Tb}

    if nck == nc0
        Ixv = - 2 / sqrtpi * (vGk[1] - vGk[end]) * nai
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
        # μk = chebyshevmoments1(T, ocp)
        wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
        k = 1
        nk = nvlevel[k]
        nk1 = 1
        nk9 = ocp
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
        if nk > ocp
            nks = ocp
            while nk - nks > 0
                nks += (ocp - 1)
                nk1 = nk9
                nk9 = nk1 + ocp - 1
                vk = vGk[nk1:nk9]
                Ixvi = - 2/sqrtpi * (vk[1] - vk[end])
                if j == -2
                    Msk += Ixvi * dot(wcck, fLn[nk1:nk9])
                elseif j == -1
                    Msk += Ixvi * dot(wcck, (vk .* fLn[nk1:nk9]))
                elseif j == 0
                    Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9]))
                elseif j == 1
                    Msk += Ixvi * dot(wcck, (vk.^3 .* fLn[nk1:nk9]))
                else
                    Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9] .* vk.^j))
                end
            end
        end
        for k in 2:nc0-1
            nk = nvlevel[k]
            nk1 = nk9
            nk9 = nk1 + ocp - 1
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
            if nk > ocp
                nks = ocp
                while nk - nks > 0
                    nks += (ocp - 1)
                    nk1 = nk9
                    nk9 = nk1 + ocp - 1
                    vk = vGk[nk1:nk9]
                    Ixvi = - 2/sqrtpi * (vk[1] - vk[end])
                    if j == -2
                        Msk += Ixvi * dot(wcck, fLn[nk1:nk9])
                    elseif j == -1
                        Msk += Ixvi * dot(wcck, (vk .* fLn[nk1:nk9]))
                    elseif j == 0
                        Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9]))
                    elseif j == 1
                        Msk += Ixvi * dot(wcck, (vk.^3 .* fLn[nk1:nk9]))
                    else
                        Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9] .* vk.^j))
                    end
                end
            end
        end
        Msk *= nai
        return Msk
    end
end

# `vthi = 1` and `nai = 1`: the normalized moments
function Mslevels(Msk::T,fLn::AbstractVector{T},vGk::AbstractVector{Tb},
    nvlevel::AbstractVector{Int},nc0::Int,nck::Int,ocp::Int,j::Int) where{T<:Real,Tb}

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
        # μk = chebyshevmoments1(T, ocp)
        wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
        k = 1
        nk = nvlevel[k]
        nk1 = 1
        nk9 = ocp
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
        if nk > ocp
            nks = ocp
            while nk - nks > 0
                nks += (ocp - 1)
                nk1 = nk9
                nk9 = nk1 + ocp - 1
                vk = vGk[nk1:nk9]
                Ixvi = - 2/sqrtpi * (vk[1] - vk[end])
                if j == -2
                    Msk += Ixvi * dot(wcck, fLn[nk1:nk9])
                elseif j == -1
                    Msk += Ixvi * dot(wcck, (vk .* fLn[nk1:nk9]))
                elseif j == 0
                    Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9]))
                elseif j == 1
                    Msk += Ixvi * dot(wcck, (vk.^3 .* fLn[nk1:nk9]))
                else
                    Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9] .* vk.^j))
                end
            end
        end
        for k in 2:nc0-1
            nk = nvlevel[k]
            nk1 = nk9
            nk9 = nk1 + ocp - 1
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
            if nk > ocp
                nks = ocp
                while nk - nks > 0
                    nks += (ocp - 1)
                    nk1 = nk9
                    nk9 = nk1 + ocp - 1
                    vk = vGk[nk1:nk9]
                    Ixvi = - 2/sqrtpi * (vk[1] - vk[end])
                    if j == -2
                        Msk += Ixvi * dot(wcck, fLn[nk1:nk9])
                    elseif j == -1
                        Msk += Ixvi * dot(wcck, (vk .* fLn[nk1:nk9]))
                    elseif j == 0
                        Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9]))
                    elseif j == 1
                        Msk += Ixvi * dot(wcck, (vk.^3 .* fLn[nk1:nk9]))
                    else
                        Msk += Ixvi * dot(wcck, (vk.^2 .* fLn[nk1:nk9] .* vk.^j))
                    end
                end
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

  Inputs:
    Ms:

  Outputs:
    Ms = Mslevels(Ms,fLn,vGk,nvlevel,nc0,nck,ocp,nai,vthi)
    Ms = Mslevels(Ms,fvL,vGk,nvlevel,nc0,nck,ocp,nai,vthi)
    Ms = Mslevels(Ms,fvL,vGk,nvlevel,nc0,nck,ocp,nai,vthi,ns)
"""

# 1.5D,
function Mslevels(Ms::AbstractArray{T,N2},fLn::AbstractVector{T},
    vGk::AbstractVector{Tb},nvlevel::AbstractVector{Int},
    nc0::Int,nck::Int,ocp::Int,nai::T,vthi::T) where{T<:Real,Tb,N2}

    if norm(fLn) > epsT5
        nj = 0
        for j in Ms[:,2]
            nj += 1
            Ms[nj,1] = Mslevels(Ms[nj,1],fLn,vGk,nvlevel,nc0,nck,ocp,Int(j),nai,vthi)
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
"""

# 2.5DL, the first `jᵗʰ`-order moments of the first `ℓᵗʰ`-order coefficient of
# distribution function on the entire velocity axis domain.
function Mslevels(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vGk::AbstractVector{Tb},nvlevel::AbstractVector{Int},
    nc0::Int,nck::Int,ocp::Int,nai::T,vthi::T) where{T<:Real,Tb,N2}

    for L1 in 1:1:size(fvL,2)
        if norm(fvL[:,L1]) > epsT5
            nj = 0
            for j in Ms[:,end]
                nj += 1
                Ms[nj,L1] = Mslevels(Ms[nj,L1],fvL[:,L1],vGk,
                      nvlevel,nc0,nck,ocp,Int(j),nai,vthi)
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
"""

# 2.5Disp and 3.5D, the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function Mslevels(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vGk::AbstractVector{Tb},nvlevel::AbstractVector{Int},nc0::Int,nck::Int,ocp::Int,
    nai::AbstractVector{T},vthi::AbstractVector{T},ns::Int) where{T<:Real,Tb,N2}

    if N2 == 2
        for isp in 1:ns
            if norm(fvL[:,isp]) > epsT5
                nj = 0
                for j in Ms[:,end]
                    nj += 1
                    Ms[nj,isp] = Mslevels(Ms[nj,isp],fvL[:,isp],vGk,
                          nvlevel,nc0,nck,ocp,Int(j),nai[isp],vthi[isp])
                end
            else
                Ms[:,isp] .=  0.0
            end
        end
        return Ms
    elseif N2 == 3
        for isp in 1:ns
            jvec = Ms[:,end,isp]
            for L1 in 1:1:size(fvL,2)
                if norm(fvL[:,L1,isp]) > epsT5
                    nj = 0
                    for j in jvec
                        nj += 1
                        Ms[nj,L1,isp] = Mslevels(Ms[nj,L1,isp],fvL[:,L1,isp],vGk,
                              nvlevel,nc0,nck,ocp,Int(j),nai[isp],vthi[isp])
                    end
                else
                    Ms[:,L1,isp] .=  0.0
                end
            end
        end
        return Ms
    else
        rehtjy
    end
end
