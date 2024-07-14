
"""
  Romberg integration, normalized moments

  # Notes: `j` may be not the correct form of your need here, make the version of your own!

  Generally for single sublevel when 'nck = ∑ₖ nk', the `jᵗʰ`-order normalized moments of
  the `ℓᵗʰ`-order normalized coefficient of distribution function, f̂ₗ(v̂), will be:

    𝓜ⱼ(fₗ) = 4π * ∫₀⁹(v̂²⁺ʲf̂ₗ(v̂))dv̂

  Especially, the first few orders of normalized moments such as n̂, Î and K̂ are:

    na: = 4π * ∫₀⁹(v̂²f₀(v))dv̂        , L=0, j = 0
    Iₐ: = 4π / 3 ∫₀⁹(v̂³f₁(v))dv̂      , L=1, j = 1
    Ka: = 4π / ∫₀⁹(v̂⁴f₀(v))dv̂        , L=0, j = 2

  Here, `ma` is not included in the procedure.

  On evenly equally spaced grids with number power of `2ⁿ+1` points,
  normalized moments of normalized distributions function will be calculated
  by using a Romberg integration which combining trapezoidal integration
  with Richardson extrapolation for improved accuracy.

        modules: Romberg.jl and Richardson.jl are used.


    Coefficient needed to be taken into account

        cf0 = 4π / π^(3/2),

    The factor `4π` is come from the shperical surface integration of velocity space and
    `π^(-3/2)` is the coefficient of the distribution function.


  Inputs:
    Ms: = [j,Msa, Msb], where 'Ms[j,1] = [Mⱼ(fₗ(v̂))]'
    vGe: The equal spacing grid points
    fLne: The function values on points `vGe`

  Outputs:
    Ms = MsEvens(Msk,fLn,vGe,j)

"""
# 0D, the `jᵗʰ`-order moment of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domian.
function MsnEvens(Msk::T,fLn::AbstractVector{T},
    vGe::AbstractVector{Tb},j::Int) where{T<:Real,Tb}

    if j == -2
        Msk,errI = romberg(vGe, fLn)
    elseif j == -1
        Msk,errI = romberg(vGe, (vGe .* fLn))
    elseif j == 0
        Msk,errI = romberg(vGe, (vGe.^2 .* fLn))
    elseif j == 1
        Msk,errI = romberg(vGe, (vGe.^3 .* fLn))
    else
        Msk,errI = romberg(vGe, (vGe.^2 .* fLn .* vGe.^j))
    end
    Msk *= ( 4 / sqrtpi)
    return Msk
end

"""
  The `jᵗʰ`-order moments (Ms[j,1]) of the `ℓᵗʰ`-order coefficient of distribution function
  on the entire velocity axis domian.

  where

    jvec = Ms[:,end].
    Ms[j,1] = Mⱼ(fₗ(v̂))

  Inputs:
    j:

    Ms: = [j,Msa, Msb], where

  Outputs:
    Ms = MsnEvens(Ms,fvL,vGe)
"""

# 0.5D, [njMs]
function MsnEvens(Ms::AbstractArray{T,N2},fLn::AbstractVector{T},vGe::AbstractVector{Tb}) where{T<:Real,Tb,N2}

    if norm(fLn) > epsT5
        nj = 0
        for j in Ms[:,2]
            nj += 1
            Ms[nj,1] = MsnEvens(Ms[nj,1],fLn,vGe,Int(j))
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
    Ms = MsnEvens(Ms,fvL,vGe,LM)
"""

# 1.5DL, [njMs,LM1], the first `jᵗʰ`-order moments of the first `ℓᵗʰ`-order coefficient of
# distribution function on the entire velocity axis domian.
function MsnEvens(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vGe::AbstractVector{Tb},LM::Int) where{T<:Real,Tb,N2}

    for L1 in 1:LM
        if norm(fvL[:,L1]) > epsT5
            nj = 0
            for j in Ms[:,end]
                nj += 1
                Ms[nj,L1] = MsnEvens(Ms[nj,L1],fvL[:,L1],vGe,Int(j))
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
    Ms = MsnEvens(Ms,fvL,vGe,ns)
"""

# 1.5Disp,[njMs,ns], the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domian.
function MsnEvens(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vGe::AbstractVector{Tb},ns::Int) where{T<:Real,Tb,N2}

    for isp in 1:ns
        if norm(fvL[:,isp]) > epsT5
            nj = 0
            for j in Ms[:,end]
                nj += 1
                Ms[nj,isp] = MsnEvens(Ms[nj,isp],fvL[:,isp],vGe,Int(j))
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
    Ms = MsnEvens(Ms,fvL,vGk,nvlevel,nc0,nck,LM,ns)
"""

# 2.5D, [njMs,LM1,ns], the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domian.
function MsnEvens(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vGe::AbstractVector{Tb},LM::Vector{Int},ns::Int) where{T<:Real,Tb,N2}

    for isp in 1:ns
        jvec = Ms[:,end,isp]
        for L1 in 1:LM[isp]
            if norm(fvL[:,L1,isp]) > epsT5
                nj = 0
                for j in jvec
                    nj += 1
                    Ms[nj,L1,isp] = MsnEvens(Ms[nj,L1,isp],fvL[:,L1,isp],vGe,Int(j))
                end
            else
                Ms[:,L1,isp] .=  0.0
            end
        end
    end
    return Ms
end
