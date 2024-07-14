
"""
  Romberg integration for general moments

  Generally for single sublevel when 'nck = ∑ₖ nk', the `jᵗʰ`-order normalized moments of
  the `ℓᵗʰ`-order normalized coefficient of distribution function, f̂ₗ(v̂), will be:

    𝓜ⱼ(fₗ) = 4π * ∫₀⁹(v̂²⁺ʲf̂ₗ(v̂))dv̂

  Especially, the first few orders of normalized moments such as n̂, Î and K̂ are:

    na: = 4π * ∫₀⁹(v̂²f₀(v))dv̂        , L=0, j = 0
    Iₐ: = 4π / 3 ∫₀⁹(v̂³f₁(v))dv̂      , L=1, j = 1
    Ka: = 4π / ∫₀⁹(v̂⁴f₀(v))dv̂        , L=0, j = 2

  Here, `ma` is not included in the procedure. 
  
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

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
    Ms: where 'Ms[j,l] = [Mⱼ(fₗ(v̂))]'
    vhe: The equal spacing grid points
    fLne: The function values on points `vhe`
    is_renorm: if `is_renorm` is true, `CjLL2 = 1`
               else, `CjLL2(j,L)`

  Outputs:
    Ms, errMs = MsnnEvens(fLn,vhe,j,L;is_renorm=is_renorm)

"""
# 0D, [Msn], the `jᵗʰ`-order moment of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.

function MsnnEvens(fLn::AbstractVector{T},
    vhe::AbstractVector{Tb},j::Int64,L::Int64;
    is_renorm::Bool=true,is_err_renorm::Bool=true) where{T<:Real,Tb}

    # Check whether `vhe[end]^j * fLn[end] ≪ eps(T)`
    if j ≥ 0 && (fLn[end] * vhe[end]^(j+2) ≥ epsT / 10)
        @warn("Error: `vhe[end]^j * fLn[end] > epsT/10` which may cause errors of the higher-order moments when `j=",j)
        @show fLn[end] * vhe[end]^(j+2)
    end
    if j == -2
        Msk,errMs = romberg(vhe, fLn)
    elseif j == -1
        Msk,errMs = romberg(vhe, (vhe .* fLn))
    elseif j == 0
        Msk,errMs = romberg(vhe, (vhe.^2 .* fLn))
    elseif j == 1
        Msk,errMs = romberg(vhe, (vhe.^3 .* fLn))
    else
        Msk,errMs = romberg(vhe, (vhe.^2 .* fLn .* vhe.^j))
    end
    if is_err_renorm
        if abs(errMs) ≥ epsT100
            if abs(Msk) > 1e-10
                errMs /= Msk
            else
                @show L, j, Msk, errMs
                @show fLn[1:5]
            end
        end
    end
    if is_renorm
        Msk *= ( 4 / sqrtpi / CjLL2(j,L))
    else
        Msk *= ( 4 / sqrtpi)
    end
    return Msk,errMs
end

"""
  The `jᵗʰ`-order moments (Ms[j,1]) of the `ℓᵗʰ`-order coefficient of distribution function
  on the entire velocity axis domain.

  where

    jvec = Ms[:,end].
    Ms[j] = Mⱼ(fₗ(v̂))

  Inputs:
    j:
    Ms: = [Msa, Msb], where

  Outputs:
    Ms = MsnnEvens(Ms,fvL,vhe,njMs,L;is_renorm=is_renorm)
"""

# 0.5D, Msn [njMs]
function MsnnEvens(Ms::AbstractVector{T},fLn::AbstractVector{T},
    vhe::AbstractVector{Tb},njMs::Int64,L::Int64;is_renorm::Bool=true) where{T<:Real,Tb}

    for k in 1:njMs
        j = L + 2(k-1)
        Ms[k] = MsnnEvens(fLn,vhe,Int(j),L;is_renorm=is_renorm)
    end
    return Ms
end


"""
  Inputs:
    fvL: fvL[:,:,isp]
    Ms: [MsL1, MsL2,..., jvec]

  Outputs:
    Ms = Msnorm(Ms,fvL,vhe,njMs,LM,LM1;is_renorm=is_renorm)
"""

# 1.5D, [njMs,LM1], the first `jᵗʰ`-order moments of the first `ℓᵗʰ`-order coefficient of
# distribution function on the entire velocity axis domain.
function MsnnEvens(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vhe::AbstractVector{Tb},njMs::Int64,LM::Int64,LM1::Int;is_renorm::Bool=true) where{T<:Real,Tb,N2}

    for L in 0:LM
        L1 = L + 1
        for k in 1:njMs
            j = L + 2(k-1)
            Ms[k,L1] = MsnnEvens(fvL[:,L1],vhe,Int(j),L;is_renorm=is_renorm)
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
    Ms = MsnnEvens(Ms,fvL,vhe,njMs,L,ns;is_renorm=is_renorm)
"""

# 1.5D,[njMs,ns], the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function MsnnEvens(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vhe::AbstractVector{Tb},njMs::Int64,L::Int64,ns::Int64;is_renorm::Bool=true) where{T<:Real,Tb,N2}

    for isp in 1:ns
        for k in 1:njMs
            j = L + 2(k-1)
            Ms[k,isp] = MsnnEvens(fvL[:,isp],vhe,Int(j),L;is_renorm=is_renorm)
        end
    end
    return Ms
end

"""
  Computing re-normalzied moments when `j = L:2:N⁺`.

  Inputs:
    ua: = û / v̂thᵢ

  Outputs:
    Ms = MsnnEvens(Ms,fvL,vhe,njMs,LM,LM1,ns;is_renorm=is_renorm)
"""

# 2.5D, [njMs,LM1,ns], the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function MsnnEvens(Ms::AbstractArray{T,N2},fvL::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
    njMs::Int64,LM::Vector{Int},LM1::Int,ns::Int64;is_renorm::Bool=true) where{T<:Real,N2}

    for isp in 1:ns
        for L in 0:LM[isp]
            L1 = L + 1
            for k in 1:njMs
                j = L1 + 2k - 3
                Ms[k,L1,isp] = MsnnEvens(fvL[isp][:,L1],vhe[isp],Int(j),L;is_renorm=is_renorm)
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
    Msnnt = MsnnEvens(Msnnt,fvL,vhe,njMs,LM,ns,jtype,dj;is_renorm=is_renorm)
"""
# 3.0D, [njMs,LM1,ns,nMod], jtype, dj
function MsnnEvens(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},vhe::AbstractVector{Tb},njMs::Int64,
    LM::Vector{Int},LM1::Int64,ns::Int64,jtype::Symbol,dj::Int64=1;is_renorm::Bool=true) where{T<:Real,Tb,N2}

    if jtype == :L
        for isp in 1:ns
            Ms[:,:,isp] = MsnnEvens(Ms[:,:,isp],fvL[:,:,isp],vhe,njMs,LM[isp],LM1;is_renorm=is_renorm)
        end
        return Ms
    elseif jtype == :n2
        for isp in 1:ns
            for L in 0:LM[isp]
                L1 = L + 1
                for k in 1:njMs
                    j = dj * (k - 1) - 2
                    Ms[k,L1,isp] = MsnnEvens(fvL[:,L1,isp],vhe,Int(j),L;is_renorm=is_renorm)
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
                    Ms[k,L1,isp] = MsnnEvens(fvL[:,L1,isp],vhe,Int(j),L;is_renorm=is_renorm)
                end
            end
        end
        return Ms
    end
end
