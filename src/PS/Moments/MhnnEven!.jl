
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
    Mh: where 'Mh[j,l] = [M̂ⱼ(fₗ(v̂))]'
    vhe: The equal spacing grid points
    fLne: The function values on points `vhe`
    is_renorm: if `is_renorm` is true, `CjLL2 = 1`
               else, `CjLL2(j,L)`

  Outputs:

"""

"""
  Inputs:
    ua: = û / v̂thᵢ

  Outputs:
    MhnnEvens(fLn,vhe,j,L;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
"""
# 0D, [Mh, errI], [1]
function MhnnEvens(fLn::AbstractVector{T},
    vhe::AbstractVector{Tb},j::Int64,L::Int64;
    is_renorm::Bool=true,is_err_renorm::Bool=true) where{T<:Real,Tb}

    # Check whether `vhe[end]^j * fLn[end] ≪ eps(T)`
    if j ≥ 0 && (fLn[end] * vhe[end]^(j+2) ≥ epsT / 10)
        @warn("Error: `vhe[end]^j * fLn[end] > epsT/10` which may cause errors of the higher-order moments when `j=",j)
        @show fLn[end] * vhe[end]^(j+2)
    end
    if j == -2
        Mhk, errMh = romberg(vhe, fLn)
    elseif j == -1
        Mhk, errMh = romberg(vhe, (vhe .* fLn))
    elseif j == 0
        Mhk, errMh = romberg(vhe, (vhe.^2 .* fLn))
    elseif j == 1
        Mhk, errMh = romberg(vhe, (vhe.^3 .* fLn))
    else
        Mhk, errMh = romberg(vhe, (vhe.^2 .* fLn .* vhe.^j))
    end
    if is_err_renorm
        if abs(errMh) ≥ epsT100
            if abs(Mhk) > 1e-10
                errMh /= Mhk
            else
                @show L, j, Mhk, errMh
                @show fLn[1:5]
            end
        end
    end
    if is_renorm
        Mhk *= ( 4 / sqrtpi / CjLL2(j,L))
    else
        Mhk *= ( 4 / sqrtpi)
    end
    return Mhk, errMh
end

"""
  Computing re-normalzied moments when `j = L:2:N⁺`.

  Inputs:
    ua: = û / v̂thᵢ

  Outputs:
    MhnnEvens!(Mh,errMh,fvL,vhe,njMs,LM,ns;is_renorm=is_renorm,is_err_renorm=is_err_renorm,L_Mh_limit=L_Mh_limit)
"""

# 2.5D, [Mh, errI], [njMs,LM1,ns],  the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function MhnnEvens!(Mh::Vector{Matrix{T}},errMh::Vector{Matrix{T}},
    fvL::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
    njMs::Vector{Int64},LM::Vector{Int64},ns::Int64;
    is_renorm::Bool=true,is_err_renorm::Bool=true,L_Mh_limit::Int64=0) where{T<:Real}

    if L_Mh_limit == 0
        for isp in 1:ns
            for L in 0:LM[isp]
                L1 = L + 1
                if norm(fvL[isp][:,L1]) ≥ epsT1000
                    L13 = L1 - 3
                    for k in 1:njMs[isp]
                        j = L13 + 2k
                        Mh[isp][k,L1], errMh[isp][k,L1] = MhnnEvens(fvL[isp][:,L1],vhe[isp],j,L;
                                is_renorm=is_renorm,is_err_renorm=is_err_renorm)
                    end
                else
                    Mh[isp][:,L1] .= 0.0
                    errMh[isp][:,L1] .= 0.0
                end
            end
        end
    else
        for isp in 1:ns
            for L in 0:min(LM[isp],L_Mh_limit)
                L1 = L + 1
                if norm(fvL[isp][:,L1]) ≥ epsT1000
                    L13 = L1 - 3
                    for k in 1:njMs[isp]
                        j = L13 + 2k
                        Mh[isp][k,L1], errMh[isp][k,L1] = MhnnEvens(fvL[isp][:,L1],vhe[isp],j,L;
                                is_renorm=is_renorm,is_err_renorm=is_err_renorm)
                    end
                else
                    Mh[isp][:,L1] .= 0.0
                    errMh[isp][:,L1] .= 0.0
                end
            end
        end
    end
end

# 1.5D, [Mh, errI], [njMs,LM1],  the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function MhnnEvens!(Mh::AbstractArray{T,N},errMh::AbstractArray{T,N},fvL::Matrix{T},
    vhe::StepRangeLen,njMs::Int64,LM::Int64;
    is_renorm::Bool=true,is_err_renorm::Bool=true,L_Mh_limit::Int64=0) where{T<:Real,N}

    if L_Mh_limit == 0
        for L in 0:LM
            L1 = L + 1
            if norm(fvL[:,L1]) ≥ epsT1000
                L13 = L1 - 3
                for k in 1:njMs
                    j = L13 + 2k
                    Mh[k,L1], errMh[k,L1] = MhnnEvens(fvL[:,L1],vhe,j,L;
                            is_renorm=is_renorm,is_err_renorm=is_err_renorm)
                end
            else
                Mh[:,L1] .= 0.0
                errMh[:,L1] .= 0.0
            end
        end
    else
        for L in 0:min(LM,L_Mh_limit)
            L1 = L + 1
            if norm(fvL[:,L1]) ≥ epsT1000
                L13 = L1 - 3
                for k in 1:njMs
                    j = L13 + 2k
                    Mh[k,L1], errMh[k,L1] = MhnnEvens(fvL[:,L1],vhe,j,L;
                            is_renorm=is_renorm,is_err_renorm=is_err_renorm)
                end
            else
                Mh[:,L1] .= 0.0
                errMh[:,L1] .= 0.0
            end
        end
    end
end

"""
  Inputs:
    j:

    Mh: = [Mha, Mhb], where

  Outputs:
    MhnnEvens!(MhkerrI,fvL,vhe,j,L,ns;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
    MhnnEvens!(MhkerrI,fvL,vhe,Rvth,j,L,ns;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
"""
# 1D, [Mh,errI] [ns]
# function MhnnEvens!(MhkerrI::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    # vhe::AbstractVector{Tb},j::Int64,L::Int64,ns::Int;
    # is_renorm::Bool=true,is_err_renorm::Bool=true) where{T<:Real,Tb,N2}

#     # for isp in 1:ns
#     #     MhnnEvens!(MhkerrI[:,isp],fvL[:,isp],vhe,j,L;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
#     #     # @show isp,MhkerrI[:,isp]
#     # end
#     isp = 1
#     MhnnEvens!(MhkerrI[:,isp],fvL[:,isp],vhe,j,L;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
#     isp = 2
#     MhnnEvens!(MhkerrI[:,isp],fvL[:,isp],vhe,j,L;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
# end

# 1D, [Mh,errI] [ns]
function MhnnEvens!(MhkerrI::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vhe::AbstractVector{Tb},Rvth::AbstractVector{T},j::Int64,L::Int64,ns::Int;
    is_renorm::Bool=true,is_err_renorm::Bool=true) where{T<:Real,Tb,N2}
    
    buiihil
    for isp in 1:ns
        vheup = vhe * Rvth[isp]
        v2 = vheup .^2
        if j == -2
            MhkerrI[1,isp], MhkerrI[2,isp] = romberg(vheup, fvL[:,isp])
        elseif j == -1
            MhkerrI[1,isp], MhkerrI[2,isp] = romberg(vheup, (vheup .* fvL[:,isp]))
        elseif j == 0
            MhkerrI[1,isp], MhkerrI[2,isp] = romberg(vheup, (v2 .* fvL[:,isp]))
        elseif j == 1
            MhkerrI[1,isp], MhkerrI[2,isp] = romberg(vheup, (v2 .* vheup .* fvL[:,isp]))
        elseif j == 2
            MhkerrI[1,isp], MhkerrI[2,isp] = romberg(vheup, (v2 .* fvL[:,isp] .* v2))
        else
            if iseven(j)
                MhkerrI[1,isp], MhkerrI[2,isp] = romberg(vheup, (v2 .* fvL[:,isp] .* v2.^(j/2)))
            else
                MhkerrI[1,isp], MhkerrI[2,isp] = romberg(vheup, (v2 .* fvL[:,isp] .* v2.^((j-1)/2) .* vheup))
            end
        end
    end
    if is_renorm
        MhkerrI[1,:] *= ( 4 / sqrtpi / CjLL2(j,L))
    else
        MhkerrI[1,:] *= ( 4 / sqrtpi)
    end
end
