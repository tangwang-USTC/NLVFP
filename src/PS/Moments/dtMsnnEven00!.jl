
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
    dtfLn: The function values on points `vhe`
    is_renorm: if `is_renorm` is true, `CjLL2 = 1`
               else, `CjLL2(j,L)`

  Outputs:

"""

"""
  Computing re-normalzied moments when `j = L:2:N⁺`.

  Inputs:
    ua: = û / v̂thᵢ

  Outputs:
    dtMsnnEvens!(Mh,errMh,fvL,vhe,njMs,LM,ns;is_renorm=is_renorm)
"""

# 2.5D, [Msn, errI], [njMs,LM1,ns],  the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function dtMsnnEvens!(Mh::Vector{Matrix{T}},errMh::Vector{Matrix{T}},
    fvL::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
    njMs::Vector{Int64},LM::Vector{Int64},ns::Int64;
    is_renorm::Bool=true,is_err_renorm::Bool=true) where{T}

    for isp in 1:ns
        for L in 0:LM[isp]
            L1 = L + 1
            if norm(fvL[isp][:,L1]) ≥ epsT1000
                L13 = L1 - 3
                for k in 1:njMs[isp]
                    j = L13 + 2k
                    Mh[isp][k,L1], errMh[isp][k,L1] = dtMsnnEvens!(fvL[isp][:,L1],vhe[isp],j,L;
                            is_renorm=is_renorm,is_err_renorm=is_err_renorm)
                end
            else
                Mh[isp][:,L1] .= 0.0
                errMh[isp][:,L1] .= 0.0
            end
        end
    end
end

# 1.5D, [Msn, errI], [njMs,LM1],  the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function dtMsnnEvens!(Mh::AbstractArray{T,N},errMh::AbstractArray{T,N},fvL::Matrix{T},
    vhe::AbstractVector{T},nvG::Int64,njMs::Int64,LM::Int64;
    is_renorm::Bool=true,is_err_renorm::Bool=true) where{T,N}

    for L in 0:LM
        L1 = L + 1
        L13 = L1 - 3
        for k in 1:njMs
            j = L13 + 2k
            a = [Mh[k,L1], errMh[k,L1]]
            dtMsnnEvens!(a,fvL[:,L1],vhe,nvG,j,L;is_renorm=is_renorm)
            Mh[k,L1] = a[1]
            errMh[k,L1] = a[2]
        end
    end
end

function dtMsnnEvens!(Mh::AbstractArray{T,N},fvL::Matrix{T},
    vhe::AbstractVector{T},nvG::Int64,njMs::Int64,LM::Int64;is_renorm::Bool=true) where{T,N}

    for L in 0:LM
        L1 = L + 1
        L13 = L1 - 3
        for k in 1:njMs
            j = L13 + 2k
            Mh[k,L1] = dtMsnnEvens(fvL[:,L1],vhe,nvG,j,L;is_renorm=is_renorm)
        end
    end
end

function dtMsnnEvens!(Mh::AbstractArray{T,N},errMh::AbstractArray{T,N},fvL::Matrix{T},
    vhe::StepRangeLen,njMs::Int64,LM::Int64;is_renorm::Bool=true) where{T,N}

    for L in 0:LM
        L1 = L + 1
        L13 = L1 - 3
        for k in 1:njMs
            j = L13 + 2k
            a = [Mh[k,L1], errMh[k,L1]]
            dtMsnnEvens!(a,fvL[:,L1],vhe,j,L;is_renorm=is_renorm)
            Mh[k,L1] = a[1]
            errMh[k,L1] = a[2]
        end
    end
end

function dtMsnnEvens!(Mh::Matrix{T},fvL::Matrix{T},vhe::StepRangeLen,
    njMs::Int64,LM::Int64;is_renorm::Bool=true) where{T}

    for L in 0:LM
        L1 = L + 1
        L13 = L1 - 3
        for k in 1:njMs
            j = L13 + 2k
            a = [Mh[k,L1], 0.0]
            dtMsnnEvens!(a,fvL[:,L1],vhe,j,L;is_renorm=is_renorm)
            Mh[k,L1] = a[1]
        end
    end
end

"""
  Inputs:
    ua: = û / v̂thᵢ

  Outputs:
    dtMsnnEvens!(MskerrI,dtfLn,vhe,j,L;is_renorm=is_renorm)
"""
# 0D, [Msn, errI], [1]
function dtMsnnEvens!(MskerrI::AbstractVector{T},dtfLn::AbstractVector{T},
    vhe::AbstractVector{T},nvG::Int64,j::Int64,L::Int64;is_renorm::Bool=true) where{T}

    # Check whether `vhe[end]^j * dtfLn[end] ≪ eps(T)`
    a = abs(dtfLn[end]) * vhe[end]^(j+2)
    if j ≥ 0 && (a ≥ atol_vjdtfLn9)
        @warn("Error: `vhe[end]^j * dtfLn[end] > atol_vjdtfLn9` which may cause errors of the higher-order moments when `j=",(j,a))
    end

    Ixv = - 2 / sqrtpi * (vhe[1] - vhe[end])
    wcck = clenshawcurtisweights(chebyshevmoments1(T, nvG))
    vec = 1:2:nvG
    wcck2 = clenshawcurtisweights(chebyshevmoments1(T, Int((nvG+1)/2)))
    if j == -2
        MskerrI[1] = Ixv * dot(wcck, dtfLn)
        MskerrI[2] = Ixv * dot(wcck2, dtfLn[vec]) - MskerrI[1]                      # error
    elseif j == -1
        MskerrI[1] = Ixv * dot(wcck, (vhe .* dtfLn))
        MskerrI[2] = Ixv * dot(wcck2, (vhe[vec] .* dtfLn[vec])) - MskerrI[1]
    elseif j == 0
        MskerrI[1] = Ixv * dot(wcck, (vhe.^2 .* dtfLn))
        MskerrI[2] = Ixv * dot(wcck2, (vhe[vec].^2 .* dtfLn[vec])) - MskerrI[1]
    elseif j == 1
        MskerrI[1] = Ixv * dot(wcck, (vhe.^3 .* dtfLn))
        MskerrI[2] = Ixv * dot(wcck2, (vhe[vec].^3 .* dtfLn[vec])) - MskerrI[1]
    else
        MskerrI[1] = Ixv * dot(wcck, (vhe.^2 .* dtfLn .* vhe.^j))
        MskerrI[2] = Ixv * dot(wcck2, (vhe[vec].^2 .* dtfLn[vec] .* vhe[vec].^j)) - MskerrI[1]
    end
    if is_renorm
        MskerrI[1] /= CjLL2(j,L)
    end
end

function dtMsnnEvens!(MskerrI::AbstractVector{T},dtfLn::AbstractVector{T},
    vhe::StepRangeLen,j::Int64,L::Int64;is_renorm::Bool=true) where{T<:Real}

    # Check whether `vhe[end]^j * dtfLn[end] ≪ eps(T)`
    a = dtfLn[end] * vhe[end]^(j+2)
    if j ≥ 0 && (abs(a) ≥ atol_vjdtfLn9)
        @warn("Error: `vhe[end]^j * dtfLn[end] > atol_vjdtfLn9` which may cause errors of the higher-order moments when `j=",(j,a))
    end
    if j == -2
        MskerrI[1], MskerrI[2] = romberg(vhe, dtfLn)
    elseif j == -1
        MskerrI[1], MskerrI[2] = romberg(vhe, (vhe .* dtfLn))
    elseif j == 0
        MskerrI[1], MskerrI[2] = romberg(vhe, (vhe.^2 .* dtfLn))
    elseif j == 1
        MskerrI[1], MskerrI[2] = romberg(vhe, (vhe.^3 .* dtfLn))
    else
        MskerrI[1], MskerrI[2] = romberg(vhe, (vhe.^2 .* dtfLn .* vhe.^j))
    end
    if is_renorm
        MskerrI[1] *= ( 4 / sqrtpi / CjLL2(j,L))
    else
        MskerrI[1] *= ( 4 / sqrtpi)
    end
end

function dtMsnnEvens(dtfLn::AbstractVector{T},vhe::AbstractVector{T},
    nvG::Int64,j::Int64,L::Int64;is_renorm::Bool=true) where{T}

    # Check whether `vhe[end]^j * dtfLn[end] ≪ eps(T)`
    a = dtfLn[end] * vhe[end]^(j+2)
    if j ≥ 0 && (abs(a) ≥ atol_vjdtfLn9)
        @warn("Error: `vhe[end]^j * dtfLn[end] > atol_vjdtfLn9` which may cause errors of the higher-order moments when `j=",(j,a))
    end

    Ixv = - 2 / sqrtpi * (vhe[1] - vhe[end])
    wcck = clenshawcurtisweights(chebyshevmoments1(T, nvG))
    if j == -2
        Msk = Ixv * dot(wcck, dtfLn)
    elseif j == -1
        Msk = Ixv * dot(wcck, (vhe .* dtfLn))
    elseif j == 0
        Msk = Ixv * dot(wcck, (vhe.^2 .* dtfLn))
    elseif j == 1
        Msk = Ixv * dot(wcck, (vhe.^3 .* dtfLn))
    else
        Msk = Ixv * dot(wcck, (vhe.^2 .* dtfLn .* vhe.^j))
    end
    if is_renorm
        Msk /= CjLL2(j,L)
    end
    return Msk
end

"""
  Inputs:
    j:

    Ms: = [Msa, Msb], where

  Outputs:
    dtMsnnEvens!(MskerrI,fvL,vhe,j,L,ns;is_renorm=is_renorm)
    dtMsnnEvens!(MskerrI,fvL,vhe,Rvth,j,L,ns;is_renorm=is_renorm)
"""
# 1D, [Msn,errI] [ns]
# function dtMsnnEvens!(MskerrI::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
#     vhe::AbstractVector{StepRangeLen},j::Int64,L::Int64,ns::Int;is_renorm::Bool=true) where{T<:Real,N2}

#     # for isp in 1:ns
#     #     dtMsnnEvens!(MskerrI[:,isp],fvL[:,isp],vhe,j,L;is_renorm=is_renorm)
#     #     # @show isp,MskerrI[:,isp]
#     # end
#     isp = 1
#     dtMsnnEvens!(MskerrI[:,isp],fvL[:,isp],vhe,j,L;is_renorm=is_renorm)
#     isp = 2
#     dtMsnnEvens!(MskerrI[:,isp],fvL[:,isp],vhe,j,L;is_renorm=is_renorm)
# end

# 1D, [Msn,errI] [ns]
function dtMsnnEvens!(MskerrI::AbstractArray{T,N2},fvL::AbstractArray{T,N2},
    vhe::AbstractVector{StepRangeLen},Rvth::AbstractVector{T},j::Int64,L::Int64,ns::Int;is_renorm::Bool=true) where{T<:Real,N2}
    
    syhyhyyh
    for isp in 1:ns
        vheup = vhe * Rvth[isp]
        v2 = vheup .^2
        if j == -2
            MskerrI[1,isp], MskerrI[2,isp] = romberg(vheup, fvL[:,isp])
        elseif j == -1
            MskerrI[1,isp], MskerrI[2,isp] = romberg(vheup, (vheup .* fvL[:,isp]))
        elseif j == 0
            MskerrI[1,isp], MskerrI[2,isp] = romberg(vheup, (v2 .* fvL[:,isp]))
        elseif j == 1
            MskerrI[1,isp], MskerrI[2,isp] = romberg(vheup, (v2 .* vheup .* fvL[:,isp]))
        elseif j == 2
            MskerrI[1,isp], MskerrI[2,isp] = romberg(vheup, (v2 .* fvL[:,isp] .* v2))
        else
            if iseven(j)
                MskerrI[1,isp], MskerrI[2,isp] = romberg(vheup, (v2 .* fvL[:,isp] .* v2.^(j/2)))
            else
                MskerrI[1,isp], MskerrI[2,isp] = romberg(vheup, (v2 .* fvL[:,isp] .* v2.^((j-1)/2) .* vheup))
            end
        end
    end
    if is_renorm
        MskerrI[1,:] *= ( 4 / sqrtpi / CjLL2(j,L))
    else
        MskerrI[1,:] *= ( 4 / sqrtpi)
    end
end
