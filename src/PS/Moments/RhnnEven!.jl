
"""
  Romberg integration for general moments

  Generally for single sublevel when 'nck = ∑ₖ nk', the `jᵗʰ`-order normalized moments of
  the `ℓᵗʰ`-order normalized coefficient of distribution function, dtf̂ₗ(v̂), will be:

    dt𝓜ⱼ(fₗ) = 4π * ∫₀⁹(v̂²⁺ʲf̂ₗ(v̂))dv̂

  Especially, the first few orders of normalized moments such as n̂, Î and K̂ are:

    dtna: = 4π * ∫₀⁹(v̂²f₀(v))dv̂        , L=0, j = 0
    dtIₐ: = 4π / 3 ∫₀⁹(v̂³f₁(v))dv̂      , L=1, j = 1
    dtKa: = 4π / ∫₀⁹(v̂⁴f₀(v))dv̂        , L=0, j = 2

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
    Rh: where 'Rh[j,l] = [Mⱼ(fₗ(v̂))]'
    vhe: The equal spacing grid points
    dtfLne: The function values on points `vhe`
    is_renorm: if `is_renorm` is true, `CjLL2 = 1`
               else, `CjLL2(j,L)`

  Outputs:

"""

"""
  Computing re-normalzied moments when `j = L:2:N⁺`.

  Inputs:
    ua: = û / v̂thᵢ

  Outputs:
    RhnnEvens!(Rh,errRh,dtfvL,vhe,njMs,LM,ns;is_renorm=is_renorm,is_err_renorm=is_err_renorm,L_Mh_limit=L_Mh_limit)
"""

# 2.5D, [Rh, errI], [njMs,LM1,ns],  the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function RhnnEvens!(Rh::Vector{Matrix{T}},errRh::Vector{Matrix{T}},
    dtfvL::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
    njMs::Vector{Int64},LM::Vector{Int64},ns::Int64;
    is_renorm::Bool=true,is_err_renorm::Bool=true,L_Mh_limit::Int64=0) where{T<:Real}

    for isp in 1:ns
        RhnnEvens!(Rh[isp],errRh[isp],dtfvL[isp],vhe[isp],njMs[isp],LM[isp];
                    is_renorm=is_renorm,is_err_renorm=is_err_renorm,L_Mh_limit=L_Mh_limit)
    end
    # if L_Mh_limit == 0
    #     for isp in 1:ns
    #         for L in 0:LM[isp]
    #             L1 = L + 1
    #             if norm(dtfvL[isp][:,L1]) ≥ epsT1000
    #                 L13 = L1 - 3
    #                 for k in 1:njMs[isp]
    #                     j = L13 + 2k
    #                     Rh[isp][k,L1], errRh[isp][k,L1] = RhnnEvens(dtfvL[isp][:,L1],vhe[isp],j,L;
    #                             is_renorm=is_renorm,is_err_renorm=is_err_renorm)
    #                 end
    #             else
    #                 Rh[isp][:,L1] .= 0.0
    #                 errRh[isp][:,L1] .= 0.0
    #             end
    #         end
    #     end
    # else
    #     for isp in 1:ns
    #         for L in 0:min(LM[isp],L_Mh_limit)
    #             L1 = L + 1
    #             if norm(dtfvL[isp][:,L1]) ≥ epsT1000
    #                 L13 = L1 - 3
    #                 for k in 1:njMs[isp]
    #                     j = L13 + 2k
    #                     Rh[isp][k,L1], errRh[isp][k,L1] = RhnnEvens(dtfvL[isp][:,L1],vhe[isp],j,L;
    #                             is_renorm=is_renorm,is_err_renorm=is_err_renorm)
    #                 end
    #             else
    #                 Rh[isp][:,L1] .= 0.0
    #                 errRh[isp][:,L1] .= 0.0
    #             end
    #         end
    #     end
    # end
end

# 1.5D, [Rh, errI], [njMs,LM1],  the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function RhnnEvens!(Rh::AbstractArray{T,N},errRh::AbstractArray{T,N},dtfvL::Matrix{T},
    vhe::StepRangeLen,njMs::Int64,LM::Int64;
    is_renorm::Bool=true,is_err_renorm::Bool=true,L_Mh_limit::Int64=0) where{T<:Real,N}

    if L_Mh_limit == 0
        for L in 0:LM
            L1 = L + 1
            if norm(dtfvL[:,L1]) ≥ epsT1000
                L13 = L1 - 3
                for k in 1:njMs
                    j = L13 + 2k
                    Rh[k,L1], errRh[k,L1] = RhnnEvens(dtfvL[:,L1],vhe,j,L;
                            is_renorm=is_renorm,is_err_renorm=is_err_renorm)
                end
            else
                Rh[:,L1] .= 0.0
                errRh[:,L1] .= 0.0
            end
        end
    else
        for L in 0:min(LM,L_Mh_limit)
            L1 = L + 1
            if norm(dtfvL[:,L1]) ≥ epsT1000
                L13 = L1 - 3
                for k in 1:njMs
                    j = L13 + 2k
                    Rh[k,L1], errRh[k,L1] = RhnnEvens(dtfvL[:,L1],vhe,j,L;
                            is_renorm=is_renorm,is_err_renorm=is_err_renorm)
                end
            else
                Rh[:,L1] .= 0.0
                errRh[:,L1] .= 0.0
            end
        end
    end
end

"""
  Inputs:
    ua: = û / v̂thᵢ

  Outputs:
    RhnnEvens(dtfLn,vhe,j,L;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
"""
# 0D, [Rh, errI], [1]
function RhnnEvens(dtfLn::AbstractVector{T},
    vhe::AbstractVector{Tb},j::Int64,L::Int64;
    is_renorm::Bool=true,is_err_renorm::Bool=true) where{T<:Real,Tb}

    # Check whether `vhe[end]^j * dtfLn[end] ≪ eps(T)`
    if j ≥ 0 && (dtfLn[end] * vhe[end]^(j+2) ≥ epsT / 10)
        @warn("Error: `vhe[end]^j * dtfLn[end] > epsT/10` which may cause errors of the higher-order moments when `j=",j)
        @show dtfLn[end] * vhe[end]^(j+2)
    end
    if j == -2
        Rhk, errRh = romberg(vhe, dtfLn)
    elseif j == -1
        Rhk, errRh = romberg(vhe, (vhe .* dtfLn))
    elseif j == 0
        Rhk, errRh = romberg(vhe, (vhe.^2 .* dtfLn))
    elseif j == 1
        Rhk, errRh = romberg(vhe, (vhe.^3 .* dtfLn))
    else
        Rhk, errRh = romberg(vhe, (vhe.^2 .* dtfLn .* vhe.^j))
    end
    if is_err_renorm
        if abs(errRh) ≥ epsT100
            if abs(Rhk) > 1e-10
                errRh /= Rhk
            else
                @show L, j, Rhk, errRh
                @show dtfLn[1:5]
            end
        end
    end
    if is_renorm
        Rhk *= ( 4 / sqrtpi / CjLL2(j,L))
    else
        Rhk *= ( 4 / sqrtpi)
    end
    return Rhk, errRh
end

"""
  Inputs:
    j:

    Rh: = [Rha, Rhb], where

  Outputs:
    RhnnEvens!(RhkerrI,dtfvL,vhe,j,L,ns;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
    RhnnEvens!(RhkerrI,dtfvL,vhe,Rvth,j,L,ns;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
"""
# 1D, [Rh,errI] [ns]
# function RhnnEvens!(RhkerrI::AbstractArray{T,N2},dtfvL::AbstractArray{T,N2},
    # vhe::AbstractVector{Tb},j::Int64,L::Int64,ns::Int;
    # is_renorm::Bool=true,is_err_renorm::Bool=true) where{T<:Real,Tb,N2}

#     # for isp in 1:ns
#     #     RhnnEvens!(RhkerrI[:,isp],dtfvL[:,isp],vhe,j,L;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
#     #     # @show isp,RhkerrI[:,isp]
#     # end
#     isp = 1
#     RhnnEvens!(RhkerrI[:,isp],dtfvL[:,isp],vhe,j,L;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
#     isp = 2
#     RhnnEvens!(RhkerrI[:,isp],dtfvL[:,isp],vhe,j,L;is_renorm=is_renorm,is_err_renorm=is_err_renorm)
# end

# 1D, [Rh,errI] [ns]
function RhnnEvens!(RhkerrI::AbstractArray{T,N2},dtfvL::AbstractArray{T,N2},
    vhe::AbstractVector{Tb},Rvth::AbstractVector{T},j::Int64,L::Int64,ns::Int;
    is_renorm::Bool=true,is_err_renorm::Bool=true) where{T<:Real,Tb,N2}
    
    buiihil
    for isp in 1:ns
        vheup = vhe * Rvth[isp]
        v2 = vheup .^2
        if j == -2
            RhkerrI[1,isp], RhkerrI[2,isp] = romberg(vheup, dtfvL[:,isp])
        elseif j == -1
            RhkerrI[1,isp], RhkerrI[2,isp] = romberg(vheup, (vheup .* dtfvL[:,isp]))
        elseif j == 0
            RhkerrI[1,isp], RhkerrI[2,isp] = romberg(vheup, (v2 .* dtfvL[:,isp]))
        elseif j == 1
            RhkerrI[1,isp], RhkerrI[2,isp] = romberg(vheup, (v2 .* vheup .* dtfvL[:,isp]))
        elseif j == 2
            RhkerrI[1,isp], RhkerrI[2,isp] = romberg(vheup, (v2 .* dtfvL[:,isp] .* v2))
        else
            if iseven(j)
                RhkerrI[1,isp], RhkerrI[2,isp] = romberg(vheup, (v2 .* dtfvL[:,isp] .* v2.^(j/2)))
            else
                RhkerrI[1,isp], RhkerrI[2,isp] = romberg(vheup, (v2 .* dtfvL[:,isp] .* v2.^((j-1)/2) .* vheup))
            end
        end
    end
    if is_renorm
        RhkerrI[1,:] *= ( 4 / sqrtpi / CjLL2(j,L))
    else
        RhkerrI[1,:] *= ( 4 / sqrtpi)
    end
end
