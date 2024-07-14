
"""
  Romberg integration, moments without coefficient `CjLL2(j,L)`

  Generally for single sublevel when 'nck = ∑ₖ nk', the `jᵗʰ`-order normalized moments of
  the `ℓᵗʰ`-order normalized coefficient of distribution function, f̂ₗ(v̂), will be:

    𝓜ⱼ(fₗ) = 4π * ∫₀⁹(v̂²⁺ʲf̂ₗ(v̂))dv̂

  where

     f̂ₗ(v̂) = π^(3/2) × nₐ⁻¹vₜₕ³ × fₗ(v)

  The factor `π^(-3/2)` is needed to be taken into account. Especially,
  the first few orders of normalized moments such as n̂, Î and K̂ are:

    na: = 4π * π^(3/2) * na * ∫₀⁹(v̂²f̂₀(v))dv̂                  , L=0, j = 0
    Iₐ: = 4π/3 * π^(3/2) * ma * na * vth * ∫₀⁹(v̂³f̂₁(v))dv̂     , L=1, j = 1
    Ka: = 4π * π^(3/2) * ma * na * vth²/2 *  ∫₀⁹(v̂⁴f̂₀(v))dv̂   , L=0, j = 2
  
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

  On evenly equally spaced grids with number power of `2ⁿ+1` points,
  normalized moments of normalized distributions function will be calculated
  by using a Romberg integration which combining trapezoidal integration
  with Richardson extrapolation for improved accuracy.

        modules: Romberg.jl and Richardson.jl are used.

  Inputs:
    ma: which is normalzied by `Dₐ`.
    na:
    vth:
    Ms: = [j,Msa, Msb], where 'Ms[j,1] = [Mⱼ(fₗ(v̂))]'
    vGe: The equal spacing grid points
    fLne: The function values on points `vGe`

  Outputs:
  Ms = MsrnEvens(Msk,fLn,vGe,ma,na,vth,j,L;is_renorm=is_renorm)

"""

# # 0D, the `jᵗʰ`-order moment of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
# function MsrnEvens(Msk::T,fLn::AbstractVector{T},vGe::AbstractVector{Tb},
#     ma::T,na::T,vth::T,j::Int64,L::Int64;is_renorm::Bool=true) where{T<:Real,Tb}
    #
    #   # Check whether `vGe[end]^j * fLn[end] ≪ eps(T)`
    #   if j ≥ 0 && (fLn[end] * vGe[end]^(j+2) ≥ epsT / 10)
    #      @warn("Error: `vGe[end]^j * fLn[end] > epsT/10` which may cause errors of the higher-order moments when `j=",j)
    #   end
#     if j == -2
#         Msk,errI = romberg(vGe, fLn)
#     elseif j == -1
#         Msk,errI = romberg(vGe, (vGe .* fLn))
#     elseif j == 0
#         Msk,errI = romberg(vGe, (vGe.^2 .* fLn))
#     elseif j == 1
#         Msk,errI = romberg(vGe, (vGe.^3 .* fLn))
#     else
#         Msk,errI = romberg(vGe, (vGe.^2 .* fLn .* vGe.^j))
#     end
#     if is_renorm
#         if j == 0
#             Msk *= ( 4 / sqrtpi / CjLL2(j,L))
#         elseif j == 1
#             Msk *= ( 4 / sqrtpi / CjLL2(j,L)) * (ma * na * vth)
#         elseif j == 2
#             if L == 0
#                 Msk *= ( 2 / sqrtpi / CjLL2(j,L)) * (ma * na * vth^2)
#             else
#                 Msk *= ( 4 / sqrtpi / CjLL2(j,L)) * (ma * na * vth^2)
#             end
#         else
#             Msk *= ( 4 / sqrtpi / CjLL2(j,L)) * (ma * na * vth^(j))
#         end
#     else
#         if j == 0
#             Msk *= ( 4 / sqrtpi)
#         elseif j == 1
#             Msk *= ( 4 / sqrtpi) * (ma * na * vth^(j))
#         elseif j == 2
#             if L == 0
#                 Msk *= ( 2 / sqrtpi) * (ma * na * vth^(j))
#             else
#                 Msk *= ( 4 / sqrtpi) * (ma * na * vth^(j))
#             end
#         else
#             Msk *= ( 4 / sqrtpi) * (ma * na * vth^(j))
#         end
#     end
#     return Msk
# end

function MsrnEvens(Msk::T,fLn::AbstractVector{T},vGe::AbstractVector{Tb},
    ma::T,na::T,vth::T,j::Int64,L::Int64;is_renorm::Bool=true) where{T<:Real,Tb}
    
    # Check whether `vGe[end]^j * fLn[end] ≪ eps(T)`
    if j ≥ 0 && (fLn[end] * vGe[end]^(j+2) ≥ epsT / 10)
        @warn("Error: `vGe[end]^j * fLn[end] > epsT/10` which may cause errors of the higher-order moments when `j=",j)
        @show fLn[end] * vGe[end]^(j+2)
    end
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
    if is_renorm
        if j == 0
            Msk *= (4 / sqrtpi / CjLL2(j,L))
        elseif j == 1
            Msk *= (4 / sqrtpi / CjLL2(j,L)) * (ma * na * vth)
        elseif j == 2
            if L == 0
                Msk *= (2 / sqrtpi * Mms / CjLL2(j,L)) * (ma * na * vth^2)
            else
                Msk *= (4 / sqrtpi * Mms / CjLL2(j,L)) * (ma * na * vth^2)
            end
        else
            Msk *= (4 / sqrtpi * Mms^(j-1) / CjLL2(j,L)) * (ma * na * vth^(j))
        end
    else
        if j == 0
            Msk *= (4 / sqrtpi)
        elseif j == 1
            Msk *= 4 / sqrtpi * (ma * na * vth)
        elseif j == 2
            if L == 0
                Msk *= (2 / sqrtpi * Mms) * (ma * na * vth^2)
            else
                Msk *= (4 / sqrtpi * Mms) * (ma * na * vth^2)
            end
        else
            Msk *= (4 / sqrtpi * Mms^(j-1)) * (ma * na * vth^(j))
        end
    end
    return Msk
end

"""
  The `jᵗʰ`-order moments (Ms[j,1]) of the `ℓᵗʰ`-order coefficient of distribution function
  on the entire velocity axis domain.

  where

    Ms[j] = Mⱼ(fₗ(v̂))

  Inputs:
    j:

    Ms: = [Msa, Msb], where

  Outputs:
    Ms = MsrnEvens(Ms,fvL,vGe,ma,na,vth,njMs,L;is_renorm=is_renorm)
"""

# 0.5D, [njMs]
function MsrnEvens(Ms::AbstractVector{T},fLn::AbstractVector{T},vGe::AbstractVector{Tb},
    ma::T,na::T,vth::T,njMs::Int64,L::Int64;is_renorm::Bool=true) where{T<:Real,Tb}

    for k in 1:njMs
        j = L + 2(k-1)
        Ms[k] = MsrnEvens(Ms[k],fLn,vGe,ma,na,vth,Int(j),L;is_renorm=is_renorm)
    end
    return Ms
end

"""
  Outputs:
    Ms = MsrnEvens(Ms,fvL,vGe,ma,na,vth,j,L,ns;is_renorm=is_renorm)
"""
# 0.5D,[j,ns]
function MsrnEvens(Ms::AbstractVector{T},fvL::AbstractArray{T,N2},vGe::AbstractVector{Tb},
    ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},
    j::Int64,L::Int64,ns::Int64;is_renorm::Bool=true) where{T<:Real,Tb,N2}

    for isp in 1:ns
        Ms[isp] = MsrnEvens(Ms[isp],fvL[:,isp],vGe,ma[isp],na[isp],vth[isp],j,L;is_renorm=is_renorm)
    end
    return Ms
end

"""
  Inputs:
    fvL: fvL[:,:,isp]
    Ms: [MsL1, MsL2,...]

  Outputs:
    Ms = MsrnEvens(Ms,fvL,vGe,ma,na,vth,njMs,LM,LM1;is_renorm=is_renorm)
"""

# 1.5D, [njMs,LM1], the first `jᵗʰ`-order moments of the first `ℓᵗʰ`-order coefficient of
# distribution function on the entire velocity axis domain.
function MsrnEvens(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},vGe::AbstractVector{Tb},
    ma::T,na::T,vth::T,njMs::Int64,LM::Int64,LM1::Int64;is_renorm::Bool=true) where{T<:Real,Tb,N2}

    for L in 0:LM
        L1 = L + 1
        for k in 1:njMs
            j = L + 2(k-1)
            Ms[k,L1] = MsrnEvens(Ms[k,L1],fvL[:,L1],vGe,ma,na,vth,Int(j),L;is_renorm=is_renorm)
        end
        if LM + 1 ≠ LM1
            Ms[:,LM+2:LM1] .=  0.0
        end
    end
    return Ms
end

"""
  Inputs:
    fvL: fvL[:,L1,:]
    Ms: [Ms1, Ms2]

  Outputs:
    Ms = MsrnEvens(Ms,fvL,vGe,ma,na,vth,njMs,L,ns;is_renorm=is_renorm)
"""

# 1.5D,[njMs,ns], the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function MsrnEvens(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},vGe::AbstractVector{Tb},
    ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},
    njMs::Int64,L::Int64,ns::Int64;is_renorm::Bool=true) where{T<:Real,Tb,N2}

    for isp in 1:ns
        for k in 1:njMs
            j = L + 2(k-1)
            Ms[k,isp] = MsrnEvens(Ms[k,isp],fvL[:,isp],vGe,ma[isp],
                            na[isp],vth[isp],Int(j),L;is_renorm=is_renorm)
        end
    end
    return Ms
end

"""
  Computing re-normalzied moments when `j = L:2:N⁺`.

  Inputs:
    ua: = û / v̂thᵢ

  Outputs:
    Ms = MsrnEvens(Ms,fvL,vGe,ma,na,vth,njMs,LM,LM1,ns;is_renorm=is_renorm)
"""

# 2.5D, [njMs,LM1,ns], the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
function MsrnEvens(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},vGe::AbstractVector{Tb},
    ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},njMs::Int64,
    LM::Vector{Int},LM1::Int,ns::Int64;is_renorm::Bool=true) where{T<:Real,Tb,N2}

    for isp in 1:ns
        for L in 0:LM[isp]
            L1 = L + 1
            for k in 1:njMs
                j = L1 + 2k - 3
                Ms[k,L1,isp] = MsrnEvens(Ms[k,L1,isp],fvL[:,L1,isp],vGe,
                                ma[isp],na[isp],vth[isp],Int(j),L;is_renorm=is_renorm)
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
    Msrnt = MsrnEvens(Msrnt,fvL,vGe,ma,na,vth,njMs,LM,LM1,ns,jtype,dj;is_renorm=is_renorm)
"""
# 3.0D, [njMs,LM1,ns,nMod], jtype, dj
function MsrnEvens(Ms::AbstractArray{T,N2},fvL::AbstractArray{T,N2},vGe::AbstractVector{Tb},
    ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},njMs::Int64,
    LM::Vector{Int},LM1::Int,ns::Int64,jtype::Symbol,dj::Int64=1;is_renorm::Bool=true) where{T<:Real,Tb,N2}

    werthyujk
    if jtype == :L
        for isp in 1:ns
            Ms[:,:,isp] = MsrnEvens(Ms[:,:,isp],fvL[:,:,isp],vGe,
                      ma[isp],na[isp],vth[isp],njMs,LM[isp];is_renorm=is_renorm)
        end
        return Ms
    elseif jtype == :n2
        for isp in 1:ns
            for L1 in 1:LM[isp]
                L = L1 - 1
                for k in 1:njMs
                    j = dj * (k - 1) - 2
                    Ms[k,L1,isp] = MsrnEvens(Ms[k,L1,isp],fvL[:,L1,isp],vGe,
                                    ma[isp],na[isp],vth[isp],Int(j),L;is_renorm=is_renorm)
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
                    Ms[k,L1,isp] = MsrnEvens(Ms[k,L1,isp],fvL[:,L1,isp],vGe,
                                    ma[isp],na[isp],vth[isp],Int(j),L;is_renorm=is_renorm)
                end
            end
        end
        return Ms
    end
end
