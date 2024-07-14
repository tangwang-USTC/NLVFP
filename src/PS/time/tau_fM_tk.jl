

"""
  Outputs:
    τab = tau_fM_Tk(mDa, Zq, spices, n0, T0;tau_scale=:min)
"""

# [ns=2], nMod .=1, When `mDa, n0, T0` are normalized by`Da, n20, Tk` respectively.
function tau_fM_Tk(mDa::AbstractVector{T}, Zq::Vector{Int64}, spices::Vector{Symbol}, 
    n0::AbstractVector{T}, T0::AbstractVector{T};tau_scale::Symbol=:min) where {T<:Real}

    ispt = 1
    iFvt = 2
    lnAab = lnA_const(spices[[ispt,iFvt]])
    νT = 4.41720911682e2 * (mDa[ispt] * mDa[iFvt])^0.5 * (Zq[ispt] * Zq[iFvt])^2 / (mDa[ispt] * T0[iFvt] + mDa[iFvt] * T0[ispt])^1.5 * lnAab
    # τ₀ᵃᵇ = 1 / (n0[iFvt] * νT)
    # τ₀ᵇᵃ = 1 / (n0[ispt] * νT)
    # τ₀ = (τ₀ᵃᵇ + τ₀ᵇᵃ) / 2
    if tau_scale == :min
        return 1 / (maximum(n0) * νT)
        # τmin = 1 / (maximum(n0) * νT)
    elseif tau_scale == :max
        return 1 / (minimum(n0) * νT)
    else
        return 1 / (maximum(n0) * νT), 1 / (minimum(n0) * νT)
            #    νT_min, νT_max
    end
end