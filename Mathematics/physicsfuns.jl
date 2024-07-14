
"""
  Mass normalization function to eliminate the rounding errors.
"""

function massNorm(ma::T) where{T}
    if ma / me - 1 < 10eps(Float64)
        return me |> T
    else
        return Dₐ * round(ma / Dₐ)
    end
end
