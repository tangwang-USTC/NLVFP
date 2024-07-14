
"""
  uC = uCab(ua, vth)
"""

function uCab(ua::AbstractVector{T}, vth::AbstractVector{T}) where{T}

    return sum_kbn([vth[1] * ua[2], vth[2] * ua[1]]) / sum_kbn(vth)
end

