

"""
  The evlution of normalzied moments `ℳ̂ⱼₗ⁰ = Mhjlo` in discrete with different schemes 
  according to the scheme of kinetic moment equations.

  Inputs:
    Rck[njMs+1,1,:] = ∂ₜvathk           # ∂ₜvₜₕ

  Outputs:
    1
"""

# The implicit `2ᵗʰ`-order scheme (Heun2 or Trapezoid) when `Ms` is solved by `2ᵗʰ`-order scheme.
function MhsRKn!(Mhck1::Vector{Matrix{T}}, Rhck1::Vector{Matrix{T}}, Mhck1i::Vector{Matrix{T}}, 
    vthk1::AbstractVector{T}, Rdtvthk1::AbstractVector{T}, 
    ns::Int64, LM::Vector{Int64}, nMjMs::Vector{Int64}, 
    Mhck::Vector{Matrix{T}}, Rhck::Vector{Matrix{T}}, vthk::AbstractVector{T}, Rdtvthk::AbstractVector{T}, dtk::T) where{T}

    dtkn = dtk / 2
    for isp in 1:ns
        lnvthk = log(vthk[isp])
        lnvthk1 = log(vthk1[isp])
        Rdtlnvthk = Rdtvthk * lnvthk
        Rdtlnvthk1 = Rdtvthk1 * lnvthk1
        for L in 0:LM[isp]
            L1 = L + 1
            for nj in 1:nMjMs[isp]
                j = L + 2(nj - 1)
                cTk1 = 1 + j * lnvthk1
                cTk1k = (1 + j * lnvthk) / cTk1
                Mhck1[isp][nj,L1] = (Rhck1[isp][nj,L1] + cTk1k * Rhck[isp][nj,L1])
                Mhck1[isp][nj,L1] -= j^2 / cTk1 * (Rdtlnvthk1 * Mhck1i[isp][nj,L1] + Rdtlnvthk * Mhck[isp][nj,L1])
                Mhck1[isp][nj,L1] *= dtkn
                Mhck1[isp][nj,L1] += cTk1k * Mhck[isp][nj,L1]
            end
        end
    end
end

# The `0ᵗʰ` order scheme when `Ms` is solved by Explicit Euler scheme.
function MhsRKn!(Mhck1::Vector{Matrix{T}}, Mhck::Vector{Matrix{T}}, ns::Int64) where{T}

    for isp in 1:ns
        Mhck1[isp] = Mhck[isp]
    end
end




