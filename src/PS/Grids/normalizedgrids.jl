
"""
  To balance the accuracy and efficiency, boundary grids is decided by the normalzied velocity grids where

    `v̂⁽ᵏ⁾ = v̂⁽ᵏ⁺¹⁾`

  Thus, the velocity grids at `(k+1)ᵗʰ` step is decied by:

    `v⁽ᵏ⁺¹⁾ = v⁽ᵏ⁾ * (vₜₕ⁽ᵏ⁺¹⁾ / vₜₕ⁽ᵏ⁾)`

  Intputs:
  Outputs:
    vs = normalizedgrids!(vs,vthk1,vthk,ns)
    v1 = normalizedgrids!(v1,vthk1,vthk)
"""

function normalizedgrids!(vs::AbstractArray{T,N},vthk1::AbstractVector{T},vthk::AbstractVector{T},ns::Int) where{T,N}

    for isp in 1:ns
        vs[:,isp] *= (vthk1[isp] / vthk[isp])
    end
    return vs
end

function normalizedgrids!(v1::AbstractVector{T},vthk1::Float64,vthk::Float64) where{T}

    return v1 * (vthk1 / vthk)
end
