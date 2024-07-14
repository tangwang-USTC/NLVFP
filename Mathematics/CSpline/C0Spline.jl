

"""
  The solver for the three-moment equations when the linear equations are

  Inputs:
    x:
    a=u:
    v:
    w:
    d: The right hand side of the linear equation

  Outputs:
    x = solver3M(x,A,d;method3M=method3M)

"""

function solver3M(x::AbstractVector{T},A::AbstractArray{T},
    d::AbstractVector{T};method3M::Symbol=:inv) where {T}

    if method3M == :inv
        return inv(A) * d
    else
        sdhj
    end
end

"""
  Solving equation

    𝔸𝒙 = 𝒅

  where

    𝔸 = diagm(-1 => u, 0 => v, 1=> w)

    𝔸 = 𝕃𝕌

  where

    𝕃 = diagm(-1 => a, 0 => L)
    𝕌 = diagm(0 => ones(np), 1 => b)

  Here,

    a[k] = u[k]

  Inputs:
    x:
    a=u:
    v:
    w:
    d: The right hand side of the linear equation
    np: The length of the vector `d` or `x`

  Warning: The values of `a`,`v` and `w` will be changed during performing the following procedure.

  Outputs:
    x = chasingM1(x,u,v,w,d)

"""


function chasingM1(x::AbstractVector{T},a::AbstractVector{T},
    v::AbstractVector{T},w::AbstractVector{T},d::AbstractVector{T}) where {T}

    np::Int=length(d)
    # Computing the elements of Crout decomposition and olving the Equation: `𝕃𝒚 = 𝒅`.
    k = 1
    # v[k] = copy(v[k])  # Reusing the `v` as the location of `L`
    w[k] /= v[k]         # Reusing the `w` as the location of `b`
    x[k] = d[k] / v[k]   # Reusing the `x` as the location of `y`
    for k in 2:np-1
        v[k] -= a[k-1] * w[k-1]
        w[k] /= v[k]
        x[k] = (d[k] - a[k-1] * x[k-1]) / v[k]
    end
    k = np
    v[k] -= a[k-1] * w[k-1]
    x[k] = (d[k] - a[k-1] * x[k-1]) / v[k]
    # @show LL = fmtf2.(v)
    # @show bb = fmtf2.(w)
    # @show yy = fmtf2.(x)
    ## Solving the Equation: 𝕌𝒙 = 𝒚
    # k = np
    # x[k] = copy(x[k])
    for k in np-1:-1:1
        x[k] -= w[k] * x[k+1]
    end
    return x
end


# function chasingM1(x::AbstractVector{T},a::AbstractVector{T},
#     v::AbstractVector{T},w::AbstractVector{T},d::AbstractVector{T}) where {T}
#
#     np::Int=length(d)
#      # Computing the elements of Crout decomposition.
#     k = 1
#     # v[k] = copy(v[k])  # Reusing the `v` as the location of `L`
#     w[k] /= v[k]         # Reusing the `w` as the location of `b`
#     for k in 2:np-1
#         v[k] -= a[k-1] * w[k-1]
#         w[k] /= v[k]
#     end
#     k = np
#     v[k] -= a[k-1] * w[k-1]
#     # Solving the Equation: 𝕃𝒚 = 𝒅
#     k = 1
#     x[k] = d[k] / v[k]    # Reusing the `x` as the location of `y`
#     for k in 2:np
#         x[k] = (d[k] - a[k-1] * x[k-1]) / v[k]
#     end
#     # Solving the Equation: 𝕌𝒙 = 𝒚
#     k = np
#     # x[k] = copy(x[k])
#     for k in np-1:-1:1
#         x[k] -= w[k] * x[k+1]
#     end
#     return x
# end

# function chasingM1standard(x::AbstractVector{T},a::AbstractVector{T},
#     v::AbstractVector{T},w::AbstractVector{T},d::AbstractVector{T}) where {T}
#
#     np::Int=length(d)
#     L = zeros(T,np)
#     b = zeros(T,np-1)
#     y = zeros(T,np)
#     # Computing the elements of Crout decomposition.
#     k = 1
#     L[k] = copy(v[k])
#     b[k] = w[k] / L[k]
#     for k in 2:np-1
#         L[k] = v[k] - a[k-1] * b[k-1]
#         b[k] = w[k] / L[k]
#     end
#     k = np
#     L[k] = v[k] - a[k-1] * b[k-1]
#     # Solving the Equation: 𝕃𝒚 = 𝒅
#     k = 1
#     y[k] = d[k] / L[k]
#     for k in 2:np
#         y[k] = (d[k] - a[k-1] * y[k-1]) / L[k]
#     end
#     # Solving the Equation: 𝕌𝒙 = 𝒚
#     k = np
#     x[k] = copy(y[k])
#     for k in np-1:-1:1
#         x[k] = y[k] - b[k] * x[k+1]
#     end
#     return x
# end
