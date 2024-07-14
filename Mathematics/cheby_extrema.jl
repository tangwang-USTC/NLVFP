"""
   Calculating the integration:

     `Iₜ = ∫ₐᵇ f(x) dx`

   with the Gauss-Chebyshev quadrature as:

     `Iₜ = ∑`

   Extrema and weights of chebyshev polynomials of first kind `Tₙ(x)`, which is similar to

      `x, w = GaussQuadrature.laguerre(T,n,1,both)`

   but with higher precision. The datatype of outputs is same as `domain`.


   Gauss-Chebyshev type 1 Quandrature weights:

       wc = Pi / (nc - 1) * ω(xₖ)
          = Pi / (nc - 1) * √(1 - xₖ²)
          ≈ Pi / (nc - 1) * sin.(Pi * ((1:nc) .-1) / (nc-1))

  When GQmethod is clenshawcurtis quadrature, quadrature weights are computed through module `FastTransforms.jl` as:

    μ = chebyshevmoments1(T, nc)
    wcc = clenshawcurtisweights(μ)

  with the same values of `extrema` of chebyshev polynomials of first kind.

  Intputs:
    GQmethod: [:chebyshev, :clenshawcurtis]

  Outputs:
    vc = cheby_extrema(n,domain)
    wc = cheby_extremaweights(n,domain;GQmethod=:clenshawcurtis)
"""

function cheby_extrema(n::S,domain::Vector{T} = [-1.0,1.0]) where {T, S <: Integer}

  n <= 0 ? error("The number of nodes must be positive.") : 1

  if T ≠ BigFloat
    PI = pi |> T
  else
    PI = pi |> T
  end
  nodes = zeros(T,n)
  if domain[1] == 1.0 && domain[2] == - 1.0
    if isodd(n)
      nodes[Int((n-1)/2)+1] = 0 |> T
    end

    for i = 1:div(n,2)
      x = - cos((i-1) * PI /(n-1))
      nodes[i]       = + x
      nodes[end-i+1] = - x
    end
    return nodes
  elseif domain[1] == -1.0 && domain[2] == 1.0
    if isodd(n)
      nodes[Int((n-1)/2)+1] = 0 |> T
    end
    for i = 1:div(n,2)
      x = cos((i-1) * PI /(n-1))
      nodes[i]       = + x
      nodes[end-i+1] = - x
    end
    return nodes
  else
    if isodd(n)
      nodes[Int((n-1)/2)+1] = (domain[1] + domain[2]) / 2.0
    end
    for i = 1:div(n,2)
      x = - cos((i-1) * PI /(n-1)) * (domain[1] - domain[2]) / 2.0
      nodes[i]       = (domain[1] + domain[2]) / 2.0  + x
      nodes[end-i+1] = (domain[1] + domain[2]) / 2.0  - x
    end
    return nodes
  end
end

function cheby_extremaweights(n::S,domain::Vector{T} = [-1.0,1.0];GQmethod::Symbol=:clenshawcurtis) where {T, S <: Integer}

  n <= 0 ? error("The number of nodes must be positive.") : 1

  if GQmethod == :clenshawcurtis
    return clenshawcurtisweights(chebyshevmoments1(T, n))
  elseif GQmethod == :chebyshev
    T ≠ BigFloat ? PI = pi |> Double64 : PI = pi |> T
    return PI / (n - 1) * sin.(PI * ((1:n) .- 1) / (n-1)) |> Vector{T}
  end
end
