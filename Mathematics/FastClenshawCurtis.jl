
"""
  Fast Clenshaw Curtis Quadrature

    Compute the N nodes and weights for Clenshaw-Curtis Quadrature on the interval [a,b].
  Unlike Gauss quadratures, Clenshaw-Curtis is only exact for polynomials up to order N,
  however, using the FFT algorithm, the weights and nodes are computed in linear time.
  This script will calculate for N = 2^20+1 (1048577 points) in about 5 seconds on a normal laptop computer.

  Written by: Greg von Winckel - 02/12/2005
  Contact: gregvw(at)chtm(dot)unm(dot)edu
"""

function clenshawcurtisfast(N1::Int,a::T,b::T) where {T}

    N = N1-1
    bma = b-a
    c = zeros(N1,2)
    vec = 1 .- (2:2:N).^2  |>Vector{T}
    c[1:2:N1,1] = 2 ./ [1;  vec]
    c[2,2] = 1.0
    f = real(fft([c[1:N1,:];c[N:-1:2,:]])[:,1])         # ?
    w = bma * ([f[1,1]; 2 * f[2:N,1];  f[N1,1]]) / 2
    x = 0.5 * ((b+a) .+ N * bma * f[1:N1,2])
    return x, w
end
