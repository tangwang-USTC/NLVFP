
# Difference matrices of first kind of Chebyshev polynomials

Ncheby = 33
Dc = Dc1n(Ncheby; datatype=Float64)               # ∂/∂ᵥ

Dc2 = Dc2n(Ncheby; datatype=Float64)              # ∂²/∂ᵥ²

paraM(Dc2)

DLag(L) = Dc2 + 2 ./ 