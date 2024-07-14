
"""
  The `jᵗʰ`-order re-normalzied moments of the `ℓᵗʰ`-order
  coefficient of the normalized distribution function `f̂ₗ(v̂)`
  are re-normalized by the coefficient:

    CjL(j,L) = (j+L+1)!! / (2L-1)!! / 2^((j-L)/2), j ∈ L:2:N⁺.

"""

# Generally
CjLL2(j,L) = prod((2L+1):2:(j+L+1)) / 2.0^((j-L)/2)
# `L = 0` yields:
CjLL2(j) = 2 / sqrtpi * gamma((3+j)/2)       # When `j > 7`, the errors may be so large that cannot be ignored.
# Especially, when `iseven(j)`, `j = 2N`
CjLL2even(j) = prod(3:2:j+1) / 2^(j/2)
# and when `isodd(j)`, `j = 2N + 1`
CjLL2odd(j) = prod(2:1:(j+1)/2) * 2 / sqrtpi

# jjj = 9
# fdd(jjj) =  CjLL2(jjj)/CjLL2odd(jjj)