

## Moore-Penrose pseudoinverse

"""
    pinv(M; atol::Real=0, rtol::Real=atol>0 ? 0 : n*ϵ)
    pinv(M, rtol::Real) = pinv(M; rtol=rtol) # to be deprecated in Julia 2.0

    Computes the Moore-Penrose pseudoinverse.

    For matrices `M` with floating point elements, it is convenient to compute
    the pseudoinverse by inverting only singular values greater than
    `max(atol, rtol*σ₁)` where `σ₁` is the largest singular value of `M`.

    The optimal choice of absolute (`atol`) and relative tolerance (`rtol`) varies
    both with the value of `M` and the intended application of the pseudoinverse.
    The default relative tolerance is `n*ϵ`, where `n` is the size of the smallest
    dimension of `M`, and `ϵ` is the [`eps`](@ref) of the element type of `M`.

    For inverting dense ill-conditioned matrices in a least-squares sense,
    `rtol = sqrt(eps(real(float(one(eltype(M))))))` is recommended.

    For more information, see [^issue8859], [^B96], [^S84], [^KY88].

    # Examples
    ```jldoctest
    julia> M = [1.5 1.3; 1.2 1.9]
    2×2 Matrix{Float64}:
     1.5  1.3
     1.2  1.9

    julia> N = pinv(M)
    2×2 Matrix{Float64}:
      1.47287   -1.00775
     -0.930233   1.16279

    julia> M * N
    2×2 Matrix{Float64}:
     1.0          -2.22045e-16
     4.44089e-16   1.0
```

[^issue8859]: Issue 8859, "Fix least squares", [https://github.com/JuliaLang/julia/pull/8859](https://github.com/JuliaLang/julia/pull/8859)

[^B96]: Åke Björck, "Numerical Methods for Least Squares Problems",  SIAM Press, Philadelphia, 1996, "Other Titles in Applied Mathematics", Vol. 51. [doi:10.1137/1.9781611971484](http://epubs.siam.org/doi/book/10.1137/1.9781611971484)

[^S84]: G. W. Stewart, "Rank Degeneracy", SIAM Journal on Scientific and Statistical Computing, 5(2), 1984, 403-413. [doi:10.1137/0905030](http://epubs.siam.org/doi/abs/10.1137/0905030)

[^KY88]: Konstantinos Konstantinides and Kung Yao, "Statistical analysis of effective singular values in matrix rank determination", IEEE Transactions on Acoustics, Speech and Signal Processing, 36(5), 1988, 757-763. [doi:10.1109/29.1585](https://doi.org/10.1109/29.1585)
"""

function pinV(A::AbstractMatrix{T}; atol::Real = 0.0, rtol::Real = (eps(real(float(one(T))))*min(size(A)...))*iszero(atol)) where T
    m, n = size(A)
    Tout = typeof(zero(T)/sqrt(one(T) + one(T)))
    if m == 0 || n == 0
        return similar(A, Tout, (n, m))
    end
    if isdiag(A)
        ind = diagind(A)
        dA = view(A, ind)
        maxabsA = maximum(abs, dA)
        tol = max(rtol * maxabsA, atol)
        B = fill!(similar(A, Tout, (n, m)), 0)
        B[ind] .= (x -> abs(x) > tol ? pinV(x) : zero(x)).(dA)
        return B
    end
    println("///////")
    println(typeof(A))
    SVD         = svd(A)
    tol         = max(rtol*maximum(SVD.S), atol)
    Stype       = eltype(SVD.S)
    Sinv        = fill!(similar(A, Stype, length(SVD.S)), 0)
    index       = SVD.S .> tol
    Sinv[index] .= pinV.(view(SVD.S, index))
    return SVD.Vt' * (Diagonal(Sinv) * SVD.U')
end

function pinV(x::Number)
    xi = inv(x)
    return ifelse(isfinite(xi), xi, zero(xi))
end



"""
    svd(A; full::Bool = false, alg::Algorithm = default_svd_alg(A)) -> SVD

    Compute the singular value decomposition (SVD) of `A` and return an `SVD` object.

    `U`, `S`, `V` and `Vt` can be obtained from the factorization `F` with `F.U`,
    `F.S`, `F.V` and `F.Vt`, such that `A = U * Diagonal(S) * Vt`.
    The algorithm produces `Vt` and hence `Vt` is more efficient to extract than `V`.
    The singular values in `S` are sorted in descending order.

    Iterating the decomposition produces the components `U`, `S`, and `V`.

    If `full = false` (default), a "thin" SVD is returned. For a ``M
    \\times N`` matrix `A`, in the full factorization `U` is `M \\times M`
    and `V` is `N \\times N`, while in the thin factorization `U` is `M
    \\times K` and `V` is `N \\times K`, where `K = \\min(M,N)` is the
    number of singular values.

    If `alg = DivideAndConquer()` a divide-and-conquer algorithm is used to calculate the SVD.
    Another (typically slower but more accurate) option is `alg = QRIteration()`.

    !!! compat "Julia 1.3"
        The `alg` keyword argument requires Julia 1.3 or later.

    # Examples
    ```jldoctest
    julia> A = rand(4,3);

    julia> F = svd(A); # Store the Factorization Object

    julia> A ≈ F.U * Diagonal(F.S) * F.Vt
    true

    julia> U, S, V = F; # destructuring via iteration

    julia> A ≈ U * Diagonal(S) * V'
    true

    julia> Uonly, = svd(A); # Store U only

    julia> Uonly == U
    true
    ```
"""

function svD(A::StridedVecOrMat{T}; full::Bool = false, alg::Algorithm = default_svd_alg(A)) where {T}
    svD!(copy_oftype(A, eigtype(T)), full = full, alg = alg)
end




copy_oftype(A::AbstractArray{T}, ::Type{T}) where {T} = copy(A)
copy_oftype(A::AbstractArray{T,N}, ::Type{S}) where {T,N,S} = convert(AbstractArray{S,N}, A)
