
"""
  Computing the first-order differentiation matrix of GLPs associated with
  the Gauss-Laguerre points (or Gauss-Laguerre-Radau points) `v`, which may be computed by function

  v, wv = laguerre(n,α,endptv).

    endptv is neither for Gauss-Laguerre points and
              left for    Gauss-Laguerre-Radau points.

  Use the function: Lp = laguerrePoly(n,v)

  See
   Page 251 and 252 of the book: J. Shen, T. Tang and L. Wang, Spectral Methods:
   Algorithms, Analysis and Applications, Springer Series in Compuational
   Mathematics, 41, Springer, 2011.

  Last modified on 2021/12/04

  Outputs:

    D = laguerrePolydiff(v;α=α)
"""

function laguerrePolydiff(v::AbstractVector{T};α::Real=0.0) where {T}

    # b > 0 || ArgumentError(throw("b must be a positive number"))
    # M > 0 || ArgumentError(throw("M must be a positive number"))
    N = length(v)
    if α == 0.0
        # Gauss-Laguerre-Radau points with v[1] = 0
        if v[1] ≤ 10eps(T)
            D = zeros(T,N,N)
            nn = N - 1
            nvec = 2:N
            Lp = laguerrePoly(nn,v)
            # k = j = 0
            D[1,1] = - nn / 2
            # k = 0, j > 0
            D[1,nvec] = - 1.0 ./ (v[nvec] .* Lp[nvec])
            # k > 0, j = 0
            D[nvec,1] = Lp[nvec] ./ v[nvec]
            # k ≠ j > 0
            D[nvec,nvec] = (v[nvec] ./ Lp[nvec]) * Lp[nvec]' - (1.0 ./ Lp[nvec]) * transpose(v[nvec] .* Lp[nvec])
            D[nvec,nvec] += eye(nn)
            D[nvec,nvec] = 1 ./ D[nvec,nvec]
            D[nvec,nvec] -= eye(nn)
            D[nvec,nvec] += eye(nn) / 2
            return D
            #
        else                     # Gauss-Laguerre collection
            Lp = laguerrePoly(N-1,v)
            D = (v.^2 ./ Lp) * (Lp ./ v)' - (v ./ Lp) * Lp'
            D += eye(N)
            D = 1 ./ D
            D -= eye(N)
            D += diagm(1/2 .- 1 ./ (2v))
            return D
        end
    else
    end
end

"""
  Computing the first-order differentiation matrix of GLFs associated with
  the Gauss-Laguerre points (or Gauss-Laguerre-Radau points) `v`, which may be computed by function

  v, ~ = laguerre(n;α=α,endptv=endptv) and

    wv =

    endptv is neither for Gauss-Laguerre points and
              left for    Gauss-Laguerre-Radau points.

  Use the function: Lp = laguerreFun(n,v)

  See
   Page 252 of the book: J. Shen, T. Tang and L. Wang, Spectral Methods:
   Algorithms, Analysis and Applications, Springer Series in Compuational
   Mathematics, 41, Springer, 2011.

  Last modified on 2021/12/04

  Outputs:

    D = laguerreFundiff(v;α=α)
"""

function laguerreFundiff(v::AbstractVector{T};α::Real=0.0) where {T}

    # b > 0 || ArgumentError(throw("b must be a positive number"))
    # M > 0 || ArgumentError(throw("M must be a positive number"))
    N = length(v)
    if α == 0.0
        # Gauss-Laguerre-Radau points with v[1] = 0
        if v[1] ≤ 10eps(T)
            D = zeros(T,N,N)
            nn = N - 1             # j, k ∈ 0:N
            nvec = 2:N
            Lp = laguerreFun(nn,v)
            # k = 0
            D[1,1] = - nn / 2.0 - 0.5
            D[1,nvec] = - 1.0 ./ (v[nvec] .* Lp[nvec])
            # k > 0, j = 0
            D[nvec,1] = Lp[nvec] ./ v[nvec]
            # k ≠ j > 0
            D[nvec,nvec] = (v[nvec] ./ Lp[nvec]) * Lp[nvec]' - (1.0 ./ Lp[nvec]) * transpose(v[nvec] .* Lp[nvec])
            D[nvec,nvec] += eye(nn)
            D[nvec,nvec] = 1 ./ D[nvec,nvec]
            D[nvec,nvec] -= eye(nn)
            #####################################
            # D[nvec,nvec] += eye(nn) / 2
            # D -= eye(N) / 2
            return D
        else                     # Gauss-Laguerre collection
            Lp =laguerreFun(N-1,v)
            D = (v.^2 ./ Lp) * (Lp ./ v)' - (v ./ Lp) * Lp'
            D += eye(N)
            D = 1.0 ./ D
            D -= eye(N)
            D += diagm(- 1.0 ./ (2v))     # different with LaguerrePolydiff
            return D
        end
    else
    end
end

"""
  Computing the first-order differentiation matrix of GLFs in Frequency space.
  The expansion coefficients of `f(v)` are given as `fn` which will be computed by function

    fn = laguerre(n;α=α,endptv=endptv)

  See
   Page 253 of the book: J. Shen, T. Tang and L. Wang, Spectral Methods:
   Algorithms, Analysis and Applications, Springer Series in Compuational
   Mathematics, 41, Springer, 2011.

  Last modified on 2021/12/04

  Outputs:

    dfn = laguerreFunFrediff(fn)
    dfn = laguerrePolyFrediff(fn)

    dᵥf(v) = L̂ₙ(v) * dfn
"""

function laguerreFunFrediff(fn::AbstractVector{T}) where {T}

    N = length(fn)
    dfn = zeros(T,N)
    # i = N
    dfn[N] = - 0.5fn[N]
    for i in N-1:-1:1
        dfn[i] = dfn[i+1] - 0.5(fn[i] + fn[i+1])
    end
    return dfn
end

function laguerrePolyFrediff(fn::AbstractVector{T}) where {T}

    N = length(fn)
    dfn = zeros(T,N)
    # i = N
    dfn[N] = 0.0
    for i in N-1:-1:1
        dfn[i] = dfn[i+1] - fn[i+1]
    end
    return dfn
end

"""
  The discrete Gauss-Laguerre tansforming (in Gauss-Laguerre Functions, GLFs)
  between the Physics space (`f(v)`) and Frequency space (`fn` by Gauss-Laguerre functions expansions).

  The expansion coefficients of `f(v)` are given as `fn` which will be computed by function

    fn =

  at the Gauss-Laguerre-Radau points, `vG` which will be solved by:

    vG, wv = laguerre(n;α=α,endptv=left)

    and ŵv = wv * exp(vG).

  Here, the Gauss-Laguerre Functions, GLFs, will be:

    L̂ₙᵃ(v) = exp(-v/2) * Lₙᵃ(v),

  which will be calculated by:

    Lₙᵃ(v) = laguerreFun(n,v), where n = vG - 1 = length(v) - 1

  See
   Page 250 of the book: J. Shen, T. Tang and L. Wang, Spectral Methods:
   Algorithms, Analysis and Applications, Springer Series in Compuational
   Mathematics, 41, Springer, 2011.

  Last modified on 2021/12/04

  Inputs:
     n: Number of Laguerre-Gauss-Radau points in v;
     v:
     ŵv: , where (x,w) can be computed by x,wv = laguerre(n;α=α,endptv=left) and
         ŵv = wv * exp(vG)

     iflag: (=true, defalt) which means the forward transforms where
       f: (input) physical values at collocation points and output is expansion coefficients.

         If iflag = false means backward transforms where
         f (input) is expansion coefficients and output will be physical values at collocation points.

  Outputs:
    fn = laguerreTransforms(N,v,wv,f;iflag=true)
"""

function laguerreTransforms(N::Int,v::AbstractVector{T},wv::AbstractVector{T},f::AbstractVector{T};iflag::Bool=true,polyfun=fun) where {T}

    if polyfun == true
        Lg = laguerreFunm(N-1,v)
    else
        Lg = laguerrePolym(N-1,v)
    end
    # f(v) → fn
    if iflag == 1
        # fn = (Lg .* ŵv) * f
        # InG = (Lg .* ŵv) * Lg' = Lg' * (Lg .* ŵv)
        return Lg * f .* wv
    else
        # f = Lg' * f
        return Lg' * f
    end
end
