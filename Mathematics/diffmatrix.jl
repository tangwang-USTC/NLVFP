
"""
   The code implements two strategies for enhanced accuracy suggested by W. Don and S. Solomonoff in
      SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).

   The two strategies are
     (a) the use of trigonometric identities to avoid the computation of differences `x(k)-x(j)` directly and
     (b) the use of the "flipping trick" which is necessary since `sin(t)` can be computed
         to high relative precision when `t` is small whereas sin (pi-t) cannot.

 Note added May 2003:
   It may, in fact, be slightly better not to implement the strategies (a) and (b).
   Please consult the following paper for details:

   "Spectral Differencing with a Twist", by
   R. Baltensperger and M.R. Trummer, to appear in SIAM J. Sci. Comp.

  Outputs:
    Dc = chebyshevdiff(n; M = 1,datatype=Float64)
    v, Dc = chebyshevdiff(v, n; M = 1,datatype=Float64)

"""

function chebyshevdiff(n::Int; M::Int=1,datatype::DataType=Float64)

    M > 0 || ArgumentError(throw("M must be a positive number"))

    T = BigFloat
    domain = [-1.0,1.0] |> Vector{T}
    LI = eye(Bool,n)
    n1 = floor(n/2) |> Int
    n2 = ceil(n/2) |> Int   # Indices used for flipping trick
    piBig = π |> T

    # Compute `θ` vector
    k = 0:n-1 |> Vector{Int}
    θ = k * (piBig / (n-1))
    # Compute Chebyshev points
    # v = sin.(piBig / (2(n-1)) * (n-1:-2:1-n))
    v = cheby_extrema(n,domain)
    S = repeat(θ/2,1,n)

    # Trigonometrix identity for `cos(kπ - cos(jπ))`, P17 / P481
    Dx = 2 * sin.(S' + S) .* sin.(S' - S)
    Dx2 = [Dx[1:n1,:]

    # Flipping trick when `nc` is even.  P17
    - reverse!(Dx[1:n2,:])]
    Dx[LI] = ones(n,1)

    # Toeplitz maxtix which is a non-symmetric matrix as:
    C = SymmetricToeplitz((-1.0).^k) |> Array{Float64}
    C[1,:] *= 2  # Entries c(k) / c[j]
    C[n,:] *= 2
    C[:,1] /= 2
    C[:,n] /= 2
    Z = 1 ./ Dx   # Which contains entries 1 / (v[k] - v[j])
    Z[LI] = zeros(T,n,1)

    if M == 1
        Dm = eye(T,n)      # Create the differences matrices
        for m in 1:M
            Dm = m * Z .* (C .* repeat(diag(Dm),1,n) - Dm)  # off-diagonals
            Dm[LI] .= - sum(Dm';dims=1)[:]                  # Correct main diagonal of Dm
            Dm[:,:,m] =  Dm
        end
        return Dm |> Array{datatype}
    else
        Dm = eye(T,n)
        D = zeros(datatype,n,n,M)
        for m in 1:M
            Dm = m * Z .* (C .* repeat(diag(Dm),1,n) - Dm)  # off-diagonals
            Dm[LI] .= - sum(Dm';dims=1)[:]                  # Correct main diagonal of Dm
            D[:,:,m] =  Dm
        end
        return D
    end
end

function chebyshevdiff(v::AbstractVector,n::Int; M::Int=1,datatype::DataType=Float64)

    M > 0 || ArgumentError(throw("M must be a positive number"))

    T = BigFloat
    domain = [-1.0,1.0] |> Vector{T}
    LI = eye(Bool,n)
    n1 = floor(n/2) |> Int
    n2 = ceil(n/2) |> Int   # Indices used for flipping trick
    piBig = π |> T
    # Compute `θ` vector
    k = 0:n-1 |> Vector{Int}
    θ = k * (piBig / (n-1))
    # Compute Chebyshev points
    # v = sin.(piBig / (2(n-1)) * (n-1:-2:1-n))
    v = cheby_extrema(n,domain)
    S = repeat(θ/2,1,n)
    # Trigonometrix identity for `cos(kπ - cos(jπ))`, P17 / P481
    Dx = 2 * sin.(S' + S) .* sin.(S' - S)
    Dx2 = [Dx[1:n1,:];
    - reverse(Dx[1:n2,:])]         # Flipping trick.  P17
    Dx[LI] = ones(n,1)
    # Toeplitz maxtix which is a non-symmetric matrix as:
    C = SymmetricToeplitz((-1.0).^k) |> Array{Float64}
    C[1,:] *= 2  # Entries c(k) / c[j]
    C[n,:] *= 2
    C[:,1] /= 2
    C[:,n] /= 2
    Z = 1 ./ Dx   # Which contains entries 1 / (v[k] - v[j])
    Z[LI] = zeros(T,n,1)
    # Z[LI] = zeros(n,1)
    if M == 1
        Dm = eye(T,n)      # Create the differences matrices
        for m in 1:M
            Dm = m * Z .* (C .* repeat(diag(Dm),1,n) - Dm)  # off-diagonals
            Dm[LI] .= - sum(Dm';dims=1)[:]                  # Correct main diagonal of Dm
            Dm[:,:,m] =  Dm
        end
        return v, Dm |> Array{datatype}
    else
        Dm = eye(T,n)
        D = zeros(datatype,n,n,M)
        for m in 1:M
            Dm = m * Z .* (C .* repeat(diag(Dm),1,n) - Dm)  # off-diagonals
            Dm[LI] .= - sum(Dm';dims=1)[:]                  # Correct main diagonal of Dm
            D[:,:,m] =  Dm
        end
        return v, D
    end
end

"""
  Dm = laguerrediff(v,M,b)  # Dm[1:nv,1:nv,1:M]

  Differentiation Matrix of orthogonal polynomials such as
  Chebyshev, Laguerre, Hermite and Legendre polynomials.

  Laguerre needs the Gauss-Laguerre-Radau points with v[1] = 0

  Input:
   nv: Number of points, i.e., order of differentiation matrices (integer).
        ∈ [0,v1,v2,...vₙ], include the left endpoint,
        if not, will be added automatically.
   M: (=1,default) ,0 < M < nv - 1, Number of derivatives required (integer).
   b: (=1,default) Scaling parameter (real, positive).

  Outputs:
   Dm = Dm[1:nv,1:nv,1:M] contains kᵗʰ derivative matrix, where k = 1:M
      for example, Dm1 =  DM[:,:,1] = dLn(v)/dv, k = 1.
      which include the left endpoint, v=0

   Dv = laguerrediff(v; M = 1, b = 1)
   Dm = legendrediff(μ; M = 1, b = 1)
   Dc = chebyshevdiff(v; M = 1)

  Wang Yanpeng, 2020/11/19
"""

function laguerrediff(v::AbstractVector{T}; M::Int=1, b::Real=1) where {T}

    b > 0 || ArgumentError(throw("b must be a positive number"))
    M > 0 || ArgumentError(throw("M must be a positive number"))
    # Gauss-Laguerre-Radau points with v[1] = 0
    if v[1] ≤ 10eps(T)
        nv = length(v)
        wv = exp.(-v/2)        # Compute the weights
                               # Set up the β matrix
        β = zeros(T,nv,M)        # kᵗʰ derivative of w(vⱼ) / w(vⱼ)
                               # βᵏⱼ = wᵏ(vⱼ) / w(vⱼ) , k = 1:M
        for k = 1:M
            β[:,k] .= (-1//2)^k
        end
        # kᵗʰ order differentiation matrices where k = 1:M
        Dv = polydiff(v,wv,β)   # Compute the differentiation matrix when b = 1
        if b ≠ 1
            for k in 1:M
                Dv[:,:,k] = b^k * Dv[:,:,k]
            end
        end
        return Dv
    else                     # Gauss-Laguerre collection
        n = length(v)
        Lp = laguerrePoly(n-1,v)
        D = (v.^2 ./ Lp) * (Lp ./ v)' - (v ./ Lp) * Lp'
        D += eye(n)
        D = 1 ./ D
        D -= eye(n)
        D += diagm(1/2 .- 1 ./ (2v))
        if M == 1
            return D
        elseif M == 2
            Dm = zeros(T,n,n,M)
            Dm[:,:,1] = D
            Dm[:,:,2] = D * D
            return Dm
        else
        end
    end

end

function legendrediff(μ::AbstractVector{T}; M::Int=1, b::Real=1) where {T}

    b > 0 || ArgumentError(throw("b must be a positive number"))
    M > 0 || ArgumentError(throw("M must be a positive number"))

    # kᵗʰ order differentiation matrices where k = 1:M
    Dm = polydiff(μ,M)   # Compute the differentiation matrix when b = 1
    if b ≠ 1
        μ = μ / b
        for k in 1:M
            Dm[:,:,k] = b^k * Dm[:,:,k]
        end
    end
    return Dm
end

"""
  Kerl for differentiation Matrix of orthogonal polynomials such as
     Chebyshev, Laguerre, Hermite and Legendre polynomials.

  Inputs:
   v: Vector (with length nv) of the points for the differentiation matrices.
   w: Vector of the weight values w(v).
   β: Matrix of size nv × M, where M is the highest derivative required.

      βᵏⱼ = wᵏ(vⱼ) / w(vⱼ) , k = 1:M

  Outputs:
   Dm = Dm[1:nv,1:nv,1:M] contains kᵗʰ derivative matrix, where k = 1:M
      for example, Dm1 =  DM[:,:,1] = dLn(v)/dv.

  Wang Yanpeng, 2020/11/19
"""

function polydiff(v::AbstractVector{T},wv::AbstractVector{T},β::AbstractArray{T,N}) where{T,N}

    nv = length(v)
    M = length(β[1,:])        # β = Array[nv,M]
    Dm = zeros(T,nv,nv,M)
    vv = repeat(v,1,nv)
    Dv = vv - vv'             # Dv contains entries v(k) - v(j)
                              # diag(Dv) .= 1
    for i in 1:nv
        Dv[i,i] = 1
    end
                              # Quantities c[j]
    c = wv .* prod(Dv,dims = 2)
    C = repeat(c,1,nv)

    C = C ./ C'               # Matrix with entries c[k]/c[j]
    Z = 1 ./ Dv               # Z contains entries 1 / (x[k] - x[j])
    Zdia = zeros(T,(nv-1)*nv)     # V = Z[remove(diag(Z))] = Z[nv,nv-1]
    iv = 0
    for i in 1:nv
        for j in 1:nv
            if j ≠ i
                iv += 1
                Zdia[iv] = Z[i,j]
            end
        end
    end
    V = reshape(Zdia,nv-1,nv)
    Y = ones(T,nv-1,nv)          # Y is matrix of cumulative sums
    D = eye(T,nv)          # D is the differentiation matrix of single order of k

    for k in 1:M
        # Diagnoals
        Y = cumsum([transpose(β[:,k]); k * Y[1:nv-1,:] .* V],dims=1)
        D = k*Z .* (C .* repeat(diag(D),1,nv) - D) # Off-diagonals
                            # Correct the diagonals
        for i in 1:nv
            D[i,i] = Y[nv,i]
        end
        Dm[:,:,k] = D       # Store the current D
    end
    return Dm
end

"""
  wₙ(v) := 1 = one(v)
  βᵏⱼ   := 1 =
"""

function polydiff(v::AbstractVector{T}, M::Int) where {T}

    nv = length(v)
    wv = ones(T,nv)        # β = Array[nv,M]
    β = zeros(T,nv,M)
    ####
    Dm = zeros(T,nv,nv,M)
    vv = repeat(v,1,nv)
    Dv = vv - vv'             # Dv contains entries v(k) - v(j)
                              # diag(Dv) .= 1
    for i in 1:nv
        Dv[i,i] = 1
    end
                              # Quantities c[j]
    c = wv .* prod(Dv,dims = 2)
    C = repeat(c,1,nv)

    C = C ./ C'               # Matrix with entries c[k]/c[j]
    Z = 1 ./ Dv               # Z contains entries 1 / (x[k] - x[j])
    Zdia = zeros(T,(nv-1)*nv)     # V = Z[remove(diag(Z))] = Z[nv,nv-1]
    iv = 0
    for i in 1:nv
        for j in 1:nv
            if j ≠ i
                iv += 1
                Zdia[iv] = Z[i,j]
            end
        end
    end
    V = reshape(Zdia,nv-1,nv)
    Y = ones(T,nv-1,nv)          # Y is matrix of cumulative sums
    D = eye(T,nv)          # D is the differentiation matrix of single order of k

    for k in 1:M
        # Diagnoals
        Y = cumsum([transpose(β[:,k]); k * Y[1:nv-1,:] .* V],dims=1)
        D = k*Z .* (C .* repeat(diag(D),1,nv) - D) # Off-diagonals
                            # Correct the diagonals
        for i in 1:nv
            D[i,i] = Y[nv,i]
        end
        Dm[:,:,k] = D       # Store the current D
    end
    return Dm
end

"""
  The function p = polint(vk, fk, vitp) computes the polynomial interpolant
   of the data (vk, fk).  Two or more data points are assumed

  Input  (constant weight)
   vk:    Vector of v-coordinates of data (assumed distinct).
   fk:    Vector of y-coordinates of data.
   vitp:  Vector of v-values where polynomial interpolant is to be evaluated.

  Outputs:
   Vector of interpolated values.

 The code implements the barycentric formula; see page 252 in
  P. Henrici, Essentials of Numerical Analysis, Wiley, 1982.
"""
# vk = vbhat
# fk = Hvf
# vitp = va
# Not good for general case
function polyinterp(vk::AbstractVector{T},fk,vitp) where {T}
    n = length(vk)
    mv = length(vitp)
    Dv = zeros(T,n,n)
    # Dv[:,:] = repeat(vk,1,n) # Change the DataType
    for id in 1:n
        Dv[:,id] = vk
    end
    Dv = Dv - Dv'
    for i in 1:n
        Dv[i,i] = 1
    end
    pDv = prod(Dv,dims=1)
    wv = 1 ./ pDv[:]        # Compute the weights wv(k)
    ##  Compute quantities v-v(k) and their reciprocals
    D = repeat(vitp,1,n) - transpose(repeat(vk,1,mv))
    D = 1 ./ (D + eps(T) * (D .== 0))
    # Evaluate interpolant as matrix-vector products.
    return D * (wv.*fk) ./ (D*wv)
end
# fitp = D * (wv.*fk) ./ (D*wv)
