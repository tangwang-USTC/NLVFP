"""

   Solving the Poisson equations which is second order partial differential equation with the form:

     ∂²H/∂v² + q(v) * ∂H/∂v + r(x) * H(v) = F(v),

   with two-point boundary conditions and the domain is:

     v ∈ (a,b) = (0, 10).

   Program for computing first and second derivative matrices and
   and boundary condition functions for 2 point boundary conditions

    a1 * H(a) + b1 * ∂ᵥH(a) = c1
    a9 * H(b) + b9 * ∂ᵥH(b) = c9

   Case 1: Dirichlet / Dirichlet boundary conditions

     b1 = b9 == 0, (a1, a9) is some const

   Case 2: Dirichlet / Robin boundary conditions

     b1 == 0,  b9 ≠ 0, (a1, a9) is some const

   Case 3: Robin / Dirichlet boundary conditions

     b1 ≠ 0, b9 == 0, (a1, a9) is some const

   Case 4: Robin / Robin boundary conditions

     b1, b9 ≠ 0, (a1, a9) is some const

   INPUT:
     n    =  number of Chebyshev points in `domain = [-1,1]
     g    =  boundary condition matrix = `[a1, b1, c1, a9, b9, c9]

   OUTPUT:
     xt   =  Chebyshev points corresponding to rows and columns
             of D1t and D2t
     D1t  =  1st derivative matrix incorporating g
     D2t  =  2nd derivative matrix incorporating g
     ϕ1   =  1st and 2nd derivative of g function at x=1
             (array with 2 columns)
     ϕ9   =  1st and 2nd derivative of g function at x=-1
             (array with 2 columns)

    xt,D1t,D2t,ϕ1,ϕ9 = chebyshevbc(n,g)
"""

using LinearAlgebra, LinearAlgebraX, SavitzkyGolay, ToeplitzMatrices

pathroot = "G:\\BaiduNetdiskWorkspace\\FP0D1VchebySelf"
pathdatas = "G:\\atom\\datas\\VFP0D2VchebySelf"
cd(pathroot)
include(joinpath(pathroot,"mathematics\\diffmatrix.jl"))
include(joinpath(pathroot,"mathematics\\cheby_extrema.jl"))

paraM(A) = @show size(A),rank(A), Float64.([cond(A), det(A), tr(A)])

# Two points boundary conditions
function chebyshevbc(n::Int,g::AbstractVector{T};datatype::DataType=Float64) where{T,N}

   # Get differentiation matrices
   x = zeros(T,n)
   x, DM = chebyshevdiff(x,n; M = 2,datatype=datatype)
   D0 = eye(n)
   D1 = DM[:,:,1]
   D2 = DM[:,:,2]
   # extract boundary condition coefficients
   a1, b1, c1, a9, b9, c9 = g[1], g[2], g[3], g[4], g[5], g[6]
   Pi = pi |> BigFloat
   if ((a1==0 && b1==0) | (a9==0 && b9==0))  # Case 0: Invalid boundary condition information
      ArgumentError(error("This is a single point or nature boundary condition!"))
   elseif b1==0 && b9==0 # Dirichlet / Dirichlet
      K = 2:n-1 |> Vector{Int}
      # J = K
      # D1t = D1[K,K]
      # D2t = D2[K,K]
      # ϕ1 = c1 / a1 * [D1[K,1] D2[K,1]]          # ϕ+
      # ϕ9 = c9 / a9 * [D1[K,n] D2[K,n]]          # ϕ-
      # xt = x[K]                                  # node vector
      return x[K],D1[K,K],D2[K,K],c1 / a1 * [D1[K,1] D2[K,1]],c9 / a9 * [D1[K,n] D2[K,n]]
   elseif b1≠0 && b9==0 # Dirichlet x=1, Robin x=-1

      J = 2:n-1 |> Vector{Int}
      K = 1:n-1 |> Vector{Int}
      xkcol = 2 * sin.((K .- 1) * (Pi / 2 / (n-1))).^2      # 1-xk, using trig identity
      xjinv = reshape(1 ./ (2 * sin.((J .- 1) * (Pi / 2 / (n-1))).^2),1,n-2)      # 1 / (1-xj), using trig identity
      oner = ones(length(xkcol))                # column of ones

      fac0 = oner * xjinv                # matrix -1/(1-xj)
      fac1 = xkcol * xjinv               # matrix (1-xk)/(1-xj)
      D1t = fac1 .* D1[K,J] - fac0 .* D0[K,J]
      D2t = fac1 .* D2[K,J] - 2 * fac0 .* D1[K,J]

      cfac = D1[1,1] + a1 / b1                  # compute phi'_1, phi''_1
      fcol1 = - cfac * D0[K,1] + ( 1 .+ cfac * xkcol) .* D1[K,1]
      fcol2 = - 2 * cfac * D1[K,1] + (1 .+ cfac * xkcol) .* D2[K,1]
      # D1t  = [fcol1 D1t]
      # D2t  = [fcol2 D2t]

      ϕ1= - xkcol .* D1[K,1] + D0[K,1]         # phi'_+, phi''_+
      ϕ9 = xkcol .* D1[K,n] / 2 - D0[K,n] / 2     # phi'_-, phi''_-
      # ϕ1 = c1 / b1 * [ϕ1 (- xkcol .* D2[K,1]+2 * D1[K,1])]
      # ϕ9 = c9 / a9 * [ϕ9 xkcol .* D2[K,n] / 2 - D1[K,n]]

      # xt = x[K]                             # node vector
      return x[K],[fcol1 D1t],[fcol2 D2t],c1 / b1 * [ϕ1 (- xkcol .* D2[K,1]+2 * D1[K,1])],c9 / a9 * [ϕ9 xkcol .* D2[K,n] / 2 - D1[K,n]]
      return xt,D1t,D2t,ϕ1,ϕ9
   elseif b1==0 && b9≠0 # Case 3: neumann or Robin boundary x=1 and Dirichlet at x=-1.

      J = 2:n-1
      K = 2:n |> Vector{Int}
      xkcol = 2 * cos.((K .- 1) * (Pi / 2 / (n-1))).^2      # `1+xk`, using trig identity
      xjinv = reshape(1 ./ (2 * cos.((J .- 1) * (Pi / 2 / (n-1))).^2),1,n-2)      # `1 / (1+xj)`, using trig identity
      oner = ones(length(xkcol))                  # column of ones

      fac0 = oner * xjinv                     # matrix `1/(1+xj)`
      fac1 = xkcol * xjinv                    # matrix `(1+xk)/(1+xj)`
      D1t = fac1 .* D1[K,J] + fac0 .* D0[K,J]
      D2t = fac1 .* D2[K,J] + 2 * fac0 .* D1[K,J]

      cfac = D1[n,n] + a9 / b9                # compute phi'_n, phi''_n
      lcol1 = - cfac * D0[K,n] + (1 .- cfac * xkcol) .* D1[K,n]
      lcol2 = - 2 * cfac * D1[K,n] + (1 .- cfac * xkcol) .* D2[K,n]
      # D1t  = [D1t lcol1]
      # D2t  = [D2t lcol2]

      ϕ1 = xkcol .* D1[K,1] / 2 + D0[K,1]        # compute phi'_+,phi''_+
      ϕ9 = xkcol .* D1[K,n] + D0[K,n]           # compute phi'_-,phi''_-
      # ϕ1 = c1 / a1 * [ϕ1 xkcol .* D2[K,1] / 2 + D1[K,1]]
      # ϕ9 = c9 / b9 * [ϕ9 xkcol .* D2[K,n] + 2 * D1[K,n]]

      # xt = x[K]                               # node vector
      return x[K],[D1t lcol1],[D2t lcol2],c1 / a1 * [ϕ1 xkcol .* D2[K,1] / 2 + D1[K,1]],c9 / b9 * [ϕ9 xkcol .* D2[K,n] + 2 * D1[K,n]]
      # return xt,D1t,D2t,ϕ1,ϕ9
   elseif (b1≠0 && b9≠0) # Case 4: neumann or Robin boundary conditions at both endpoints.

      J = 2:n-1  |> Vector{Int}
      K = 1:n    |> Vector{Int}
      xkcol0 = sin.((K .- 1) * (Pi / (n-1))).^2        # `1-xk^2` using trig identity
      xkcol1 = -2 * x[K]                           # -2 * xk
      xkcol2 = -2 * ones(length(xkcol0))           # -2
      xjrow = reshape(1 ./ (sin.((J .- 1) * (Pi / (n-1))).^2),1,n-2)  # `1-xj^2` using trig identity

      fac0 = xkcol0 * xjrow
      fac1 = xkcol1 * xjrow
      fac2 = xkcol2 * xjrow

      D1t = fac0 .* D1[K,J] + fac1 .* D0[K,J]
      D2t = fac0 .* D2[K,J] + 2 * fac1 .* D1[K,J]+fac2 .* D0[K,J]

      omx = sin.((K .- 1) * Pi / 2 / (n-1)).^2              # (1-xk) / 2
      opx = cos.((K .- 1) * Pi / 2 / (n-1)).^2              # (1+xk) / 2

      r0 = opx + (0.5 + D1[1,1] + a1 / b1) * xkcol0 / 2       # compute phi'_1, phi''_1
      r1 = 0.5 .- (0.5 + D1[1,1] + a1 / b1) * x
      r2 = -0.5 - D1[1,1] - a1 / b1
      rcol1 = r0 .* D1[K,1] + r1 .* D0[K,1]
      rcol2 = r0 .* D2[K,1] + 2 * r1 .* D1[K,1]+r2 .* D0[K,1]

      l0 = omx + (0.5-D1[n,n] - a9 / b9) * xkcol0 / 2       # compute phi'_n, phi''_n
      l1 = -0.5 .+ (D1[n,n] + a9 / b9-0.5) * x
      l2 = D1[n,n] + a9 / b9 - 0.5
      lcol1 = l0 .* D1[K,n] + l1 .* D0[K,n]
      lcol2 = l0 .* D2[K,n] + 2 * l1 .* D1[K,n]+l2 .* D0[K,n]

      # D1t = [rcol1 D1t lcol1]
      # D2t = [rcol2 D2t lcol2]

      phip1 = (-xkcol0 .* D1[K,1] - xkcol1 .* D0[K,1]) / 2
      phip2 = (-xkcol0 .* D2[K,1] - 2 * xkcol1 .* D1[K,1]-xkcol2 .* D0[K,1]) / 2
      phim1 = (xkcol0 .* D1[K,n] + xkcol1 .* D0[K,n]) / 2
      phim2 = (xkcol0 .* D2[K,n] + 2 * xkcol1 .* D1[K,n] + xkcol2 .* D0[K,n]) / 2
      # ϕ1 = c1 / b1 * [phip1 phip2]                # compute phi'_+, phi''_+
      # ϕ9 = c9 / b9 * [phim1 phim2]                 # compute phi'_-, phi''_-

      # xt = x[K]                                  # node vector
      return x[K],[rcol1 D1t lcol1],[rcol2 D2t lcol2],c1 / b1 * [phip1 phip2],c9 / b9 * [phim1 phim2]
      return xt,D1t,D2t,ϕ1,ϕ9
   end
end

"""
  Equation:

    P(x) * H'' + Q(x) * H' + R * H = S(x)
    P(x) = 1
    Q(x) = - 2x
    R(x) = 2
    S(x) = 4exp(-x^2)

  with boundary conditions:

    2u(a) - u'(a) = 1
    2u(b) + u'(b) = -1

  So, both Robin boundary conditions are specified at the end points.

  Inputs:
    n: number of Chebyshev grids;

  Outputs:
    xt, u = dxu224(n)
    xt, u = dxu224(2 .^(3:6))

"""

function dxu224(n::Int,g::AbstractVector{T},ff::Function;isplot::Bool=true) where{T,N}

   @time xt,D1t,D2t,ϕ1,ϕ9 = chebyshevbc(n,g)
   Qx = - 2 * xt
   Rx = 2
   p1 = ϕ1[:,2] + Qx .* ϕ1[:,1]
   p9 = ϕ9[:,2] + Qx .* ϕ9[:,1]
   D = D2t + diagm(Qx) * D1t + Rx * eye(length(xt))
   @time ff0 = D \ (ff(xt) - (p1 + p9))
   # @time ff0 = inv(D) * (ff(xt) - (p1 + p9))
   if isplot == 1
      label = string("u,nodes=2^",log2(n))
      @show n
      paraM(D1t)
      paraM(D2t)
      pp = plot(xt,ff0,label=label,line=(1,:auto))
      display(pp)
   end
   return xt, ff0
end

function dxu224(nvec::Vector{Int},g::AbstractVector{T},ff::Function) where{T,N}

   i = 1
   n = nvec[1]
   xt, ff0 = dxu224(n,g,ff;isplot=false)
   label = string("u,nodes=2^",log2(n))
   pp = plot(xt,ff0,label=label,line=(1,:auto))
   for n in nvec[2:end]
      i += 1
      xt, ff0 = dxu224(n,g,ff;isplot=false)
      label = string("u,nodes=2^",log2(n))
      pp = plot!(xt,ff0,label=label,line=(1,:auto))
   end
   display(pp)
   return xt, ff0
end

ff(x) = 4 * exp.(-x.^2)
g = [0,1,1, 0,1,-1]
xt, u = dxu224(2 .^(3:6),g,ff)
power = 4
n = 2^power
xt, u = dxu224(n,g,ff;isplot=false)
