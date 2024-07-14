using FiniteDifferences
path = "G:\\atom\\julia\\MVFP2D3V"
include(joinpath(path,"mathematics\\gradiate.jl"))
# Contruct a finite difference method at p points for qᵗʰ order direvative.

p = 5     # number of grids
q = 1     # order of derivative to estimate

## Derivatives: ∂/∂x
# Forward method
f51 = forward_fdm(p,q)
# Central method
c51 = central_fdm(p,q)
# Dealing with Singularities
c51_range = central_fdm(p,1,max_range = 1e-4)   # A range-limited central method
# Backward method
b51 = backward_fdm(p,q)
# Richardson Extrapolation
extrapolate_fdm(c51,sin,q)    # at x = 1
        # when p > p_c , Richardson may provide no futher gains.

# Method on a custom grid
gridnum = [-2,-1,0,4,5]
d1grid = FiniteDifferenceMethod(gridnum,q)

# # estimates a Qth order derivative from P function evaluations
# FiniteDifferences.UnadaptedFiniteDifferenceMethod{p,q} <: FiniteDifferences.FiniteDifferenceMethod{p,q}
# # Adapts the step size dynamically
# FiniteDifferences.AdaptedFiniteDifferenceMethod{p,q,E<:FiniteDifferenceMethod} <: FiniteDifferences.AdaptedFiniteDifferenceMethod{p,q}
# FiniteDifferences.AdaptedFiniteDifferenceMethod{p,q,E} <: FiniteDifferences.AdaptedFiniteDifferenceMethod{p,q}

# # aprroximate
# is_eq_ab = assert_approx_equal(a1,a2,Atol,Rtol)

## Gradient: ∇f(x,y) = grad(fdm,f(x,y),x,y) = ê₁*∂f/∂x + ê₂* ∂f/∂y
# # Example101: f(x) = a * x²/2 , ∂f/∂x = a * x
# nx = 3
# a = randn(nx,nx)
# a = a * a'
# f(x) = 0.5 * x' * a * x
# x = randn(nx)
# ∇f = grad(c51,f,x)[1]
# err = ∇f - a * x

# Example201: f(x,y) = y * x²/2 , ∂f/∂x = x * y,  ∂f/∂y = x²/2
nx = 3
ny = 7
f(x,y) = y.^3 .* x.^2
fy(x,y)  = 3 * y.^2 .* x.^2
fx(x,y)  = 2 * y.^3 .* x
x = collect(LinRange(1,3,nx))
y = collect(LinRange(1,5,ny)')
df = Df(c51,f,x,y)             # ∇f([x...],[y...]ᵀ)
df = Df(c51,(x,y)->f(x,y),x,y)
# a = jacobian(c51,f,x,y)

## Jacobian matrix: Matrix of all the first-order partial derivatives of a vector-valued function in several variables
# Jacobian determinant, det(J): When the matrix is square
#=
   f:Rⁿ → Rᵐ,
     J = {Jᵢⱼ} where Jᵢⱼ = ∂fᵢ/∂xⱼ

   if m = 1, that is Rⁿ → R
     J = ∇ᵀf

   f([x,y]ᵀ) = [f₁(x,y),f₂(x,y)]ᵀ

    J(x,y) = [∂f₁/∂x, ∂f₂/∂x;∂f₁/∂y, ∂f₂/∂y]
           = ∂f₁/∂x  ∂f₁/∂y
             ∂f₂/∂x  ∂f₂/∂y
=#
# jac_f = jacobian(c51,f,x)[1]
# err = jac_f - (a*x)'

##
c51 = central_fdm(5,1)     # first order derivative
b51 = backward_fdm(5,1)     # first order derivative
f51 = forward_fdm(5,1)     # first order derivative
# E2(x) = x.^5
# dEx = Df(c51,x->E2(x),x)
# dEx = Df(c51,E2,x)
## dims = 2
E2(x,y) = sin.(x) .* cos.(y)
@time dExyc = Df(c51,(x,y)->E2(x,y),x,x2)
@time dExyf = Df(f51,(x,y)->E2(x,y),x,x2)
@time dExyb = Df(b51,(x,y)->E2(x,y),x,x2)
dEx = cos.(x) .* cos.(x2)
dEy = -sin.(x) .* sin.(x2)
errxf = dEx - dExyf[1]
erryf = dEy - dExyf[2]
errx = dEx - dExyc[1]
erry = dEy - dExyc[2]



# # Method for non-scalar functions
# f(x) = x.^2
# # jac_f = jacobian(c51,f,x)[1]
# # err = jac_f - a
# #####
# nx = ny = nz = 3
# x = collect(LinRange(1,3,nx))
# y = collect(LinRange(1,5,ny)')
# f(x,y,z) = y.^3 .* x.^2 * z.^4
# fy(x,y,z) = 3 * y.^2 .* x.^2 * z.^4
# fx(x,y,z) = 2 * y.^3 .* x * z.^4
# fz(x,y,z) = 4 * y.^3 .* x.^2 * z.^3
# x = collect(LinRange(1,3,nx))
# y = zeros(1,ny)
# z = zeros(1,1,nz)
# y[:,:] = collect(LinRange(1,5,ny))
# z[:,:,:] = collect(LinRange(2,5,nz))
# a = jacobian(c51,f,x,y,z)
