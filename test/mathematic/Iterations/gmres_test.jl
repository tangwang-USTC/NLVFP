using LinearMaps, FiniteDifferences
using Test
using LinearAlgebra
using SparseArrays
using Random, Format
using IterativeSolvers
fmtf2 = generate_formatter( "%1.2e" )
fmtf4 = generate_formatter( "%1.4e" )
fmtf6 = generate_formatter( "%1.6e" )

path = "G:\\atom\\julia\\MVFP2D3V"
# cd(path)
include(joinpath(path,"src//solver//iterations//gmresJF.jl"))
# include(joinpath(path,"src//solver//iterations//gmres.jl"))
# include(joinpath(path,"src//solver//iterations//gmres0.jl"))
# include(joinpath(path,"src//solver//iterations//JFNKgmres.jl"))

n = 7
a = randn(n,n)
x = randn(n)
x0 = x
a0 = 1a
f(x) = a0 * x

dfx = a * (x ./ x)
A = jacobian(central_fdm(5,1),f,x)[1]
b0 =  f(x)

 restart = min(20, length(x0))
 maxiter = length(x0)
 initially_zero = true
## ########### A x = b
δx = zeros(n) + 0 * x0
# x1, hist1 = gmres(A, b0, log = true) # initially_zero = true
# x1, hist1 = gmres!(δx, A, b0, log = true) # initially_zero = true
## ########### 𝕁 δx = - R(x), R(x) = b - Ax = f(x) - Jx * x
δx = 0 * x0
ϵ = 1e-5
X0 =  x0 * (1 - ϵ)
Jx(x) = jacobian(central_fdm(5,1),f,x)[1]
R(x) = f(x) - Jx(x) * x
Rx = jacobian(central_fdm(5,1),R,x)[1]
vk = R(x) / norm(R(x))
JRx = Rx * vk
## ##
R1(x) = f(x)- Jx(x) * x
vk1 = R1(x) / norm(R1(x))
JRx1 = (R1(x + ϵ * vk1) - R1(x)) / ϵ
# ## ##
R2(x) = b0 - Jx(x) * x
vk2 = R2(x) / norm(R2(x))
JRx2 = (R2(x + ϵ * vk2) - R2(x)) / ϵ
##
x1 = 0 * x0
ϵ = 1e-2
X0 = (1 + ϵ) * x0
xn = 1 * X0
R0 = norm(R2(xn))
Rm = 1 * R0
abstol = 1e-12
reltol = 1e-4
ϵt = abstol + reltol * R0
#
println()
# δx = 0 * x0
# iter = gmres_iterable!(δx, x0, R2; initially_zero = true)
x1, hist1 = gmres!(xn, R2; log = true, initially_zero = true)
Rn = R2(xn)
println("Rn=",fmtf4.(Rn))
D =  A * xn - b0
println("ϵ=",ϵ,",D=",fmtf2.(D))
## ##
# x1, hist1 = gmres!(δx, X0, f, log = true, initially_zero = true) # initially_zero = true
# x1 = gmres_iterable!(δx, X0, f, initially_zero = true) # initially_zero = true
# x10 = gmres_iterable!(δx, A, b0)
# x1, hist1 = gmres!(x,A, b0, log = true)
# δx = zeros(n)
# x2, hist2 = idrs!(δx, A, b0, log = true)
# return
# dgeeeeeeeeeeeeeeeeeeeeeeeeee
# # egwg
# ## ################
# # J δx = - f
# # R = f
# # x = x0 + 0.1 * randn(n)
# R(x) = a * x
# X0 = 1 * x
# δx = 0 * x
# # gmres!(δx, X0, R)
# # x3 = jfnkgmres(x,R)
# x = x0 + 0.1 * randn(n)
# # x4 = gmres!(x,A,b0)
# x4 = jfnkgmres(x,A,b0)
# # # @test hist.isconverged
# println(fmtf4.(x3./x0))
#
# # function test1(ff,x)
# #
# #     Jx = jacobian(central_fdm(5,1),ff,x)[1]
# # end
