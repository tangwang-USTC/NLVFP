using LinearMaps, FiniteDifferences
using Test
using LinearAlgebra
using SparseArrays
using Random, Format
using IterativeSolvers
fmtf2 = generate_formatter( "%1.2e" )
fmtf4 = generate_formatter( "%1.4e" )
fmtf6 = generate_formatter( "%1.6e" )

# cd(path)
include(joinpath(path,"src\\solver\\iterations\\gmresJF.jl"))

n = 7
a = randn(n,n)
x = randn(n)
x0 = x
a0 = 1a
f(x) = a0 * x
b0 =  f(x)
Jx(x) = jacobian(central_fdm(5,1),f,x)[1]
A = Jx(x)
R2(x) = b0 - Jx(x) * x
## ########### 𝕁 δx = - R(x), R(x) = b - Ax = f(x) - Jx * x
# R1(x) = f(x)- Jx(x) * x
# vk1 = R1(x) / norm(R1(x))
# JRx1 = (R1(x + ϵ * vk1) - R1(x)) / ϵ
# ## ##
# vk2 = R2(x) / norm(R2(x))
# JRx2 = (R2(x + ϵ * vk2) - R2(x)) / ϵ
##
abstol = 1e-12
reltol = 1e-5
maxiterN = 20
ϵ = 1e-5       # for Maxtrix-Vector products, Jv

ϵ0 = 1e-1
X0 = (1 + ϵ0) * x0
xn = 1 * X0
R0 = norm(R2(xn))
Rm = 1 * R0
ϵt = abstol + reltol * R0
#
println()
println("kn=",00,",ϵt=",ϵt,",xn=",fmtf6.(x0))
x1, hist1 = gmres!(xn, R2; ϵ = ϵ, reltol = reltol,maxiterN=maxiterN,log = false)
Rn = R2(xn)
println("Rn=",fmtf4.(Rn))
D =  A * xn - b0
println("ϵ=",ϵ,",D=",fmtf2.(D))
