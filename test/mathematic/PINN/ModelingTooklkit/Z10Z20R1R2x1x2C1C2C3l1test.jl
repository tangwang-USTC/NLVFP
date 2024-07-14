
using DifferentialEquations, ModelingToolkit
using Plots
using ModelingToolkit: t_nounits as t
# using ModelingToolkit: t_nounits as t, D_nounits as Dt

# @parameters cc, R0, f1, f2, l1

cc = 3e8
R0 = 50.0
f1 = 81.36e6
f2 = 1.6272e8

w1 = 2π * f1
w2 = 2π * f2
lam(f,l) = l / (cc / f)
# lam1 = l1 / (cc / f1)
# lam2 = l1 / (cc / f2)
@variables x(t) y(t) z(t) l1(t)           # C1, C2, C3, l1
# Dt = Differential(t)

function Tunsz(Z,f,l;R0=R0)

    return R0 * ((Z .+ im * R0 * tan.(2π * lam(f,l)))) ./ (R0 .+ im * Z * tan.(2π * lam(f,l)))
end

Z1(C3) = R0 + 1 / (im * w1 * C3)
Z3(C2) = im * w1 * C2
Z2(C3,l1) = Tunsz(Z1(C3),f1,lam(f1,l1);R0=R0)
Z4(C2,l1) = Tunsz(Z3(C2),f1,lam(f2,l1);R0=R0)
Z5(C2,C3,l1) = Z2(C3,l1) * Z4(C2,l1) / (Z2(C3,l1) + Z4(C2,l1))
Z6(C2,C3,l1) = Tunsz(Z5(C2,C3,l1),f1,lam(f1,l1);R0=R0)
Z7(C1) = im * w1 * C1
Z8(C1,l1) = Tunsz(Z7(C1),f1,lam(f2,l1);R0=R0)
Z9(C1,C2,C3,l1) = Z8(C1,l1) * Z6(C2,C3,l1) / (Z8(C1,l1) + Z6(C2,C3,l1))
Z10(C1,C2,C3,l1) = Tunsz(Z9(C1,C2,C3,l1),f1,l1 + 2 * lam(f2,l1);R0=R0)

Z11(C3) = R0 + 1 / (im * w2 * C3)
Z13(C2) = im * w2 * C2
Z12(C3,l1) = Tunsz(Z11(C3),f2,lam(f1,l1);R0=R0)
Z14(C2,l1) = Tunsz(Z13(C2),f2,lam(f2,l1);R0=R0)
Z15(C2,C3,l1) = Z12(C3,l1) * Z14(C2,l1) / (Z12(C3,l1) + Z14(C2,l1))
Z16(C2,C3,l1) = Tunsz(Z15(C2,C3,l1),f2,lam(f1,l1);R0=R0)
# Z17(C1) = im * w2 * C1
function Z17(C1)
    
    println(rand(1))
    im * w2 * C1
end
Z18(C1,l1) = Tunsz(Z17(C1),f2,lam(f2,l1);R0=R0)
Z19(C1,C2,C3,l1) = Z18(C1,l1) * Z16(C2,C3,l1) / (Z18(C1,l1) + Z16(C2,C3,l1))
Z20(C1,C2,C3,l1) = Tunsz(Z19(C1,C2,C3,l1),f2,l1 + 2 * lam(f2,l1);R0=R0)

Z10Re(C1,C2,C3,l1) = real(Z10(C1,C2,C3,l1))
Z10Im(C1,C2,C3,l1) = imag(Z10(C1,C2,C3,l1))
Z20Re(C1,C2,C3,l1) = real(Z20(C1,C2,C3,l1))
Z20Im(C1,C2,C3,l1) = imag(Z20(C1,C2,C3,l1))
# function Z20Im(C1,C2,C3,l1)
    
#     println(rand(1))
#     imag(Z20(C1,C2,C3,l1))
# end

l1 = 1e-2
C1 = 1.0e-1
C2 = 1.0e-1
C3 = 1.0e-1
aaa = -7:0.01:5
Cn = 3
if Cn == 1
    C1 = exp.(aaa)
    xlabel = "C1"
    Cvec = C1
elseif Cn == 2
    C2 = exp.(aaa)
    xlabel = "C2"
    Cvec = C2
elseif Cn == 3
    C3 = exp.(aaa)
    xlabel = "C3"
    Cvec = C3
else
    l1 = exp.(aaa)
    xlabel = "l1"
    Cvec = l1
end
# xscale = :norm
xscale = :log10
yscale = :norm
ylabel = "R1(C1,C2,C3,l1)"
pZ10Re = plot(Cvec, Z10Re.(C1,C2,C3,l1),xscale=xscale,yscale=yscale,
                xlabel=xlabel,ylabel=ylabel)
ylabel = "x1(C1,C2,C3,l1)"
pZ10Im = plot(Cvec, Z10Im.(C1,C2,C3,l1),xscale=xscale,yscale=yscale,
                xlabel=xlabel,ylabel=ylabel)
ylabel = "R2(C1,C2,C3,l1)"
pZ20Re = plot(Cvec, Z20Re.(C1,C2,C3,l1),xscale=xscale,yscale=yscale,
                xlabel=xlabel,ylabel=ylabel)
ylabel = "x2(C1,C2,C3,l1)"
pZ20Im = plot(Cvec, Z20Im.(C1,C2,C3,l1),xscale=xscale,yscale=yscale,
                xlabel=xlabel,ylabel=ylabel)
display(plot(pZ10Re,pZ20Re,pZ10Im,pZ20Im,layout=(2,2)))

# R1, x1 = 10.0, 3.0
# R2, x2 = 18.0, 1.0

# eqs = [0 ~ Z10Re(x,y,z,l1) - R1,
#        0 ~ Z10Im(x,y,z,l1) - x1,
#        0 ~ Z20Re(x,y,z,l1) - R2,
#        0 ~ Z20Im(x,y,z,l1) - x2]

# # @named de = ODESystem(eqs,t,[x,y,z],[σ,ρ,β],tspan=(0, 10.0))

# @mtkbuild sys = ODESystem(eqs, t)

# u0 = [x => 1.0,      # 
#       y => 0.2,
#       z => 0.1,
#       l1 => 1.0]

# # p = [cc => 3e8,
# #     R0 => 50.0,
# #     f1 => 81.36e6,
# #     f2 => 1.6272e8]

# p = []

# tspan = (0.0, 1e-5)
# # prob = ODEProblem(sys, u0, tspan, p)
# prob = ODEProblem(sys, u0, tspan, p, jac = true)

# sol = solve(prob)

# title = "ModelingToolkit"
# pt = plot(sol,line=(1,:auto),title=title)
# pxyt = plot(sol, idxs = (x, y))
# display(plot(pt,pxyt,layout=(2,1)))
