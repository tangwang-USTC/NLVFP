using DifferentialEquations, ModelingToolkit
using Plots
using ModelingToolkit: t_nounits as t, D_nounits as D

@parameters σ ρ β
@variables t x(t) y(t) z(t)
D = Differential(t)

eqs = [0 ~ σ*(y-x),
       0 ~ x*(ρ-z)-y,
       0 ~ x*y - β*z]

tspan = (0.0, 55)
# @named de = ODESystem(eqs,t,[x,y,z],[σ,ρ,β],tspan=tspan)

@mtkbuild sys = ODESystem(eqs, t)

u0 = [D(x) => 2.0,
    x => 1.0,
    y => 1.0,
    z => 0.0]

p = [σ => 1228.1,
    ρ => 1200.1,
    β => 129 / 3]


prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob)

title = "ModelingToolkit"
pt = plot(sol,line=(1,:auto),title=title)
pxyt = plot(sol, idxs = (x, y))
display(plot(pt,pxyt,layout=(2,1)))
