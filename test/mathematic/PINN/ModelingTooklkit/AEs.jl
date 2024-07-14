
using DifferentialEquations, ModelingToolkit
using Plots
# using ModelingToolkit: t_nounits as tn, D_nounits as Dt
using ModelingToolkit: t_nounits as tn

@parameters σ ρ β
@variables x(tn) y(tn) z(tn)
# Dt = Differential(tn)

eqs = [0 ~ σ*(y-x),
       0 ~ x*(ρ-z)-y,
       0 ~ x*y - β*z]

# @named de = ODESystem(eqs,tn,[x,y,z],[σ,ρ,β],tspan=(0, 10.0))

@mtkbuild sys = ODESystem(eqs, tn)

u0 = [x => 1.0,
      y => 1.0,
      z => 0.0]

p = [σ => 1228.1,
    ρ => 1200.1,
    β => 129 / 3]


tspan = (0.0, 55)
prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob)

title = "ModelingToolkit"
pt = plot(sol,line=(1,:auto),title=title)
pxyt = plot(sol, idxs = (x, y))
display(plot(pt,pxyt,layout=(2,1)))
