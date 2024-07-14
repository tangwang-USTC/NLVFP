using DifferentialEquations, ModelingToolkit
using Plots
using ModelingToolkit: t_nounits as t, D_nounits as D

####################################################### DifferentialEquations
function lorentz!(du,u,p,t)

    σ = p[1]
    ρ = p[2]
    β = p[3]
    # du[1] = u[4]                              # dx
    # du[2] = u[1] * (ρ - u[3]) - u[2]          # dy
    # du[3] = u[1] * u[2] - β * u[3]            # dz
    # du[4] = σ * (u[2] - u[1])                 # dw
    
    du[1] = σ * (u[2] - u[4])                 # dw
    du[2] = u[4] * (ρ - u[3]) - u[2]          # dy
    du[3] = u[4] * u[2] - β * u[3]            # dz
    du[4] = u[1]                              # dx
end
# u00 = [1.0,0.0,0.0,2.0]       # [x,y,z,w]
u00 = [2.0,0.0,0.0,1.0]       # [w,y,z,x]
p0 = [12, 200, 9/3]


tspan = (0.0, 55)
prob0 = ODEProblem(lorentz!, u00, tspan, p0)
sol0 = solve(prob0)
title = "ODE"
# label = ["x" "y" "z" "x_t"]
label = ["x_t" "y" "z" "x"]
pt0 = plot(sol0,line=(1,:auto),label=label,title=title)
pxyt0 = plot(sol0, idxs = (4,2),label="(x,y)")
xlabel!("x")
ylabel!("y")
# display(plot(pt0,pxyt0,layout=(2,1)))

############################################################### ModelingToolkit
@parameters σ ρ β
@variables x(t) y(t) z(t)

eqs = [D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z]

@mtkbuild sys = ODESystem(eqs, t)

# u0 = [D(x) => 2.0,
#     x => 1.0,
#     y => 0.0,
#     z => 0.0]

# p = [σ => 1228.1,
#     ρ => 1200.1,
#     β => 129 / 3]

# u0 = [D(x) => u00[4],
#     x => u00[1],
#     y => u00[2],
#     z => u00[3]]

u0 = [D(x) => u00[1],
    x => u00[4],
    y => u00[2],
    z => u00[3]]

p = [σ => p0[1],
    ρ => p0[2],
    β => p0[3]]

# tspan = (0.0, 55)
prob = ODEProblem(sys, u0, tspan, p, jac = true)
sol = solve(prob)

title = "ModelingToolkit"
pt = plot(sol,line=(1,:auto),title=title)
pxyt = plot(sol, idxs = (x, y))
# display(plot(pt,pxyt,layout=(2,1)))
display(plot(pt0,pt,pxyt0,pxyt,layout=(2,2)))
