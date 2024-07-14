using DifferentialEquations
using Plots

function lorentz!(du,u,p,t)

    σ = p[1]
    ρ = p[2]
    β = p[3]
    du[1] = u[4]                              # dx
    du[2] = u[1] * (ρ - u[3]) - u[2]          # dy
    du[3] = u[1] * u[2] - β * u[3]            # dz
    du[4] = σ * (u[2] - u[1])                 # dw
end
u00 = [1.0,0.0,0.0,2.0]       # [x,y,z,w]
p0 = [12, 20, 9/3]


tspan = (0.0, 55)
prob0 = ODEProblem(lorentz!, u00, tspan, p0)
sol0 = solve(prob0)

pt0 = plot(sol0,line=(1,:auto))
pxyt0 = plot(sol0, idxs = (1, 2))
xlabel!("x")
ylabel!("y")
display(plot(pt0,pxyt0,layout=(2,1)))
