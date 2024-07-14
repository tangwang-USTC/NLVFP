
using FiniteVolumeMethod, DelaunayTriangulation, CairoMakie

# To solve this problem, the first step is to define the mesh.
a, b, c, d = 0.0, 2.0, 0.0, 2.0
nx, ny = 50, 50
tri = triangulate_rectangle(a, b, c, d, nx, ny, single_boundary=true)
mesh = FVMGeometry(tri)


# This mesh is shown below.

fig, ax, sc = triplot(tri)
display(fig)


# We now need to define the boundary conditions. We have a homogeneous Dirichlet condition:
bc = (x, y, t, u, p) -> zero(u)
BCs = BoundaryConditions(mesh, bc, Dirichlet)

# We can now define the actual PDE. We start by defining the initial condition and the diffusion function.
f = (x, y) -> y ≤ 1.0 ? 50.0 : 0.0
initial_condition = [f(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
D = (x, y, t, u, p) -> 1 / 9

# We can now define the problem:
final_time = 0.5
prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition, final_time)

# Note that in `prob`, it is not a diffusion function that is used but instead it is a flux function:
prob.flux_function

# solve
# sol = solve(prob, Tsit5(), saveat=0.05)
sol = solve(prob, Tsit5())
# sol |> tc #hide

# To visualise the solution, we can use `tricontourf!` from Makie.jl.
u = Observable(sol.u[1])
figs, ax, sc = tricontourf(tri, u, levels=0:5:50, colormap=:matter)
tightlimits!(ax)
record(figs, "anim.gif", eachindex(sol)) do i
    u[] = sol.u[i]
    # display(plot(u[]))
end
display(figs)

# To visualise the solution at jᵗʰ step where `j ∈ [1, 56, 91,175]`.
figs3 = Figure(fontsize=38)
for (i, j) in zip(1:4, (1, 56, 91,175))
    local ax
    ax = Axis(figs3[1, i], width=600, height=600,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[j])",
        titlealign=:left)
    tricontourf!(ax, tri, sol.u[j], levels=0:5:50, colormap=:matter)
    tightlimits!(ax)
end
resize_to_layout!(figs3)
figs3
display(figs3)

local ax




