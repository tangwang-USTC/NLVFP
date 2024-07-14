
using DelaunayTriangulation, FiniteVolumeMethod, ElasticArrays
using OrdinaryDiffEq, LinearSolve, CairoMakie, Test, Plots

# gr()
# pyplot()
# geometry with `Triangulation`, `refine!` and `CircularArc`
α = π / 4
points = [(0.0, 0.0), (1.0, 0.0), (cos(α), sin(α))]
bottom_edge = [1, 2]
arc = CircularArc((1.0, 0.0), (cos(α), sin(α)), (0.0, 0.0))
upper_edge = [3, 1]
boundary_nodes = [bottom_edge, [arc], upper_edge]
tri = triangulate(points; boundary_nodes)
A = get_area(tri)
refine!(tri; max_area=1e-4A)
mesh = FVMGeometry(tri)

# This is the mesh we've constructed.
fig, ax, sc = triplot(tri)
display(fig)


# To confirm that the boundary is now in three parts
get_boundary_nodes(tri)

# Boundary conditions
# provide `Tuple`s, where the `i`th element of the `Tuple`s refers to the `i`th part of the boundary.
lower_bc = arc_bc = upper_bc = (x, y, t, u, p) -> zero(u)
types = (Neumann, Dirichlet, Neumann)
BCs = BoundaryConditions(mesh, (lower_bc, arc_bc, upper_bc), types)

# PDE: reaction-diffusion formulation, specifying the diffusion function as a constant
f = (x, y) -> 1 - sqrt(x^2 + y^2)
D = (x, y, t, u, p) -> one(u)
initial_condition = [f(x, y) for (x, y) in DelaunayTriangulation.each_point(tri)]
final_time = 0.1
prob = FVMProblem(mesh, BCs; diffusion_function=D, initial_condition, final_time)


# If you did want to use the flux formulation, you would need to provide
flux = (x, y, t, α, β, γ, p) -> (-α, -β)

# Solve
sol = solve(prob, TRBDF2(linsolve=KLUFactorization()), saveat=0.01, parallel=Val(false))

ind = findall(DelaunayTriangulation.each_point_index(tri)) do i #hide
    !DelaunayTriangulation.has_vertex(tri, i) #hide
end #hide
@test sol[ind, :] ≈ reshape(repeat(initial_condition, length(sol)), :, length(sol))[ind, :] # make sure that missing vertices don't change #hide
# sol |> tc #hide

# Plot
figs3 = Figure(fontsize=38)
for (i, j) in zip(1:3, (1, 6, 11))
    local ax
    ax = Axis(figs3[1, i], width=600, height=600,
        xlabel="x", ylabel="y",
        title="t = $(sol.t[j])",
        titlealign=:left)
    tricontourf!(ax, tri, sol.u[j], levels=0:0.01:1, colormap=:matter)
    tightlimits!(ax)
end
resize_to_layout!(figs3)
figs3

local ax
display(figs3)
