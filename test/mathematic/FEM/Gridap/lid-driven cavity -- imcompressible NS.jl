

"""
  Details in:
    https://gridap.github.io/Tutorials/dev/pages/t008_inc_navier_stokes/
"""

using Gridap
n = 100
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)

# boundary tags
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri1",[6,])
add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

# Lagrangian FE space
D = 2
order = 2    # second order interpolation
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,labels=labels,dirichlet_tags=["diri0","diri1"])

# interpolation space for the pressure
reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)
Q = TestFESpace(model,reffeₚ,conformity=:L2,constraint=:zeromean)

# trial multi-field FE spaces
uD0 = VectorValue(0,0)
uD1 = VectorValue(1,0)
U = TrialFESpace(V,[uD0,uD1])
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

# triangulation and integration measure
degree = order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)

# nonlinear weak form
const Re = 10.0
conv(u,∇u) = Re*(∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

# bilinear form
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ

# nonlinear term and its Jacobian
c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

# Navier-Stokes weak form residual and Jacobian
res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v)
jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)

# nonlinear FE problem
op = FEOperator(res,jac,X,Y)

using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)

# solve the problem without providing an initial guess
uh, ph = solve(solver,op)

# write the results for visualization
writevtk(Ωₕ,"ins-results",cellfields=["uh"=>uh,"ph"=>ph])


