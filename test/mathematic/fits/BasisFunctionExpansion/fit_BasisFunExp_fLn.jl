using Plots, BasisFunctionExpansions
using ReverseDiff
using DSP            # filt
# plotly()
gr()
# using Test, Statistics, Random, LinearAlgebra, Statistics
# using BenchmarkTools

v0 = vGk[nvlevel0] # Scheduling signal
y0 = fLnt[nvlevel0] # Signal to be approximated
is_ys = y0 .> eps(Float64) * 1e-0
v = v0[is_ys] #
y = y0[is_ys] #
Nv = length(v)

Nv  = 55 # Number of basis functions
rbf = UniformRBFE(v,Nv, normalize=true) # Approximate using radial basis functions with constant width
bfa = BasisFunctionApproximation(y,v,rbf,1e-1) # Create approximation object
ŷ   = bfa(v0) # Reconstruct signal using approximation object
py = scatter(v0,y0, lab="Signal")
py = plot!(v0,ŷ, lab="Reconstruction")

erry = ŷ - y0
Rerry = ŷ ./ y0 .- 1
perry = plot(v0,erry, lab="erry")
pRerry = plot(v0,Rerry, lab="erry")
display(plot(py,perry,pRerry,layout=(3,1)))
