using Plots, BasisFunctionExpansions
using ReverseDiff
using DSP            # filt
plotly()
# using Test, Statistics, Random, LinearAlgebra, Statistics
# using BenchmarkTools

# Multidim Diagonal
N    = 1000
x    = range(0, stop=4pi, length=N)

v    = [cos.(x) sin.(x)] .*x
y    = randn(N)
y    = filt(ones(500)/500,[1],y)
Nv  = [10,10]   # Number of basis functions along each dimension
rbf = MultiUniformRBFE(v,Nv, normalize=true) # Approximate using radial basis functions with constant width (Not isotropic, but all functions have the same diagonal covariance matrix)
bfa = BasisFunctionApproximation(y,v,rbf,0.0001) # Create approximation object
ŷ   = bfa(v) # Reconstruct signal using approximation object

# scatter3d(v[:,1],v[:,2],y, lab="Signal")
# plot!(v[:,1],v[:,2],ŷ, lab="Reconstruction")



plot(v[:,1],v[:,2],y, lab="Signal")
plot!(v[:,1],v[:,2],ŷ, lab="Reconstruction")
