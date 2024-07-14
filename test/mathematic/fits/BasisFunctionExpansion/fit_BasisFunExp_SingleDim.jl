using Plots, BasisFunctionExpansions
using ReverseDiff
using DSP            # filt
plotly()
# using Test, Statistics, Random, LinearAlgebra, Statistics
# using BenchmarkTools

N = 1000
v = range(0,stop = 10, length = N) # Scheduling signal
y = randn(N) # Signal to be approximated
y = filt(ones(500)/500,[1],y)

Nv  = 10 # Number of basis functions
rbf = UniformRBFE(v,Nv, normalize=true) # Approximate using radial basis functions with constant width
bfa = BasisFunctionApproximation(y,v,rbf,1) # Create approximation object
ŷ   = bfa(v) # Reconstruct signal using approximation object
scatter(v,y, lab="Signal")
scatter!(v,ŷ, lab="Reconstruction")


# plot(v[:,1],v[:,2],y, lab="Signal")
# plot!(v[:,1],v[:,2],ŷ, lab="Reconstruction")
