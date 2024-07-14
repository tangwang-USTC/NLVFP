using BasicBSpline
# using BasicInterpolators

function interpolate(xs::AbstractVector, fs::AbstractVector{T};
    degrees::Int=2) where T
    # Cubic open B-spline space
    p = 3
    k = KnotVector(xs) + KnotVector([xs[1],xs[end]]) * p
    P = BSplineSpace{p}(k)

    # dimensions
    m = length(xs)
    n = dim(P)

    # The interpolant function has a f''=0 property at bounds.
    ddP = BSplineDerivativeSpace{degrees}(P)
    dda = [bsplinebasis(ddP,j,xs[1]) for j in 1:n]
    ddb = [bsplinebasis(ddP,j,xs[m]) for j in 1:n]

    # Compute the interpolant function (1-dim B-spline manifold)
    M = [bsplinebasis(P,j,xs[i]) for i in 1:m, j in 1:n]
    M = vcat(dda', M, ddb')
    y = vcat(zero(T), fs, zero(T))
    return BSplineManifold(M\y, P)
end
# plotly()
gr()
# Example inputs

degrees = 1  # ∈ [0,1,2,3]
LL1 = 1
y_isp = 1

xs = vG0
fs = fvL[nvlevel0,L1,y_isp]
ys0 = fvL[:,L1,y_isp]
f = interpolate(xs,fs;degrees=degrees)
is_xs0 = vGk .≤ xs[end]
is_xs0 = ys0 .≥ eps(Float64) * 1e-3
xs0 = vGk[is_xs0]
ys0 = ys0[is_xs0]
ys = f.(xs0)
erry = ys - ys0
Rerry = ys ./ ys0 .- 1
# Plot
pfs = scatter(xs, fs)
pfs = plot!(t->f(t))
label = string("errf")
pdfs = plot(xs0,erry,label=label)
label = string("Rerrf")
pRdfs = plot(xs0,Rerry,label=label)
display(plot(pfs,pdfs,pRdfs,layout=(3,1)))
