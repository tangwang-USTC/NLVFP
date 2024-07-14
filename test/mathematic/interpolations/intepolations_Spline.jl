
# Interpolations and Smoothing
using Dierckx
using Loess
# using Smoothers
using DataInterpolations, NumericalIntegration
# using SmoothingSplines
# using LocalFilters

"""
      Dierckx.jl: Spline1D
                  derivative
                  extropolate
      DataInterpolations.jl: QuadraticInterpolation
                             CubicSpline()
      SmoothingSpline.jl
        spl = fit(SmoothingSpline,v,dG[:,iu],1e-3)
        dG[:,iu] = predict(spl)

"""


n = 20
α = 0.0
endptv = neither
k = 5            # order of the interpolation polynomials
degrees = 5
λ = 0.51

v , w = laguerre(n,α,endptv)

v0 = v[1:n-6]
v5 = v[n-5:n]

order = 5
f = v .^ order

printstyled("order=",order,",k=",k,",n=",n,color=:yellow)
println()

## ## Dierckx.jl

f0 = 1f
f0[n-5:n] .= 0
itpDL = Dierckx.Spline1D(v0,f0[1:n-6];k=k,s=1,bc="extrapolate")
f0[n-5:n] = itpDL.(v5)
@show norm(f0-f)
@show norm(f0./f.-1)

# f1 = 1f
# f1[n-5:n] .= 0
# model = Loess.loess(v0,f1[1:n-6];span=λ,degree=degrees)
# dHLn = Loess.predict(model,v5)
# @show norm(f1-f)


##################        # worse for f(v) = v^3
f2 = 1f
f2[n-5:n] .= 0
itpDL = QuadraticInterpolation(f2,v0)
f2[n-5:n] = itpDL.(v5)
@show norm(f2-f)
@show norm(f2./f.-1)

##################         # worse for f(v) = v^2
f3 = 1f
f3[n-5:n] .= 0
itpDL = CubicSpline(f3,v0)
f3[n-5:n] = itpDL.(v5)
@show norm(f3-f)
@show norm(f3./f.-1)
