
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
degrees = 5
λ = 0.51
k = 2            # order of the interpolation polynomials

v , w = laguerre(n,α,endptv)

v0 = vG0
v5 = v[n-5:n]

order = 5
f = v .^ order

vs = vGk[nvlevel0] # Scheduling signal
fs = fLnt[nvlevel0] # Signal to be approximated
is_ys = fs .> eps(Float64) * 1e5
v0 = vs[is_ys] #
f = fs[is_ys] #

v5 = vGk[vGk .≤ v0[end]]
f5 = fLnt[vGk .≤ v0[end]]
printstyled("order=",order,",k=",k,",n=",n,color=:yellow)
println()

## ## Dierckx.jl

k = 3
s = 0e-6
n = length(v0)
f0 = 1f
itpDL = Dierckx.Spline1D(v0,f0;k=k,s=s,bc="extrapolate")  # [extrapolate, nearest]
f0k = itpDL.(v5)

errf = f0k-f5
Rerrf = f0k./f5.-1
@show norm(errf)
@show norm(Rerrf)

xlabel = string("v,(k,s)=",(k,s))
pf = plot(v0,f)
pf = plot!(v5,f0k)
perrf = plot(v5,errf)
pRerrf = plot(v5[1:end-10],Rerrf[1:end-10],xlabel=xlabel)
display(plot(pf,perrf,pRerrf,layout=(3,1)))

##
degrees = 19
λ = 1.8e-1
# model = Loess.loess(v0,f0;span=λ,degree=degrees)
model = Loess.loess(v0,f0;degree=degrees)
f0k = Loess.predict(model,v5)
errf = f0k-f5
Rerrf = f0k./f5.-1
@show norm(errf)
@show norm(Rerrf)

xlabel = string("v,(deg,λ)=",(degrees,λ))
pf = plot(v0,f)
pf = plot!(v5,f0k)
perrf = plot(v5,errf)
pRerrf = plot(v5[1:end-10],Rerrf[1:end-10],xlabel=xlabel)
display(plot(pf,perrf,pRerrf,layout=(3,1)))


##################        # worse for f(v) = v^3
itpDL = QuadraticInterpolation(f0,v0)
f0k = itpDL.(v5)

errf = f0k-f5
Rerrf = f0k./f5.-1
@show norm(errf)
@show norm(Rerrf)

xlabel = string("v,(Qk,s)=",(2,s))
pf = plot(v0,f)
pf = plot!(v5,f0k)
perrf = plot(v5,errf)
pRerrf = plot(v5[1:end-10],Rerrf[1:end-10],xlabel=xlabel)
display(plot(pf,perrf,pRerrf,layout=(3,1)))


##################         # worse for f(v) = v^2
itpDL = CubicSpline(f0,v0)
f0k = itpDL.(v5)

errf = f0k-f5
Rerrf = f0k./f5.-1
@show norm(errf)
@show norm(Rerrf)

xlabel = string("v,(Ck,s)=",(3,s))
pf = plot(v0,f)
pf = plot!(v5,f0k)
perrf = plot(v5,errf)
pRerrf = plot(v5[1:end-10],Rerrf[1:end-10],xlabel=xlabel)
display(plot(pf,perrf,pRerrf,layout=(3,1)))
