using SmoothingSplines
using RDatasets
plotly()

λ = 0.0 |> Float64  # smooth degree, bigger λ means higher smoothness

cars = dataset("datasets","cars")
X = cars.Speed |> Vector{Float64}
Y = cars.Dist |> Vector{Float64}
# spl = fit(SmoothingSpline,X,Y,λ)
Ypred2(λ) = predict(fit(SmoothingSpline,X,Y,λ))             # fitted vector
plot(X,Y,xlabel="speed",ylabel="Dist",label="Datas")
# label = string("λ=",λ)
# plot!(X,Ypred2(λ),xlabel="speed",ylabel="Dist",label=label)
λ = [0,1,2,5,10,20,50,100,200] |> Vector{Float64}
plot!(X,Ypred2.(λ),xlabel="speed",ylabel="Dist",label=label)

Ypred2(λ) = predict(fit(SmoothingSpline,X,Y,λ))

using Dierckx # library for 1-d and 2-d splines, wrapper Fortran library

spl = Spline1D(vremesh,dg1;k=3,s=1e-1,bc="extrapolate")
dg11 = evaluate(spl,vremesh)
plot!(log.(vremesh),dg11,line=(1,:auto))
