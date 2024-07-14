using SmoothingSplines
using RDatasets
plotly()

λ = 0.0 |> Float64  # smooth degree, bigger λ means higher smoothness

L1 = 2
isp = 1
nspF = nsp_vec[nsp_vec .≠ isp]
iFv = nspF[1]
vabth = vth[isp] / vth[iFv]
va = vremesh * vabth
## ##############


using Dierckx # library for 1-d and 2-d splines, wrapper Fortran library

dg1 = ddGvL[:,L1,isp]
plot(log.(va),dg1,line=(1,:auto))
spl = Spline1D(va,dg1;k=3,s=1e-1,bc="extrapolate")
dg11 = evaluate(spl,va)
plot!(log.(va),dg11,line=(1,:auto))

## ##########
dg1 = ddGvL[:,L1,isp]
Ypred2(λ) = predict(fit(SmoothingSpline,va,dg1,λ))             # fitted vector
# plot(log.(va),dg1,xlabel="speed",ylabel="Dist",label="Datas")
# plot!(X,Ypred2(λ),xlabel="speed",ylabel="Dist",label=label)

λ = 0.0
label = string("L=",L1-1,",λ=",λ)
plot(log.(va),Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))
λ = 1e-5
label = string("L=",L1-1,",λ=",λ)
plot!(log.(va),Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))
λ = 1e-4
label = string("L=",L1-1,",λ=",λ)
plot!(log.(va),Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))
λ = 1e-3
label = string("L=",L1-1,",λ=",λ)
plot!(log.(va),Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))
λ = 1e-2
label = string("L=",L1-1,",λ=",λ)
plot!(log.(va),Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))
λ = 1e-1
label = string("L=",L1-1,",λ=",λ)
px = plot!(log.(va),Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))



λ = 0.0
label = string("L=",L1-1,",λ=",λ)
plot(log.(va),dg1 - Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))
λ = 1e-5
label = string("L=",L1-1,",λ=",λ)
plot!(log.(va),dg1 - Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))
λ = 1e-4
label = string("L=",L1-1,",λ=",λ)
plot!(log.(va),dg1 - Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))
λ = 1e-3
label = string("L=",L1-1,",λ=",λ)
plot!(log.(va),dg1 - Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))
λ = 1e-2
label = string("L=",L1-1,",λ=",λ)
plot!(log.(va),dg1 - Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))
λ = 1e-1
label = string("L=",L1-1,",λ=",λ)
pRx = plot!(log.(va),dg1 - Ypred2.(λ),xlabel="va",ylabel="ddGvL",label=label,line=(1,:auto))
