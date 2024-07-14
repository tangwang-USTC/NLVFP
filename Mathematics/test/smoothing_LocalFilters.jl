using DSP
using LocalFilters
using RDatasets
using Plots

plotly()

λ = 0.0 |> Float64  # smooth degree, bigger λ means higher smoothness

cars = dataset("datasets","cars")
X = cars.Speed |> Vector{Float64}
Y = cars.Dist |> Vector{Float64}
plot(X,Y,xlabel="speed",ylabel="Dist",label="Datas")
## SmoothingSplines
# # spl = fit(SmoothingSpline,X,Y,λ)
# Ypred2(λ) = predict(fit(SmoothingSpline,X,Y,λ))             # fitted vector
# # label = string("λ=",λ)
# # plot!(X,Ypred2(λ),xlabel="speed",ylabel="Dist",label=label)
# λ = [0,1,2,5,10,20,50,100,200] |> Vector{Float64}
# plot!(X,Ypred2.(λ),xlabel="speed",ylabel="Dist",label=label)

## LocalFilters
ndata = 5
ndata = 3:5
Yfilter(ndata) = localmean(Y,ndata)
Yfilter(ndata) = erode(Y,ndata)        # performs an erosion (local minimum) of A by B
Yfilter(ndata) = dilate(Y,ndata)       # performs a dilation (local maximum) of A by B
Yfilter(ndata) = localextrema(Y,ndata) # yields the erosion and the dilation of A by B
Yfilter(ndata) = opening(Y,ndata)      # performs an erosion followed by a dilation
Yfilter(ndata) = closing(Y,ndata)      # performs a dilation followed by an erosion
label = string("n=",ndata)
plot!(X,Yfilter.(ndata),xlabel="speed",ylabel="Dist",label=label)
# ## DSP
# responsetype = Bandpass(10, 40; fs=100)
# designmethod = Butterworth(4)
# Ydsp = filt(digitalfilter(responsetype, designmethod), Y)
# label = string("DSP,n=",ndata)
# plot(X,Ydsp,xlabel="speed",ylabel="Dist",label=label)
