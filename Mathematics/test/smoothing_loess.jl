using Smoothers    # p = np where np ∈ [5:nt-1], which will affect the result
using Loess        #
using Plots


plotly()

t = Array(LinRange(-π,π,100))
nt = length(t)
# signal = sin.(t)
# signal = exp.(-t.^2)
signal = exp.(-t.^2) .* sin.(t)
noise = 0.25rand(nt)
x = signal .+ noise

w = 65
sw = string(w)

legendlocation = :bottomright
# Datas
label = string("sin(t)+ϵ")
plot(t,x,line=(10,:auto),alpha=0.3,label=label,xlabel="t",ylabel="x",title="Smoothers",legend=legendlocation)

# Signal
label = string("sin(t)")
plot!(t,signal,line=(2,:auto),label=label)

# Henderson Moving Average Filter
label = "hma(x,"*sw*")"
plot!(t,hma(x,w), label=label,line=(2,:auto))

# Locally Estimated Scatterplot Smoothing with Smoothers.jl
label = "Smoothers(q="*sw*")"
plot!(t,Smoothers.loess(t,x;q=w)(t), label=label,line=(2,:auto))

# # Locally Estimated Scatterplot Smoothing with Loess.jl
label = "loess"
model = Loess.loess(t,x)
plot!(t,Loess.predict(model,t), label=label,line=(2,:auto))

# # Moving Average Filter with Matlab/Octave 'filter'
# b = ones(w)/w; a = [1];
# label = "filter(1,[1/"*sw*",...],x)"
# plot!(t,filter(b,a,x), label=label,line=(2,:auto))

# # Simple Moving Average
# label = "sma(x,"*sw*",true)"
# plot!(t, sma(x,w,true), label=label,line=(2,:auto))

# ########################## Loess
#
# xs = 10 .* rand(100)
# ys = sin.(xs) .+ 0.5 * rand(100)
#
# model = Loess.loess(xs, ys)
#
# us = range(extrema(xs)...; step = 0.1)
# vs = predict(model, us)
#
# using Gadfly
# p = Gadfly.plot(x=xs, y=ys, Geom.point, Guide.xlabel("x"), Guide.ylabel("y"))
# # p = Gadfly.plot(x=xs, y=ys, Geom.point, Guide.xlabel("x"), Guide.ylabel("y"),
# #          layer(Geom.line, x=us, y=vs))
# # draw(SVG("loess.svg", 6inch, 3inch), p)
