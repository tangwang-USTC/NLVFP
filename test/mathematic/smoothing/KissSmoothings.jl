# using PyPlot
using KissSmoothing
t = LinRange(0,2pi,1000) |>Vector{Float64}
ty = sin.(t)
y = ty .+ randn(length(t)) .*0.05
cps = LinRange(0,2pi,20) |>Vector{Float64}

LL1 = 1
y_isp = 1
vs0 = vG0
ys0 = fvL[nvlevel0,LL1,y_isp]
is_ys = ys0 .> eps(Float64) * 1e7
t = vs0[is_ys]
ty = ys0[is_ys]
y = 1ty
y[15] *= 1.4
nt = length(t)
cps = LinRange(t[1],t[end],nt) |>Vector{Float64}
cps = t[1:3:end]
fn = fit_rbf(t,y,cps)
yfit = fn(t)
erry = ty - y
erryfit = ty - yfit
erryy = y - yfit
py = scatter(t, y, color="gray",s=2,label="noisy",line=(1,:auto))
py = plot!(t, yfit, color="red",lw=1.5,label="rbf estimate")
py = plot!(t,ty, color="blue",lw=1.0,label="true",line=(2,:auto))
# plot!(t,y, color="blue",lw=1.0,label="true")

legend()
xlabel!("X")
ylabel!("Y")

perryy = plot(t,erryy, color="green",lw=1.0,label="erry",line=(2,:auto))
perry = plot(t,erry, color="green",lw=1.0,label="erry",line=(2,:auto))
perryfit = plot(t,erryfit, color="green",lw=1.0,label="erryfit",line=(2,:auto))
display(plot(py,perry,perryy,layout=(3,1)))
