using NumericalIntegration


dx = 0.01
x = 0:dx:2π

y1(x) = sin(x)

yx = y1.(x)

yite = cumul_integrate(x,yx) .- cos(x[1])  # = ∫₀^x (sin(x))dx

py = plot(x,yx)
plot(x,yite)
pIy = plot!(x, - cos.(x))
err = yite + cos.(x)
perr = plot(x,err)
plot(py,pIy,perr,layout=(3,1))
