using Polynomials

xend = 14
xs = 0:0.5:xend
ff(xs) = @. exp(-xs) * sin(xs)
ys = ff(xs)
xss = 0.0:0.1:xend
ys[16] *= 1.0 + 1e-2

yf1 = fit(xs, ys) |> p -> round.(coeffs(p), digits=4) |> Polynomial


yf2 = fit(ChebyshevT, xs, ys, 2) |> p -> round.(coeffs(p), digits=4) |> ChebyshevT

yf1 = fit(xs, ys)
yf2 = fit(ChebyshevT, xs, ys, 2)

yss = ff(xss)
yft = yf1.(xss)
# yfc = yf2.(xs)
erry = yss - yft
Rerry = yss ./ yft .- 1

is_ys = xss .< 5.0
label = string("ys")
py = plot(xss[is_ys], yss[is_ys],label=label,line=(2,:auto))
label = string("ys_Tay")
py = plot!(xss[is_ys], yft[is_ys],label=label,line=(2,:auto))
# label = string("ys_Che")
# py = plot(xs, ysc,label=label,line=(2,:auto))


label = string("errys")
perry = plot(xss[is_ys], erry[is_ys],label=label,line=(2,:auto))

label = string("Rerrys")
pRerry = plot(xss[is_ys], Rerry[is_ys],label=label,line=(2,:auto))
display(plot(py,perry,pRerry,layout=(3,1)))
