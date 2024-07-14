

vath = vth[isp3]
vbth = vth[iFv3]
va = vG0 * vath
vb = vG0 * vbth

nab = [1, 10] * n20
cf3  = nab ./ vth.^3 / Pi^(3/2)

fLn0c = cf3[isp3] * fLn0t
FLn0c = cf3[iFv3] * FLn0t

xlabel = string("v")
label = string("fLn(v̂)")
plot(vva, fLn0c,xlabel=xlabel,label=label)
label = string("FLn(v̂)")
p0 = plot!(vvb, FLn0c,xlabel=xlabel,label=label)


ylabel = string("fLn(v) + FLn(v)")
p1 = plot(vG0, fLn0c + FLn0c,xlabel=xlabel,ylabel=ylabel)

display(plot(p0,p1,layout=(2,1)))


