n = 3
k = 1
v = v[v .< 5]
y(v) = exp.(-v.^n)
dy(v) = - n * v.^(n-1) .* exp.(-v.^n)

# y(v) = sin.(v)
# dy(v) = cos.(v)

yv = y(v)

spl = Spline1D(v,yv;k=k)
dyn = derivative(spl,v,1)
dy2 = (yv[2:end] - yv[1:end-1]) ./ (v[2:end] - v[1:end-1])
dy2 = ([0;dy2] + [dy2;0]) / 2
dy3 = derivationCDS(yv,v)
label = string("dy_exc")
plot(dy(v),label=label,line=(3,:dot))
label = string("dyn,k=",k)
pdy = plot!(dyn,label=label,line=(2,:solid))
# label = string("dy2")
# pdy = plot!(dy2,label=label,line=(2,:dash))
label = string("dy3")
pdy = plot!(dy3,label=label,line=(2,:dash))
Rerr = (dy(v) - dyn) ./ dyn
label = string("Rerr")
perr = plot(Rerr,label=label)
plot(pdy,perr,layout=(2,1))
