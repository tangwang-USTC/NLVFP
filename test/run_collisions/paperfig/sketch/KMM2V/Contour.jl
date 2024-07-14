
theta = pi / 2 * μ

isp333 = 1
rr = vhk[isp333]

veccc = rr .< 2
zz = fvL[isp333][veccc,:] * Mun

xx = rr[veccc] .* cos.(theta)'
yy = rr[veccc] .* sin.(theta)'

contour(xx, yy, zz, levels=10, color=:turbo, clabels=true, cbar=false, lw=2)
title!(L"Plot of $f(v_x, v_y)$")
xlabel!(L"v_x")
ylabel!(L"v_y")
