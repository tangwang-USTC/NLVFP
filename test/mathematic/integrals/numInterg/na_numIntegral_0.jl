using NumericalIntegration # Trapeoidal, SimpsonEven
              # SimpsonEven nedds a power of `2ⁿ+1` points
              # (i.e., nd ∈ [5,9,17,33,65,127,...])
using Trapz   # 1ᵗʰ-order with at genetic unequally spaced points.
using Romberg # 1+ᵗʰ-order at equal points.
              # Combines trapezoidal integration with Richardson extrapolation for improved accuracy.
              #
xs = copy(vGk)
ys = copy(fLnt)
xd = copy(vG0)
yd = copy(fLn0)
xd2 = range(vGe[1],stop=vGe[end],length=nvG)
yd2 = zero.(xd2)
function filterequal(yd2,xd2,yd,xd)
    kkk = 1
    for i in 1:nc0
        if abs(xd[i] - xd2[kkk]) < epsT10
            yd2[kkk] = copy(yd[i])
            kkk += 1
        end
    end
    return yd2
end
yd2 = filterequal(yd2,xd2,yd,xd)
j2 = 2
# M0 =  4π ∫₀^∞ (x^2 * yd)dx ≡ 1

M0R2, errM0R2 = Romberg.romberg(xd2,xd2.^j2 .* yd2)
errM0R2 = M0R2 * 4/sqrtpi - 1

M0Ts = Trapz.trapz(xs,xs.^j2 .* ys) * 4/sqrtpi - 1
M0Ns = NumericalIntegration.integrate(xs,xs.^j2 .* ys) * 4/sqrtpi - 1

M0N2 = NumericalIntegration.integrate(xd2,xd2.^j2 .* yd2,SimpsonEven()) * 4/sqrtpi - 1

M0T1 = Trapz.trapz(xd,xd.^j2 .* yd) * 4/sqrtpi - 1
M0N1 = NumericalIntegration.integrate(xd,xd.^j2 .* yd) * 4/sqrtpi - 1

# M0N12 = NumericalIntegration.integrate(xd,xd.^j2 .* yd,SimpsonEven()) * 4/sqrtpi - 1

# py = plot(x,yx)
# plot(x,yite)
# pIy = plot!(x, - cos.(x))
# err = yite + cos.(x)
# perr = plot(x,err)
# plot(py,pIy,perr,layout=(3,1))
#
#
# x = range(0, stop=π, length=2^8)
# y = sin.(x)
# a = romberg(x, y)
