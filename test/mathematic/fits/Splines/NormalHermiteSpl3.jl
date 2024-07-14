using NormalHermiteSplines

x = collect(1.0:1.0:20)       # function nodes
u = x.*0.0                    # function values in nodes
for i in 6:10
    u[i] = 1.0
end
for i in 11:14
    u[i] = -0.2 * i + 3.0
end
methodRK = RK_H0()

# Build a continuous spline by values of function in nodes
# Here value of the 'scaling parameter' ε is estimated in the interpolate procedure.
spline = NormalHermiteSplines.interpolate(x, u, methodRK)

# A value of the 'scaling parameter' the spline was built with
ε = get_epsilon(spline)

# An estimation of the Gram matrix condition number
cond = estimate_cond(spline)


# An estimation of the interpolation accuracy -
# number of significant digits in the function value interpolation result.
significant_digits = estimate_accuracy(spline)

p = collect(1.0:0.2:20)        # evaluation points
σ = NormalHermiteSplines.evaluate(spline, p)

erru = u - σ[1:5:end]

Rerru = u ./ σ[1:5:end] .- 1

title = string(methodRK,",cond=",cond,",ε=",fmtf2(ε),",sign=",significant_digits)
label = string("origin")
pfff = plot(x,u,label=label)
label = string("fit")
pfff = plot!(p,σ,label=label,xlabel=title)
