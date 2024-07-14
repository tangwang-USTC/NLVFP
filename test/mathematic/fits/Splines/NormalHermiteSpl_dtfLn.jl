using NormalHermiteSplines

v0 = vGk[nvlevel0][2:end] # Scheduling signal
y0 = dtfvLa[:,1][2:end] # Signal to be approximated
is_ys = y0 .> eps(Float64) * 1e-0
v = v0[is_ys] #
y = y0[is_ys] #

p = vGk[vGk .≤ v[end]]       # evaluation points
yk = fLnt[vGk .≤ v[end]]

methodRK = RK_H0()

# Build a continuous spline by values of function in nodes
# Here value of the 'scaling parameter' ε is estimated in the interpolate procedure.
spline = NormalHermiteSplines.interpolate(v, y, methodRK)

# A value of the 'scaling parameter' the spline was built with
ε = get_epsilon(spline)

# An estimation of the Gram matrix condition number
cond = estimate_cond(spline)


# An estimation of the interpolation accuracy -
# number of significant digits in the function value interpolation result.
significant_digits = estimate_accuracy(spline)

σ = NormalHermiteSplines.evaluate(spline, p)

erry = yk - σ

Rerry = yk ./ σ .- 1

title = string(methodRK,",cond=",cond,",ε=",fmtf2(ε),",sign=",significant_digits)
label = string("origin")
pfff = plot(v,y,label=label)
label = string("fit")
pfff = plot!(p,σ,label=label,xlabel=title)
