x = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]
y = log.(x) + sqrt.(x)
gradients = missing

using SchumakerSpline
using Plots
########################
# Linear Extrapolation #
spline = Schumaker(x,y; extrapolation = (Linear, Linear))
# Now plotting the spline
xrange = collect(range(-5, stop=10, length=100))
vals = SchumakerSpline.evaluate.(spline, xrange)
derivative_values = SchumakerSpline.evaluate.(spline, xrange, 1 )
second_derivative_values = SchumakerSpline.evaluate.(spline, xrange , 2 )
plot(xrange , vals; label = "Spline")
plot!(xrange, derivative_values; label = "First Derivative")
plot!(xrange, second_derivative_values; label = "Second Derivative")


vals = spline.(xrange)
derivative_values = spline.(xrange, 1)
second_derivative_values = spline.(xrange , 2)
