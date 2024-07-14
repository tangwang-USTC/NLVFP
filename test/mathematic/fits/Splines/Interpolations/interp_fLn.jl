using Interpolations

A = rand(20)
A_x = 1.0:2.0:40.0
nodes = (A_x,)
itp = Interpolations.interpolate(nodes, A, Gridded(Linear()))

plot(A_x,A)
xsc = 1.0:1.0:20
scatter!(xsc,itp(xsc))
