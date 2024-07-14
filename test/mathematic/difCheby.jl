using ChebyshevApprox,Plots

n = 50
orders = 5
domain = [-1,1]
nodes = chebyshev_extended(n,domain)
v = vCmapping(nodes,vGdom[1],vGdom[2];isinv=true)
y = exp.(-v.^2)
p = chebyshev_polynomial(orders,nodes)
w = chebyshev_weights_extrema(y,nodes,orders,domain)
