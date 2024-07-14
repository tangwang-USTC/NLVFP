using Optim

f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

# function f(x)

#     [1 - x[1], 100 * (x[2]-x[1]^2)]
# end

function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

x0 = [0.0, 0.0]
result = optimize(f, x0)

# optimize(f, g, x0; inplace = false)

solver = Optim.summary(result)
xfit = Optim.minimizer(result)
xfi = Optim.minimum(result)
niter = Optim.iterations(result)
is_converged = Optim.converged(result)
is_xconverged = Optim.x_converged(result)
is_fconverged = Optim.f_converged(result)
is_gconverged = Optim.g_converged(result)