using LeastSquaresOptim


para = [4,100]
function rosenbrock_f!(out, x;p=para)
 out[1] = p[1] - x[1]           # p[1] = 1
 out[2] = p[2] * (x[2]-x[1]^2)  # p[2] = 100
end

(optimizer, optimizer_abbr) = (LeastSquaresOptim.Dogleg, :dl)
# (optimizer, optimizer_abbr) = (LeastSquaresOptim.LevenbergMarquardt, :lm)
factor = LeastSquaresOptim.QR()
# factor = LeastSquaresOptim.Cholesky()
# factor = LeastSquaresOptim.LSMR()
autodiffs = :central
# autodiffs = :forward

x0 = zeros(2)

nls = LeastSquaresProblem(x=x0,f! = rosenbrock_f!,output_length=length(x0),autodiff=:central)
# res = optimize!(nls, Dogleg())
res = optimize!(nls,optimizer(factor))
# res = optimize!(nls,optimizer())

xfit = res.minimizer         # the vector of best model1 parameters
niter= res.iterations
is_converged = res.converged
xssr = res.ssr                         # sum(abs2, fcur)
is_xconverged = res.x_converged
is_fconverged = res.f_converged
is_gconverged = res.g_converged
optim_method = res.optimizer
