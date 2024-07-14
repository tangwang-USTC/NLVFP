using LeastSquaresOptim

"""
  ∑ₖ nₖ = 1
  
  Ml0fit = zero.(Ml0)
  king!(Ml0fit, xfit; Ml0=Ml0,l=0)
"""

nModel = 1
Ml0 = MsnnE[2:3nModel]
Ml0 .= 1.0
function king!(out, x; Ml0=Ml0,l=0)
    
    # nh = x[1]
    # uh = x[1] 
    # vhth = x[2] 
    out[1] = x[2] ^2 * (1 + 2/3 * ((x[1]  / x[2] )^2)) - Ml0[1]
    out[2] = x[2] ^4 * (1 + 4/3 * ((x[1]  / x[2] )^2) + 4/15 * ((x[1]  / x[2] )^4)) - Ml0[2]
end


function king_g!(J, x; Ml0=Ml0,l=0)
    
  J[1,1] = 4/3 * x[1]
  J[1,2] = 2 * x[2]
  J[2,1] = 8/3 * x[1] * (x[2] ^2 + 2/5 * x[1]^2)
  J[2,2] = 4 * x[2] * (x[2]^2 + 2/3 * x[1]^2)
end


p_tol=1e-50
f_tol=1e-36
g_tol=1e-50
(optimizer, optimizer_abbr) = (LeastSquaresOptim.Dogleg, :dl)
# (optimizer, optimizer_abbr) = (LeastSquaresOptim.LevenbergMarquardt, :lm)
factor = LeastSquaresOptim.QR()
# factor = LeastSquaresOptim.Cholesky()
# factor = LeastSquaresOptim.LSMR()
autodiffs = :central
# autodiffs = :forward

x0 = ones(2nModel)
x0 = [0e-3,0.1]
# nls = LeastSquaresProblem(x=x0,f! = king!,output_length=length(x0),autodiff=autodiffs)
nls = LeastSquaresProblem(x=x0,f! = king!,g! = king_g!,output_length=length(x0),autodiff=autodiffs)
# res = optimize!(nls, Dogleg(),x_tol=p_tol,f_tol=f_tol,g_tol=g_tol)
# res = optimize!(nls,optimizer(factor),x_tol=p_tol,f_tol=f_tol,g_tol=g_tol)
res = optimize!(nls,optimizer(),x_tol=p_tol,f_tol=f_tol,g_tol=g_tol)

xfit = res.minimizer         # the vector of best model1 parameters
niter = res.iterations
is_converged = res.converged
xssr = res.ssr                         # sum(abs2, fcur)
is_xconverged = res.x_converged
is_fconverged = res.f_converged
is_gconverged = res.g_converged
optim_method = res.optimizer
@show is_xconverged,is_fconverged,is_gconverged;
is_converged,niter,xssr,xfit