using LeastSquaresOptim

"""
  ∑ₖ nₖ = 1
  
  Ml0fit = zero.(Ml0)
  king!(Ml0fit, xfit; Ml0=Ml0,l=0)
"""

nModel = nMod[isp3]
Ml0 = MsnnE[1:3nModel]
function king!(out, x; Ml0=Ml0,l=0)
    
  nh = x[1]
  uh = x[2] 
  vhth = x[3] 
  out[1] = nh - Ml0[1]
  out[2] = (nh * vhth^2) * (1 + 2/3 * (uh^2 / vhth^2)) - Ml0[2]
  out[3] = (nh * vhth^4) * (1 + 4/3 * (uh^2 / vhth^2) + 4/15 * (uh^4 / vhth^4)) - Ml0[3]
end


function king_g!(J, x; Ml0=Ml0,l=0)
    
  nh = x[1]
  uh = x[2] 
  vhth = x[3] 
  J[1,1] = 1.0
  J[1,2] = 0.0
  J[1,3] = 0.0
  J[2,1] = vhth^2 + 2/3 * uh^2
  J[2,2] = 4/3 * nh * uh
  J[2,3] = 2nh * vhth
  J[3,1] = vhth^4 + 4/3 * uh^2 * vhth^2 + 4/15 * uh^4
  J[3,2] = 8/3 * nh * uh * (vhth ^2 + 2/5 * uh^2)
  J[3,3] = 4nh * vhth * (vhth ^2 + 2/3 * uh^2)
end

# The initial solution
p_noise_ratio = 1e-1 # Relative error of the initial guess parameters `p`
p_noise_abs = 1e-15 # Absolute error of the initial guess parameters `p`.
p_noise_rel = rand(3nModel) * p_noise_ratio
x0 = zeros(3nModel)
x0[1:3:end] .= nai0[isp3]
x0[2:3:end] .= uai0[isp3]
x0[3:3:end] .= vthi0[isp3]
x0 .*= (1.0 .+ p_noise_rel ) + p_noise_abs * rand(3nModel)
x0[2:3:end] .= 0.0

# The parameter limits for MCF plasma.
lbs = zeros(3nModel)
lbs[1:3:end] .= 0.0
lbs[2:3:end] .= -uhMax
lbs[3:3:end] .= 1/vhthMax
ubs = zeros(3nModel)
ubs[1:3:end] .= nhMax
ubs[2:3:end] .= uhMax
ubs[3:3:end] .= vhthMax

p_tol=1e-37
f_tol=1e-37
g_tol=1e-37
maxIterKing = 1000
(optimizer, optimizer_abbr) = (LeastSquaresOptim.Dogleg, :dl)
# (optimizer, optimizer_abbr) = (LeastSquaresOptim.LevenbergMarquardt, :lm)
factor = LeastSquaresOptim.QR()
# factor = LeastSquaresOptim.Cholesky()
# factor = LeastSquaresOptim.LSMR()
autodiffs = :central
# autodiffs = :forward

# nls = LeastSquaresProblem(x=x0,f! = king!,output_length=length(x0),autodiff=:central)
nls = LeastSquaresProblem(x=x0,f! = king!,g! = king_g!,output_length=length(x0),autodiff=:central)
# res = optimize!(nls, Dogleg(),iterations=maxIterKing,show_trace=show_trace,
#                 x_tol=p_tol,f_tol=f_tol,g_tol=g_tol,lower=lbs,upper=ubs)
# res = optimize!(nls,optimizer(factor),iterations=maxIterKing,show_trace=show_trace,
#                 x_tol=p_tol,f_tol=f_tol,g_tol=g_tol,lower=lbs,upper=ubs)
res = optimize!(nls,optimizer(),iterations=maxIterKing,show_trace=show_trace,
                x_tol=p_tol,f_tol=f_tol,g_tol=g_tol,lower=lbs,upper=ubs)

xfit = res.minimizer         # the vector of best model1 parameters
niter = res.iterations
is_converged = res.converged
xssr = res.ssr                         # sum(abs2, fcur)
is_xconverged = res.x_converged
is_fconverged = res.f_converged
is_gconverged = res.g_converged
optim_method = res.optimizer

naifit = xfit[1:3:end]
uaifit = xfit[2:3:end]
vthifit = xfit[3:3:end]
datafit = DataFrame(ni=nai0[isp3],nfit=naifit,ui=uai0[isp3],ufit=uaifit,vthi=vthi0[isp3],vthifit=vthifit)

yfit = zeros(3nModel)
king!(yfit, xfit; Ml0=Ml0,l=0)

Jfit = zeros(3nModel,3nModel)
king_g!(Jfit, xfit; Ml0=Ml0,l=0)

@show datafit;
@show is_xconverged,is_fconverged,is_gconverged,norm(yfit);
is_converged,niter,xssr,fmtf12.(xfit)