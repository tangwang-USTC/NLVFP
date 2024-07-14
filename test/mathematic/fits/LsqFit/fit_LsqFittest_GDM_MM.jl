using LsqFit

"""
  Implements box constraints as described in Kanzow, Yamashita, Fukushima (2004;
   J Comp & Applied Math).

  # Keyword arguments
  * `x_tol::Real=1e-8`: search tolerance in x
  * `g_tol::Real=1e-12`: search tolerance in gradient
  * `maxIter::Integer=1000`: maximum number of iterations
  * `show_trace::Bool=false`: print a status summary on each iteration if true
  * `min_step_quality=1e-3`: for steps below this quality, the trust region is shrinked
  * `good_step_quality=0.75`: for steps above this quality, the trust region is expanded
  * `lambda::Real=10`: (inverse of) initial trust region radius
  * `tau=Inf`: set initial trust region radius using the heuristic : tau*maximum(jacobian(df)'*jacobian(df))
  * `lambda_increase=10.0`: `lambda` is multiplied by this factor after step below min quality
  * `lambda_decrease=0.1`: `lambda` is multiplied by this factor after good quality steps
  * `lower,upper=[]`: bound solution to these limits

  if show_trace, the following messages will be printed while running the algorithm:

    (iterCt, sum(abs2, value(df)), g_norm)
    g(x):: = g_norm = norm(J' * value(df), Inf)
    lambda:: = min(lambda_ * lambda, _LAMBDA)
           where `lambda_` denotes `lambda_decrease` or `lambda_increase`
           and `_LAMBDA` denotes `MIN_LAMBDA` or `MAX_LAMBDA`
           Here `MIN_LAMBDA = 1e-16` gives the minimum trust region radius
           and `MAX_LAMBDA = 1e16` gives the maximum trust region radius.
           Notes that `MIN_DIAGONAL = 1e-6` donotes lower bound on values of diagonal matrix used to regularize the trust region step

    dx:: = copy(delta_x) which is limited by `x_tol` and will be updated according to:
          delta_x = ( J'*J + lambda * Diagonal(DtD) ) \ ( -J'*value(df) )

"""
# ua > 0
LL1 = 1
ℓ = LL1 - 1
model1exp(v,p) = exp.(- (v.^2 .+ p[3] ^2)) ./ (2p[3] * v).^0.5 .* besseli.(p[2],2p[3] * v)
model1(v,p) = p[1] * model1exp(v,p)
# Instead of applying the finite difference method
uMax = 20
u = ûa0[isp3]
xvec = 2:nc0
xs = copy(vG0[xvec])
ys = copy(fvL[nvlevel0,LL1,isp3][xvec])
isJacob = false
atolrn = 1e-8

p01 = (2ℓ + 1) / 2 * √(2π)
pp0 = [p01,1/2+ℓ, ûa0[isp3]]  # the initial guess
# initial parameters: `p0 = [wᵢ, 1/2+ℓ, uᵢ]`
p0 = [1.0, 0.5, 1.0]  # the initial guess
np = length(p0)
# Optional upper and/or lower bounds on the free parameters can be passed as an argument
lbs = [0.0, 0, 0.0]
ubs = [Inf, 20.0, 10.0]
lbs = lbs[1:np]
ubs = ubs[1:np]
# Finding out the composite type of fit
# which includes the degrees of freedom (dof()), coefficients (coef(),
# the vector of residuals (.resid) and estimated Jacobian at solution (.jacobian)

# @time  poly = curve_fit(model1,xs,ys,p0,lower=lbs, upper=ubs)
# @time poly = curve_fit(model1,xs,ys,p0;autodiff=:finiteforward)
@time  poly = curve_fit(model1,xs,ys,p0)
p = poly.param         # the array of best model1 parameters
pdof = dof(poly)
presid = poly.resid
pjacob = poly.jacobian
# To estimate errors on the fit parameters.
sigma = stderror(poly)  # Standard error (stderror) of each parameter by estimating errors on the fit parameters.
# to get margin of error and confidence interval of each parameter at 5% significance level
error_margin = margin_error(poly)
# The product of standard error and critical value of each parameter at `alpha` significance level.
confid_interval = confidence_interval(poly,0.05;)
# confidence_interval = confidence_interval(poly,alpha=0.5;atolfit,rtolfit)

# The parameter covariance matrix evaluated at the best fit point.
covar = estimate_covar(poly)
@show ℓ, ûa0
@show p - pp0
yfit = model1(xs,p)
ymodel = model1(xs,pp0)


# # using module LeastSqyaresOptim.jl
# fopt = optimeze(model1(xs,p0))

pM,sigmaM, rnM, modelDMM,MnDMkM = fvLmodelDM(xs,ys,ℓ,u,uMax;atolrn=atolrn)
ppp,sigma0, rn0, modelDM,MnDMk  = fvLmodelDM(xs,ys,ℓ;isJacob=isJacob)
pa, sigmaa, rna, modela,MnDMka = fvLmodel(xs,ys,ℓ,u,uMax,nc0;atolrn=atolrn)
# yfit2 = ppp[1] * modelDMexp(xs,ppp[2:3],ℓ)
yfit2 = modelDM(xs,ppp)
yfitM = modelDMM(xs,pM)
@show pp0 - ppp
@show sigma - sigma0
# Plotting
label = string("y,ℓ=",ℓ)
py= plot(xs,ys,label=label,line=(1,:auto))
label = "y_fit"
py = plot!(xs,yfit,label=label,line=(1,:auto))
label = "y_model"
py = plot!(xs,ymodel,label=label,line=(1,:auto))
label = "y_fit2"
py = plot!(xs,yfit2,label=label,line=(1,:auto))
label = "y_fitM"
py = plot!(xs,yfitM,label=label,line=(1,:auto))

xlabel = string("vG0,u=",fmtf2(ûa0[isp3]))
label = string("err_y")
perry = plot(xs,presid,label=label,line=(1,:auto))

xlabel = string("σ=",fmtf2(norm(sigma)),", σ₀=",fmtf2(norm(sigma0)))
label = string("ys-yMod")
erryMod = ys - ymodel
perryMod = plot(xs,erryMod,label=label,xlabel=xlabel,line=(1,:auto))
label = string("ys-yfit2")
erry_fit2 = ys - yfit2
perryfit2 = plot!(xs,erry_fit2,label=label,line=(1,:auto))
label = string("ys-yfitM")
erry_fitM = ys - yfitM
perryfitM = plot!(xs,erry_fitM,label=label,line=(1,:auto))

xlabel = string("vG0,u=",fmtf2(ûa0[isp3]))
label = string("Rerr_y")
Rerry = (ys ./ yfit .- 1)
pRerry = plot(xs,Rerry,label=label,xlabel=xlabel,line=(1,:auto))
label = string("Rerr_yMod")
RerryMod = (ys ./ ymodel .- 1)
pRerryMod = plot!(xs,RerryMod,label=label,xlabel=xlabel,line=(1,:auto))
label = string("Rerr_yfit2")
Rerryfit2 = (ys ./ yfit2 .- 1)
pRerryfit2 = plot!(xs,Rerryfit2,label=label,xlabel=xlabel,line=(1,:auto))
label = string("Rerr_yfitM")
RerryfitM = (ys ./ yfitM .- 1)
pRerryfitM = plot!(xs,RerryfitM,label=label,xlabel=xlabel,line=(1,:auto))
display(plot(py,perry,perryMod,pRerry,layout=(2,2)))
