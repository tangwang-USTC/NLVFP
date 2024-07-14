using LsqFit

model(v,p) = p[1] * modelMexp(v,p[2:3])

function jacobian_model(v::AbstractVector{T},p;L=1.0) where{T}

  J = Array{T}(undef,length(v),length(p))
  J[:,1] = modelMexp(v,p[2:3])                         # dmodel/dp[1]
  J[:,2] = - (v .- p[3]).^2 * p[1] .* J[:,1]    # dmodel/dp[2]
  J[:,3] = 2p[1] * p[2] * (v .- p[3]) .* J[:,1] # dmodel/dp[3]
  J
end

uMax = 20.0
u = ûa0[isp3]
LL1 = 1
xs = copy(vG0)
ys = copy(fvL[nvlevel0,LL1,isp3])
isJacob = false
# isJacob = true
atolrn = 1e-8

pp0 = [1.0,1.0,0.0]
p0 = [0.1, 0.1, 0.1]  # the initial guess
p0 = 0p0

# Optional upper and/or lower bounds on the free parameters can be passed as an argument
lbs = [0.0, 0.0, -10.0]
ubs = [20.0, 20.0, 10.0]
# Finding out the composite type of fit
# which includes the degrees of freedom (dof()), coefficients (coef(),
# the vector of residuals (.resid) and estimated Jacobian at solution (.jacobian)

# @time poly = curve_fit(model,xs,ys,p0;autodiff=:finiteforward)

"""
  Implements box constraints as described in Kanzow, Yamashita, Fukushima (2004;
   J Comp & Applied Math).

  # Keyword arguments of the Levenberg-Marquardt (L-M) algorithm.
  * `x_tol::Real=1e-8`: search tolerance in x, the parameters of model.
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
           which is limited by `g_tol`

    lambda:: = min(lambda_ * lambda, _LAMBDA)
           where `lambda_` denotes `lambda_decrease` or `lambda_increase`
           and `_LAMBDA` denotes `MIN_LAMBDA` or `MAX_LAMBDA`
           Here `MIN_LAMBDA = 1e-16` gives the minimum trust region radius
           and `MAX_LAMBDA = 1e16` gives the maximum trust region radius.
           Notes that `MIN_DIAGONAL = 1e-6` donotes lower bound on values of diagonal matrix used to regularize the trust region step

    dx:: = copy(delta_x)
          which is limited by `x_tol` and will be updated according to:
"""

show_trace = true
maxIter = 100
x_tol = eps(Float64)
g_tol = eps(Float64)
@time  poly = curve_fit(model,xs,ys,p0;
         show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIter)

# @time  poly = curve_fit(model,jacobian_model,xs,ys,p0)

# @time  poly = curve_fit(model,xs,ys,p0,lower=lbs, upper=ubs)

p = poly.param   # `= coef(poly)`, the array of best model parameters
pdof = dof(poly)
presid = poly.resid
pjacob = poly.jacobian
# The parameter covariance matrix evaluated at the best fit point.
covar = estimate_covar(poly)
# To estimate errors on the fit parameters: `sigm = √(abs(diag(covar)))`
sigma = stderror(poly)  # Standard error (stderror) of each parameter by estimating errors on the fit parameters.
# to get margin of error and confidence interval of each parameter at 5% significance level
error_margin = margin_error(poly)
# The product of standard error and critical value of each parameter at `alpha` significance level.
confid_interval = confidence_interval(poly,0.001;)
# confidence_interval = confidence_interval(poly,alpha=0.5;atolfit,rtolfit)

@show LL1, ûa0
@show p - p0
yfit = model(xs,p)

# # using module LeastSqyaresOptim.jl
# fopt = optimeze(model(xs,p0))

# two-temperature model
@time pM2,sigmaM2, rnM2, modelM2,MnDMkM2 = fvLmodelDM(xs,ys,uMax,atolrn;
          show_trace=show_trace,p_tol=x_tol,g_tol=g_tol,maxIterLM=maxIterLM)
# Single Maxwellian model
@time pM1,sigmaM1, rnM1, modelM1,MnDMkM1 = fvLmodelDM(xs,ys;
          show_trace=show_trace,p_tol=x_tol,g_tol=g_tol,maxIterLM=maxIterLM)

@time pM, sigmaM, rnM, modelM,modelv,MnDMkM = fvLmodel(xs,ys,ℓ,u,uMax,atolrn;
               isnormal=isnormal,isforward=isforward,
               show_trace=show_trace,p_tol=x_tol,g_tol=g_tol,maxIterLM=maxIterLM)
# yfit1 = pM1[1] * modelMexp(xs,pM1[2:3])
yfit1 = modelM1(xs,pM1)
yfit2 = modelM2(xs,pM2)
yfitM = modelM(xs,pM)
pMw = [pM2[1],pM2[4]]
pMws = sum(pMw)
pppM = [pMws,(pMw' *[pM2[2],pM2[5]] / pMws),(pMw' *[pM2[3],pM2[6]] / pMws)]
@show p - pppM
@show p - pM1
@show p - pM
@show sigma - sigmaM1

label = "y"
py= plot(xs,ys,label=label,line=(1,:auto))
label = "y_fit"
py = plot!(xs,yfit,label=label,line=(1,:auto))
label = "y_fit1"
py = plot!(xs,yfit1,label=label,line=(1,:auto))
label = "y_fit2"
py = plot!(xs,yfit2,label=label,line=(1,:auto))
label = "y_fitM"
py = plot!(xs,yfitM,label=label,line=(1,:auto))

xlabel = string("vG0,u=",fmtf2(ûa0[isp3]))
label = "Rerr_fit"
Rerry = (ys ./ yfit .- 1)
pRerry = plot(xs,Rerry,label=label,xlabel=xlabel,line=(1,:auto))
label = "Rerr_fit1"
Rerry1 = (ys ./ yfit1 .- 1)
pRerry = plot!(xs,Rerry1,label=label,xlabel=xlabel,line=(1,:auto))
label = "Rerr_fit2"
Rerry2 = (ys ./ yfit2 .- 1)
pRerry = plot!(xs,Rerry2,label=label,xlabel=xlabel,line=(1,:auto))
label = "Rerr_fitM"
RerryM = (ys ./ yfitM .- 1)
pRerry = plot!(xs,RerryM,label=label,xlabel=xlabel,line=(1,:auto))
display(plot(py,pRerry,layout=(2,1)))
