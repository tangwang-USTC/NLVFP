using LsqFit

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
uMax = 20.0
atolrn = 1e-8
show_trace = true
maxIter = 100
x_tol = eps(Float64)
g_tol = eps(Float64)
x_tol = 1e-8
g_tol = 1e-12


LL1 = 2
ℓ = LL1 - 1

modeltype = 1
# Instead of applying the finite difference method
u = ûa0[isp3]
xvec = 4:nc0
xs = copy(vG0[xvec])
ys = copy(fvL[nvlevel0,LL1,isp3][xvec])

if modeltype == 1
        model1exp(v,p) = exp.(-p[2] * (v.^2 .+ p[3] ^2)) ./ v.^0.5 .* besseli.(1/2+ℓ,2p[3] * v)
        model1(v,p) = p[1] * model1exp(v,p)
        p01 = (2ℓ + 1) / 2 * √(2π) / (2u).^0.5
        p02 = 1.0
        pp0 = [p01,p02, ûa0[isp3]]  # the initial guess
elseif modeltype == 2
        model1exp(v,p) = exp.(-p[2] * (v.^2 .+ p[3] ^2)) ./ (2p[3] * v).^0.5 .* besseli.(1/2+ℓ,2p[3] * v)
        model1(v,p) = p[1] * model1exp(v,p)
        p01 = (2ℓ + 1) / 2 * √(2π)
        p02 = 1.0
        pp0 = [p01,p02, ûa0[isp3]]  # the initial guess
elseif modeltype == 3
        model1exp(v,p) = exp.(- (v.^2 .+ p[3] ^2)) ./ (2p[3] * v).^0.5 .* besseli.(p[2],2p[3] * v)
        model1(v,p) = p[1] * model1exp(v,p)
        p01 = (2ℓ + 1) / 2 * √(2π)
        p02 = 1 / 2 + ℓ
        pp0 = [p01,p02, ûa0[isp3]]  # the initial guess
elseif modeltype == 4
        model1exp(v,p) = exp.(-(p[2] * v.^2 .+ p[3] ^2)) ./ (2p[3] * v).^0.5 .* besseli.(1/2+ℓ,2p[3] * v)
        model1(v,p) = p[1] * model1exp(v,p)
        p01 = (2ℓ + 1) / 2 * √(2π)
        p02 = 1.0
        pp0 = [p01,p02, ûa0[isp3]]  # the initial guess
elseif modeltype == 5
        # p3 = p3 * p2
        model1exp(v,p) = exp.(-(p[2] * v.^2 .+ p[3] ^2)) ./ v.^0.5 .* besseli.(1/2+ℓ,2p[3] * v)
        model1(v,p) = p[1] * model1exp(v,p)
        p01 = (2ℓ + 1) / 2 * √(2π) / (2u).^0.5
        p02 = 1.0
        pp0 = [p01,p02, ûa0[isp3]]  # the initial guess
elseif modeltype == 6
        model1exp(v,p) = exp.(-(p[2] * v.^2 .+ p[3] ^2)) ./ v.^0.5 .* besseli.(1/2+ℓ,2p[3] * v)
        model1(v,p) = p[1] * model1exp(v,p)
        p01 = (2ℓ + 1) / 2 * √(2π) / (2u).^0.5
        p02 = 1.0
        pp0 = [p01,p02, ûa0[isp3]]  # the initial guess
else
        model1exp(v,p) = exp.(- (v.^2 .+ p[3] ^2)) ./ v.^0.5 .* besseli.(p[2],2p[3] * v)
        model1(v,p) = p[1] * model1exp(v,p)
        p01 = (2ℓ + 1) / 2 * √(2π) / (2u).^0.5
        p02 = 1 / 2 + ℓ
        pp0 = [p01,p02, ûa0[isp3]]  # the initial guess
end
# initial parameters: `p0 = [wᵢ, 1/2+ℓ, uᵢ]`
p0 = [1.0, 0.3, 0.9]  # the initial guess
# p0 = [1.253314, 1/2+ℓ, u]  # the initial guess
# p0 = ones(length(p0))
p0 = pp0 * (1 + 0e-9)
np = length(p0)
# Optional upper and/or lower bounds on the free parameters can be passed as an argument
lbs = [0.0, 0, 0.0]
ubs = [20.0, 20.0, 10.0]
lbs = lbs[1:np]
ubs = ubs[1:np]
# Finding out the composite type of fit
# which includes the degrees of freedom (dof()), coefficients (coef(),
# the vector of residuals (.resid) and estimated Jacobian at solution (.jacobian)

# @time  poly = curve_fit(model1,xs,ys,p0,lower=lbs, upper=ubs)
# @time poly = curve_fit(model1,xs,ys,p0;autodiff=:finiteforward)

@time  poly = curve_fit(model1,xs,ys,p0;
         show_trace=show_trace)
# @time  poly = curve_fit(model1,xs,ys,p0;
#          show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIter)
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
ymodel = model1(xs,pp0)
yfit = model1(xs,p)
erryMod = ys - ymodel
erry_fit = ys - yfit
@show ℓ, ûa0
@show pp0 - p

# # using module LeastSqyaresOptim.jl
# fopt = optimeze(model1(xs,p0))

# Single Harmonic model
@time pM1,sigmaM1, rnM1, modelM1,MnDMkM1 = fvLmodelDM(xs,ys,ℓ,u,uMax;
          show_trace=show_trace,p_tol=x_tol,g_tol=g_tol,maxIterLM=maxIterLM)

@time pM, sigmaM, rnM, modelM,modelv,MnDMkM = fvLmodel(xs,ys,ℓ,u,uMax,atolrn;
               isnormal=isnormal,isforward=isforward,
               show_trace=show_trace,p_tol=x_tol,g_tol=g_tol,maxIterLM=maxIterLM)

# # two-temperature model
# @time pM2,sigmaM2, rnM2, modelM2,MnDMkM2 = fvLmodelDM(xs,ys,ℓ,u,uMax,atolrn;
#           show_trace=show_trace,p_tol=x_tol,g_tol=g_tol,maxIterLM=maxIterLM)

pM2,sigmaM2, rnM2, modelM2,MnDMkM2 = pM1,sigmaM1, rnM1, modelM1,MnDMkM1
# yfit2 = ppp[1] * modelDMexp(xs,ppp[2:3],ℓ)
yfit1 = modelM1(xs,pM1)
yfit2 = modelM2(xs,pM2)
yfitM = modelM(xs,pM)
@show pp0 - pM1
if length(pM) == length(pp0)
        @show pp0 - pM
end
if length(pM2) ≠ length(pp0)
        pMw = [pM2[1],pM2[4]]
        pMws = sum(pMw)
        pppM = [pMws,(pMw' *[pM2[2],pM2[5]] / pMws),(pMw' *[pM2[3],pM2[6]] / pMws)]
        @show pp0 - pppM
end
@show sigma - sigmaM1
# Plotting
label = string("y,ℓ=",ℓ)
py= plot(xs,ys,label=label,line=(1,:auto))
label = "y_fit"
py = plot!(xs,yfit,label=label,line=(1,:auto))
label = "y_fit1"
py = plot!(xs,yfit1,label=label,line=(1,:auto))
label = "y_fit2"
py = plot!(xs,yfit2,label=label,line=(1,:auto))
label = "y_fitM"
py = plot!(xs,yfitM,label=label,line=(1,:auto))
label = "y_model"
py = plot!(xs,ymodel,label=label,line=(1,:auto))

xlabel = string("vG0,u=",fmtf2(ûa0[isp3]))
label = string("presid")
perry = plot(xs,presid,label=label,line=(1,:auto))

xlabel = string("σ=",fmtf2(norm(sigma)),", σ₀=",fmtf2(norm(sigmaM1)))
label = string("ys-yMod")
perryMod = plot(xs,erryMod,label=label,xlabel=xlabel,line=(1,:auto))
label = string("ys-yfit")
perryfit1 = plot!(xs,erry_fit,label=label,line=(1,:auto))
display(perryMod)
label = string("ys-yfit1")
erry_fit1 = ys - yfit1
perryfit1 = plot!(xs,erry_fit1,label=label,line=(1,:auto))
label = string("ys-yfit2")
erry_fit2 = ys - yfit2
perryfit2 = plot!(xs,erry_fit2,label=label,line=(1,:auto))
label = string("ys-yfitM")
erry_fitM = ys - yfitM
perryfitM = plot!(xs,erry_fitM,label=label,line=(1,:auto))

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
label = string("Rerr_yMod")
RerryMod = (ys ./ ymodel .- 1)
pRerryMod = plot!(xs,RerryMod,label=label,xlabel=xlabel,line=(1,:auto))
display(plot(py,perry,perryMod,pRerry,layout=(2,2)))
