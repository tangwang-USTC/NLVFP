using LsqFit

using LinearAlgebra
using BenchmarkTools
# using Zygote
using Plots
include(joinpath(pathroot,"src\\Collision\\weightfunctions_vMp1.jl"))

# fileweightfunction = :weight2uv

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

    dx:: = copy(delta_x=δx) which is limited by `x_tol` and will be updated according to:

  Optimization will be terminated when

    `g(x) ≤ g_tol` or `max(δx) ≤ x_tol`.
"""

# Key points: The stagnation points
# Important affects: The guessing values

"""
  When `p0 = rand(length(pp0))`, there are three possible rerults of the curve_fitting functions of `ys`:

  I, Success with high precision;
  II, Failure with low precision
  III, Failure owing to singularity.
"""
# ua > 0
LL1 = 1
# # parameters
# fileweightfunction = :weightv  # [:weightv , :weight2uv]
issolvefDM = false
show_trace = true
isnvlevel0 = true
istol = true
isnormal = false
isforward = false
timeType = :btime    # [:time, :btime, :nothing]
timeType == :btime ? show_trace = false : nothing
maxIterLM = 200
n10 = 7             # [-5:1:10],  ys_min = 10^(-5 * n10) which sets the minimum value of vector `ys`.
dnvs = 1
p_noise_ratio = 1e-1 # Relative error of the initial guess parameters `p0`
p_noise_abs = 1e-2 # Absolute error of the initial guess parameters `p0`.
x_tol = eps(Float64) * 1e-1
g_tol = eps(Float64) * 1e-1
if LL1 == LL1
    # Instead of applying the finite difference method
    # x_tol = 1e-8
    # g_tol = 1e-12
    atolrn = 1e-5
    u = ûa0[isp3]
    uMax = min(20.0,3u)
    if isnvlevel0 == 1
        vs0 = copy(vG0)
        vs0[1] == 0.0 ? (xvec = 2:nc0) : (xvec = 1:nc0)
        vs0 =vs0[xvec]
        ys0 = copy(fvL[nvlevel0,LL1,isp3][xvec])
    else
        vs0 = copy(vGk)
        vs0[1] == 0.0 ? (xvec = 2:nck) : (xvec = 1:nck)
        vs0 =vs0[xvec]
        ys0 = copy(fvL[:,LL1,isp3][xvec])
    end
    norm(ys0) > eps(Float64) || error("`norm(ys) = 0.0` and no fitting is needed.")
    # Normalization
    ymax = maximum(ys0)
    ymax = 10^(round(log10(ymax)))
    # ymax = 1.0
    ℓ = LL1 - 1
    if isnormal == 1 && ymax ≠ 1.0
        ys0 = ys0 / ymax
    else
        ymax = 1.0
    end
    # filter
    if n10 < 0
        is_ys_0 = ys0 .> 2eps(Float64) * 10.0^-(n10)
    else
        is_ys_0 = ys0 .> 2eps(Float64) * 10.0^-(5 * n10)
    end
    vs = vs0[is_ys_0]
    ys = ys0[is_ys_0]
    nvs = length(vs)
    if dnvs ≥ 2
        vs = vs[1:dnvs:end]
        ys = ys[1:dnvs:end]
    end
end
# `p1` in theory
cp1t = 1.0 / ymax
# cp1t = 1.0 / u^0.5 / ymax
pp0 =[cp1t, 1.0, u]
# p=[1.0, 1.0, 1.02]
p0 = [1.0, 1.0, 0.4u]
# p0 = ones(3)
p_noise_rel = rand(3) * p_noise_ratio
# p_noise_rel0 = p_noise_rel
# p_noise_rel = p_noise_rel0
p0 = pp0 .* (1.0 .+ p_noise_rel ) + p_noise_abs * rand(3)
# p0[3] = 0.6
# p0 = rand(3)
# p0pp = 1p0
# p0 = p0pp
# p0 = pp0
lbs = [0.0, 0.0, 0.0]
uMax < p0[3] ? uMax = 2p0[3] : nothing
ubs = [+Inf, 20.0, uMax]

@show (ℓ, nvs , fmtf2(ymax)), fmtf2.(u) ,fmtf2.(p_noise_rel)
modelDM(v,p) = p[1] * modelDMexp(v,p[2:3],ℓ)
# modelDM(v,p) = p[1] * modelDMexpp1(v,p[2:3],ℓ)
println()
if isforward == 0
    if istol == 1
        if timeType == :time
            @time  poly = curve_fit(modelDM,vs,ys,p0,lower=lbs, upper=ubs;
                      show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
        elseif timeType == :btime
            @time  poly = curve_fit(modelDM,vs,ys,p0,lower=lbs, upper=ubs;
                      show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
            @btime  poly = curve_fit(modelDM,vs,ys,p0,lower=lbs, upper=ubs;
                      show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
        else
            poly = curve_fit(modelDM,vs,ys,p0,lower=lbs, upper=ubs;
                      show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
        end
    else
        ertfgh
        if timeType == :time
        elseif timeType == :btime
        else
        end
        @btime  poly = curve_fit(modelDM,vs,ys,p0,lower=lbs, upper=ubs;show_trace=show_trace)
    end
else
    wrtgnh
    if istol == 1
        if timeType == :time
        elseif timeType == :btime
        else
        end
        @btime  poly = curve_fit(modelDM,vs,ys,p0,lower=lbs, upper=ubs;autodiff=:forwarddiff,
                  show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
        # @btime  poly = curve_fit(modelDM,vs,ys,p0,lower=lbs, upper=ubs;
        #           show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
    else
        if timeType == :time
        elseif timeType == :btime
        else
        end
        @btime  poly = curve_fit(modelDM,vs,ys,p0,lower=lbs, upper=ubs;autodiff=:forwarddiff)
    end
end
p = poly.param         # the array of best model1 parameters
pdof = dof(poly)
presid = poly.resid
pjacob = poly.jacobian
if 1 == 1  # momentums
    # To estimate errors on the fit parameters. but failure when model is independent on some parameters.
    sigma = stderror(poly)  # Standard error (stderror) of each parameter by estimating errors on the fit parameters.
    # to get margin of error and confidence interval of each parameter at 5% significance level
    error_margin = margin_error(poly)
    # The product of standard error and critical value of each parameter at `alpha` significance level.
    confid_interval = confidence_interval(poly,0.05;)
    # confidence_interval = confidence_interval(poly,alpha=0.5;atolfit,rtolfit)
end
println()
if issolvefDM == 1
    if istol == 1
        @time pM1,sigmaM1, rnM1, modelM1,modelM1v0,nDM = fvLmodelDM(vs,ys,ℓ,u,uMax;p=p0,
                show_trace=show_trace,p_tol=x_tol,g_tol=g_tol,maxIterLM=maxIterLM)
    else
        @time pM1,sigmaM1, rnM1, modelM1,modelM1v0,nDM = fvLmodelDM(vs,ys,ℓ,u,uMax;p=p0,show_trace=show_trace)
    end
else
    pM1 = p
end
@show (ℓ, nvs , fmtf2(ymax)), fmtf2.(u) ,fmtf2.(p_noise_rel)
println()
@show p ./ pp0 .- 1

##
ymodel = modelDM(vs0,pp0)
yfit = modelDM(vs0,p)
erryMod = ys0 - ymodel
erry_fit = ys0 - yfit
RerryMod = (ys0 ./ ymodel .- 1)
Rerry = (ys0 ./ yfit .- 1)
if issolvefDM == 1
    yfit1 = modelM1(vs0,pM1)
else
    yfit1 = yfit
end
@show pM1 ./ pp0 .- 1
erry_fit1 = ys0 - yfit1
Rerry1 = (ys0 ./ yfit1 .- 1)

# Plotting
wline = 3
title = string("dnx=",dnvs,",n10=",n10,",yM=",fmtf2(ymax),",p_n=",fmtf2(p_noise_ratio))
if 1 == 1
    1
    label = string("y,ℓ=",ℓ)
    py= plot(vs0,ys0,label=label,line=(wline,:auto),title=title)
    label = "y_fit"
    py = plot!(vs0,yfit,label=label,line=(wline,:auto))
    label = "y_fit1"
    py = plot!(vs0,yfit1,label=label,line=(wline,:auto))
    label = "y_model"
    py = plot!(vs0,ymodel,label=label,line=(wline,:auto))
    2
    xlabel = string("vG0,u=",fmtf2(ûa0[isp3]))
    label = string("presid")
    if length(vs) == length(presid)
        perry = plot(vs,presid,label=label,line=(wline,:auto))
    else
        perry = plot()
    end
    3
    # xlabel = string("σ=",fmtf2(norm(sigma)),", σ₀=",fmtf2(norm(sigmaM1)))
    xlabel = string("σ=",fmtf2(norm(sigma)))
    label = string("ys-yMod")
    perryMod = plot(vs0,erryMod,label=label,xlabel=xlabel,line=(wline,:auto))
    label = string("ys-yfit")
    perryfit = plot!(vs0,erry_fit,label=label,line=(wline,:auto))
    # display(perryMod)
    label = string("ys-yfit1")
    perryfit1 = plot!(vs0,erry_fit1,label=label,line=(wline,:auto))
    4
    xlabel = string("vG0,u=",fmtf2(ûa0[isp3]))
    label = string("Rerr_yMod")
    pRerryMod = plot(vs0,RerryMod,legend=legendtL,label=label,xlabel=xlabel,line=(wline,:auto))
    label = "Rerr_fit"
    pRerry = plot!(vs0,Rerry,label=label,xlabel=xlabel,line=(wline,:auto))
    label = "Rerr_fit1"
    pRerry = plot!(vs0,Rerry1,label=label,xlabel=xlabel,line=(wline,:auto))
    display(plot(py,perry,perryMod,pRerry,layout=(2,2)))

end
