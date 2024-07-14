using LsqFit

using LinearAlgebra
using BenchmarkTools
# using Zygote

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

"""
  When `p0 = rand(length(pp0))`, there are three possible rerults of the curve_fitting functions of `ys`:

  I, Success with high precision;
  II, Failure with low precision
  III, Failure owing to singularity.
"""

"""
  Generally in the harmonic models, the `ℓᵗʰ`-order harmonic of drift-Maxwellian distribution will be:

    `f̂ₗ(v̂,û) := (2L + 1)/2 * √2π * exp(- (v̂² + û²)) / ξ * besseli(1/2+ℓ,ξ)`.

  When `û → 0`, the model can be approximated as:

    `f̂ₗ(v̂,û→0) := 1 / (2L - 1)!! * ξᴸ * exp(- p₁ * v̂²) * (1 - û² * (1 / p₁ - 2/(2L + 3) * v̂²)) + 𝒪(û⁴)`

  and when `û = 0` strictly gives:

    `f̂ₗ(v̂;û=0) := 1 / (2L - 1)!! * ξᴸ * exp(- p₁ * v̂²) ≡ 0, L ≥ 1`.

  That means `f̂ₗ(v̂,û=0)` will be zero except `ℓ = 0`, which can be rewritten as:

    `f̂ₗ(v̂) := exp(- p₁ * v̂²)) * δₗ⁰`

  Here

    `ξ = 2û * v̂`.

  Thus, the model when `ℓ ≥ 1` will be:

    ` 𝒻̂ₗ(v̂,û) = ∑ₖ(f̂ₗ(v̂ₖ,ûₖ)), k ∈ [1:nDM]`.

  and when `ℓ = 0`,

    ` 𝒻̂₀(v̂,û) = ∑ₖ(f̂₀(v̂ₖ)) + ∑ₛ(f̂₀(v̂ₖ,ûₖ)), k ∈ [1:nM], s ∈ [1:nDM]`.

"""
# # Harmonic models for drift-Maxwellian distribution.
# `p = [p1, p2]` where `p1 ~ vₜₕ` and `p2 ~ û`.
modelDMexp(v,p,ℓ::Int) = exp.(- (p[1] * v.^2 .+ p[2]^2 / p[1])) ./ v.^(ℓ + 0.5) .* besseli.(0.5+ℓ,2p[2] * v)

# ua > 0
LL1 = 2
# # parameters
# fileweightfunction = :weightv  # [:weightv , :weight2uv]
issolvefDM = false
show_trace = true
istol = true
isnormal = false
isforward = false
timeType = :time    # [:time, :btime, :nothing]
maxIterLM = 100
n10 = -7             # [-5:10],  ys_min = 10^(-5 * n10) which sets the minimum value of vector `ys`.
dnxs = 1
p_noise_ratio = 1e-6 # Relative error of the initial guess parameters `p0`
p_noise_abs = 1e-6 # Relative error of the initial guess parameters `p0`.
x_tol = eps(Float64) * 1e-4
g_tol = eps(Float64) * 1e-4
fileweightfunction == :weightv
if LL1 == LL1
    ℓ = LL1 - 1
    # Instead of applying the finite difference method
    atolrn = 1e-5
    x_tol = eps(Float64)
    g_tol = eps(Float64) * 1e-1
    # x_tol = 1e-8
    # g_tol = 1e-12

    u = ûa0[isp3]
    uMax = min(20.0,3u)
    uMax = 1.0
    xs0 = copy(vG0)
    xs0[1] == 0.0 ? (xvec = 2:nc0) : (xvec = 1:nc0)
    xs0 =xs0[xvec]
    if ℓ == 0
        ys0 = copy(fvL[nvlevel0,LL1,isp3][xvec])
    elseif ℓ == 1
        ys0 = copy(fvL[nvlevel0,LL1,isp3][xvec]) ./ vG0[xvec]
    else
        ys0 = copy(fvL[nvlevel0,LL1,isp3][xvec]) ./ vG0[xvec].^ℓ
    end
    norm(ys0) > eps(Float64) || error("`norm(ys) = 0.0` and no fitting is needed.")
    # Normalization
    ymax = maximum(ys0)
    ymax = 10^(round(log10(ymax)))
    # ymax = 1.0
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
    xs = xs0[is_ys_0]
    ys = ys0[is_ys_0]
    nxs = length(xs)
    if dnxs ≥ 2
        xs = xs[1:dnxs:end]
        ys = ys[1:dnxs:end]
    end
    # # # using module LeastSqyaresOptim.jl
    # # fopt = optimeze(model1(xs,p0))
    # is_avv = 0  # (= 0 default) which is not proposed to be used.
    # isJacob = 0
end
if u == 0.0
    cp1t = 1.0
else
    if fileweightfunction == :weight2uv
        cp1t = (2ℓ+1)/2.0^0.5 * sqrtpi / ymax
    elseif fileweightfunction == :weightv
        cp1t = (2ℓ+1)/2 * sqrtpi / u.^0.5 / ymax
    end
end
pp0 =[cp1t, 1.0, u]
# p=[1.0, 1.0, 1.02]
p0 = [1.0, 1.0, 0.4u]
# p0 = ones(3)
p_noise_rel = rand(3) * p_noise_ratio
# p_noise_rel0 = p_noise_rel
# p_noise_rel = p_noise_rel0
p0 = pp0 .* (1.0 .+ p_noise_rel ) + p_noise_abs * ones(3)
p0[3] = 0.1
# p0 = [0.7353054693001071, 0.9668478296540347, 0.32480727659949726]
# p0 = rand(3)
# p0 = [1.0, 0.5, u]
lbs = [0.0, 0.0, 0.0]
ubs = [+Inf, 20.0, uMax]
# lbs = [0.0, 1e-5, 0.1u]
# ubs = [Inf, 5.0, uMax]
@show (ℓ, nxs , fmtf2(ymax)), fmtf2.(u) ,fmtf2.(p_noise_rel)
modelDM(v,p) = p[1] * modelDMexp(v,p[2:3],ℓ)
println()
if isforward == 0
    if istol == 1
        if timeType == :time
            @time  poly = curve_fit(modelDM,xs,ys,p0,lower=lbs, upper=ubs;
                      show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
        elseif timeType == :btime
            @time  poly = curve_fit(modelDM,xs,ys,p0,lower=lbs, upper=ubs;
                      show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
            @btime  poly = curve_fit(modelDM,xs,ys,p0,lower=lbs, upper=ubs;
                      show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
        else
            poly = curve_fit(modelDM,xs,ys,p0,lower=lbs, upper=ubs;
                      show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
        end
    else
        ertfgh
        if timeType == :time
        elseif timeType == :btime
        else
        end
        @btime  poly = curve_fit(modelDM,xs,ys,p0,lower=lbs, upper=ubs;show_trace=show_trace)
    end
else
    wrtgnh
    if istol == 1
        if timeType == :time
        elseif timeType == :btime
        else
        end
        @btime  poly = curve_fit(modelDM,xs,ys,p0,lower=lbs, upper=ubs;autodiff=:forwarddiff,
                  show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
        # @btime  poly = curve_fit(modelDM,xs,ys,p0,lower=lbs, upper=ubs;
        #           show_trace=show_trace,x_tol=x_tol,g_tol=g_tol,maxIter=maxIterLM)
    else
        if timeType == :time
        elseif timeType == :btime
        else
        end
        @btime  poly = curve_fit(modelDM,xs,ys,p0,lower=lbs, upper=ubs;autodiff=:forwarddiff)
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
        @time nDM,pM1,sigmaM1, rnM1, modelM1 = fvLmodelDM(xs,ys,ℓ,u,uMax,atolrn;
                p=p0,isforward=isforward,show_trace=show_trace,p_tol=x_tol,g_tol=g_tol,maxIterLM=maxIterLM)
    else
        @time nDM,pM1,sigmaM1, rnM1, modelM1 = fvLmodelDM(xs,ys,ℓ,u,uMax;p=p0,show_trace=show_trace)
    end
else
    pM1 = p
end
@show (ℓ, nxs , fmtf2(ymax)), fmtf2.(u) ,fmtf2.(p_noise_rel)
println()
@show p ./ pp0 .- 1
ymodel = modelDM(xs0,pp0)
yfit = modelDM(xs0,p)
erryMod = ys0 - ymodel
erry_fit = ys0 - yfit
RerryMod = (ys0 ./ ymodel .- 1)
Rerry = (ys0 ./ yfit .- 1)
if issolvefDM == 1
    yfit1 = modelM1(xs0,pM1)
else
    yfit1 = yfit
end
@show pM1 ./ pp0 .- 1
erry_fit1 = ys0 - yfit1
Rerry1 = (ys0 ./ yfit1 .- 1)

# Plotting
wline = 3
title = string("dnx=",dnxs,",n10=",n10,",yM=",fmtf2(ymax),",p_n=",fmtf2(p_noise_ratio))
if 1 == 1
    1
    label = string("y,ℓ=",ℓ)
    py= plot(xs0,ys0,label=label,line=(wline,:auto),title=title)
    label = "y_fit"
    py = plot!(xs0,yfit,label=label,line=(wline,:auto))
    label = "y_fit1"
    py = plot!(xs0,yfit1,label=label,line=(wline,:auto))
    label = "y_model"
    py = plot!(xs0,ymodel,label=label,line=(wline,:auto))
    2
    xlabel = string("vG0,u=",fmtf2(ûa0[isp3]))
    label = string("presid")
    if length(xs) == length(presid)
        perry = plot(xs,presid,label=label,line=(wline,:auto))
    else
        perry = plot()
    end
    3
    # xlabel = string("σ=",fmtf2(norm(sigma)),", σ₀=",fmtf2(norm(sigmaM1)))
    xlabel = string("σ=",fmtf2(norm(sigma)))
    label = string("ys-yMod")
    perryMod = plot(xs0,erryMod,label=label,xlabel=xlabel,line=(wline,:auto))
    label = string("ys-yfit")
    perryfit = plot!(xs0,erry_fit,label=label,line=(wline,:auto))
    # display(perryMod)
    label = string("ys-yfit1")
    perryfit1 = plot!(xs0,erry_fit1,label=label,line=(wline,:auto))
    4
    xlabel = string("vG0,u=",fmtf2(ûa0[isp3]))
    label = string("Rerr_yMod")
    pRerryMod = plot(xs0,RerryMod,legend=legendtL,label=label,xlabel=xlabel,line=(wline,:auto))
    label = "Rerr_fit"
    pRerry = plot!(xs0,Rerry,label=label,xlabel=xlabel,line=(wline,:auto))
    label = "Rerr_fit1"
    pRerry = plot!(xs0,Rerry1,label=label,xlabel=xlabel,line=(wline,:auto))
    display(plot(py,perry,perryMod,pRerry,layout=(2,2)))
end

label = string("y/v^ℓ,ℓ=",ℓ)
pdp1 = plot!(xs0,modelDMexp(xs0,pp0[2:3],ℓ),label=label,line=(wline,:auto))

pdp1 = plot!(xs0,modelDMexp(xs0,p[2:3],ℓ),label=label,line=(wline,:auto))
