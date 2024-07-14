using LeastSquaresOptim
using LsqFit
using Optim

using LinearAlgebra
using BenchmarkTools
using Statistics, Test
using Plots

include(joinpath(pathroot,"src\\Collision\\weightfunctions_vMp1.jl"))

"""
  @printf io "Results of Optimization Algorithm\n"
  @printf io " * Algorithm: %s\n" r.optimizer
  @printf io " * Minimizer: [%s]\n" join(r.minimizer, ",")
  @printf io " * Sum of squares at Minimum: %f\n" r.ssr = sum(abs2, fres)
  @printf io " * Iterations: %d\n" r.iterations
  @printf io " * Convergence: %s\n" converged(r)
  @printf io " * |x - x'| < %.1e: %s\n" r.x_tol r.x_converged
  @printf io " * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" r.f_tol r.f_converged
  @printf io " * |g(x)| < %.1e: %s\n" r.g_tol r.g_converged
  @printf io " * Function Calls: %d\n" r.f_calls
  @printf io " * Gradient Calls: %d\n" r.g_calls
  @printf io " * Multiplication Calls: %d\n" r.mul_calls
"""

# ua > 0
LL1 = 1                # When `ℓ ≫ 1`, i.e. `ℓ ~ 10`, local optimal solution is given freqently.
restart_fit_count = 0  # `= 0` default which is `∈[0, 1]`, means the first outer iteration.
                       # or else,


is_DL_finished = false
# is_DL_finished = true

##
if restart_fit_count == 0
    restart_fit = 0       # (= 20 default) which is `∈ [0, N⁺]`, the number of restart iterations of the fit procedure.
                          # The best value is influenced by the guess values `p0` and `maxIterLM`.
                          # which is equivalent to the number of initial guessing.
                          # Even though, a local optimal solution may be given and
                          # restart with another algorithm during `Dogleg` and `LevenbergMarquardt maybe is beneficial.
                          # 'LevenbergMarquardt' is recommended as the initial algorithm
                          # owing to its robusrness than `Dogleg` when the guess values are so poor.
                          # If `restart_fit= 0`, means no restart is used when fitting.
else
    restart_fit = 1
end
if restart_fit_count == 0
    is_maxIterLM = true    # (default = true) Applying `LevenbergMarquardt` algorithm firstly.
                           # Whether the innermost iteration of fitting reaches the `maxIterLM`.
else
    @warn("`is_maxIterLM = true` means `restart_fit_count = 0` firstly")
end
## # parameters
# fileweightfunction = :weightv  # [:weightv , :weight2uv]
fitpkg = :LeastSquaresOptim     # Better than `LsqFit.jl` in most cases.
# fitpkg = :LsqFit              # Instead by package `LeastSquaresOptim.jl`
if is_maxIterLM == true && is_DL_finished ≠ true
    fitmethod = :LevenbergMarquardt
    if restart_fit_count == 0
        restart_fit = 0
    else
        restart_fit = 1
    end
else
    fitmethod = :Dogleg    # When the guess values is near the real solution,
                           # i.e. the relative errors is `< 1e-6`, the precision
                           # is no worse than `LevenbergMarquardt' method;
                           # However, when the guess values is very poor, i.e. the relative errors is `1e-0`,
                           # `Dogleg' method is so poor that always fails to converge.
                           # When `ℓ ≫ 0`, i.e., `ℓ = 10`, worse than `LevenbergMarquardt` method.
    restart_fit = 1
end
isforward = true
autodiff = :forward  #
autodiff = :central  # A little more complicated but maybe obtaining no benefits.
factorMethod = :QR       # factorMethod = [:QR, :Cholesky, :LSMR]
# factorMethod = :Cholesky
# factorMethod = :LSMR
issolvefDM = true
show_trace = true
isnvlevel0 = true
istol = true
isnormal = true
# isnormal = false
timeType = :time    # [:time, :btime, :nothing]
timeType == :btime ? show_trace = false : nothing
maxIterLM = 100
ratioeps = 1e-0
p_tol = eps(Float64) * ratioeps
g_tol = eps(Float64) * ratioeps
f_tol = eps(Float64) * ratioeps
n10 = 0             # [-5:1:10],  ys_min = 10^(-5 * n10) which sets the minimum value of vector `ys`.
dnvs = 1
p_noise_ratio = 1e-1 # Relative error of the initial guess parameters `p`
p_noise_abs = 1e-1 # Absolute error of the initial guess parameters `p`.
# p_tol=1e-50
# f_tol=1e-36
# g_tol=1e-50
if LL1 == LL1
    # Instead of applying the finite difference method
    # p_tol = 1e-8
    # g_tol = 1e-12
    atolrn = 1e-5
    u = ûa0[isp3]
    uMax = min(20.0,3u)
    fvLa = fvL[:,:,isp3]
    if isnvlevel0 == 1
        vs0 = copy(vG0)
        vs0[1] == 0.0 ? (xvec = 2:nc0) : (xvec = 1:nc0)
        vs0 =vs0[xvec]
        ys0 = copy(fvLa[nvlevel0,LL1][xvec])
    else
        vs0 = copy(vGk)
        vs0[1] == 0.0 ? (xvec = 2:nck) : (xvec = 1:nck)
        vs0 =vs0[xvec]
        ys0 = copy(fvLa[:,LL1][xvec])
    end
    norm(ys0) > eps(Float64) || error("`norm(ys) = 0.0` and no fitting is needed.")
    # Normalization
    ymax = maximum(ys0)
    ymax = 10^(round(log10(ymax)))
    ℓ = LL1 - 1
    if isnormal == 1 && ymax ≠ 1.0
        yscale = copy(ymax)
        ys0 = ys0 / yscale
    else
        yscale = 1.0
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
if ymax ≠ 1
    p_tol = eps(Float64) * ymax
    g_tol = eps(Float64) * ymax
    f_tol = eps(Float64) * ymax
end
# `p1` in theory
cp1t = 1.0 / yscale
# cp1t = 1.0 / u^0.5 / yscale
pp0 =[cp1t, 1.0, u]
# p=[1.0, 1.0, 1.02]
# p0 = [1.0, 1.0, 0.4u]
# p0 = ones(3)
p_noise_rel = rand(3) * p_noise_ratio
# p_noise_rel0 = p_noise_rel
# p_noise_rel = p_noise_rel0
p0 = pp0 .* (1.0 .+ p_noise_rel ) + p_noise_abs * rand(3)
# # p0 = rand(3)
if restart_fit > 0
    p0pp = 1p
    p0 = copy(p0pp)
elseif restart_fit == 0
else
    # p0pp = 1p0
    p0 = copy(p0pp)
end
# p0 = pp0
# p0[1:2] = pp0[1:2]
# p0[3] = 0.6
@show p0
lbs = [0.0, 0.0, 0.0]
uMax < p0[3] ? uMax = 2p0[3] : nothing
ubs = [10/yscale, 20.0, uMax]
if fitpkg == :LsqFit
    fitmethod = :LevenbergMarquardt
    optimizer_abbr = :LsqFit_lm
elseif fitpkg == :LeastSquaresOptim
    if fitmethod == :LevenbergMarquardt
        (optimizer, optimizer_abbr) = (LeastSquaresOptim.LevenbergMarquardt, :LSO_LM)
    elseif fitmethod == :Dogleg
        (optimizer, optimizer_abbr) = (LeastSquaresOptim.Dogleg, :LSO_DL)
    end
    if factorMethod == :QR
        factor = LeastSquaresOptim.QR()
    elseif factorMethod == :Cholesky
        factor = LeastSquaresOptim.Cholesky()
    elseif factorMethod == :LSMR
        factor = LeastSquaresOptim.LSMR()
    end
elseif fitpkg == :Optim
    optimizer = :NelderMead
    optimizer = :LBFGS
end
println()
modelDM(v,p) = p[1] * modelDMexp(v,p[2:3],ℓ)
# modelDM(v,p) = p[1] * modelDMexpp1(v,p[2:3],ℓ)
function fres!(res,modelDM,ys,vs,p)

    # for i in 1:length(vs)
    #     res[i] = ys[i] - modelDM(vs[i],p)
    # end
    # res = ys - modelDM(vs,p)   # not correct, by why ?
    res[:] = ys - modelDM(vs,p)
end
p = copy(p0)
if fitpkg == :LsqFit
    if isforward == 0
        if istol == 1
            if timeType == :time
                @time  poly = curve_fit(modelDM,vs,ys,p,lower=lbs, upper=ubs;
                          show_trace=show_trace,x_tol=p_tol,g_tol=g_tol,maxIter=maxIterLM)
            elseif timeType == :btime
                @time  poly = curve_fit(modelDM,vs,ys,p,lower=lbs, upper=ubs;
                          show_trace=show_trace,x_tol=p_tol,g_tol=g_tol,maxIter=maxIterLM)
                @btime  poly = curve_fit(modelDM,vs,ys,p,lower=lbs, upper=ubs;
                          show_trace=show_trace,x_tol=p_tol,g_tol=g_tol,maxIter=maxIterLM)
            else
                poly = curve_fit(modelDM,vs,ys,p,lower=lbs, upper=ubs;
                          show_trace=show_trace,x_tol=p_tol,g_tol=g_tol,maxIter=maxIterLM)
            end
        else
            ertfgh
            if timeType == :time
            elseif timeType == :btime
            else
            end
            @btime  poly = curve_fit(modelDM,vs,ys,p,lower=lbs, upper=ubs;show_trace=show_trace)
        end
    else
        if istol == 1
            if timeType == :time
                @time  poly = curve_fit(modelDM,vs,ys,p,lower=lbs,upper=ubs;autodiff=autodiff,
                          show_trace=show_trace,x_tol=p_tol,g_tol=g_tol,maxIter=maxIterLM)
            elseif timeType == :btime
                @time  poly = curve_fit(modelDM,vs,ys,p,lower=lbs, upper=ubs;autodiff=autodiff,
                          show_trace=show_trace,x_tol=p_tol,g_tol=g_tol,maxIter=maxIterLM)
                @btime  poly = curve_fit(modelDM,vs,ys,p,lower=lbs, upper=ubs;autodiff=autodiff,
                          show_trace=show_trace,x_tol=p_tol,g_tol=g_tol,maxIter=maxIterLM)
            else
                poly = curve_fit(modelDM,vs,ys,p,lower=lbs, upper=ubs;autodiff=autodiff,
                          show_trace=show_trace,x_tol=p_tol,g_tol=g_tol,maxIter=maxIterLM)
            end
        else
            ertfgh
            if timeType == :time
            elseif timeType == :btime
            else
            end
            @btime  poly = curve_fit(modelDM,vs,ys,p,lower=lbs, upper=ubs;show_trace=show_trace)
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
elseif fitpkg == :LeastSquaresOptim
    if isforward == 0
        if istol == 1
            nls = LeastSquaresProblem(x=p,f! =(res,p) -> fres!(res,modelDM,ys,vs,p),
                   output_length=length(vs))
            if timeType == :time
                @time  poly = optimize!(nls,optimizer(factor),x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,
                                        iterations=maxIterLM,show_trace=show_trace,lower=lbs,upper=ubs)
            elseif timeType == :btime
                @time  poly = optimize!(nls,optimizer(factor),x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,
                                        iterations=maxIterLM,show_trace=show_trace,lower=lbs,upper=ubs)
                @btime  poly = optimize!(nls,optimizer(factor),x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,
                                        iterations=maxIterLM,show_trace=show_trace,lower=lbs,upper=ubs)
            else
                poly = optimize!(nls,optimizer(factor),x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,
                                        iterations=maxIterLM,show_trace=show_trace,lower=lbs,upper=ubs)
            end
        else
            ertfgh
        end
    else
        if istol == 1
            nls = LeastSquaresProblem(x=p,f! =(res,p) -> fres!(res,modelDM,ys,vs,p),
                   autodiff=autodiff,output_length=length(vs))
            if timeType == :time
                @time  poly = optimize!(nls,optimizer(factor),x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,
                                        iterations=maxIterLM,show_trace=show_trace,lower=lbs,upper=ubs)
            elseif timeType == :btime
                @time  poly = optimize!(nls,optimizer(factor),x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,
                                        iterations=maxIterLM,show_trace=show_trace,lower=lbs,upper=ubs)
                @btime  poly = optimize!(nls,optimizer(factor),x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,
                                        iterations=maxIterLM,show_trace=show_trace,lower=lbs,upper=ubs)
            else
                poly = optimize!(nls,optimizer(factor),x_tol=x_tol,g_tol=g_tol,f_tol=f_tol,
                                        iterations=maxIterLM,show_trace=show_trace,lower=lbs,upper=ubs)
            end
        else
            ertfgh
        end
    end
    p = poly.minimizer         # the array of best model1 parameters
    inner_maxIterLM = poly.iterations
    is_converged = poly.converged
    # is_x_converged = poly.x_converged
    # is_f_converged = poly.f_converged
    # is_g_converged = poly.g_converged
    # pssr = poly.ssr      # sum(abs2, fcur)
    # optim_method = poly.optimizer
    if inner_maxIterLM == maxIterLM || inner_maxIterLM == 1
        is_maxIterLM = true
    else
        is_maxIterLM = false
    end
elseif fitpkg == :Optim
    pcat = p' |> Array{Float64}
    @time  poly = Optim.optimize(f! =(res,p) -> fres!(res,modelDM,ys,vs,p),pcat,
                  NelderMead())
end
maxIterTR = 50
restartfit::Vector{Int}=[0,0,100]
poly = fitTRMs(modelDM,vs,ys,p;restartfit=restartfit,maxIterTR=maxIterTR,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,autodiff=autodiff,
            factorMethod=factorMethod,show_trace=show_trace,lower=lbs,upper=ubs)

poly.minimizer[1] *= yscale
p5 = poly.minimizer         # the vector of best model1 parameters
inner_maxIterLM5 = poly.iterations
is_converged5 = poly.converged
pssr5 = poly.ssr                         # sum(abs2, fcur)
is_x_converged5 = poly.x_converged
is_f_converged5 = poly.f_converged
is_g_converged5 = poly.g_converged
optim_method5 = poly.optimizer

# LeastSquaresResult("LevenbergMarquardt", x, ssr, iterations , converged,
#                     x_converged, x_tol, f_converged, f_tol, g_converged, g_tol, tr,
#                     f_calls, g_calls, mul_calls)

##
p[1] /= yscale
ymodel = modelDM(vs0,pp0)
yfit = modelDM(vs0,p)
yfit5 = modelDM(vs0,p5)
erryMod = (ys0 - ymodel) * neps
erry_fit = (ys0 - yfit) * neps
erry_fit5 = (ys0 - yfit5) * neps
RerryMod = (ys0 ./ ymodel .- 1)
Rerry = (ys0 ./ yfit .- 1)
Rerry5 = (ys0 ./ yfit5 .- 1)
# if issolvefDM == 1
#     yfit1 = modelM1(vs0,pM1)
# else
#     yfit1 = yfit
# end
# @show pM1 ./ pp0 .- 1
# erry_fit1 = ys0 - yfit1
# Rerry1 = (ys0 ./ yfit1 .- 1)

wline = 3
# Plotting
title = string("dnx=",dnvs,",n10=",n10,",p_n=",fmtf2(p_noise_ratio))
parames = title,string(",yM,yscale=",fmtf2.([ymax,yscale]))
@show parames
title2 = string(optimizer_abbr,",",autodiff)
if 1 == 1
    1
    label = string("y,ℓ=",ℓ)
    py= plot(vs0,ys0,label=label,line=(wline,:auto),title=title)
    label = "y_fit"
    py = plot!(vs0,yfit,label=label,line=(wline,:auto))
    label = "y_fit5"
    py = plot!(vs0,yfit5,label=label,line=(wline,:auto))
    label = "y_model"
    py = plot!(vs0,ymodel,label=label,line=(wline,:auto))
    2
    xlabel = string("vG0,u=",fmtf2(ûa0[isp3]))
    label = string("presid")
    # if length(vs) == length(presid)
    #     perry = plot(vs,presid,label=label,line=(wline,:auto))
    # else
        perry = plot(title=title2)
    # end
    3
    # xlabel = string("σ=",fmtf2(norm(sigma)),", σ₀=",fmtf2(norm(sigmaM1)))
    xlabel = string("vs",";yM,yscale=",fmtf2.([ymax,yscale]))
    label = string("ys-yMod")
    perryMod = plot(vs0,erryMod,label=label,xlabel=xlabel,line=(wline,:auto))
    label = string("(ys-yfit)/eps")
    perryfit = plot!(vs0,erry_fit,label=label,line=(wline,:auto))
    label = string("ys-yfit5")
    perryfit = plot!(vs0,erry_fit5,label=label,line=(wline,:auto))
    # display(perryMod)
    4
    xlabel = string("vG0,u=",fmtf2(ûa0[isp3]))
    label = string("Rerr_yMod")
    pRerryMod = plot(vs0,RerryMod,legend=legendtL,label=label,xlabel=xlabel,line=(wline,:auto))
    label = "Rerr_fit"
    pRerry = plot!(vs0,Rerry,label=label,xlabel=xlabel,line=(wline,:auto))
    label = "Rerr_fit5"
    pRerry = plot!(vs0,Rerry5,label=label,xlabel=xlabel,line=(wline,:auto))
    display(plot(py,perry,perryMod,pRerry,layout=(2,2)))
end
@show fitpkg, optimizer_abbr,autodiff
@show (ℓ,nvs,fmtf2(yscale)),fmtf2.(u),fmtf2.(p_noise_rel)
@show p0 ./ pp0 .- 1
@show p ./ pp0 .- 1
