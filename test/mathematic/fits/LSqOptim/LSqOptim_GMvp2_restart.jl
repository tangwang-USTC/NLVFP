using LeastSquaresOptim
using LsqFit
using Optim

using LinearAlgebra
using BenchmarkTools
using Statistics, Test
using Plots

include(joinpath(pathroot,"src\\Collision\\weightfunctions_L.jl"))

# Gaussian models for Maxwellian distribution
modelDM(v,p) = p[1] * modelMexp(v,p[2])

"""
  Outputs:
    r.ssr = sum(abs2, fres)


  @printf io "Results of Optimization Algorithm\n"
  @printf io " * Algorithm: %s\n" r.optimizer
  @printf io " * Minimizer: [%s]\n" join(r.minimizer, ",")
  @printf io " * Sum of squares at Minimum: %f\n"
  @printf io " * Iterations: %d\n" r.iterations
  @printf io " * Convergence: %s\n" converged(r)
  @printf io " * |x - x'| < %.1e: %s\n" r.x_tol r.x_converged
  @printf io " * |f(x) - f(x')| / |f(x)| < %.1e: %s\n" r.f_tol r.f_converged
  @printf io " * |g(x)| < %.1e: %s\n" r.g_tol r.g_converged
  @printf io " * Function Calls: %d\n" r.f_calls
  @printf io " * Gradient Calls: %d\n" r.g_calls
  @printf io " * Multiplication Calls: %d\n" r.mul_calls
"""

# ua = 0
LL1 = 1
isokk = 1
is_plot_scale = true

## # parameters
maxIterLM = 100
restartfit::Vector{Int}=[0,10,maxIterLM]
      # = [fit_restart_p0,fit_restart_p,maxIterLM] = [0,0,100]
      # which will decide the process of restart process of fitting process.
y_isp = iFv3
# y_isp = isp3
p_noise_ratio = 1e-1 # Relative error of the initial guess parameters `p`
if 1 == 1
    # autodiff = :forward  #
    autodiff = :central  # A little more complicated but maybe obtaining no benefits.
    factorMethod = :QR       # factorMethod = [:QR, :Cholesky, :LSMR]
    # factorMethod = :Cholesky
    show_trace = false
    isnormal = true
    maxIterTR = 500
    p_tol = eps(Float64)
    g_tol = eps(Float64)
    f_tol = eps(Float64)
    n10 = 0             # [-5:1:10],  ys_min = 10^(-5 * n10) which sets the minimum value of vector `ys`.
    dnvs = 1
    p_noise_abs = 1e-5 # Absolute error of the initial guess parameters `p`.
end
# variables
if 2 == 2
    ℓ = LL1 - 1
    # `p0` in theory
    pp0 = [1.0, 1.0]
    lbs = [0.0, 0.0]
    ubs = [10, 20.0]
    p_noise_rel = rand(2) * p_noise_ratio

    fvLa = fvL[:,:,y_isp]
    vs0 = copy(vG0)
    vs0[1] == 0.0 ? (xvec = 2:nc0) : (xvec = 1:nc0)
    vs0 =vs0[xvec]
    ys0 = copy(fvLa[nvlevel0,LL1][xvec])
    norm(ys0) > eps(Float64) || error("`norm(ys) = 0.0` and no fitting is needed.")
end
nvs0 = length(ys0)
p0 = pp0 .* (1.0 .+ p_noise_rel ) + p_noise_abs * rand(2)
if isokk == 1
    yscale, counts, poly = fitTRMs(modelDM,vs0,ys0,copy(p0),lbs,ubs,isnormal;restartfit=restartfit,
                maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
else
    # Normalization of the original datas `ys`
    if 3 == 3
        if isnormal
            yscale, ys = normalfLn(ys0)
        else
            yscale = 1.0
        end
        # filter
        ys, vs = filterfLn(ys,vs0;n10=n10,dnvs=dnvs)
        if yscale ≠ 1.0
            p0[1] /= yscale
            ubs[1] /= yscale
        end
        nvs = length(vs)
    end
    println()
    counts, poly = fitTRMs(modelDM,vs,ys,copy(p0),lbs,ubs;restartfit=restartfit,
                maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)
end
p = poly.minimizer         # the vector of best model1 parameters
inner_maxIterLM = poly.iterations
is_converged = poly.converged
pssr = poly.ssr                         # sum(abs2, fcur)
is_x_converged = poly.x_converged
is_f_converged = poly.f_converged
is_g_converged = poly.g_converged
optim_method = poly.optimizer

## The normalized datas `ys → ys / yscale`
if is_plot_scale == 0
    p[1] *= yscale
else
    pp0[1] /= yscale
    ys0 /= yscale
end
ymodel = modelDM(vs0,pp0)
yfit = modelDM(vs0,p)
erryMod = (ys0 - ymodel) * neps^0
erry_fit = (ys0 - yfit) * neps^0
RerryMod = (ys0 ./ ymodel .- 1)
Rerry = (ys0 ./ yfit .- 1)
# Plotting
wline = 3
title = string("dnx=",dnvs,",n10=",n10,",p_n=",fmtf2(p_noise_ratio))
parames = title,string(",yM,yscale=",fmtf2.(yscale))
@show parames
title2 = string(factorMethod,",",autodiff,",Niter_all=",counts)
if 1 == 1
    1
    label = string("y,ℓ=",ℓ)
    py= plot(vs0,ys0,label=label,line=(wline,:auto),title=title)
    label = "y_fit"
    py = plot!(vs0,yfit,label=label,line=(wline,:auto))
    label = "y_model"
    py = plot!(vs0,ymodel,label=label,line=(wline,:auto))
    2
    xlabel = string("vG0,u=",fmtf2(ûa0[y_isp]))
    label = string("presid")
    # if length(vs) == length(presid)
    #     perry = plot(vs,presid,label=label,line=(wline,:auto))
    # else
        perry = plot(title=title2)
    # end
    3
    # xlabel = string("σ=",fmtf2(norm(sigma)),", σ₀=",fmtf2(norm(sigmaM1)))
    label = string("(ys-yfit)/eps")
    perryfit = plot(vs0,erry_fit,label=label,line=(wline,:auto))
    xlabel = string("vs",";yscale=",fmtf2.(yscale))
    # label = string("ys-yMod")
    # perryMod = plot!(vs0,erryMod,label=label,xlabel=xlabel,line=(wline,:auto))
    # display(perryMod)
    4
    xlabel = string("vG0,u=",fmtf2(ûa0[y_isp]))
    label = "Rerr_fit"
    pRerry = plot(vs0,Rerry,label=label,xlabel=xlabel,line=(wline,:auto))
    # label = string("Rerr_yMod")
    # pRerryMod = plot!(vs0,RerryMod,legend=legendtL,label=label,xlabel=xlabel,line=(wline,:auto))
    display(plot(py,perry,perryfit,pRerry,layout=(2,2)))
end
println()
@show title2
@show (ℓ,nvs0),fmtf2.(p_noise_rel)
@show p0 ./ pp0 .- 1
@show p ./ pp0 .- 1
println()
@show poly
