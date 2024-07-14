using LeastSquaresOptim
using LsqFit
using Optim

using LinearAlgebra
using BenchmarkTools
using Statistics, Test
using Plots

include(joinpath(pathroot,"src\\Collision\\weightfunctions_L.jl"))

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
μu = 1

"""
  When `u ≫ 0`, i.e., `u > 0.5`, It will be the best one that
  `yscale0` will be up to `1e-9`.

  When `u ≪ 1.0 → 0.0`, i.e., `u < 1e-3`, It will be the best one that
  `yscale0` will be up to `1e-15` for `ℓ ≥ 1`.
  But for order `ℓ = 0`, it is so poor that convergence can be achieved when `p_noise_ratio = 1e-1`;
     `p_noise_ratio < 1e-5` or more strict is needed.
"""

# ua > 0
LL1 = 1   # When `ℓ ≫ 1` up to `ymax ≤ (1e-6 ~ 1e-9)`, when 'p_noise_ratio = 1e-1' ,
          # Or else, local optimal solution is given freqently.
is_plot_scale = true
isokk = 1        # [0,1,2,3,4]
p_noise_ratio = 1e-1 # Relative error of the initial guess parameters `p`
ratio_eps = 1e-0
y_isp = iFv3
# y_isp = isp3
is_p0_immutabel = false
# is_p0_immutabel = true
if is_p0_immutabel
    p0_up = copy(p)
end
is_inner_p0 = false
# is_inner_p0 = true
## # parameters
maxIterLM = 50
if 1 == 1
    restartfit::Vector{Int}=[0,10,maxIterLM]
          # = [fit_restart_p0,fit_restart_p,maxIterLM] = [0,0,100]
          # which will decide the process of restart process of fitting process.
          # Parameter `maxIterLM` is the maximum iteration of the standard Trust Region algorithm.
    maxIterTR = 500        # The total maximum number of iterations in a specified fitting process.
    # autodiff = :forward  #
    autodiff = :central  # A little more complicated but maybe obtaining no benefits.
    factorMethod = :QR       # factorMethod = [:QR, :Cholesky, :LSMR]
    # factorMethod = :Cholesky
    show_trace = false
    isnormal = true
    # isnormal = false
    p_tol = eps(Float64)*ratio_eps
    g_tol = eps(Float64)*ratio_eps
    f_tol = eps(Float64)*ratio_eps
    # filtering parameters
    n10 = 5             # [-5:1:10],  ys_min = 10^(-5 * n10) which sets the minimum value of vector `ys`.
    dnvs = 2            # `ys → ys[1:dnvs:end]`
    p_noise_abs = 1e-15 # Absolute error of the initial guess parameters `p`.
end
if is_p0_immutabel == 0
    # variables
    if 2 == 2
        ℓ = LL1 - 1
        modelDM(v,p) = p[1] * modelDMexp(v,μu,p[2:3],ℓ)
        u = ûa0[y_isp]
        uMax = min(20.0,3u)
        lbs = [0.0, 0.0, 0.0]
        ubs = [10, 20.0, uMax]
        # lbs = [1.0, 0.0, 0.0]
        # ubs = [1.0, 20.0, uMax]
        # `p0` in theory
        p0nt =
        pp0 = [1.0, 1.0, u]
        p_noise_rel = rand(3) * p_noise_ratio
        # p01 = pp0 .* (1.0 .+ p_noise_rel ) + p_noise_abs * rand(3)
        p0 = p0DM_guess(u,3;pDM_noise_ratio=p_noise_ratio,pDM_noise_abs=p_noise_abs)
        # p0_up = p0
        # p0 = p0_up
        fvLa = fvL[:,:,y_isp]
        fvLa0 = fvLa[nvlevel0,:]
        vs0 = copy(vG0)
        vs0[1] == 0.0 ? (xvec = 2:nc0) : (xvec = 1:nc0)
        yscale0, ~ = normalfLn(fvLa0[:,LL1][xvec])
        vs0 =vs0[xvec]
        ys0 = copy(fvLa0[:,LL1][xvec])
        norm(ys0) > eps(Float64) || error("`norm(ys) = 0.0` and no fitting is needed.")
    end
    nvs0 = length(ys0)
    if isokk == 1
        yscale, counts, poly = fitTRMs(modelDM,vs0,ys0,copy(p0),lbs,ubs,isnormal;restartfit=restartfit,
                    maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
        if yscale ≠ 1.0
            p0[1] /= yscale
            ubs[1] /= yscale
        end
        # p0 = p
    elseif isokk == 0
        # Normalization of the original datas `ys`
        if 3 == 3
            if isnormal
                yscale, ys = normalfLn(ys0)
            else
                ys = copy(ys0)
                yscale = 1.0
            end
            # filter
            ys, vs = filterfLn(ys,vs0;n10=n10,dnvs=dnvs)
            if  yscale ≠ 1.0
                p0[1] /= yscale
                ubs[1] /= yscale
            end
            # p0[1] = 1.0 / yscale
            # lbs[1] = 1.0 / yscale - 0eps(Float64)
            # ubs[1] = 1.0 / yscale + 0eps(Float64)
            # p0[2] = 1.0
            # lbs[2] = 1.0
            # ubs[2] = 1.0
            nvs = length(vs)
        end
        ubs[3] < p0[3] ? ubs[3] = 2p0[3] : nothing
        println()
        counts, poly = fitTRMs(modelDM,vs,ys,copy(p0),lbs,ubs;restartfit=restartfit,
                    maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)
    else
        if isokk == 2
            # Normalization of the original datas `ys`
            if 3 == 3
                if isnormal
                    yscale, ys = normalfLn(ys0)
                else
                    yscale = 1.0
                end
                # filter
                ys, vs = filterfLn(ys,vs0;n10=n10,dnvs=dnvs)
                nvs = length(vs)
            end
            counts,poly,modelM,MnDMk = fvLmodelDM(vs,ys,ℓ,u,uMax;
                        yscale=yscale,p=copy(p0),restartfit=restartfit,
                        maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
        else
            if isnormal
                yscale, ys = normalfLn(ys0)
            else
                yscale = 1.0
            end
            if isokk == 3
                is_inner_p0 ? p0 .= 0.0 : nothing
                counts,poly,modelM,~,MnDMk = fvLmodel(copy(vs0),copy(ys0),ℓ,0.0,uMax,atolrn;
                            is_p0=false,isnormal=isnormal,p=copy(p0),restartfit=restartfit,
                            maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
            else
                is_inner_p0 ? p0 .= 0.0 : nothing
                counts = nothing
                ddfLn_new,dfLn_new,fLn0_new = zero.(fLn0),zero.(fLn0),zero.(fLn0)
                ddfLn_new,dfLn_new,fLn0_new = FfvLCS(ddfLn_new,dfLn_new,fLn0_new,ys0,vs0,nc0,np,ocp,
                                    nvlevel0,bc,u,ℓ;method3M=method3M,isrenormalization=isrenormalization,
                                    uMax=uMax,atolrn=atolrn,isnormal=isnormal,p=copy(p0),restartfit=restartfit,
                                    maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

            end
        end
    end
else
    if isokk == 1
        yscale, counts, poly = fitTRMs(modelDM,vs0,ys0,copy(p0_up),lbs,ubs,isnormal;restartfit=restartfit,
                    maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    elseif isokk == 0
        counts, poly = fitTRMs(modelDM,vs,ys,copy(p0_up),lbs,ubs;restartfit=restartfit,
                    maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)
    elseif isokk == 2
        counts,poly,modelM,MnDMk = fvLmodelDM(vs,ys,ℓ,u,uMax;
                    yscale=yscale,p=copy(p0_up),restartfit=restartfit,
                    maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    elseif isokk == 3
        is_inner_p0 ? p0 .= 0.0 : nothing
        counts,poly,modelM,~,MnDMk = fvLmodel(copy(vs0),copy(ys0),ℓ,0.0,uMax,atolrn;
                    is_p0=false,isnormal=isnormal,p=copy(p0_up),restartfit=restartfit,
                    maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    else
        is_inner_p0 ? p0 .= 0.0 : nothing
        ddfLn_new,dfLn_new,fLn0_new = zero.(fLn0),zero.(fLn0),zero.(fLn0)
        ddfLn_new,dfLn_new,fLn0_new = FfvLCS(ddfLn_new,dfLn_new,fLn0_new,ys0,vs0,nc0,np,ocp,
                            nvlevel0,bc,u,ℓ;method3M=method3M,isrenormalization=isrenormalization,
                            uMax=uMax,atolrn=atolrn,isnormal=isnormal,p=copy(p0_up),restartfit=restartfit,
                            maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    end
end
if isokk ≤ 3
    p = poly.minimizer         # the vector of best model1 parameters
    inner_maxIterLM = poly.iterations
    is_converged = poly.converged
    pssr = poly.ssr                         # sum(abs2, fcur)
    is_x_converged = poly.x_converged
    is_f_converged = poly.f_converged
    is_g_converged = poly.g_converged
    optim_method = poly.optimizer
else
    p = p0DM_guess(u,3)
end
## The normalized datas `ys → ys / yscale`
# p
if isokk == 3
    if is_plot_scale == 0
    else
        p[1] /= yscale
        pp0[1] /= yscale
        ys0 /= yscale
    end
else
    if is_plot_scale == 0
        p[1] *= yscale
    else
        pp0[1] /= yscale
        ys0 /= yscale
    end
end

# yfit
if isokk == 4
    if is_plot_scale == 0
        yfit = fLn0_new
    else
        yfit = fLn0_new / yscale
    end
else
    ymodel = modelDM(vs0,pp0)
    yfit = modelDM(vs0,p)
end
# if isokk ≤ 1
#     ymodel = modelDM(vs0,pp0)
#     yfit = modelDM(vs0,p)
# else
#     ymodel = modelDM(vs0,pp0)
#     if isokk ≤ 3
#         yfit = modelM(vs0,p)
#     elseif isokk == 4
#         if is_plot_scale == 0
#             yfit = fLn0_new
#         else
#             yfit = fLn0_new / yscale
#         end
#     end
# end
erryMod = (ys0 - ymodel) * neps
erry_fit = (ys0 - yfit) * neps
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
    xlabel = string("vs",";yscale,scl0=",fmtf2.([yscale,yscale0]))
    label = string("ys-yMod")
    perryMod = plot(vs0,erryMod,label=label,xlabel=xlabel,line=(wline,:auto))
    label = string("(ys-yfit)/eps")
    perryfit = plot!(vs0,erry_fit,label=label,line=(wline,:auto))
    # display(perryMod)
    4
    xlabel = string("vG0,u=",fmtf2(ûa0[y_isp]))
    label = string("Rerr_yMod")
    pRerryMod = plot(vs0,RerryMod,legend=legendtL,label=label,xlabel=xlabel,line=(wline,:auto))
    label = "Rerr_fit"
    pRerry = plot!(vs0,Rerry,label=label,xlabel=xlabel,line=(wline,:auto))
    display(plot(py,perry,perryMod,pRerry,layout=(2,2)))
end
println()
@show title2
@show (ℓ,nvs0),fmtf2.(u),fmtf2.(p_noise_rel)
@show p0 ./ pp0 .- 1
@show p ./ pp0 .- 1
# if show_trace && isokk ≤ 3
if isokk ≤ 3
    # println()
    # @show poly
end
@show p0
fvLa0 = fvLa[nvlevel0,:]
label = string("fLn/v^ℓ,",ℓ)
pfv(nn0,nn9,L) = plot(vG0[nn0:nn9],fvLa0[nn0:nn9,L+1]./vG0[nn0:nn9].^L,label=string("fLn/v^ℓ,",L),line=(wline,:auto))

# function LSqOptim_fLn_testing(modelDM,fvL,vG0,nvlevel0,u,nc0;LL1=1,isokk=1,is_plot_scale=true,
#     uMax::Float64=10.0,isnormal::Bool=true,restartfit::Vector{Int}=[0,10,100],
#     maxIterTR::Int=500,autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
#     p_tol::Float64=1e-16,f_tol::Float64=1e-16,g_tol::Float64=1e-16,n10::Int=5,dnvs::Int=2,
#     p_noise_ratio=1e-2,p_noise_abs=1e-15,y_isp=2) where{T,Tb,N}
#
#     # variables
#     if 2 == 2
#         ℓ = LL1 - 1
#         uMax = min(20.0,3u)
#         lbs = [0.0, 0.0, 0.0]
#         ubs = [10, 20.0, uMax]
#         # `p0` in theory
#         pp0 = [1.0, 1.0, u]
#         p_noise_rel = rand(3) * p_noise_ratio
#         # p01 = pp0 .* (1.0 .+ p_noise_rel ) + p_noise_abs * rand(3)
#         p0 = p0DM_guess(u,3;pDM_noise_ratio=p_noise_ratio,pDM_noise_abs=p_noise_abs)
#         fvLa = fvL[:,:,y_isp]
#         vs0 = copy(vG0)
#         vs0[1] == 0.0 ? (xvec = 2:nc0) : (xvec = 1:nc0)
#         vs0 =vs0[xvec]
#         ys0 = copy(fvLa[nvlevel0,LL1][xvec])
#         norm(ys0) > eps(Float64) || error("`norm(ys) = 0.0` and no fitting is needed.")
#     end
#     nvs0 = length(ys0)
#     if isokk == 1
#         yscale, counts, poly = fitTRMs(modelDM,vs0,ys0,copy(p0),lbs,ubs,isnormal;restartfit=restartfit,
#                     maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
#                     p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
#         if yscale ≠ 1.0
#             p0[1] /= yscale
#             ubs[1] /= yscale
#         end
#         # p0 = p
#     elseif isokk == 0
#         # Normalization of the original datas `ys`
#         if 3 == 3
#             if isnormal
#                 yscale, ys = normalfLn(ys0)
#             else
#                 ys = copy(ys0)
#                 yscale = 1.0
#             end
#             # filter
#             ys, vs = filterfLn(ys,vs0;n10=n10,dnvs=dnvs)
#             if  yscale ≠ 1.0
#                 p0[1] /= yscale
#                 ubs[1] /= yscale
#             end
#             # p0[1] = 1.0 / yscale
#             # lbs[1] = 1.0 / yscale - 0eps(Float64)
#             # ubs[1] = 1.0 / yscale + 0eps(Float64)
#             # p0[2] = 1.0
#             # lbs[2] = 1.0
#             # ubs[2] = 1.0
#             nvs = length(vs)
#         end
#         ubs[3] < p0[3] ? ubs[3] = 2p0[3] : nothing
#         println()
#         counts, poly = fitTRMs(modelDM,vs,ys,copy(p0),lbs,ubs;restartfit=restartfit,
#                     maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
#                     p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)
#     else
#         if isokk == 2
#             # Normalization of the original datas `ys`
#             if 3 == 3
#                 if isnormal
#                     yscale, ys = normalfLn(ys0)
#                 else
#                     yscale = 1.0
#                 end
#                 # filter
#                 ys, vs = filterfLn(ys,vs0;n10=n10,dnvs=dnvs)
#                 nvs = length(vs)
#             end
#             counts,poly,modelM,MnDMk = fvLmodelDM(vs,ys,ℓ,u,uMax;
#                         yscale=yscale,p=copy(p0),restartfit=restartfit,
#                         maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
#                         p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
#         else
#             if isnormal
#                 yscale, ys = normalfLn(ys0)
#             else
#                 yscale = 1.0
#             end
#             if isokk == 3
#                 is_inner_p0 ? p0 .= 0.0 : nothing
#                 counts,poly,modelM,~,MnDMk = fvLmodel(copy(vs0),copy(ys0),ℓ,0.0,uMax,atolrn;
#                             is_p0=false,isnormal=isnormal,p=copy(p0),restartfit=restartfit,
#                             maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
#                             p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
#             else
#                 is_inner_p0 ? p0 .= 0.0 : nothing
#                 counts = nothing
#                 ddfLn_new,dfLn_new,fLn0_new = zero.(fLn0),zero.(fLn0),zero.(fLn0)
#                 ddfLn_new,dfLn_new,fLn0_new = FfvLCS(ddfLn_new,dfLn_new,fLn0_new,ys0,vs0,nc0,np,ocp,
#                                     nvlevel0,bc,u,ℓ;method3M=method3M,isrenormalization=isrenormalization,
#                                     uMax=uMax,atolrn=atolrn,isnormal=isnormal,p=copy(p0),restartfit=restartfit,
#                                     maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
#                                     p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
#
#             end
#         end
#     end
#     if isokk ≤ 3
#         p = poly.minimizer         # the vector of best model1 parameters
#         # inner_maxIterLM = poly.iterations
#         # is_converged = poly.converged
#         # pssr = poly.ssr                         # sum(abs2, fcur)
#         # is_x_converged = poly.x_converged
#         # is_f_converged = poly.f_converged
#         # is_g_converged = poly.g_converged
#         # optim_method = poly.optimizer
#     else
#         p = p0DM_guess(u,3)
#     end
#     ## The normalized datas `ys → ys / yscale`
#     if is_plot_scale == 0
#         p[1] *= yscale
#         yfit = fLn0_new
#     else
#         pp0[1] /= yscale
#         ys0 /= yscale
#         isokk == 4 ? yfit = fLn0_new / yscale : nothing
#     end
#     if isokk ≤ 1
#         ymodel = modelDM(vs0,pp0)
#         yfit = modelDM(vs0,p)
#     else
#         ymodel = modelDM(vs0,pp0)
#         if isokk ≤ 3
#             yfit = modelM(vs0,p)
#         end
#     end
#     erryMod = (ys0 - ymodel) * neps
#     erry_fit = (ys0 - yfit) * neps
#     RerryMod = (ys0 ./ ymodel .- 1)
#     Rerry = (ys0 ./ yfit .- 1)
#     # Plotting
#     wline = 3
#     title = string("dnx=",dnvs,",n10=",n10,",p_n=",fmtf2(p_noise_ratio))
#     parames = title,string(",yM,yscale=",fmtf2.(yscale))
#     @show parames
#     title2 = string(factorMethod,",",autodiff,",Niter_all=",counts)
#     if 1 == 1
#         1
#         label = string("y,ℓ=",ℓ)
#         py= plot(vs0,ys0,label=label,line=(wline,:auto),title=title)
#         label = "y_fit"
#         py = plot!(vs0,yfit,label=label,line=(wline,:auto))
#         label = "y_model"
#         py = plot!(vs0,ymodel,label=label,line=(wline,:auto))
#         2
#         xlabel = string("vG0,u=",fmtf2(ûa0[y_isp]))
#         label = string("presid")
#         # if length(vs) == length(presid)
#         #     perry = plot(vs,presid,label=label,line=(wline,:auto))
#         # else
#             perry = plot(title=title2)
#         # end
#         3
#         # xlabel = string("σ=",fmtf2(norm(sigma)),", σ₀=",fmtf2(norm(sigmaM1)))
#         xlabel = string("vs",";yscale=",fmtf2.(yscale))
#         label = string("ys-yMod")
#         perryMod = plot(vs0,erryMod,label=label,xlabel=xlabel,line=(wline,:auto))
#         label = string("(ys-yfit)/eps")
#         perryfit = plot!(vs0,erry_fit,label=label,line=(wline,:auto))
#         # display(perryMod)
#         4
#         xlabel = string("vG0,u=",fmtf2(ûa0[y_isp]))
#         label = string("Rerr_yMod")
#         pRerryMod = plot(vs0,RerryMod,legend=legendtL,label=label,xlabel=xlabel,line=(wline,:auto))
#         label = "Rerr_fit"
#         pRerry = plot!(vs0,Rerry,label=label,xlabel=xlabel,line=(wline,:auto))
#         display(plot(py,perry,perryMod,pRerry,layout=(2,2)))
#     end
#     println()
#     @show title2
#     @show (ℓ,nvs0),fmtf2.(u),fmtf2.(p_noise_rel)
#     @show p0 ./ pp0 .- 1
#     @show p ./ pp0 .- 1
#     # if show_trace && isokk ≤ 3
#     if isokk ≤ 3
#         # println()
#         # @show poly
#     end
#     @show p0
#
# end
