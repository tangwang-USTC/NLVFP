
"""
  Inputs:
    modelDM:
    xs:
    ys:
    p:
    yscale:
    maxIterTR: (=1000 default), the maximum number of the total iterations of fitting steps.
    restartfit: = [fit_restart_p0,fit_restart_p,maxIterLM] = [0,0,100]
                    which will decide the process of restart process of fitting process.

             `fit_restart_p0` is the number of outter iteration of restart fitting process
                    which will re-update the initial guess value `p0`;
             `fit_restart_p` is the number of inner iteration of restart fitting process
                    which will re-update the parameters `p` which is the last step;
             `maxIterLM`is the maximum number of iteration step of the standard Trust Regions algorithm
                    which is without restart process.
    kwargs:
      autodiff::Symbol ∈ [:forward, :central]
      show_trace::Bool ∈ [true, false]
      factorMethod::Symbol ∈ [:QR, :Cholesky]

  Outputs:
    s
"""

function fres!(res,modelDM,ys,vs,p)

    res[:] = ys - modelDM(vs,p)
end

"""
  Outputs:
    counts, poly = fitTRMs(modelDM,vs,ys,p,lbs,ubs;restartfit=restartfit,
                maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)

"""

function fitTRMs(modelDM,vs::AbstractVector,ys::AbstractVector,p::AbstractVector,
    lbs::AbstractVector,ubs::AbstractVector;restartfit::Vector{Int}=[0,0,100],
    maxIterTR::Int64=1000,autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18)

    fit_restart_p0,fit_restart_p,maxIterLM = restartfit
    optimizer = LeastSquaresOptim.LevenbergMarquardt
    nls = LeastSquaresProblem(x=p,f! =(res,p) -> fres!(res,modelDM,ys,vs,p),
                                    autodiff=autodiff,output_length=length(vs))
    if factorMethod == :QR
        factor = LeastSquaresOptim.QR()
    elseif factorMethod == :Cholesky
        factor = LeastSquaresOptim.Cholesky()
    else
        asgefgbb
    end
    if fit_restart_p0 == 0
        if fit_restart_p > 0
            Numres = round(Int,maxIterTR/maxIterLM)
            if Numres ≤ 1
                poly = optimize!(nls,optimizer(factor),iterations=maxIterTR,show_trace=show_trace,
                        x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,lower=lbs,upper=ubs)
                return poly.iterations, poly
            else
                isfalse = true
                counts = 0
                poly = optimize!(nls,optimizer(factor),iterations=maxIterLM,show_trace=show_trace,
                        x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,lower=lbs,upper=ubs)
                p = poly.minimizer         # the vector of best model1 parameters
                inner_maxIterLM = poly.iterations
                # is_converged = poly.converged
                is_g_converged = poly.g_converged
                is_f_converged = poly.f_converged
                # pssr = poly.ssr                         # sum(abs2, fcur)
                # is_x_converged = poly.x_converged
                # optim_method = poly.optimizer
                is_fit_DL = false  # No `Dogleg` method has been used.
                counts += inner_maxIterLM
                nIter = 1
                if is_g_converged && is_f_converged
                    return counts, poly
                else
                    p_up = deepcopy(p)
                    while isfalse
                        if is_fit_DL == 0
                            if inner_maxIterLM == maxIterLM || inner_maxIterLM == 1
                                optimizer = LeastSquaresOptim.LevenbergMarquardt
                            else
                                optimizer = LeastSquaresOptim.Dogleg
                                is_fit_DL = true
                            end
                        end
                        nls = LeastSquaresProblem(x=p,f! =(res,p) -> fres!(res,modelDM,ys,vs,p),
                                        autodiff=autodiff,output_length=length(vs))
                        # poly = optimize!(nls,optimizer(factor),iterations=maxIterLM,show_trace=show_trace,
                        #         x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,lower=lbs,upper=ubs)
                        if is_fit_DL == 1
                            espT1 = p_tol
                            poly = optimize!(nls,optimizer(factor),iterations=maxIterLM,show_trace=show_trace,
                                    x_tol=espT1,g_tol=espT1,f_tol=espT1,lower=lbs,upper=ubs)
                        else
                            poly = optimize!(nls,optimizer(factor),iterations=maxIterLM,show_trace=show_trace,
                                    x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,lower=lbs,upper=ubs)
                        end
                        p = poly.minimizer
                        inner_maxIterLM = poly.iterations
                        # is_converged = poly.converged
                        is_g_converged = poly.g_converged
                        is_f_converged = poly.f_converged
                        nIter += 1
                        counts += inner_maxIterLM
                        if (is_g_converged && is_f_converged)
                            break
                        elseif norm(p - p_up) < 1e-15
                            break
                        else
                            if counts ≥ maxIterTR
                                break
                            end
                        end
                        p_up = deepcopy(p)
                    end
                end
                return counts, poly
            end
        else
            poly = optimize!(nls,optimizer(factor),iterations=maxIterTR,show_trace=show_trace,
                    x_tol=p_tol,g_tol=g_tol,f_tol=f_tol,lower=lbs,upper=ubs)
            return poly.iterations, poly
        end
    else
        rhtgfhm
    end
end

"""
  Intputs:
    n10::Int64 = 0 default, limiting the smallest value of datas `ys` by
         `ys0 .> eps(Float64) * 10.0^-(n10)`
    dnvs::INt = 1 default, limiting the number of datas `ys` by `ys = ys0[1:dnvs:end]`

  Outputs:
    counts, poly = fitTRMs(modelDM,vs,ys,p,lbs,ubs,isnormal;restartfit=restartfit,
                maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""
# isnormalized
function fitTRMs(modelDM,vs0::AbstractVector,ys0::AbstractVector,p::AbstractVector,
    lbs::AbstractVector,ubs::AbstractVector,isnormal::Bool;restartfit::Vector{Int}=[0,0,100],
    maxIterTR::Int64=1000,autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int64=1,dnvs::Int64=1)

    # Normalization of the original datas `ys0`
    if isnormal
        yscale, ys = normalfLn(ys0)
        # filter
        ys, vs = filterfLn(ys,vs0;n10=n10,dnvs=dnvs)
    else
        yscale = 1.0
        # filter
        ys, vs = filterfLn(ys0,vs0;n10=n10,dnvs=dnvs)
    end
    if yscale ≠ 1.0
        p[1] /= yscale
        ubs[1] /= yscale
    end
    counts, poly = fitTRMs(modelDM,vs,ys,deepcopy(p),lbs,ubs;restartfit=restartfit,
                maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)
    return yscale, counts, poly
end

