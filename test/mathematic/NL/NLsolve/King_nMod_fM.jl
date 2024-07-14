using NLsolve

"""
  ∑ₖ nₖ = 1

  kwargs:

"""
show_trace = true
# show_trace = false


m_anderson = 5

# The initial solution
p_noise_rel = 1e-15      # `p_noise_rel ≤ 1e-3` default. Relative error of the initial guess parameters `p`
                        # 
p_noise_abs = 1e-15     # Absolute error of the initial guess parameters `p`.

p_tol = epsT
f_tol = epsT
maxIterKing = 200
NLsolve_method = :trust_region    # [:trust_region, :newton, :anderson]
# NLsolve_method = :anderson
# NLsolve_method = :newton
autodiff = :forward

isp33 = iFv3
nModel = nMod[isp33]
if is_nai_const
    naifit = nai0[isp33]
    Mhcsl0 = MsnnE3[2:nModel+1,1,isp33]
    f!(out, x) = king_fM!(out, x, naifit; Mhcsl0=Mhcsl0, nModel=nModel)
    J!(J, x) = king_fM_g!(J, x, naifit; nModel=nModel)
    x0 = vthi0[isp33]
    x00 = copy(x0)
    p_noise_ratio = rand(nModel) * p_noise_rel
    x0 .*= (1.0 .+ p_noise_ratio ) + p_noise_abs * rand(nModel)
else
    Mhcsl0 = MsnnE3[1:2nModel,1,isp33]
    f!(out, x) = king_fM!(out, x; Mhcsl0=Mhcsl0, nModel=nModel)
    J!(J, x) = king_fM_g!(J, x; nModel=nModel)
    x0 = zeros(2nModel)
    x0[1:2:end] = nai0[isp33]
    x0[2:2:end] = vthi0[isp33]
    x00 = copy(x0)
    p_noise_ratio = rand(2nModel) * p_noise_rel
    x0 .*= (1.0 .+ p_noise_ratio ) + p_noise_abs * rand(2nModel)
    @show x0
end

# fJ!(out, x) = king_fM_gs!(out, x; Mhcsl0=Mhcsl0, nModel=nModel)

#res = nlsolve(f!, x0)                      # `J!` is computed by `trust_region` algorithm
#res = nlsolve(f!, x0, autodiff=autodiff)   # `J!` is computed by automatic differentiation `forward`

nls = OnceDifferentiable(f!, J!, x0, similar(x0))
# nls = OnceDifferentiable(f!, J!, fJ!, x0, similar(x0))
if NLsolve_method == :trust_region
    res = nlsolve(nls, x0, method=NLsolve_method,factor=1.0,autoscale=true,xtol=p_tol,ftol=f_tol,
                iterations=maxIterKing,show_trace=show_trace)
elseif NLsolve_method == :anderson
    res = nlsolve(nls, x0, method=NLsolve_method,beta=1,m=m_anderson,xtol=p_tol,ftol=f_tol,
                iterations=maxIterKing,show_trace=show_trace)
elseif NLsolve_method == :newton
    res = nlsolve(nls, x0, method=NLsolve_method,xtol=p_tol,ftol=f_tol,
                iterations=maxIterKing,show_trace=show_trace)
else
    res = nlsolve(nls, x0)
end

xfit = res.zero         # the vector of best model1 parameters
niter= res.iterations

if is_nai_const
    vthifit = xfit[:]
    datafit = DataFrame(vthi=vthi0[isp33],vthifit=vthifit)
    
    dataerrfit = DataFrame(Dvthi=vthi0[isp33]-vthifit)
    
    yfit = zeros(nModel)
    king_fM!(yfit, xfit, naifit; Mhcsl0=Mhcsl0, nModel=nModel)
    yfit0 = zeros(nModel)
    king_fM!(yfit0, x00, naifit; Mhcsl0=Mhcsl0, nModel=nModel)
    
    Jfit = zeros(nModel,nModel)
    king_fM_g!(Jfit, xfit, naifit; nModel=nModel)
else
    naifit = xfit[1:2:end]
    vthifit = xfit[2:2:end]
    datafit = DataFrame(ni=nai0[isp33],nfit=naifit,vthi=vthi0[isp33],vthifit=vthifit)
    
    dataerrfit = DataFrame(Dni=nai0[isp33]-naifit,Dvthi=vthi0[isp33]-vthifit)
    
    yfit = zeros(2nModel)
    king_fM!(yfit, xfit; Mhcsl0=Mhcsl0, nModel=nModel)
    yfit0 = zeros(nModel)
    king_fM!(yfit0, x00; Mhcsl0=Mhcsl0, nModel=nModel)
    
    Jfit = zeros(2nModel,2nModel)
    king_fM_g!(Jfit, xfit; nModel=nModel)
end

# fLn1efit = zero.(fLn1e)
# fLn1efit = fvLDMz(fLn1efit,vGe,ℓ,naifit,uai,vthifit,nModel)
# DfLn1e = fLn1e - fLn1efit
# @show norm(DfLn1e)

if NLsolve_method == :anderson
    @show m_anderson ;
end
@show p_noise_rel,p_noise_abs;
@show x0
@show datafit;
@show dataerrfit;
@show yfit;
@show yfit0;
println()
@show paraM(Jfit);
println()
@show res;
