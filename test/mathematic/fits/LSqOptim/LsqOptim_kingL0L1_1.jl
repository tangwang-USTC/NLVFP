using LeastSquaresOptim

"""
  ∑ₖ nₖ = 1

  kwargs:
    autodiff::Symbol ∈ [:forward, :central]
    show_trace::Bool ∈ [true, false]
    factorMethod::Symbol ∈ [:QR, :Cholesky]

"""
is_x0_noise = true
is_Jacobian = true
show_trace = true


nModel = nMod[isp3]
Ml0 = MsnnE3[1:3nModel,1,isp3]
Ml1 = MsnnE3[1:3nModel,2,isp3]
Ml01 = zeros(2length(Ml0))
Ml01[1:2:end] = Ml0
Ml01[2:2:end] = Ml1

function king!(out, x; Ml01=Ml01, nModel=nModel)

    nh = x[1:3:end]
    uh = x[2:3:end]
    vhth = x[3:3:end]
    nj = 1
    # (l,j) = (0,0)
    out[nj] = sum(nh) - Ml01[nj]
    nj += 1
    # (l,j) = (1,1)
    out[nj] = sum(nh .* uh) - Ml01[nj]
    nj += 1
    # (l,j) = (0,2)
    out[nj] = sum((nh .* vhth .^2) .* (1 .+ 2 / 3 * ((uh ./ vhth) .^2))) - Ml01[nj]
end

function king_g!(J, x; Ml01=Ml01, nModel=nModel)

    fill!(J, 0.0)
    nh = x[1:3:end]
    uh = x[2:3:end]
    vhth = x[3:3:end]
    nj = 1
    # (l,j) = (0,0)
    for s in 1:nModel
        J[nj, 3(s-1)+1] = 1.0
    end
    nj += 1
    # (l,j) = (1,1)
    for s in 1:nModel
        s3 = 3(s-1)
        J[nj, s3+1] = uh[s]
        J[nj, s3+2] = nh[s]
    end
    nj += 1
    # (l,j) = (0,2)
    for s in 1:nModel
        s3 = 3(s-1)
        J[nj, s3+1] = vhth[s]^2 + 2 / 3 * uh[s]^2
        J[nj, s3+2] = 4 / 3 * nh[s] * uh[s]
        J[nj, s3+3] = 2nh[s] * vhth[s]
    end
end

show_trace = false
is_Jacobian = false
# is_x0_noise = false
########## The initial solution
p_noise_rel = 1e-0      # `p_noise_rel ≤ 1e-3` default. Relative error of the initial guess parameters `p`
                        # 
p_noise_abs = 1e-15     # Absolute error of the initial guess parameters `p`.
if 1 == 1
    if is_x0_noise
        p_noise_ratio = rand(3nModel) * p_noise_rel
        x0 = zeros(3nModel)
        x0[1:3:end] .= nai0[isp3]
        x0[2:3:end] .= uai0[isp3]
        x0[3:3:end] .= vthi0[isp3]
        x0t = copy(x0)
        x0 .*= (1.0 .+ p_noise_ratio) + p_noise_abs * rand(3nModel)
        # x0[2:3:end] .= epsT1000
        x00 = copy(x0)
    else
        x0 = copy(x00)
    end
    @show x0
    # The parameter limits for MCF plasma.
    lbs = zeros(3nModel)
    lbs[1:3:end] .= 0.0
    lbs[2:3:end] .= -uhMax
    lbs[3:3:end] .= 1 / vhthMax
    ubs = zeros(3nModel)
    ubs[1:3:end] .= nhMax
    ubs[2:3:end] .= uhMax
    ubs[3:3:end] .= vhthMax

    p_tol = 1e-37
    f_tol = 1e-37
    g_tol = 1e-37
    maxIterKing = 1000
    (optimizer, optimizer_abbr) = (LeastSquaresOptim.Dogleg, :dl)
    # (optimizer, optimizer_abbr) = (LeastSquaresOptim.LevenbergMarquardt, :lm)
    factor = LeastSquaresOptim.QR()
    # factor = LeastSquaresOptim.Cholesky()
    # factor = LeastSquaresOptim.LSMR()
    autodiffs = :central
    # autodiffs = :forward
end
if is_Jacobian
    nls = LeastSquaresProblem(x=x0,f! = king!,g! = king_g!,output_length=length(x0),autodiff=:central)
else
    nls = LeastSquaresProblem(x=x0,f! = king!,output_length=length(x0),autodiff=:central)
end
# res = optimize!(nls,optimizer(factor),iterations=maxIterKing,show_trace=show_trace,
#                 x_tol=p_tol,f_tol=f_tol,g_tol=g_tol,lower=lbs,upper=ubs)
res = optimize!(nls,optimizer(),iterations=maxIterKing,show_trace=show_trace,
                x_tol=p_tol,f_tol=f_tol,g_tol=g_tol,lower=lbs,upper=ubs)


xfit = res.minimizer         # the vector of best model1 parameters
niter= res.iterations
is_converged = res.converged
xssr = res.ssr                         # sum(abs2, fcur)
is_xconverged = res.x_converged
is_fconverged = res.f_converged
is_gconverged = res.g_converged
optim_method = res.optimizer

naifit = xfit[1:3:end]
uaifit = xfit[2:3:end]
vthifit = xfit[3:3:end]
K̂afit = 3/2 * sum(naifit .* vthifit.^2) + sum(naifit .* uaifit.^2)
n̂afit = sum(naifit)
ûafit = sum(naifit .* uaifit)
T̂afit0 = sum(naifit .* vthifit.^2)
T̂afit = sum(naifit .* (vthifit.^2 + 2/3 * uaifit.^2)) - 2/3 * ûafit^2
KTEkfit = sum(naifit .* vthifit.^2) + 1/3 * (ûafit.^2 - sum(naifit .* uaifit.^2))

datafit = DataFrame(ni=nai0[isp3],nfit=naifit,ui=uai0[isp3],ufit=uaifit,vthi=vthi0[isp3],vthifit=vthifit)

dataerrfit = DataFrame(Dni=nai0[isp3]-naifit,Dui=uai0[isp3]-uaifit,Dvthi=vthi0[isp3]-vthifit)

yfit = zeros(3nModel)
king!(yfit, xfit; Ml01=Ml01, nModel=nModel)

Jfit = zeros(3nModel,3nModel)
king_g!(Jfit, xfit; Ml01=Ml01, nModel=nModel)

dataJfit = DataFrame(Jfit,:auto)
# fLn1efit = zero.(fLn1e)
# fLn1efit = fvLDMz(fLn1efit,vGe,ℓ,naifit,uaifit,vthifit,nModel)
# DfLn1e = fLn1e - fLn1efit

@show x0
@show dataJfit;
@show paraM(Jfit);
@show datafit;
@show dataerrfit;
@show norm(DfLn1e),n̂afit-1,KTEkfit - 1
@show p_noise_rel,p_noise_abs;
@show is_xconverged,is_fconverged,is_gconverged,norm(yfit);
println()
is_converged,niter,xssr,fmtf12.(xfit)
