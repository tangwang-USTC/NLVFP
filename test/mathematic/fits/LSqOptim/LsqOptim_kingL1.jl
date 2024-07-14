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
Ml1 = MsnnE3[1:3nModel,2,isp3]

"""
  king!(yjj, xfit; Ml1=Ml1, l=1, nModel=nModel)
  king_g!(Jjj, xfit; Ml1=Ml1, l=1, nModel=nModel)
"""

function king!(out, x; Ml1=Ml1, l=1, nModel=nModel)

    nh = x[1:3:end]
    uh = x[2:3:end]
    vhth = x[3:3:end]
    # j = l + 0
    nj = 1
    out[1] = sum(nh .* uh) - Ml1[1]

    j = l + 2
    nj += 1
    out[nj] = sum((nh .* uh .* vhth .^ 2) .* (1 .+ 2 / 5 * (uh ./ vhth) .^ 2)) - Ml1[nj]
    
    j = l + 4
    nj += 1
    out[nj] = sum((nh .* uh .* vhth .^ 4) .* (1 .+ 4 / 5 * (uh ./ vhth) .^ 2 .+ 4 / 35 * (uh ./ vhth) .^ 4)) - Ml1[nj]
    for kM in 2:nModel
        for i in 1:3
            j = l + 6(kM-1) + 2(i-1)
            nj += 1
            # N = (j - l) / 2 |> Int
            N = 3(kM-1) + (i-1)
            k = 1:N
            ck = [2^k * binomial(N, k) / prod(5:2:2(l+k)+1) for k in 1:N]
            out[nj] = sum((nh .* uh .* vhth .^ (j-1)) .* (1 .+ [sum(ck .* (uh[s] ./ vhth[s]) .^ (2k)) for s in 1:nModel])) - Ml1[nj]
        end
    end
end

function king_g!(J, x; Ml1=Ml1, l=1, nModel=nModel)

    nh = x[1:3:end]
    uh = x[2:3:end]
    vhth = x[3:3:end]
    # j = l + 0
    nj = 1
    for s in 1:nModel
        s3 = 3(s-1)
        J[nj, s3+1] = uh[s]
        J[nj, s3+2] = nh[s]
        J[nj, s3+3] = 0.0
    end
    # j = l + 2
    nj = 2
    for s in 1:nModel
        s3 = 3(s-1)
        J[nj, s3+1] = uh[s] * (vhth[s]^2 + 2 / 5 * uh[s]^2)
        J[nj, s3+2] = nh[s] * (vhth[s]^2 + 6 / 5 * uh[s]^2)
        J[nj, s3+3] = 2nh[s] * uh[s] * vhth[s]
    end
    j = l + 4
    nj += 1
    for s in 1:nModel
        s3 = 3(s-1)
        J[nj, s3+1] = uh[s] * vhth[s]^4 * (1 + 4 / 5 * (uh[s] / vhth[s])^2 + 4 / 35 * (uh[s] / vhth[s])^4)
        J[nj, s3+2] = nh[s] * vhth[s]^4 * (1 + 12 / 5 * (uh[s] / vhth[s])^2 + 4 / 7 * (uh[s] / vhth[s])^4)
        J[nj, s3+3] = 4nh[s] * uh[s] * vhth[s]^3 * (1 + 2 / 5 * (uh[s] / vhth[s])^2)
    end
    # @show norm(J-Jfit)
    for kM in 2:nModel
        for i in 1:3
            j = l + 6(kM-1) + 2(i-1)
            nj += 1
            N = (j - l) / 2 |> Int
            k = 1:N
            ck = [2^k * binomial(N, k) / prod((2l+3):2:2(l+k)+1) for k in 1:N]
            for s in 1:nModel
                s3 = 3(s-1)
                J[nj, s3+1] = uh[s] * vhth[s]^(j-l) * (1 + sum(ck .* (uh[s] / vhth[s]) .^ (2k)))
                J[nj, s3+2] = nh[s] * vhth[s]^(j-l) * (l + sum((2k.+l).*ck .* (uh[s] / vhth[s]) .^ (2k)))
                J[nj, s3+3] = nh[s] * uh[s] * vhth[s]^(j-2) * ((j-l) - sum((2k .+ (l-j)) .* ck .* (uh[s] / vhth[s]) .^ (2k)))
            end
        end
    end
end

show_trace = false
# is_Jacobian = false
# is_x0_noise = false
########## The initial solution
p_noise_rel = 1e-12      # `p_noise_rel ≤ 1e-3` default. Relative error of the initial guess parameters `p`
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
king!(yfit, xfit; Ml1=Ml1,l=0)

Jfit = zeros(3nModel,3nModel)
king_g!(Jfit, xfit; Ml1=Ml1,l=0)

dataJfit = DataFrame(Jfit,:auto)
fLn1efit = zero.(fLn1e)
fLn1efit = fvLDMz(fLn1efit,vGe,ℓ,naifit,uaifit,vthifit,nModel)
DfLn1e = fLn1e - fLn1efit

@show x0
@show dataJfit;
@show paraM(Jfit);
@show datafit;
@show dataerrfit;
@show norm(DfLn1e),n̂afit-1,KTEkfit - 1
@show p_noise_rel,p_noise_abs,is_Jacobian;
@show is_xconverged,is_fconverged,is_gconverged,norm(yfit);
println()
is_converged,niter,xssr,fmtf12.(xfit)
