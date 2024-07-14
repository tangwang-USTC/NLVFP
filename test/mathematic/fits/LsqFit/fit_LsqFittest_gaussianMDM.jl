using LsqFit

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

    dx:: = copy(delta_x) which is limited by `x_tol` and will be updated according to:
          delta_x = ( J'*J + lambda * Diagonal(DtD) ) \ ( -J'*value(df) )

"""

# ua > 0
LL1 = 15
ℓ = LL1 - 1
# Instead of applying the finite difference method
uMax = 20.0
u = ûa0[isp3]
xs = copy(vG0)
xs[1] == 0.0 ? (xvec = 2:nc0) : (xvec = 1:nc0)
xs = copy(xs[xvec])
ys = copy(fvL[nvlevel0,LL1,isp3][xvec])
ymax = maximum(ys)
ymax = 10^(round(log10(ymax)))
# ymax = 1.0
isJacob = false
atolrn = 1e-5

@show ℓ, ûa0, ymax
# # using module LeastSqyaresOptim.jl
# fopt = optimeze(model1(xs,p0))

ppp,sigma0, rn0, modelDM,MnDMk  = fvLmodelDM(xs,ys/ymax,ℓ;isJacob=isJacob)
# Notes: `u` and `uMax` are referrance datas to optimeze the convergence of the model by
# giveing a better guess parameters of the `neurons` in the model.
# The model follows the principle of simplicity with no need to be a accurate model,
# which just acts as a weight function to scaling the original function to be a low-order polynomial function.
pa, sigmaa, rna, modela,MnDMka = fvLmodel(xs,ys,ℓ,u,uMax,nc0;atolrn=atolrn)
# yfit2 = ppp[1] * modelDMexp(xs,ppp[2:3],ℓ)
yfit2 = modelDM(xs,ppp)
yfita = modela(xs,pa)
@show fmtf2.(sigma0)
@show fmtf2.(sigmaa)
# Plotting
label = string("y,ℓ=",ℓ)
py= plot(xs,ys,label=label,line=(1,:auto))
label = "y_fit2"
py = plot!(xs,yfit2,label=label,line=(1,:auto))
label = "y_fita"
py = plot!(xs,yfita,label=label,line=(1,:auto))

xlabel = string("σa=",fmtf2(norm(sigmaa)),", σ₀=",fmtf2(norm(sigma0)))
label = string("ys-yfit2")
erry_fit2 = ys - yfit2
perryfit2 = plot(xs,erry_fit2,label=label,line=(1,:auto),xlabel=xlabel)
label = string("ys-yfita")
erry_fita = ys - yfita
perryfita = plot(xs,erry_fita,label=label,line=(1,:auto),xlabel=xlabel)

xlabel = string("vG0,u=",fmtf2(ûa0[isp3]),",model=",MnDMk)
label = string("Rerr_yfit2")
Rerryfit2 = (ys ./ yfit2 .- 1)
pRerryfit2 = plot(xs,Rerryfit2,label=label,xlabel=xlabel,line=(1,:auto))

xlabel = string("vG0,u=",fmtf2(ûa0[isp3]),",model=",MnDMka)
label = string("Rerr_yfita")
Rerryfita = (ys ./ yfita .- 1)
pRerryfita = plot(xs,Rerryfita,label=label,xlabel=xlabel,line=(1,:auto))
display(plot(py,perryfit2,pRerryfit2,pRerryfita,layout=(2,2)))
