"""
  Optimization of the amplitude function `fhl0` by using the `King` functions 
    to finding the optimized parameters `(n̂ₛ,ûₛ,v̂ₜₕₛ) = (nai,uai,vthi)`.
"""

# is_x0_noise = true
# is_Jacobian = true
show_trace = false
# p_tol = 1e-37
# f_tol = 1e-37
# g_tol = 1e-37
# (optimizer, optimizer_abbr) = (LeastSquaresOptim.Dogleg, :dl)
# # (optimizer, optimizer_abbr) = (LeastSquaresOptim.LevenbergMarquardt, :lm)
# factor = LeastSquaresOptim.QR()               # `=QR(), default`
# # factor = LeastSquaresOptim.Cholesky()
# # factor = LeastSquaresOptim.LSMR()
# autodiffs = :central
# # autodiffs = :forward
# # ########## The initial solution
# maxIterKing = 200
nModel = nMod[isp3]
nMjMs = ceil(Int,3nModel / 2)
Mhsl0 = MsnnE3[1:nMjMs,1,isp3]
Mhsl1 = MsnnE3[1:nMjMs,2,isp3]
# Ml1[2:5] *= 1.0 + 1e-5
Mhsl01 = zeros(3nModel)
Mhsl01[1:2:end] = Mhsl0[1:nMjMs]
Mhsl01[2:2:end] = Mhsl1[1:(3nModel-nMjMs)]
nais, uais, vthis = fl0king01(nai0[isp3],uai0[isp3],vthi0[isp3],Mhsl01,nModel;
            optimizer=optimizer,factor=factor,autodiffs=autodiffs,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)



