
using Metaheuristics

f(x) = 10length(x) + sum( x.^2 - 10cos.(2π*x) ) # objective function
bounds = [-5ones(10) 5ones(10)]' # limits/bounds
information = Information(f_optimum = 0.0); # information on the minimization problem
options = Options(f_calls_limit = 9000*10, f_tol = 1e-5); # generic settings
algorithm = ECA(information = information, options = options) # metaheuristic used to optimize
@btime result = Metaheuristics.optimize(f, bounds, algorithm) # start the minimization proccess
mpp = minimizer(result)

# function fres(res,modelDM,ys,vs,p)
#
#     # for i in 1:length(vs)
#     #     res[i] = ys[i] - modelDM(vs[i],p)
#     # end
#     # res = ys - modelDM(vs,p)   # not correct, by why ?
#     res[:] = ys - modelDM(vs,p)
#     return res
# end
# algorithm = ECA() # metaheuristic used to optimize
# lbs = [0.0, 0.0, 0.0]
# ubs = [5pp0[1], 20.0, 10]
# bounds = reshape([lbs ubs],2,3)
# @time result = Metaheuristics.optimize(f, bounds, algorithm) # start the minimization proccess
# mp = minimizer(result)
