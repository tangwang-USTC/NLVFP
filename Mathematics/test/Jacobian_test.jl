
using AlgebraicMultigrid

HLn = HvL[:,4,isp3]
JHLn = zeros(nvremesh,nvremesh)
JHLn = jacobianDD(JHLn,HLn,va)
JHLn[abs.(JHLn) .< 1e-3] .= 0
A = sparse(JHLn)
LUa = ilu(A, τ = 1e-3)
rateLU = nnz(LUa) / nnz(A)  #
ml = ruge_stuben(A)
ml2 = smoothed_aggregation(A)
p = aspreconditioner(ml)
# x = zero(HLn)
# sor!(x,JHLn,HLn,1.2)

plot(HLn[211:nvremesh])
