using AlgebraicMultigrid
import IterativeSolvers: cg
import IterativeSolvers: gmres

n = 1000
A = poisson(n)
ml = ruge_stuben(A)
ml2 = smoothed_aggregation(A)
p = aspreconditioner(ml)
c = cg(A, A * ones(n), Pl=p)
c2 = gmres(A, A * ones(n); Pl=p)
