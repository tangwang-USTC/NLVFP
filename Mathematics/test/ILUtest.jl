using IterativeSolvers, IncompleteLU
using SparseArrays, LinearAlgebra
using BenchmarkTools
using Plots

"""
Benchmarks a non-symmetric n × n × n problem
with and without the ILU preconditioner.
"""
function mytest(n = 64)
   n = n
    N = n^3

    A = spdiagm(
      -1 => fill(-1.0, n - 1),
       0 => fill(3.0, n),
       1 => fill(-2.0, n - 1)
    )
    Id = sparse(1.0I, n, n)
    A = kron(A, Id) + kron(Id, A)
    A = kron(A, Id) + kron(Id, A)
    x = ones(N)
    b = A * x

    LU = ilu(A, τ = 0.1)
    @show nnz(LU) / nnz(A)

    # Benchmarks
    prec = @benchmark ilu($A, τ = 0.1)
    @show prec
    with = @benchmark bicgstabl($A, $b, 2, Pl = $LU, max_mv_products = 2000)
    @show with
    without = @benchmark bicgstabl($A, $b, 2, max_mv_products = 2000)
    @show without

    # Result
    x_with, hist_with = bicgstabl(A, b, 2, Pl = LU, max_mv_products = 2000, log = true)
    x_without, hist_without = bicgstabl(A, b, 2, max_mv_products = 2000, log = true)

    @show norm(b - A * x_with) / norm(b)
    @show norm(b - A * x_without) / norm(b)

    plot(hist_with[:resnorm], yscale = :log10, label = "With ILU preconditioning", xlabel = "Iteration", ylabel = "Residual norm (preconditioned)", mark = :x)
    plot!(hist_without[:resnorm], label = "Without preconditioning", mark = :x)
end
