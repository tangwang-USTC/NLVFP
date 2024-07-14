using DiffEqDiffTools, Test, LinearAlgebra

# Jacobian tests
function f(fvec,x)
    fvec[1] = (x[1]+3)*(x[2]^3-7)+18
    fvec[2] = sin(x[2]*exp(x[1])-1)
end
x = rand(2); y = rand(2)
f(y,x)

J_ref = [[-7+x[2]^3 3*(3+x[1])*x[2]^2]; [exp(x[1])*x[2]*cos(1-exp(x[1])*x[2]) exp(x[1])*cos(1-exp(x[1])*x[2])]]
J = zero(J_ref)
df = zero(x)
df_ref = diag(J_ref)
epsilon = zero(x)
forward_cache = DiffEqDiffTools.JacobianCache(x,Val{:forward})
central_cache = DiffEqDiffTools.JacobianCache(x)
complex_cache = DiffEqDiffTools.JacobianCache(x,Val{:complex})
f_in = copy(y)

@time @testset "Jacobian StridedArray real-valued tests" begin
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, forward_cache), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, forward_cache, relstep=sqrt(eps())), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, forward_cache, f_in), J_ref) < 1e-4
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, central_cache), J_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x), J_ref) < 1e-8
    @test err_func(DiffEqDiffTools.finite_difference_jacobian(f, x, complex_cache), J_ref) < 1e-14
end
