
include(joinpath(pathroot,"test/run_collisions/paras_phys.jl"))
        
include(joinpath(pathroot,"test/run_collisions/paras_alg.jl"))

include(joinpath(pathroot,"test/run_collisions/algorithm/initials.jl")) 

include(joinpath(pathroot,"test/run_collisions/paras_sol_RKn.jl"))

include(joinpath(pathroot,"test/run_collisions/sol_MsIK.jl"))

"""
  1, PseudoSpectral method: Legendre polynomials for the velocity angular spaces;
  
  2, Multi-level conservations is achieved through the Gaussian quadrature;
  
  3, Orthogonalization based on `Besseli` functions is used to smoothing the coefficients function `f̂ₗ(v̂)`.
  
     Regularization (正则化) method: for ill-posed equations, by adding suitable constraint equations,
                     using a set posed equations which is equivalent to
                     the original equations to approximate the solution.
  
                     I, Tikhonov regularization which is based on the variational principle.
                     II, Iterative regularization.
     Renormalization (重整化): In physics, it is any of a collection of techniques used to
                    treat infinities arising in calculated quantities.
  
     Orthogonalization (正交化): The process of finding a set of prthogonal vectors that span a particular subspace.
                     -- Krylov subspace methods
  
                     A, Householder transformation, which used reflection
                        giving all the vectors only at the end step.
  
                        * More numberically stable, i.e. rounding errors tend to have less serious effects.
  
                     B, Gram-Schmidt process, which uses projection
                        producing the `jᵗʰ` orthogonlized vector after the `jᵗʰ` iteration.
                        * More applicable for iterative methods like the
                        Arnoldi iteration.
  
                     C, Givens rotation
                        * The Givens rotation is more easily parallelized than Householder transformations.
  
  4, Miltidomians scheme is used on the velocity axis;
  
  5, `LMS / Chasing / QR / GMRES` method are used to solve the nonlinear algebraic equations;
  
  6, In the process of pseudo-spectral on angular space, symmetries of matrix `Mμ`
     and `Mun` should be used to improve the performance of the pseudo-spectral method.
  
  7, Explict Euler solver and Explict RK4 solver are used to solve the nonlinear FPS equation 
  
  Questions:
  
  I, In the Lagrange frames, what is the transforned distribution function of the one in Newton frames?
  
  II, How to adaptive mesh generation include fine-tune `LM` and `vGmax`?
  
  III, How to reduce the number of meshgrids `nvG` and `vGmax`?
  
  IV, How to speedup the optimization process by using the historic message `nvc3`?
  
  V, How to optimize the number of submoments `nMod` adaptively to improve the performance of the algorithm.
  
  VI. The round-off errors and cumulative errors of different solving strategies: 
      * The direct kinetic solver
      * The kinetic micro-moments solver
  
  VII. The model errors and discretization errors.
      * LM, nvG, vGmax and nMod
  
  VIII, When `nMod==1`, the algorithm is correct and effecient. What is performance when `nMod ≥ 1`?
  
  IIX, Whether there are some relations which is similar to the conservation laws for higher-order no-conservative moments? 
  
"""
