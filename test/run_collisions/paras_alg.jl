
is_dMhck1_Mhck = false
is_dMhck1_abs = true
is_NK_nai_const = true
# ## Parameters of algorithms, models and solvers
if is_sol_fvLc
    functionName = :fvL
    # i_iter_rs2 = 0
    rtol_vGdom = 9e-1
end
is_err_renorm = true              # in `MsnnEvens.jl`, `err /= Mhc`
is_njMs_fix = true                # When the number of observative moments if equivalent which means `nMjMs .= njMs`
is_MjMs_max = true                # (default := true) which means `njMs :> 2nMod` if for `Mck` and `nMod` is to decided parameters `nai`, `uai` and `vthi`. 
is_fit_f = false                  # (default: false) whether optimize `nai,uai,vthi` according to `fvL`
is_Ms_nuT = false                 # When `nMod = 1` in `Ms` version, enforcing the higher-order moments to be function of `n, u, vth`
is_Cerror_dtnIKTs = true          # Whether to save the local conservation errors of `dtn, dtIs, dtKs` during the Fokker-Planck collision processes.
njMs_fix = 8                     # (=0, default),  for `njMs` in `vʲ⁺²`. The number of observative moments for every harmonic, `∀ℓ` which will changed according to the characteristics of `fvL`

if is_fvL_CP
    is_moments_out = false
else
    is_moments_out = true         # (default: false) in `Ms` version
end

#### for testting version
is_check_conservation_dtM = true
is_check_vth = true               # The covergence checking

# is_qconservation = false
is_plotMs = 0

## Parameters: Optimization solver
## Algorithm `King`: Optimizations for `fvL`, `jMsmax = ℓ + dj * (3(nMod-1) + (3-1))`.

NL_solve = :LeastSquaresOptim     # [:LeastSquaresOptim, :NL_solve, :JuMP]
# NL_solve = :NLsolve

if NL_solve == :NLsolve
    NL_solve_method = :trust_region    # [:trust_region, :newton, :anderson]
    # NL_solve_method = :newton        # always to be falure.
elseif NL_solve == :LeastSquaresOptim
    NL_solve_method = :trust_region    # [:trust_region, :newton, :anderson]
else
end 
rtol_tk = 1e-8                         # Relative tolerance for time
## Parameters: velocity space with key parameters (LM,vGmax,nnv0,ocp)
if 1 == 1
    ##### velocity axis and remeshing for v  #
    # Initial grid points on velocity axis. When `nvG` is too large, the CFL condition where `v̂ → 0` will be critical.
    nvG0 = 2 .^ nnv0 + 1   # [17, 33, 65, 129, 256, 512, ,2049]
    # For `vGmax` which is decided by hyperparameters `L_shape` and `jMax` inner adaptively.
    eps_fup, eps_flow = 1e-25, 1e-25 # (=epsT, default)
    maxiter_vGm = 100      # (=10, default), The maxinum number of iteration to find out the vaule of `vGmax`
    if e == e
        vGm_limit = [5.0,20.0]                 # The limit of the value of `vGm`
        
        # when `ocp ≤ 6`, the integrals will be incorrect.
        # The limit of `ocp is dependent on the initial grids with number of `nvG`
        # @warn("When `ocp = 9`, there may be a problem in the step of `FFTW.jl` before julia v1.5.4!")

        method3M = :chasing
        # method3M = :inv 
        # is_nvlimit0 = true   # (=true, nc_k = nv_limit0)
        # nv_limit0 = 200      # (=60,default) maximum of nv for v-grid,
        GQmethod = :clenshawcurtis # (default = :clenshawcurtis) which is ∈ [:clenshawcurtis, :chebyshev]
        # GQmethod = :chebyshev

        # #### # gridv_type = :uniform   # [:uniform, :uAdaptive, :cAdaptive , chebyshev, shiftedchebyshev]
        # gridv_type ≠ :chebyshev || (vadaptlevels = 0)
        # L_shape = 1                     # (=0, default), The order of `ℓ` to find the maximum velocity `vGmax`and
        #                                 # refining the velocity axis grids to find out `nc0`
        #                                 # Which will be decided by `uai`, `L_shape = max(0, floor(Int64,LM / 2))`
        #                                 # The hyperparameter to determining the time-step `dtk` for version `:Ms` and `IK`
        vadaptlevels ≤ 2 || @warn("vadaptlevels ≥ 3 may be given a not proper grids for this version owing to `nck::Int64` in the procedures. ")
        domain = [-1.0, 1.0] |> Vector{datatype}        #

        orderVconst = [0, 5]   # nck: (default: = [0,10], which is ∈ [0, N⁺]) for Rosenbluth potentials `H(v)` and `G(v)`
                               # which probably ensures the conservations with orders [0:10]
        orderVconstlimit = -1  # (=-1, defaultt) when `orderVconst[2]` lower than `orderVconstlimit`, 
                               # #####there will be no refinement which equivalent to `vadaptlevels=0`.

        ##### L-m spaces、、
        L_limit = 50           # limit of L for all spices, (Ta, Tb) = (1.0, 4.3) for single harmonic model:

        if 9 == 9
            # 
            uhvec = [3.0,2.0,1.45,1.414, 1.0, 0.965, 0.707,0.682, 0.482,  0.341,0.3, 0.145, 0.1, 0.048,    3e-2,     1e-2,    1.45e-2,    3e-3,   4.82e-3,   1.45e-3,   1e-3]
            lvec =  [45,34,  28,27,    25 , 23,   20, 20,  17,   14,14, 10, 10,   8,     7,           6,       6,         5,        5,        4,        4]
            # K:     9.0  4.0   9.0  2.0       1.0     4.0   0.5   2.0      1.0     0.5  9e-2   9e-2    1e-2
            # û:      3.0  2.0  1.45  1.414     1.0   0.965 0.707  0.682   0.482    0.341  0.3   0.145   0.1
            # L_limit: 45  34    28    27        23      25   20     20      17      14    14     10     10
            # index_L: 37  28    23  (21,-13)  (17,-13)  17   15   (13,-13) (11,-13) 10  (9,-12) (3,-12) (2~3,-12)
            # ℓ>(,)[1]: -2  -2   -2  (18,-3)   (18,-3)   -3   -3   (14,-3)  (12,-3)  -3    -3    -3~-6   (≥4,-3~-7)
            # index_uL:-13 -13   -13   -13      -13     -13   -13    -13     -13     -13   -13  (11,-13) (8+,-13)    ✓✓✓
            #                                                                                            (L≥10,1e-4)
            # K:        1e-2                 9e-4       1e-4     9e-4      9e-6    1e-4      9e-6     1e-6
            # û:         0.1       0.048      3e-2       1e-2    1.45e-2    3e-3   4.82e-3   1.45e-3   1e-3
            # L_limit:   10          8         7           6       6         5        5        4        4
            # index_L: (2~3,-12)  (1+,-12)   (0,-3)     (0,-3)   (0,-3)    (0,-5)   (0,-5)   (0,-7)  (ℓ=0,-7)(≥1,-13)  ✓✓
            #         (≥4,-3~-7) (≥3,-3~-7) (≥1,-5~-8) (-7~-10)  (-6~-10) (-9~-10) (-9~-11) (-10~-12) (≥1,-13)
            # index_uL: (8+,-13)   (2,-12)    (0,-3)    (0,-4)   (0,-3)   (0,-5)   (0,-5)   (0,-7)    -8            ✓✓
            #          (L≥10,1e-4) (3,1e-4)  (1,1e-4)   (-4~-5)  (-4~-5)  (-5~-7)  (-5~-6)  (-7~-8)
            # fMvp2,(ℓ=0):-1(abs-6) -2(-8)   -3(-8)    -5(-10)   -5(-10)  -7(-12)  -6(-12)  -9(-14)  -9(-14)
            # fMvp3,(ℓ=0):-1(abs-6) -2(-8)   -3(-9)    -5(-10)   -5(-10)  -7(-13)  -6(-12)  -9(-14)  -9(-14)
            #
            # K:        1e-6            1e-8         1e-10        1e-14        1e-16
            # û:         1e-3    4.82e-4 1e-4 4.82e-5 1e-5 4.82e-6 1e-7 4.82e-8 1e-8  4.82e-9
            # L_limit:    4        4      3     3      2     2      2     2     1     1
            # index_L: (≥1,-13) (≥1,-13) -13   -13    -13   -13    -13   -13   -13   -13     ✓✓✓
            #          (ℓ=0,-7) (ℓ=0,-9)
            # index_uL:  -8        -9    -11   -13    -13   -13    -13   -13   -13   -13     ✓✓
            # fMvp2,(ℓ=0): -9     -10    -13   -13    -14   -14    -14   -14   -14   -14
            # fMvp3,(ℓ=0): -9     -10    -13   -13    -14   -14    -14   -14   -14   -14

            # The relative error of fitting will be: `10^(index_)`, (ℓ→0,ℓₛ~LM1)

            # Interval `û ∈ [0.3,1e-3]` need special method to deciding the fitting process.
            # When `û ≤ 1e-3`, procedure `weightfunctions_v` is the best one.
            # When `û ≥ 0.3`, procedure `weightfunctions_uL` is the best one.
            # But when there are `û[1] ≤ 1e-3` and `û[2] ≥ 0.3` at the same time, how to decide the fitting process?

        end
        is_LM1_full = true         # (= false, default), when is `true` means `LM .= LM1`
        # is_LM1_full = false
        const rel_dfLM = epsT10     # maximum rate of change of `f(v)= ∑fL(v) for LM`
        const abs_dtfLM = 1e-27    # maximum rate of change of `∂ₜf(v)`, owing to `f(v) \times F(v)`
        const abs_dfLM = eps0
        abstol_dtfLM = epsT  # (=epsT, default) which is balanced with 
        # the change rate of the conservation moments and 
        # the relative error of `(f̂ₗᵐ)⁻¹∂ₜf̂ₗᵐ(v̂ᵢ→0)`
        # axisymmetric in velocity space
        m = 0                #
    end
    is_boundaryv0 = false  # whether applying the boundary condition procedure in code `Boundaries`
    is_resetv0 = false     # whether to set the values of `δtfvL(v̂ᵢ → 0)` to be zeros when its relative value is less than `1e-n`
end

# `LM` space and King space
if 1 == 1
    
    # # parameter for moments of `f̂ₗᵐ(v̂)` when `l ≥ 2`
    # j1 = 0                # The initial moment
    # dj = 2                # 
    # jMsmax = 6            # The maximum order of moments
                            # nMod   = 1,  2,   3
                            # jMsmax = 4,  10,  16 when `dj = 2 and j0 = ℓ`
    is_renorm = true        # (=true, default) whether applying re-normalization which is divided by the coefficient `CjLL2(j,L)` for normalized kinetic moments `Mhc`

    L_Mh_limit = 0          # (=0, default) limit of `L` for `Mh`
    is_IKh_const = true     # (=true, default) whether keep the values of `Ih` and `Kh` to be constants for spice `a` during optimization of the `King` function.

    # ########## The initial solution noises
    is_Jacobian = true      # (=true, default) Whether Jacobian matrix will be used to improve the performance of the optimizations.
    show_trace = false      # (=false, default) 
    maxIterKing = 100       # (=200 default) The maximum inner iteration number for King's optimization
    p_tol = epsT / 1000           # (= 1e-13, default)
    g_tol = epsT / 1000 
    f_tol = epsT / 1000 

    # NL_solve == :LeastSquaresOptim
    (optimizer, optimizer_abbr) = (LeastSquaresOptim.Dogleg, :dl)
    # (optimizer, optimizer_abbr) = (LeastSquaresOptim.LevenbergMarquardt, :lm)

    factorMethod = :QR     # factorMethod = [:QR, :Cholesky, :LSMR]
    # factorMethod = :Cholesky
    (factor, factor_abbr) = (LeastSquaresOptim.QR(), :QR)     # `=QR(), default`
    # factor = LeastSquaresOptim.Cholesky()
    # factor = LeastSquaresOptim.LSMR()

    # autodiff = :forward  #
    autodiff = :central    # A little more complicated but maybe obtaining no benefits.
    ############################ AI model: weight function

    # Corrections for conservative moments: `n, I, K`
    is_corrections = [true, true, true] # whether `n, I, K` will preserve the conservation laws 
    # by absorbing the residuals in a posterior analysis.
    is_corrections .= false
    residualMethod_FP0D = 1   # (= 1, default) `∈ [1,2]` which denotes applying dichotomy method or
    # geometric ratio method to absorb the residuals of conservative moments.
    #

end


## Algorithm `RδtfvL`: Optimizations `δtfvL` according to  `y(v̂→0) → Cₗᵐ` and `Rd1y(v̂→0) → 0`
is_δtfvLaa = 1          # [-1,    0,     1]. The outputs of the collision operators.
# [δtfaa, δtfab, δtfa]
if 1 == 1
    is_optimδtfvL = true    # Whether to optimize the function `δtfvL`
    is_optimδtfvL = false
    is_normal = true
    is_normδtf = false      # (= true, default), whether `cf3 = π^(-3/2) * na / vath^3` is included in the main collision procedures.
    order_dvδtf = 2         # (=2, default), `∈ [-1, 1, 2]` which denotes `[BackwardDiff, ForwardDiff, CentralDiff]`
    Nsmooth = 3             # (=3, default), which is `∈ N⁺ + 1`, the number of points to smooth the function `δtfvL`.
    order_smooth = 3        # (=2, default), which is `∈ N⁺`, the highest order of smoothness to smooth the function `δtfvL`.
    order_smooth_itp = 1    # (=0, default), (0,1,2) → (y=Rδtf,Rd1y,Rd2y), the order of function to be extrapolated to smooth the function `δtfvL`.
    order_nvc_itp = 4       # (=2, default), (1,2,3,N⁺≥4) → (nvcd1,nvcd2,nvcd3,max(nvcd2,nvcd3))
    is_boundv0 = zeros(Bool, order_smooth)
    is_boundv0[1] = true    # (::Vector{Bool}=[true,false], default). 
    # When it is `true`, the value `Rdiy(v[1]=0.0)` will be `true`.
    # It is `true` when `v[1] == 0.0` and the value `Rdiy[1] = 0.0`
    k_δtf = 2               # (=2, default), the order of the Spline Interpolations for `δtfLn(v̂→0)`
    Nitp = 10               # (=10, default), the number of grid points to generate the interpolating function for `δtfLn(v̂→0)`
    nvc0_limit = 4          # (=4, default), `nvc0_limit ∈ N⁺` which is the lower bound of
    # `nvc(order_nvc_itp)` to applying to the extrapolation for `δtfLn(v̂→0)`
    L1nvc_limit = 3         # (=3, default), `L1nvc_limit ∈ N⁺` which is the lower bound of `L` to applying to the extrapolation for `δtfLn(v̂→0)`
    if order_smooth == 2
        abstol_Rdy = [0.95, 0.5]      # ∈ (0, 1.0 → 10). 
        # When `abstol_Rdy > 0.5,` the change of `δtfvL` will be very sharp and localized.
        # When `û → 0.0`, the higher-order components 
        # When `û ≥ 0.5 ≫ 0.0`, lower value , `abstol_Rdy < 0.5` is proposed.
    elseif order_smooth == 3
        abstol_Rdy = [0.45, 0.45, 0.45]
    else
        erfghjmk
    end
end

## Algorithm `TGM`: Keyword arguments of the Trust Region algorithm, i.e., `Levenberg-Marquardt (LM)` or `Dogleg (DL)` algorithm.
if 1 == 1
    maxIterLM = 100
    restartfit::Vector{Int} = [0, 10, maxIterLM]
    # = [fit_restart_p0,fit_restart_p,maxIterLM] = [0,0,100]
    # which will decide the process of restart process of fitting process.
    # Parameter `maxIterLM` is the maximum iteration of the standard Trust Region algorithm.
    maxIterTR = 500        # The total maximum number of iterations in a specified fitting process.
    # filtering parameters
    n10 = 0                # [-5:1:10],  ys_min = 10^(-5 * n10) which sets the minimum value of vector `ys`.
    dnvs = 1               # `ys → ys[1:dnvs:end]`
    ############################ AI model: weight function
end

## Algorithm `GMRES`: Keyword arguments of `GMRE`S for optimization of `fvL`
if 1 == 1
    ϵ = 1.0e-7             # (=1e-4 default; not affect the result almost) for Matrix-Vector products, Jv
    #  ### # Newton convergence
    if datatype == Float64
        abstol = 1e-14     # (1e-8 default) for Poisson equation of Rosenbluth potential in GMRES optimization:
        abstolf = 1e-14    # (1e-8 default) for optimization of `fvL` with GMRES method:   optimRf
    elseif datatype == BigFloat
        abstol = 1e-14     # (1e-8 default) for Poisson equation of Rosenbluth potential in GMRES optimization:
        abstolf = 1e-14    # (1e-8 default) for optimization of `fvL` with GMRES method:   optimRf
    else
        tufuj
    end
    # abstolf = abstol
    reltol = 1e-5       # (1e-2 default) for GMRES, should not be too small
    restart = 20        # default `= 20` which is ∈ [1-nc:0:nc] for GMRES algorithm,
    # if `restart= 0`, means no restart is used.
    # `restart < 0`, means `restart += nc`
    maxiterN = 15       # (=5, default), maximum iteration number of Newton process for `H` and `G`
    maxiterA = 15       # (=15, default), "alp step": to find `alp` with for the optimized `x0` after the GMRES! step
    # parameters for optimization of `fvL`
    maxiterf = 0        # (=0, default), which is ∈ [0, N⁺]
    # = 0 denotes`δₜfvL` is conservations collisions without optimizations of  `fvL`.
    maxiterNf = 15      # (=5, default), maximum iteration number of Newton process for `fvL`
    iter_MsMax = 17     # Maximum number of iterations for conservations convergence
    isRel = :unit       # ∈ [:unit, :Max, :Maxd0, :Maxd1, :Maxd2], Normalization scheme for Rosenbluth potentials
    mode = :max         # = [:norm, :max]        # For optimRX,
    # mode = :norm   # = [:norm, :max]
end
 
## Parameters: physics model according to n[i]
if e == e
    # is_PR = 0            # power of radiation
    # is_fus = 0           # fusion reactions
    # n0_c = 1e-10           # n[i] < n0_c means spices[i] is absent
    # i-e collision: maybe Rtol_ds > 0
end

# Lagrange_uC = :zero               # [:zero, :momentum, :normalzied], the scheme to calculate the relative velocity `uC`
#                                   # i. `Lagrange_uC == :zero` denotes no Lagrange coodinate will be used;
#                                   # ii. `Lagrange_uC == :momentum` denotes `uC = (ma * na * ua + mb * nb * ub) / (ma * na + mb * nb)`
#                                   # iii. `Lagrange_uC == :normalzied` denotes `uC = (vbth * ua + vath * ub) / (vath + vbth)`
1

# dtk_limit = 1e-6              # 
# errRdtM = epsT1000
# atol_dMs = epsT1000   # abstol for conservations.
# RerrDMc = epsT1000                # = (Mc ./ Mc_copy) .- 1
# Ratio_DMc = 1e-10                 # = (RerrDMc ./ RerrDMc_copy) .- 1
# Rtol_Ms_termination = 1e-4
# is_checking_nIKT_dt = false

# println("//// abstol=", abstol, ", rtol=", reltol, ",mode=", mode)

# ##             switchs which will be removed in the fulture
if 2e == e
    # is_check_Mcs = false
    # is_check_Mhc = false
    # is_check_RMcs = false
    # is_check_dtMcs = false

    isdtfcon = false
    is_vsmapping = false
    is_plot = 0
    is_Optimdfddf = 0
    is_plotf = 0
    is_plotF = 0
    is_plotLagfc = 0
    is_plotdf = 0
    is_plotIJ = 0
    is_dfFHG_testing = true
    is_plotdH = 0
    is_plotdG = 0
    is_plotdfHG = 0
    is_plotCHG = 0
    is_plotdtf = 0     # When solving VFP
    is_plotdtfp = 0     # When solving VFP
    is_smoothdtf = 0
    is_plotdtf8 = 0
    is_dtfv_method = 0
    is_plotdtfL1 = 0
    is_plotdtCFL = 0
    is_solver_t = 1
    is_plot_nuT = 0    # for plotting of moments
end
