
"""
   ODEEquations.jl
   DifferentialEquations.jl

   Stiffness: [:auto, :siff, :nonstiff]
              [:interpolant, :memorybound]
  
   Inputs:
     out: (= :alg, default), [:alg, :name, :both]
   Outputs:
     alg, algname = solverODEAlg(;out=out,IMEXplicit=IMEXplicit,is_multistep=is_multistep,
           is_stiff=is_stiff,is_fixed_timestep=is_fixed_timestep,order_stage=order_stage)
"""

function solverODEAlg(;out::Symbol=:alg, IMEXplicit::Symbol=:explicit, is_multistep::Bool=false, 
   is_fixed_timestep::Bool=false,is_stiff::Bool=false, order_stage::Int64=1, isparallel::Bool=false)

   if IMEXplicit == :explicit
      # For non-stiff equations
      if is_stiff
         ##### Stabilized explicit methods for stiff equations
         alg, algname = RKC(),  RKC         # Second order stabilized Runge-Kutta method. Exhibits high stability for real eigenvalues and is smoothened to allow for moderate sized complex eigenvalues.
         # alg, algname = ROCK2(), ROCK2
         # alg, algname = ROCK4(), ROCK4    # Fourth order
         # alg, algname = SERK2(), SERK2    # Second order stabilized extrapolated Runge-Kutta method.
         # alg, algname = ESERK5(), ESERK5  # Fifth order
      else
         if is_multistep == false
            # most of single-step algorithms are belong to Runge-Kutta (RK) methods
   
            # For non-stiff problems
            # Mostly, these high order RK methods are more robust than the high order Adams-Bashforth (AB) methods 
            # to discontinuities and achieve very high precision, and are much more efficient than the extrapolation methods.
   
            # # Notes: Tsit5() for most non-stiff problems
            #          BS3() for fast testing
            #          OwrenZen3() for fast testing when the interpolation error is important
            #          OwrenZen5() if at moderate tolerances and the interpolation error is very important
            #          BS5() for more robust error control is required
            #          Vern6() for high accuracy but with the range of (10⁻⁸ ~ 10⁻¹²)
            #          Vern7()
            #          Vern8()
            #          Vern9() for high accuracy but with the range of (≤ 10⁻¹²) with a high order interpolant
            #          Feagin12() for extremely high accuracy (≤ 10⁻²⁸) without high order interpolant
            #          Feagin14(), Feagin methods are the only high-order optimized methods
            #          Feagin14(), Feagin methods are the only high-order optimized methods
   
            # If strict error bounds are needed, then adaptive methods with defect controls are required.
            #          RK4() is a good choice for medium accuracy calculations.
            
            if is_fixed_timestep
               if order_stage == 1
                  ##############################################
                  # alg, algname = Euler(),Euler          # The canonical forward Euler method
                  # alg, algname = Midpoint(),Midpoint    # Uses embedded Euler method for adaptivity
                  # alg, algname = Heun(),Heun            # Uses embedded Euler method for adaptivity
                  # alg, algname = Ralston(),Ralston      # The optimized second order midpoint method, Uses embedded Euler method for adaptivity
            
                  # alg, algname = BS3(),BS3              # Bogacki-Shampine 3/2 method.
                  # alg, algname = RK4(),RK4              # The canonical Runge-Kutta Order 4 method.
                  # alg, algname = BS5(),BS5              # Bogacki-Shampine 5/4 Runge-Kutta method. (lazy 5th order interpolant).
                  # alg, algname = OwrenZen3(),OwrenZen3  # Owren-Zennaro optimized interpolation 3/2 method (free 3rd order interpolant).
                  # alg, algname = OwrenZen4(),OwrenZen4
                  alg, algname = OwrenZen5(),OwrenZen5
                  alg, algname = DP5(),DP5            # Dormand-Prince's 5/4 Runge-Kutta method. (free 4th order interpolant).
                  alg, algname = Tsit5(),Tsit5        # Tsitouras 5/4 Runge-Kutta method. (free 4th order interpolant).
                  # alg, algname = KuttaPRK2p5(),KuttaPRK2p5 # A 5 parallel, 2 processor explicit Runge-Kutta method of 5th order.
                  # alg, algname = Anas5(0),Anas5        # Anas5(w), 4th order Runge-Kutta method designed for periodic problems.Requires a periodicity estimate `w`
                  # alg, algname = MSRK5(),MSRK5         # Stepanov 5th-order Runge-Kutta method.
                  # alg, algname = RKO65(),RKO65         # Tsitouras' Runge-Kutta-Oliver 6 stage 5th order method which is robust on problems which have a singularity at `t =0`
                  # alg, algname = FRK65(0),FRK65        # FRK65(w), which `w=0 default`, Zero Dissipation Runge-Kutta of 6th order.
                  # alg, algname = MSRK6(),MSRK6
                  # alg, algname = Alshina6(),Alshina6
                  # alg, algname = TanYam7(),TanYam7    # Tanaka-Yamashita 7 Runge-Kutta method.
                  # alg, algname = DP8(),DP8            # Hairer's 8/5/3 adaption of the Dormand-Prince Runge-Kutta method.
                  # alg, algname = PFRK87(0),PFRK87     # Phase-fitted Runge-Kutta of 8th order.
                  # alg, algname = TsitPap8(),TsitPap8  # Tsitouras-Papakostas 8/7 Runge-Kutta method.
                  # alg, algname = Feagin10(),Feagin10  # Feagin's 10th-order Runge-Kutta method.
                  # alg, algname = Feagin12(),Feagin12
                  # alg, algname = Feagin14(),Feagin14
                  # alg, algname = Stepanov5(),Stepanov5 # Stepanov adaptive 5th-order Runge-Kutta method.
                  # alg, algname = SIR54(),SIR54         # 5th order explicit Runge-Kutta method suited for SIR-type epidemic models.
                  # alg, algname = Alshina2(),Alshina2   # Alshina 2nd-order Runge-Kutta method.
                  # alg, algname = Alshina3(),Alshina3
     
                  ## Tableau Method
                  # alg, algname = ExplicitRK(tableau = constructDormandPrince()),ExplicitRK
   
                  #### The following algorithms have a lazy interpolant to achieve higher accuracy (10⁻⁸ ~ 10⁻¹²) efficiently
                  # alg, algname = BS5(),BS5
                  # alg, algname = Vern6(),Vern6       # Verner's “Most Efficient” 6/5 Runge-Kutta method. (lazy 6th order interpolant).
                  # alg, algname = Vern7(),Vern7       # ManifoldProjection(conservationFun)
                  # alg, algname = Vern8(),Vern8
                  # alg, algname = Vern9(),Vern9
               else
                  ############ Explicit strong-stability preserving Runge-Kutta methods for Hyperbolic PDEs
                  # # multistage stability preserving (SSP) method of Shu and Osher
                  # Zhang and Shu (Zhang, Xiangxiong, and Chi-Wang Shu. "Maximum-principle-satisfying and positivity-preserving high-order schemes for conservation laws: survey and new developments." Proceedings of the Royal Society of London A: Mathematical, Physical and Engineering Sciences. The Royal Society, 2011.).
                  # # # SSPXY(stage_limiter!, step_limiter!)

                  # alg, algname = SSPRK22(), SSPRK22   # The two-stage, second order strong stability preserving (SSP) method of Shu and Osher
                  # alg, algname = SSPRK33(), SSPRK33   # (SSP coefficient 1, free 2nd order SSP interpolant)
                  # alg, algname = SSPRK432(),SSPRK432  # A 3/2 adaptive strong stability preserving (SSP) method with five stages(2, 2)
                  # alg, algname = SSPRK43(),SSPRK43    # The main method is the same as SSPRK432, but the embedded method has a larger stability region.
                  alg, algname = SSPRK53(), SSPRK53     #  The five-stage, third order strong stability preserving (SSP) method of Ruuth(SSP coefficient 2.65, free 3rd order Hermite interpolant)
                  alg, algname = SSPRK54(), SSPRK54     # (1.108, 4),  (SSP) method of Spiteri and Ruuth
                  alg, algname = SSPRK63(), SSPRK63     # (SSP coefficient 3.518, free 3rd order Hermite interpolant)
                  alg, algname = SSPRK73(), SSPRK73     # (4.2879, 3)
                  alg, algname = SSPRK83(), SSPRK83     # (5.107, 3)
                  alg, algname = SSPRK932(),SSPRK932    # A 3/2 adaptive strong stability preserving (SSP) method with nine stages(6, 3)
                  alg, algname = SSPRK104(),SSPRK104      # (6, 4)       (SSP) method of Ketcheson
                  alg, algname = SSPRKMSVS32(),SSPRKMSVS32   # (0.5, 2)
                  alg, algname = SSPRKMSVS43(),SSPRKMSVS43   # (0.33, 3)

                  ##### Low-Storage methods
                  
                  alg, algname = CKLLSRK43_2(), CKLLSRK43_2        # 4-stage, third order low-storage scheme, optimized for compressible Navier–Stokes equations.
                  alg, algname = CKLLSRK54_3C(), CKLLSRK54_3C
                  alg, algname = CKLLSRK54_3C_3R(), CKLLSRK54_3C_3R # 5-stage, fourth order low-storage scheme, optimized for compressible Navier–Stokes equations.
                  alg, algname = CKLLSRK54_3M_3R(), CKLLSRK54_3M_3R
                  alg, algname = CKLLSRK54_3N_3R(), CKLLSRK54_3N_3R
                  alg, algname = CKLLSRK54_3N_4R(), CKLLSRK54_3N_4R
                  alg, algname = CKLLSRK54_3M_4R(), CKLLSRK54_3M_4R
                  alg, algname = CKLLSRK65_4M_4R(), CKLLSRK65_4M_4R
                  alg, algname = CKLLSRK75_4M_5R(), CKLLSRK75_4M_5R
                  alg, algname = CKLLSRK85_4C_3R(), CKLLSRK85_4C_3R
                  alg, algname = CKLLSRK85_4M_3R(), CKLLSRK85_4M_3R
                  alg, algname = CKLLSRK85_4P_3R(), CKLLSRK85_4P_3R
                  alg, algname = CKLLSRK85_4FM_4R(), CKLLSRK85_4FM_4R
                  alg, algname = CKLLSRK95_4S(), CKLLSRK95_4S
                  alg, algname = CKLLSRK95_4C(), CKLLSRK95_4C
                  alg, algname = CKLLSRK95_4M(), CKLLSRK95_4M
                  
                  # alg, algname = RDPK3Sp35(), RDPK3Sp35              #  5-stage, third order low-storage scheme with embedded error estimator, optimized for compressible fluid mechanics
                  # alg, algname = RDPK3SpFSAL35(), RDPK3SpFSAL35
                  alg, algname = RDPK3Sp49(), RDPK3Sp49
                  alg, algname = RDPK3SpFSAL49(),RDPK3SpFSAL49 
                  alg, algname = RDPK3Sp510(), RDPK3Sp510
                  alg, algname = RDPK3SpFSAL510(), RDPK3SpFSAL510

                  # ##alg, algname = ORK256(), ORK256           # 5-stage, second order low-storage method for wave propagation equations.
                  # alg, algname = SSPRK53_2N1(), SSPRK53_2N1   # 5-stage, third order low-storage methods with large SSP coefficients. 
                  # alg, algname = SSPRK53_2N2(), SSPRK53_2N2
                  alg, algname = CarpenterKennedy2N54(), CarpenterKennedy2N54  # (5,4) Designed for hyperbolic PDEs (stability properties).
                  alg, algname = NDBLSRK124(), NDBLSRK124   # (12,4)  optimized stability regions for advection-dominated problems.
                  alg, algname = NDBLSRK134(), NDBLSRK134   #         advection-dominated problems.
                  alg, algname = NDBLSRK144(), NDBLSRK144   #         advection-dominated problems.
                  alg, algname = SHLDDRK64(), SHLDDRK64     # low-stage, low-dissipation, low-dispersion scheme. 
                  alg, algname = RK46NL(), RK46NL
                  alg, algname = CFRLDDRK64(), CFRLDDRK64   # 6-stage, fourth order low-storage, low-dissipation, low-dispersion scheme. 
                  alg, algname = DGLDDRK73_C(), DGLDDRK73_C # low-storage low-dissipation, low-dispersion scheme for discontinuous Galerkin space discretizations applied to wave propagation problems, 
                  alg, algname = TSLDDRK74(), TSLDDRK74        #  fourth order low-storage low-dissipation, low-dispersion scheme with maximal accuracy and stability limit along the imaginary axes
                  alg, algname = DGLDDRK84_C(), DGLDDRK84_C # optimized for PDE discretizations when maximum spatial step is small due to geometric features of computational domain. 
                  alg, algname = DGLDDRK84_F(), DGLDDRK84_F # optimized for PDE discretizations when the maximum spatial step size is not constrained. 
                  
                  ###### Low-Storage methods
                    # alg, algname = ParsaniKetchesonDeconinck3S32(), ParsaniKetchesonDeconinck3S32 # 3-stage, second order (3S) low-storage scheme, optimized for the spectral difference method applied to wave propagation problems.
                    # alg, algname = ParsaniKetchesonDeconinck3S82(), ParsaniKetchesonDeconinck3S82
                    # alg, algname = ParsaniKetchesonDeconinck3S53(), ParsaniKetchesonDeconinck3S53
                    # alg, algname = ParsaniKetchesonDeconinck3S173(), ParsaniKetchesonDeconinck3S173
                    # alg, algname = ParsaniKetchesonDeconinck3S94(), ParsaniKetchesonDeconinck3S94
                    # alg, algname = ParsaniKetchesonDeconinck3S184(), ParsaniKetchesonDeconinck3S184
                    # alg, algname = ParsaniKetchesonDeconinck3S105(), ParsaniKetchesonDeconinck3S105
                    # alg, algname = ParsaniKetchesonDeconinck3S205(), ParsaniKetchesonDeconinck3S205 # 20-stage, fifth order (3S)
                  
               end
            else
               if order_stage == 0     # Explicit extrapolation methods
                  # The following are adaptive order, adaptive step size extrapolation methods:
      
                  # alg, algname = AitkenNeville(),AitkenNeville   # Euler extrapolation using Aitken-Neville with the Romberg Sequence.
                  # alg, algname = ExtrapolationMidpointDeuflhard(),ExtrapolationMidpointDeuflhard   # Midpoint extrapolation using Barycentric coordinates
                  alg, algname = ExtrapolationMidpointHairerWanner(),ExtrapolationMidpointHairerWanner   # Midpoint extrapolation using Barycentric coordinates, following Hairer's ODEX in the adaptivity behavior.
               elseif order_stage == 1
                  # alg, algname = Midpoint(),Midpoint    # Uses embedded Euler method for adaptivity
                  # alg, algname = Heun(),Heun            # Uses embedded Euler method for adaptivity
                  # alg, algname = Ralston(),Ralston      # The optimized second order midpoint method, Uses embedded Euler method for adaptivity
                  # alg, algname = Alshina2(),Alshina2   # Alshina 2nd-order Runge-Kutta method.
                  # alg, algname = Alshina3(),Alshina3
                  
                  # alg, algname = BS3(),BS3              # Bogacki-Shampine 3/2 method.
                  # alg, algname = OwrenZen3(),OwrenZen3  # Owren-Zennaro optimized interpolation 3/2 method (free 3rd order interpolant).
                  # alg, algname = OwrenZen4(),OwrenZen4
                  # alg, algname = RK4(),RK4              # The canonical Runge-Kutta Order 4 method.
                  # alg, algname = SIR54(),SIR54         # 5th order explicit Runge-Kutta method suited for SIR-type epidemic models.
                  # alg, algname = BS5(),BS5              # Bogacki-Shampine 5/4 Runge-Kutta method. (lazy 5th order interpolant).
                  # alg, algname = Stepanov5(),Stepanov5 # Stepanov adaptive 5th-order Runge-Kutta method.
                  # alg, algname = OwrenZen5(),OwrenZen5
                  # alg, algname = DP5(),DP5            # Dormand-Prince's 5/4 Runge-Kutta method. (free 4th order interpolant).
                  alg, algname = Tsit5(),Tsit5        # Tsitouras 5/4 Runge-Kutta method. (free 4th order interpolant).
                  # alg, algname = Alshina6(),Alshina6
                  # alg, algname = TanYam7(),TanYam7    # Tanaka-Yamashita 7 Runge-Kutta method.
                  # alg, algname = DP8(),DP8            # Hairer's 8/5/3 adaption of the Dormand-Prince Runge-Kutta method.
                  # alg, algname = TsitPap8(),TsitPap8  # Tsitouras-Papakostas 8/7 Runge-Kutta method.
                  # alg, algname = Feagin10(),Feagin10  # Feagin's 10th-order Runge-Kutta method.
                  # alg, algname = Feagin12(),Feagin12
                  # alg, algname = Feagin14(),Feagin14
                  
                  ## Tableau Method
                  # alg, algname = ExplicitRK(tableau = constructDormandPrince()),ExplicitRK
   
                  #### The following algorithms have a lazy interpolant to achieve higher accuracy (10⁻⁸ ~ 10⁻¹²) efficiently
                  # alg, algname = BS5(),BS5
                  # alg, algname = Vern6(),Vern6       # Verner's “Most Efficient” 6/5 Runge-Kutta method. (lazy 6th order interpolant).
                  # alg, algname = Vern7(),Vern7       # ManifoldProjection(conservationFun)
                  # alg, algname = Vern8(),Vern8
                  # alg, algname = Vern9(),Vern9

                  # Parallel explicit RK methods
                     # alg, algname = KuttaPRK2p5(),KuttaPRK2p5
               else
                  ############ Explicit strong-stability preserving Runge-Kutta methods for Hyperbolic PDEs
                  # # multistage stability preserving (SSP) method of Shu and Osher
                  # Zhang and Shu (Zhang, Xiangxiong, and Chi-Wang Shu. "Maximum-principle-satisfying and positivity-preserving high-order schemes for conservation laws: survey and new developments." Proceedings of the Royal Society of London A: Mathematical, Physical and Engineering Sciences. The Royal Society, 2011.).
                  
                  # alg, algname = SSPRK432(),SSPRK432       # A 3/2 adaptive strong stability preserving (SSP) method with five stages(2, 2)
                  # alg, algname = SSPRK43(),SSPRK43        # The main method is the same as SSPRK432, but the embedded method has a larger stability region.
                  # alg, algname = SSPRK932(),SSPRK932       # A 3/2 adaptive strong stability preserving (SSP) method with nine stages(6, 3)

                  ###### Low-Storage methods
                  # alg, algname = CKLLSRK43_2(), CKLLSRK43_2        # 4-stage, third order low-storage scheme, optimized for compressible Navier–Stokes equations.
                  # alg, algname = CKLLSRK54_3C(), CKLLSRK54_3C
                  # alg, algname = CKLLSRK54_3C_3R(), CKLLSRK54_3C_3R # 5-stage, fourth order low-storage scheme, optimized for compressible Navier–Stokes equations.
                  # alg, algname = CKLLSRK54_3M_3R(), CKLLSRK54_3M_3R
                  # alg, algname = CKLLSRK54_3N_3R(), CKLLSRK54_3N_3R
                  # alg, algname = CKLLSRK54_3N_4R(), CKLLSRK54_3N_4R
                  # alg, algname = CKLLSRK54_3M_4R(), CKLLSRK54_3M_4R
                  # alg, algname = CKLLSRK65_4M_4R(), CKLLSRK65_4M_4R
                  # alg, algname = CKLLSRK75_4M_5R(), CKLLSRK75_4M_5R
                  # alg, algname = CKLLSRK85_4C_3R(), CKLLSRK85_4C_3R
                  # alg, algname = CKLLSRK85_4M_3R(), CKLLSRK85_4M_3R
                  # alg, algname = CKLLSRK85_4P_3R(), CKLLSRK85_4P_3R
                  # alg, algname = CKLLSRK85_4FM_4R(), CKLLSRK85_4FM_4R
                  # alg, algname = RDPK3Sp35(), RDPK3Sp35              #  5-stage, third order low-storage scheme with embedded error estimator, optimized for compressible fluid mechanics
                  # alg, algname = RDPK3SpFSAL35(), RDPK3SpFSAL35
                  # alg, algname = RDPK3Sp49(), RDPK3Sp49
                  # alg, algname = RDPK3SpFSAL49(),RDPK3SpFSAL49 
                  # alg, algname = RDPK3Sp510(), RDPK3Sp510
                  # alg, algname = RDPK3SpFSAL510(), RDPK3SpFSAL510
                  # alg, algname = CKLLSRK95_4M(), CKLLSRK95_4M
                  # alg, algname = CKLLSRK95_4S(), CKLLSRK95_4S
                  # alg, algname = CKLLSRK95_4C(), CKLLSRK95_4C
               end
            end
         else
            # # @ Multistep method using the approximation at more than one previous mesh point to determine the approximation at the next point,
            # which tend to be more efficient as the size of the system or the cost of `f` increases.
            # # Adams-Bashforth explicit methods with multistep
            if is_fixed_timestep
               # # # The following algorithm Require a choice of `dt=dt_initial` for initial step
                 alg, algname = AB3(),AB3     # The 3-step third order multistep method. Ralston's Second Order Method is used to calculate starting values.
                 alg, algname = AB4(),AB4     # The 4-step fourth order multistep method. Runge-Kutta method of order 4 is used to calculate starting values
                 alg, algname = AB5(),AB5     # The 5-step fifth order multistep method. Runge-Kutta method of order 4 is used to calculate starting values.
                 alg, algname = ABM32(),ABM32   # It is third order method. In ABM32, AB3 works as predictor and Adams Moulton 2-steps method works as Corrector. Ralston's Second Order Method is used to calculate starting values.
                 alg, algname = ABM43(),ABM43
                 alg, algname = ABM54(),ABM54


               # # # # Adaptive step size Adams explicit methods with multistage
               # VCABM() method can be a good choice for high accuracy when the system of equations is very large (≥1000).
               #         but the function calculation is very expensive, or the solution is very smooth.
   
               alg, algname = VCAB3(),VCAB3     # The 3rd order Adams method. Bogacki-Shampine 3/2 method is used to calculate starting values.
               alg, algname = VCAB4(),VCAB4     # The 4th order Adams method. Runge-Kutta 4 is used to calculate starting values.
               alg, algname = VCAB5(),VCAB5     # 
               alg, algname = VCABM3(),VCABM3    # The 3rd order Adams-Moulton method. Bogacki-Shampine 3/2 method is used to calculate starting values.
               alg, algname = VCABM4(),VCABM4    # The 4th order Adams-Moulton method. Runge-Kutta 4 is used to calculate starting values.
               alg, algname = VCABM5(),VCABM5
               alg, algname = VCABM(),VCABM     # ode113, An adaptive order adaptive time Adams Moulton method. It uses an order adaptivity algorithm is derived from Shampine's DDEABM.
               # alg, algname = AN5(),AN5       # An adaptive 5th order fixed-leading coefficient Adams method in Nordsieck form.
               # alg, algname = JVODE_Adams(),JVODE_Adams # An adaptive time adaptive order fixed-leading coefficient Adams method in Nordsieck form.
            else
               # # # The following algorithm Require a choice of `dt=dt_initial` for initial step
                 alg, algname = AB3(),AB3     # The 3-step third order multistep method. Ralston's Second Order Method is used to calculate starting values.
                 alg, algname = AB4(),AB4     # The 4-step fourth order multistep method. Runge-Kutta method of order 4 is used to calculate starting values
                 alg, algname = AB5(),AB5     # The 5-step fifth order multistep method. Runge-Kutta method of order 4 is used to calculate starting values.
                 alg, algname = ABM32(),ABM32   # It is third order method. In ABM32, AB3 works as predictor and Adams Moulton 2-steps method works as Corrector. Ralston's Second Order Method is used to calculate starting values.
                 alg, algname = ABM43(),ABM43
                 alg, algname = ABM54(),ABM54


               # # # # Adaptive step size Adams explicit methods with multistage
               # VCABM() method can be a good choice for high accuracy when the system of equations is very large (≥1000).
               #         but the function calculation is very expensive, or the solution is very smooth.
   
               alg, algname = VCAB3(),VCAB3     # The 3rd order Adams method. Bogacki-Shampine 3/2 method is used to calculate starting values.
               alg, algname = VCAB4(),VCAB4     # The 4th order Adams method. Runge-Kutta 4 is used to calculate starting values.
               alg, algname = VCAB5(),VCAB5     # 
               alg, algname = VCABM3(),VCABM3    # The 3rd order Adams-Moulton method. Bogacki-Shampine 3/2 method is used to calculate starting values.
               alg, algname = VCABM4(),VCABM4    # The 4th order Adams-Moulton method. Runge-Kutta 4 is used to calculate starting values.
               alg, algname = VCABM5(),VCABM5
               alg, algname = VCABM(),VCABM     # ode113, An adaptive order adaptive time Adams Moulton method. It uses an order adaptivity algorithm is derived from Shampine's DDEABM.
               alg, algname = AN5(),AN5       # An adaptive 5th order fixed-leading coefficient Adams method in Nordsieck form.
               alg, algname = JVODE_Adams(),JVODE_Adams # An adaptive time adaptive order fixed-leading coefficient Adams method in Nordsieck form.
            end
         end
      end
   elseif IMEXplicit == :implicit
      ##### OrdinaryDiffEq.jl for stiff equations
      if is_multistep == false
         if is_fixed_timestep
            alg, algname = TRBDF2(),TRBDF2          # 4th order L-stable Rosenbrock-W method

            #### Exponential Runge-Kutta Methods
            # alg, algname = LawsonEuler(),LawsonEuler # First order exponential Euler scheme.
            # alg, algname = NorsettEuler(),NorsettEuler # First order exponential-RK scheme. Alias: ETD1.
            # alg, algname = ETD2(),ETD2
            # alg, algname = ETDRK2(),ETDRK2          # 2nd order exponential-RK scheme.
            # alg, algname = ETDRK3(),ETDRK3
            # alg, algname = ETDRK3(),ETDRK4
            # alg, algname = HochOst4(),HochOst4      # 4th order exponential-RK scheme with stiff order 4.

            #### Exponential Propagation Iterative Runge-Kutta Methods (EPIRK)
            # alg, algname = Exp4(),Exp4              # 4th order EPIRK scheme.
            # alg, algname = EPIRK4s3A(),EPIRK4s3A    # 4th order EPIRK scheme with stiff order 4.
            # alg, algname = EPIRK4s3B(),EPIRK4s3B
            # alg, algname = EPIRK5P1(),EPIRK5P1
            # alg, algname = EPIRK5P2(),EPIRK5P2
            # alg, algname = EPIRK5P3(),EPIRK5P3      # 5th order “horizontal” EPIRK scheme with stiff order 5.
            # alg, algname = EXPRB53s3(),EXPRB53s3    # 5th order EPIRK scheme with stiff order 5.
            
            ##### SDIRK (singly-diagonally implicit RK) methods, but not suitable for BigFloat due to ForwardDiff
           
            alg, algname = ImplicitEuler(;autodiff=false),ImplicitEuler         # (1, A-B-L-stable), Strong-stability preserving (SSP).
            # alg, algname = ImplicitMidpoint(;autodiff=false),ImplicitMidpoint   # (2, A-stable, symplectic, symmetric), Good for highly stiff equations which need symplectic integration.
            # alg, algname = Trapezoid(;autodiff=false),Trapezoid  # (2, , A-stable, symmetric), Good for highly stiff equations which are non-oscillatory. "Almost symplectic" without numerical dampening. Also known as Crank-Nicolson (C-N) when applied to PDEs.
            # alg, algname = TRBDF2(;autodiff=false),TRBDF2        # (2,A-B-L-S-stable one-step) , smoothed derivatives for highly stiff and oscillatory problems.
            # alg, algname = SDIRK2(;autodiff=false),SDIRK2        # (2,A-B-L-stable)
            # alg, algname = Kvaerno3(;autodiff=false),Kvaerno3    # (3,A-L-stable), stiffly-accurate 
            # alg, algname = KenCarp3(;autodiff=false),KenCarp3    # (3,A-L-stable), stiffly-accurate with splitting
            # alg, algname = Cash4(;autodiff=false),Cash4          # (4,A-L-stable)
            # alg, algname = Hairer4(;autodiff=false),Hairer4      # (4,A-L-stable)
            # alg, algname = Hairer42(;autodiff=false),Hairer42    # (4,A-L-stable)
            # alg, algname = Kvaerno4(;autodiff=false),Kvaerno4    # (4,A-L-stable), stiffly-accurate 
            # alg, algname = KenCarp4(;autodiff=false),KenCarp4    # (4,A-L-stable), stiffly-accurate with splitting
            # alg, algname = Kvaerno5(;autodiff=false),Kvaerno5    # (5,A-L-stable), stiffly-accurate 
            # alg, algname = KenCarp5(;autodiff=false),KenCarp5    # (5,A-L-stable), stiffly-accurate with splitting
 
            # alg, algname = KenCarp47(;autodiff=false),KenCarp47  # (4,A-L-stable), stiffly-accurate, seven-stage ESDIRK method with splitting
            # alg, algname = KenCarp58(;autodiff=false),KenCarp58  # (5,A-L-stable), stiffly-accurate, eigth-stage ESDIRK method with splitting
            # alg, algname = ESDIRK54I8L2SA(;autodiff=false),ESDIRK54I8L2SA
            # alg, algname = ESDIRK436L2SA2(;autodiff=false),ESDIRK436L2SA2
            # alg, algname = ESDIRK437L2SA(;autodiff=false),ESDIRK437L2SA
            # alg, algname = ESDIRK547L2SA2(;autodiff=false),ESDIRK547L2SA2
 
            #### Fully-implicit Runge-Kutta methods (FIRK)
            # alg, algname = RadauIIA3(;autodiff=false),RadauIIA3  # (3, A-B-L stable), with internal tableau complex basis transform for efficiency.
            # alg, algname = RadauIIA5(;autodiff=false),RadauIIA5
 
            #### Parallel Diagonally Implicit Runge-Kutta Methods
            # alg, algname = PDIRK44(;autodiff=false),PDIRK44      #  A 2 processor 4th order diagonally non-adaptive implicit method.
 
            ##### Rosenbrock methods, but not suitable for BigFloat due to ForwardDiff
            # alg, algname = ROS3P(;autodiff=false),ROS3P         # (3, A-stable), stiffly stable which Keeps high accuracy on discretizations of nonlinear parabolic PDEs.
            # alg, algname = Rodas3(;autodiff=false),Rodas3       # (3, A-stable), stiffly stable
            # alg, algname = RosShamp4(;autodiff=false),RosShamp4 # (4, A-stable)
            # alg, algname = Veldd4(;autodiff=false),Veldd4       # (4, D-stable)
            # alg, algname = Velds4(;autodiff=false),Velds4       # (4, A-stable)
            # alg, algname = GRK4T(;autodiff=false),GRK4T         # (4, efficient)
            # alg, algname = GRK4A(;autodiff=false),GRK4A         # (4, A-stable), Essentially "anti-L-stable" but efficient.
            # alg, algname = Ros4LStab(;autodiff=false),Ros4LStab # (4, L-stable)
            # alg, algname = Rodas4(;autodiff=false),Rodas4       # (4, A-stable), stiffly stable
            # alg, algname = Rodas42(;autodiff=false),Rodas42     # (4, A-stable), stiffly stable
            # alg, algname = Rodas4P(;autodiff=false),Rodas4P     # (4, A-stable), stiffly stable
            # alg, algname = Rodas5(;autodiff=false),Rodas5       # (5, A-stable), stiffly stable
            # alg, algname = Rodas5P(;autodiff=false),Rodas5P     # (5, A-stable), stiffly stable

            #### Implicit Strong-Stability Preserving Runge-Kutta Methods for Hyperbolic PDEs (Conservation Laws)
            # alg, algname = SSPSDIRK2(),SSPSDIRK2    (2, A-L-stable, symplectric) with the strong stability preserving (SSP) property (SSP coefficient 2)

            # #### Sympletic integrators

            #### GeometricIntegrators

         else
            # alg, algname = lsoda(),lsoda          # a well-known method which uses switching to solve both stiff and non-stiff equations.
            
             ##### SDIRK (singly-diagonally implicit RK) methods, but not suitable for BigFloat due to ForwardDiff
            
            # alg, algname = ImplicitEuler(;autodiff=false),ImplicitEuler         # (1, A-B-L-stable), Strong-stability preserving (SSP).
            # alg, algname = ImplicitMidpoint(;autodiff=false),ImplicitMidpoint   # (2, A-stable, symplectic, symmetric), Good for highly stiff equations which need symplectic integration.
            # alg, algname = Trapezoid(;autodiff=false),Trapezoid  # (2, , A-stable, symmetric), Good for highly stiff equations which are non-oscillatory. "Almost symplectic" without numerical dampening. Also known as Crank-Nicolson (C-N) when applied to PDEs.
            # alg, algname = TRBDF2(;autodiff=false),TRBDF2        # (2,A-B-L-S-stable one-step) , smoothed derivatives for highly stiff and oscillatory problems.
            # alg, algname = SDIRK2(;autodiff=false),SDIRK2        # (2,A-B-L-stable)
            # alg, algname = Kvaerno3(;autodiff=false),Kvaerno3    # (3,A-L-stable), stiffly-accurate 
            # alg, algname = KenCarp3(;autodiff=false),KenCarp3    # (3,A-L-stable), stiffly-accurate with splitting
            # alg, algname = Cash4(;autodiff=false),Cash4          # (4,A-L-stable)
            # alg, algname = Hairer4(;autodiff=false),Hairer4      # (4,A-L-stable)
            # alg, algname = Hairer42(;autodiff=false),Hairer42    # (4,A-L-stable)
            # alg, algname = Kvaerno4(;autodiff=false),Kvaerno4    # (4,A-L-stable), stiffly-accurate 
            # alg, algname = KenCarp4(;autodiff=false),KenCarp4    # (4,A-L-stable), stiffly-accurate with splitting
            # alg, algname = Kvaerno5(;autodiff=false),Kvaerno5    # (5,A-L-stable), stiffly-accurate 
            # alg, algname = KenCarp5(;autodiff=false),KenCarp5    # (5,A-L-stable), stiffly-accurate with splitting

            # alg, algname = KenCarp47(;autodiff=false),KenCarp47  # (4,A-L-stable), stiffly-accurate, seven-stage ESDIRK method with splitting
            alg, algname = KenCarp58(;autodiff=false),KenCarp58  # (5,A-L-stable), stiffly-accurate, eigth-stage ESDIRK method with splitting
            # alg, algname = ESDIRK54I8L2SA(;autodiff=false),ESDIRK54I8L2SA
            # alg, algname = ESDIRK436L2SA2(;autodiff=false),ESDIRK436L2SA2
            # alg, algname = ESDIRK437L2SA(;autodiff=false),ESDIRK437L2SA
            # alg, algname = ESDIRK547L2SA2(;autodiff=false),ESDIRK547L2SA2

            #### Fully-implicit Runge-Kutta methods (FIRK)
            # alg, algname = RadauIIA3(;autodiff=false),RadauIIA3  # (3, A-B-L stable), with internal tableau complex basis transform for efficiency.
            # alg, algname = RadauIIA5(;autodiff=false),RadauIIA5

            #### Parallel Diagonally Implicit Runge-Kutta Methods
            # alg, algname = PDIRK44(;autodiff=false),PDIRK44      #  A 2 processor 4th order diagonally non-adaptive implicit method.

            ##### Rosenbrock methods, but not suitable for BigFloat due to ForwardDiff
            # alg, algname = ROS3P(;autodiff=false),ROS3P         # (3, A-stable), stiffly stable which Keeps high accuracy on discretizations of nonlinear parabolic PDEs.
            # alg, algname = Rodas3(;autodiff=false),Rodas3       # (3, A-stable), stiffly stable
            # alg, algname = RosShamp4(;autodiff=false),RosShamp4 # (4, A-stable)
            # alg, algname = Veldd4(;autodiff=false),Veldd4       # (4, D-stable)
            # alg, algname = Velds4(;autodiff=false),Velds4       # (4, A-stable)
            # alg, algname = GRK4T(;autodiff=false),GRK4T         # (4, efficient)
            # alg, algname = GRK4A(;autodiff=false),GRK4A         # (4, A-stable), Essentially "anti-L-stable" but efficient.
            # alg, algname = Ros4LStab(;autodiff=false),Ros4LStab # (4, L-stable)
            # alg, algname = Rodas4(;autodiff=false),Rodas4       # (4, A-stable), stiffly stable
            # alg, algname = Rodas42(;autodiff=false),Rodas42     # (4, A-stable), stiffly stable
            # alg, algname = Rodas4P(;autodiff=false),Rodas4P     # (4, A-stable), stiffly stable
            # alg, algname = Rodas5(;autodiff=false),Rodas5       # (5, A-stable), stiffly stable
            alg, algname = Rodas5P(;autodiff=false),Rodas5P     # (5, A-stable), stiffly stable
         end
      else

         ##### Multistep methods, but not suitable for BigFloat due to ForwardDiff
         #### Quasi-constant stepping is the time stepping strategy which matches the classic GEAR, LSODE, and ode15s integrators.
         if is_fixed_timestep
            alg, algname = MEBDF2(;autodiff=false),MEBDF2
            
            # alg, algname = QNDF1(;autodiff=false),QNDF1   # (1, L-stable) numerical differentiation function (NDF) method.
            # alg, algname = QBDF1(;autodiff=false),QBDF1   # (1, L-stable) which is equivalent to implicit Euler but using the BDF error estimator.
            # alg, algname = ABDF2(;autodiff=false),ABDF2   # (2, L-stable) fixed leading coefficient multistep BDF method.
            # alg, algname = QNDF2(;autodiff=false),QNDF2
            # alg, algname = QBDF2(;autodiff=false),QBDF2
            alg, algname = QNDF(;autodiff=false),QNDF     # An adaptive order quasi-constant timestep NDF method. Similar to ode15s.
            # alg, algname = QBDF(;autodiff=false),QBDF     # An adaptive order quasi-constant timestep BDF method.
            # alg, algname = FBDF(;autodiff=false),FBDF     #  A fixed-leading coefficient adaptive-order adaptive-time BDF method,similar to ode15i or CVODE_BDF
            # alg, algname = (;autodiff=false),
         else
            # alg, algname = QNDF1(;autodiff=false),QNDF1   # (1, L-stable) numerical differentiation function (NDF) method.
            # alg, algname = QBDF1(;autodiff=false),QBDF1   # (1, L-stable) which is equivalent to implicit Euler but using the BDF error estimator.
            # alg, algname = ABDF2(;autodiff=false),ABDF2   # (2, L-stable) fixed leading coefficient multistep BDF method.
            # alg, algname = QNDF2(;autodiff=false),QNDF2
            # alg, algname = QBDF2(;autodiff=false),QBDF2
            alg, algname = QNDF(;autodiff=false),QNDF     # An adaptive order quasi-constant timestep NDF method. Similar to ode15s.
            # alg, algname = QBDF(;autodiff=false),QBDF     # An adaptive order quasi-constant timestep BDF method.
            # alg, algname = FBDF(;autodiff=false),FBDF     #  A fixed-leading coefficient adaptive-order adaptive-time BDF method,similar to ode15i or CVODE_BDF
            # alg, algname = (;autodiff=false),
            # alg, algname = (;autodiff=false),
            ## Sundials.jl
         end
      end
      # For stiff equations
   else
      # #### The Implicit-Explicit (IMEX) ODE is a SplitODEProblem with two functions: `dₜu - A * u + f(t,u)`

      # #### The Implicit-Explicit (IMEX) ODE is a SplitODEProblem with two functions: `dₜu - f₁(t,u) + f₂(t,u)`
      ## where the first function is the stiff part and the second function is the non-stiff part (implicit integration on `f₁`, explicit integration on `f₂`).
      if is_fixed_timestep
         alg, algname = IMEXEuler(;autodiff=false),IMEXEuler    # (1)
         # alg, algname = CNAB2(;autodiff=false),CNAB2            # (2), Crank-Nicolson Adams Bashforth Order 2.
         # alg, algname = CNLF(;autodiff=false),CNLF              # (2), Crank-Nicolson Leapfrog of Order 2
         # alg, algname = SBDF2(;autodiff=false),SBDF2            # (2), 2nd order IMEX BDF method.
         # alg, algname = SBDF3(;autodiff=false),SBDF3            # (2), 
         # alg, algname = SBDF4(;autodiff=false),SBDF4            # (2), 
         # alg, algname = CNLF(;autodiff=false),CNLF              # (2), 
         # alg, algname = CNLF(;autodiff=false),CNLF              # (2), 
         # 
      else
         alg, algname = SplitEuler(;autodiff=false),SplitEuler    # 1st order fully explicit method. Used for testing accuracy of splits.
         # 
      end
   end
   ## Tableau Method
   # alg, algname = ExplicitRK(tableau = constructDormandPrince())

   # ODEInterface.jl

   # LSODA.jl
   
   # TaylorIntegration.jl, an implementation of an adaptive order Taylor series method for high accuracy integration of ODEs

   # NeuralPDE.jl

   #### Pre-Built Stiffness Detecting and Auto-Switching Algorithms
   # alg, algname = AutoTsit5(Rosenbrock23()),AutoTsit5
   # alg, algname = AutoDP5(),AutoDP5
   # alg, algname = AutoVern6(),AutoVern6
   # alg, algname = AutoVern7(),AutoVern7
   # alg, algname = AutoVern8(),AutoVern8
   # alg, algname = AutoVern9(),AutoVern9



   ##### Rosenbrock-W methods, but not suitable for BigFloat due to ForwardDiff
   # alg, algname = Rosenbrock23(),Rosenbrock23
   # alg, algname = Rosenbrock32(),Rosenbrock32
   # alg, algname = ROS34PW1a(),ROS34PW1a
   # alg, algname = ROS34PW1b(),ROS34PW1b
   # alg, algname = ROS34PW2(),ROS34PW2
   # alg, algname = ROS34PW3(),ROS34PW3

   @show algname
   if out == :alg
      return alg
   elseif out == :name
      return algname
   else
      return alg, algname
   end
end
