
datatype = Float64
domain = [-1.0, 1.0] |> Vector{datatype}     # For Chebyshev grids

## procedures 
pathroot = "D:/BaiduSyncdisk/2023FP/FP0D2Vd/eDTA_fM_lnARKNK"
pathdatas = "D:/atom/datas/2023FP/FP0D2Vd/eDTA_fM_lnARKNK"
# ispath(pathroot) || mkpath(pathroot)
ispath(pathdatas) || mkpath(pathdatas)

cd(pathroot)
include(joinpath(pathroot,"test/run_collisions/algorithm/modules.jl"))
include(joinpath(pathroot,"test/run_collisions/algorithm/main.jl"))

path_paper = "test/run_collisions/paperfig/ee/fM_nMod1"
path_paper_fM = "test/run_collisions/paperfig/plotting/fM"

gr()            # The default plot
# plot() 
# pyplot()
# pythonplot()

# plotly() 
# legend = false, :outertopright, :topleft, :bottomright
is_sol_Brag = true
is_plot_DflKing = true
is_sol_fvLc = true
is_Case_C01 = true

is_MultiCase = true
# is_MultiCase = false

is_plot_only = false
# is_plot_only = true

# is_skip_solve = true
is_skip_solve = false           # (=false, default), when `=true` means skip the solving process for testing case when the results have been achieved.

is_plot_only == true ? is_skip_solve = true : nothing
println()

# # Time scales: τ₀ₑₑ, τ₀ₑᵢ, τ₀ᵢᵢ, 
#                τₛₑₑ, τₛₑᵢ, τₛᵢᵢ,   
#                τᵤₑₑ, τᵤₑᵢ, τᵤᵢᵢ,   
# τ₀ₑₑ ≪ τ₀ᵢᵢ ≪ τₛₑᵢ                                   # Number of time scales: Nτ₀ = ns + binomial(ns,2)
# τ₀ᵢᵢ: τ₀_DD ~ τ₀_TT < τ₀_DT < τ₀_AA < τ₀_DA ~  τ₀_TA
# τₛₑᵢ: τ₀_eD ~ τ₀_eT < τ₀_eA
 
nMod0 = ones(Int,2)
nModavec = [3]
nModbvec = [3]

nModavec = [2,3]
nModbvec = [1,2,3]

nModavec = [1,2,3] 
nModbvec = [1,2,3]

# nModavec = [1,2]
# nModbvec = [1,2]

nModavec = [1] 
nModbvec = [1]

# iterEmbeded = 0                 # ∈ [0; N⁺], If `iterEmbeded ≤ 0`, the implicit method will degenerate into be the explicit Euler method when `rs = 2`
#                                 #            If `iterEmbeded == 1`, the Heun's second-order method will be used when `rs = 2`
#                                 ## ------------------------------------------------------------------------------------------------------------------------
orderRK, rsRK, iterRK = 2, 2, 0                 # [:Euler(1,:,N), :Trapz(2,:,N⁺), :RadauI3(3,2,N), :GL4(4,2,N), :LobattoIIIA4(4,3,N), :LobattoIIIA6(6,4,N), ⋯]
orderEmbeded, rsEmbeded, iterEmbeded = 2, 2, 10  # [:Euler(1,:,N), :Trapz(2,:,N⁺), :Kutta3(3,3,N⁺), :Heun3(3,4,N⁺), :Runge416(4,3,N⁺), :Runge438(4,4,N⁺), :Runge42(4,4(5),N⁺), :RK5(5,5,N⁺), :Lawson5(5,5(13),N⁺), :Runge5(5,6,N⁺)]
is_explicit = false
if orderRK == 4 && rsRK == 3
    orderEmbeded, rsEmbeded = 3, 3
end
i_iter_rs2 = iterRK
is_fixed_timestep = true
if is_sol_fvLc
    is_fixed_timestep == true
end

is_Mhck_reconstruct = false
is_name_ab = true
if is_MultiCase
    MultiType = :dt                    # [:nnv, :dt]

    if MultiType == :nnv
        n21 = 0                        # ∈ Z
        Nτ_fix = 2^n21 * 2 .^4         # (=20, default) Number of timestep during one characteristic time `τ₀`
        nnvocpM = [6, 7, 8]
        NCase = length(nnvocpM)    # (nnv0, ocp0) = [(6,6), (7,7), (8,8)]

        iCase = 3
        nnv0 = nnvocpM[iCase]
        ocp0 = 1nnv0
        if iCase == 1
            nvG =  [65, 65]
        else
            if iCase == 2
                nvG =  [129, 129]
            else
                if iCase == 3
                    nvG =  [257, 257]
                else
                    if iCase == 4
                        nvG =  [513, 513]
                    else
                        ggfgg
                    end
                end
            end
        end

        vec_sweep = nnvocpM
        is_plot_DflKing = false
    elseif MultiType == :dt
        nnvocpM = [(6,6), (7,7), (8,8), (9,9)]
        iCase77 = 2
        if iCase77 == 1
            nvG =  [65, 65]
        else
            if iCase77 == 2
                nvG =  [129, 129]
            else
                if iCase77 == 3
                    nvG =  [257, 257]
                else
                    if iCase77 == 4
                        nvG =  [513, 513]
                    else
                        ggfgg
                    end
                end
            end
        end
        iCase77 = 2
        nnv0, ocp0 = nnvocpM[iCase77]
    
        n21 = 0                       # ∈ Z
        Nτ_fixvec = 2^n21 * 2 .^(1:6)
        NCase = length(Nτ_fixvec)    # (nnv0, ocp0) = [(6,6), (7,7), (8,8)]

        iCase = 6
        Nτ_fix = Nτ_fixvec[iCase]
        if NCase == 1
            is_MultiCase == false
        else
            is_MultiCase == true
        end
        vec_sweep = Nτ_fixvec
    else
        edfgbn 
    end
    Nvec_sweep = length(vec_sweep) 
    is_Case_C01 = true
    NCasep = NCase             # For plotting: The maximum number of plotting case.

    if iCase == NCase
        is_output_convergence = true
        pyplot()
    else
        is_output_convergence = false
    end
    # is_output_convergence = true
    is_display_King = false
    is_display_RDfamax = false
    is_display_dtCFL = false
    
    Cnm_name = :C0        # `:C0: RDKingvec = (Kingk1 - flnk1) / (flnk1 - flnk)`
    # Cnm_name = :C2        # `:C2: RDKingvec = (Kingk1 - flnk1) / (Δt * fvLc0k1[isp])`
    is_weight_DKing = false
    if Cnm_name == :C0
        Lmode = :L0              # `Lmode = :L0` means `vhe[isp][ivh]`
        is_plot_Convergence = true
    elseif Cnm_name == :C2
        Lmode = :L1              # `Lmode = :L1` means `∑ᵢₕᵥᴺᵛ (y[ivh])`
                                 # `Lmode = :L2` means `∑ᵢₕᵥᴺᵛ (√(y[ivh]^2))`
        is_plot_Convergence = false
    end
    ivh = 2                      #
    vh5 = 0.5                    # vec05 = vhek1[isp] .< vh5

    iCase ≤ NCase || (iCase = NCase)
    is_plot_King_convergence = true    # (=true, default)
    is_L2dtfl0 = true                  # (=true, default)
    is_L2dtKing = true                 # (=true, default)
    is_L2dtDKing = false
    is_CPUtime = false
    is_CPUnnvtime = false
    is_save_nvG_NCase = false
    if iCase == 1
        RDKing33vec = zeros(NCase,2+2)
        RDKing00vec = zeros(NCase,2+4)    # [a, b, tk, k, dtkCFL, dtk]
        (Lmode == :L0) || (is_save_nvG_NCase = true)
        if is_save_nvG_NCase
            RDKingvec = zeros(nvG[1]+1,Nvec_sweep,2)
        else
            RDKingvec = zeros(1+1,Nvec_sweep,2)
        end
        if is_L2dtfl0
            fl0L2dtv = zeros(nvG[1],Nvec_sweep)             # For the exact species
        end
        if is_L2dtKing
            KingL2dtv = zeros(nvG[1],Nvec_sweep)            # For the exact species
        end
        if is_L2dtDKing
            DKingL2dtv = zeros(nvG[1],Nvec_sweep+1)         # For the exact species
        end
        if is_CPUtime
            CPUvec = zeros(Nvec_sweep+1)                    # [s]
        end
        if is_CPUnnvtime
            CPUnnvvec = zeros(Nvec_sweep)                            # [s]
        end
    end
    RDKing00vec[iCase,6] = 1 / Nτ_fix
    if is_CPUtime 
        if iCase == NCase
      # iCase = 1
            CPUvec[iCase] = 5.02
    
            CPUvec1order = CPUvec[NCase] .* (2^1) .^(0:NCase-1)
            CPUvec1order *= CPUvec[NCase] / CPUvec1order[NCase]
            CPUvec05order = CPUvec[NCase] .* (2^0.5) .^(0:NCase-1)
            CPUvec05order *= CPUvec[NCase] / CPUvec05order[NCase]
    
            ylabel = L"CPU [s]"
            xlabel = L"\Delta t"
            label = L"Numerical"
            pCPUt = plot(1 ./ Nτ_fixvec, CPUvec[1:NCase],label=label,line=(2,:auto),
                        ylabel=ylabel,yscale=:log10,
                        xlabel=xlabel,xscale=:log10)
            label = L"1^{st} order"
            pCPUt = plot!(1 ./ Nτ_fixvec, CPUvec1order[1:NCase],
                        label=label,line=(2,:auto),legend=legendbL)
            label = L"(1/2)^{th} order"
            pCPUt = plot!(1 ./ Nτ_fixvec, CPUvec05order[1:NCase],label=label,line=(2,:auto))
            savefig(string(file_fig_file,"_CPUdt",NCase,".png"))
            display(plot(pCPUt))
        end
    end
    if is_CPUnnvtime && iCase == NCase

        CPUnnvvec[iCase] = 11.35

        CPUnnvvec1order = CPUnnvvec[Nvec_sweep] .* (2^1) .^(0:Nvec_sweep-1)
        CPUnnvvec1order *= CPUnnvvec[Nvec_sweep] / CPUnnvvec1order[Nvec_sweep]

        ylabel = L"CPU [s]"
        xlabel = L"\Delta t"
        label = L"Numerical"
        pCPUnnvt = plot(1 ./ Nτ_fixvec, CPUnnvvec,label=label,line=(2,:auto),
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel,xscale=:log10)
        label = L"1^{st} order"
        pCPUnnvt = plot!(1 ./ Nτ_fixvec, CPUnnvvec1order,label=label,line=(2,:auto))
    end
else
    NCase = 1
    Nτ_fix = 20                 # (=20, default) Number of timestep during one characteristic time `τ₀`
    nnv0 = 7                    # [5,  6,  7,   8,   9,   10]
    ocp0 = 7                    # (default: = 8), the order of Chebyshev polynomials on the subdivision for Gaussian-type quadrature, i.e. Clenshaw-Curtis quadrature.
end
is_dt_unit = true               # true ? dt = 1 / Nτ_fix : (tspan[2] - tspan[1]) / Nτ_fix
################################# For timestep
is_enforce_errdtnIKab = true    # (=true, default) espacilly when `vath ≫ vbth` owing to the dis disperity of `ma` or `Ta`
is_enforce_errdtnIKab = false
if Nτ_fix ≤ 4
    is_sol_Brag = false
end
rtol_DnIK = 0.01                # (=0.02, default) For the minimum timestep according to `abs(dt * Rc / Mc) ≤ rtol_DnIK`
                                #                                                   and `abs(vthik1 / vthik - 1) ≤ rtol_DnIK`
ratio_dtk1 = 1.1                # (1.2, default), The increasing rate of timestep which means `ratio_dtk1 = dtk1 / dtk`
Nspan_nuTi_maxmin = [1.01, 1.5] # (=[1.05, 1.2], default). The minimum and maximum relative span of the submoments 
                                # which will limit the timestep of the optimization process by errors `yfit`.

nτ = 2e1                       # `tmax = nτ * tau_max_0`
Nt_save = 1                     # The period number of time step to save the solution
is_dt_tau = true                # (=false, default), Whether let the timestep can be decided by `tau`

dt_initial_min = 1e-0
# For ODE solver
unit_type = :PS                  # [:PS, :Tk, :SI, :CGS]
Nτ_fix_TaTb = Nτ_fix               # (=20, default) Number of timestep during one characteristic time `τ₀`
Atolt = 1e-16                   # (=1e-9, default) Abstol tolerance of `ODE` solver                  
Rtolt = 1e-16                   # (=1e-6, default) Relative tolerance of `ODE` solver

maxiter_t = 1000

## Hyperparameters for models
is_lnA_const = true
# is_lnA_const = false          # (=true, default) The convergence may be worse when distribution function is far from Maxwellian.
is_fvL_CP = false               # `is_fvL_CP` or else `is_fvL_Ms`
gridv_type = :uniform           # [:uniform, :uAdaptive, :cAdaptive , chebyshev, shiftedchebyshev]
# gridv_type = :chebyshev
is_nvG_adapt = false            # (=false, default), Whether the hyperparameter `nvG` is adaptive.
nvG_limit = 9                   # (=9, default), ∈ N⁺, The limit of the maximum value of `nvG` when `is_nvG_adapt = true`.
vadaptlevels = 3                # (=3, default), ∈ N⁺, nc0, nck: (default: = 1, which is ∈ [0, N⁺]), the maximum level of shape function of velocity axis adaptive meshing.
                                # which gives the `vAdapt = 2^vadaptlevels + 1` relative to `nc0`.
                                # = 0, nck = nc0 = nvG, 
                                # = 1, nck ≥ nc0 = nvG,
                                # ≥ 2, nck ≥ nc0 ≥ nvG

# Criterions for the fen cha dian or the conditions to terminate
Nstep_max = 5000                # (=10000, default) The maximum timestep
rtol_dtsa_terminate = 1e-10     # (=1e-8, default) The entropy criterion condition to terminating the program
rtol_Ti = 1e-3                  # (=1e-4, default), for `cb` in the ODE solver
rtol_TiTaTb = rtol_Ti / 10

atol_uai = 1e-8                 # (=1e-8, default), The criterion condition for `uai`, or else `uai = 0`
rtol_DnuTi = 2e-8               # (=1e-3, default) for `:Ms` version. The criterion condition for characteristic values to reduce the number of `nMod`
       
if is_fvL_CP
    functionName = :IK
    rtol_dtsa = 2e-8            # (=1e-10, default) for `:IK` version. The entropy criterion condition to reduce the number of `nMod`
else
    functionName = :Ms
    is_dtk_order_Rcaa = false   # (=false, default), whether the timestep is affected by the kinetic dissipations during self-collision processes.
    dtk_order_Rc = :min         # (=:mid), which `∈ [:min, :mid, :max]`. The order of `Rⱼₗᵃᵇ` to decide the timestep during Coulomb collision processes.
                                #         `:min` denotes only the first three conservative moments `nIK` are taken into accound;
                                #                The special strategy in the procedures is `order_Rc = j = 2` in 0D1V model.
                                #         `:mid` is a strategy that balances performance with efficiency by choose the middle order when `nMod ≥ 1`;
                                #                The special strategy in the procedures is `order_Rc = j = nMod + 1` in 0D1V model.
                                #         `:max` denotes the order of kinetic dissipations is decided by parameter `nMod` adaptively;
                                #                The special strategy in the procedures is `order_Rc = j = 2nMod` in 0D1V model.
    1
end
rtol_DnuTi_warn = 1e-3
rtol_nIK_warn = 1e-10
rtol_nIK_error = 1e-3
atol_nIK = epsT1000             # 
atol_IKTh = epsT1000            # 
rtol_IKTh = 1e-6                # 
# rtol_IKTh_err = 0.01
ratio_Rdtfln = 0e1              # (=1e-1, default), The change ratio limit of harmonic of distribution function `Rdtfln = ∂ₜfLn / fLn` to limit the timestep.
                                # `ratio_Rdtfln = 0.0`means change ratio limit will not be used to update the timestep.
rtol_dtnIKs = 1e3               # (=1e-3, default), The criterion condition for conservations by `∑∂ₜIᵢ ~ 0` and `∑∂ₜKᵢ ~ 0`.
atol_nuTi_optim = 1e-10         # (=1e-10, default), The criterion condition for optimization of `nai,uai,vthi`, or else optimization is falure.
atol_u = 1e-9                   # (=1e-9, default), The criterion condition for `ua`, or else `ua = 0`
rtol_u = 1e-6                   # (=1e-6, default)
const atol_n = epsT1000         # (=epsT10, default), The criterion condition for `na`, or else `na = 0`
const rtol_n = 1e-10            # (=1e-10, default), The criterion condition for `na`, or else `na = 0`
atol_vthi = 1e-10                # I. Before reaches `iterEmbeded`, the criterion to break the iteration of `vth` optimization process.
                                # II. 
rtol_vthi = 1e-10
                                # e,   e,   H,   D,   T,    α
me0,      mp0,           mα0  = [1,   1],  [1,   2,   3],   4

ne1, ne2, np0, nD0, nT0, nα0  = 1.0, 1.0,  0.0, 0.0, 0.0,  0.0      # density of spice `e, H, D, H3`

Te1, Te2, Tp0, TD0, TT0, Tα0  = 10.0, 20.0, 1.0, 10.0, 20.0, 1000.0
Eke1,Eke2,Ekp0,EkD0,EkT0,Ekα0 = 0.0,  0.0, 0.0, 0.0, 0.0,  0.0
                                # `ua ~ ub, ∀T` are ok!
                                # `Ta ~ Tb, ∀u` are ok!
                                # `Ta ≫ Tb, ua ≫ ub` are ok!
                                # `Ta ≫ Tb, ua ≪ ub` are challenge now, which need a smart integral for `Rhcj` !
NuCase = 0           # 1,2,3, 5 for (H,D or e-e) collision
if NuCase ≠ 0
    Ekns2 = [1e-2 1e-2; 1e-2 -1e-2;
            1e-2 1e-10; 1e-10 1e-2;
            1e-2 -1e-1; -1e-1 1e-2
            ]
    Eke1, Eke2 = Ekns2[NuCase,1], Ekns2[NuCase,2]
end

# # # # The main error sources of the algorithm
is_IJ_bc = true                      # (=true, default) It will be important when the disperity of mass, temperature or the average velocity are so big.
is_extrapolate_FLn = true            # (=false, default), `=false` when `vath ≫ vbth`,; But `nnv` almost couldn't to make up the deficiency of it's vacancy.
is_extrapolate_FLn_initial = true    # (=true, default)
is_norm_error = true                 # (=true, default), Whether normalized the error of kinetic moments `Mcj` by factor `ρₐ * vₜₕʲ` to decide the stage of `enforce_errdtnIK`.
eps_FLn_limit = epsT1000             # The limit value of `FvL(𝓋̂)`
# err_dtnIK_enforce = 1e-10          # The error limit of `δₜn, δₜI, δₜK` when `vbth ≪ vath` to enforce the convergence and conservations (odd posterior conservation version).

is_warn_FLnb9 = false                # Whether to give a warning when `FLn[end]/ 10 > epsT`
atol_vjdtfLn9 = 1e-14                # (=1e-15, default), The criterion for warning the integral of dissipations when `v[end]^(j+2) * dtfLn[end] > atol_vjdtfLn9`
gridv_type_initial = :uniform
if gridv_type_initial == :chebyshev
    is_extrapolate_FLn_initial = false
end

is_vth_ode = false                   # `vth = (Ka / ρa - 2/3 * ua^2)^0.5 ` when `false`
is_vth_ode == false || @warn("Parameter `is_vth_ode = true` mean low-order algerithm for `vthk1` which will break the inner-energy restraint.")
if gridv_type == :uniform
elseif gridv_type == :chebyshev
    is_extrapolate_FLn = false
    is_extrapolate_FLn_initial = false
    # is_enforce_errdtnIKab = true
end

if is_MultiCase && iCase == 1
    include(joinpath(pathroot,path_paper_fM,"MultiCases.jl"))
end

for knMa in 1:1:length(nModavec)
    nMod0[1] = nModavec[knMa]
    for knMb in 1:1:length(nModbvec) 
        nMod0[2] = nModbvec[knMb]
        include(joinpath(pathroot,"test/run_collisions/programs.jl"))
        if is_sol_Brag
            include(joinpath(pathroot,"test/run_collisions/programsTaTb.jl"))
        end
        include(joinpath(pathroot,path_paper_fM,"plotting.jl"))
    end
end
@show is_fvL_CP
@show spices0, T0, Ek0
if is_MultiCase
    @show orderRK, rsRK, iterRK,NCase, iCase, nvG
    @show Cnm_name, Lmode, is_weight_DKing, Nτ_fix, 1/Nτ_fix
else
    @show orderRK, rsRK, iterRK
    @show Cnm_name, Lmode, is_weight_DKing, Nτ_fix, 1/Nτ_fix
end
