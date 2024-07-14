
datatype = Float64
domain = [-1.0, 1.0] |> Vector{datatype}     # For Chebyshev grids
is_ode_solver = false

## procedures 
pathroot = "D:/BaiduSyncdisk/2023FP/FP0D2Vd/eDTA_fM_lnARK"
pathdatas = "D:/atom/datas/2023FP/FP0D2Vd/eDTA_fM_lnARK"
# ispath(pathroot) || mkpath(pathroot)
ispath(pathdatas) || mkpath(pathdatas)

cd(pathroot)
include(joinpath(pathroot,"test/run_collisions/algorithm/modules.jl"))
include(joinpath(pathroot,"test/run_collisions/algorithm/main.jl"))

gr()            # The default plot
# plot() 
# pyplot()
# plotly()
# legend = false, :outertopright, :topleft, :bottomright

is_plot_only = false
# is_plot_only = true

is_skip_solve = false           # (=false, default), when `=true` means skip the solving process for testing case when the results have been achieved.
# is_skip_solve = true
is_plot_only == true ? is_skip_solve = true : nothing
println()

# # Time scales: τ₀ₑₑ, τ₀ₑᵢ, τ₀ᵢᵢ, 
#                τₛₑₑ, τₛₑᵢ, τₛᵢᵢ,   
#                τᵤₑₑ, τᵤₑᵢ, τᵤᵢᵢ,   
# τ₀ₑₑ ≪ τ₀ᵢᵢ ≪ τₛₑᵢ                                   # Number of time scales: Nτ₀ = ns + binomial(ns,2)
# τ₀ᵢᵢ: τ₀_DD ~ τ₀_TT < τ₀_DT < τ₀_AA < τ₀_DA ~  τ₀_TA
# τₛₑᵢ: τ₀_eD ~ τ₀_eT < τ₀_eA
 
ns0 = 3
nMod0 = ones(Int,ns0)
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

if ns0 ≥ 3
    nModcvec = [1]
end

################################# For timestep
rtol_DnIK = 0.01                # (=0.02, default) For the minimum timestep according to `abs(dt * Rc / Mc) ≤ rtol_DnIK`
                                #                                                   and `abs(vthik1 / vthik - 1) ≤ rtol_DnIK`
ratio_dtk1 = 1.1                # (1.2, default), The increasing rate of timestep which means `ratio_dtk1 = dtk1 / dtk`
Nspan_nuTi_maxmin = [1.01, 1.5] # (=[1.05, 1.2], default). The minimum and maximum relative span of the submoments 
                                # which will limit the timestep of the optimization process by errors `yfit`.
# For ODE solver
unit_type = :PS                  # [:PS, :Tk, :SI, :CGS]
Nτ_fix_TaTb = 100               # (=20, default) Number of timestep during one characteristic time `τ₀`
Atolt = 1e-10                   # (=1e-9, default) Abstol tolerance of `ODE` solver                  
Rtolt = 1e-10                   # (=1e-6, default) Relative tolerance of `ODE` solver

nτ = 2e1                      # `tmax = nτ * tau_max_0`
maxiter_t = 1000

## Hyperparameters for models
is_lnA_const = true
# is_lnA_const = false          # (=true, default) The convergence may be worse when distribution function is far from Maxwellian.

# Criterions for the fen cha dian or the conditions to terminate
Nstep_max = 2000                # (=10000, default) The maximum timestep
rtol_dtsa_terminate = 1e-8      # (=1e-8, default) The entropy criterion condition to terminating the program
rtol_Ti = 1e-4                  # (=1e-4, default), for `cb` in the ODE solver
rtol_TiTaTb = rtol_Ti / 10

atol_uai = 1e-8                 # (=1e-8, default), The criterion condition for `uai`, or else `uai = 0`
rtol_DnuTi = 2e-8               # (=1e-3, default) for `:Ms` version. The criterion condition for characteristic values to reduce the number of `nMod`
                                # vthi[1] / vthi[2] - 1 or `uai[1] / (uai[2] + epsT0) - 1`
rtol_dtsa = 2e-8            # (=1e-10, default) for `:IK` version. The entropy criterion condition to reduce the number of `nMod`
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
atol_vthi = 1e-8                # I. Before reaches `i_iter_rs2`, the criterion to break the iteration of `vth` optimization process.
                                # II. 
rtol_vthi = 1e-3
i_iter_rs2 = 3                  # ∈ [0; N⁺], If `i_iter_rs2 ≤ 0`, the implicit method will degenerate into be the explicit Euler method when `rs = 2`
                                #            If `i_iter_rs2 == 1`, the Heun's second-order method will be used when `rs = 2`
                                ## ------------------------------------------------------------------------------------------------------------------------
                                # e,   e,    H,   D,   T,    α
me0,      mp0,           mα0  = [1,   1],  [1,   2,   3],   4

ne1, ne2, np0, nD0, nT0, nα0  = 1.2, 0.0,  0.0, 1.0,  0.0, 0.1      # density of spice `e, H, D, H3`

Te1, Te2, Tp0, TD0, TT0, Tα0  = 1.0, 10.0, 1.0, 1.0, 10.0, 1000.0
Eke1,Eke2,Ekp0,EkD0,EkT0,Ekα0 = 0.0, 0.0, 0.0,  0.0, 0.0,  0.0
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
    EkD2,Ekα0 = Ekns2[NuCase,1], Ekns2[NuCase,2]
end

for knMa in 1:1:length(nModavec)
    nMod0[1] = nModavec[knMa]
    for knMb in 1:1:length(nModbvec) 
        nMod0[2] = nModbvec[knMb]
        if ns0 ≥ 3
            for knMc in 1:1:length(nModcvec)
                nMod0[3] = nModcvec[knMc]
                if ns0 ≥ 4
                    for knMd in 1:1:length(nModdvec)
                        nMod0[4] = nModdvec[knMd]
                        include(joinpath(pathroot,"test/run_collisions/programsTaTb.jl"))
                    end
                else
                    include(joinpath(pathroot,"test/run_collisions/programsTaTb.jl"))
                end
            end
        else
            include(joinpath(pathroot,"test/run_collisions/programsTaTb.jl"))
        end
    end
end