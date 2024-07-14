

## Parameters: Time integration (ODE solver)
t0 = 0.0
if 11 == 1
    maxiter_t = 50000
    
    nτ = tdd_scale * 1e-1
    is_skip_solveODE = false
    is_ode_solver = false
    is_implicit = false
    
    
    tmax = t_unit * nτ              # The maximal time moment.
    
    dt_initial = t_unit / Nτ_fix    # (~ τ₀) initial time step when given and time step for fixed_timestep algorithms.
    dt_ratio = dt_initial / τ₀      # which gives the arbitrary time step to be `dtk = dt_ratio * tauk[1]`
    
    if is_fixed_timestep
        dt_initial_k = dt_initial
    else
        dt_initial_k = dt_ratio * tau_min_0    # = dt_initial * ratio_tau_min # (~ tau_min_0) = 
    end
    dt_initial_max = dt_ratio * tau_max_0      # = dt_initial * ratio_tau_max # (~ tau_max_0)
    dtmax = t_unit / Ndtmax         # () maximum time step for self-adaptive algorithms
    tspan = (0.0, tmax)
    @show tspan, (nτ, Nstep_max, Ndtmax), dt_initial, dt_initial_k
    @show (tspan[2] - tspan[1]) / dt_initial
    tstops = [tmax/2, tmax]         # `tstops` is the time which is enforced to be hit on.
    Atolt = 1e-14
    Rtolt = 1e-14
    Nt_save = 1                     # The period number of time step to save the solution
    
    # Nspan_optim_nuTi = 2.0
    
    tplot_max = 30                  # `τ₀`
    
    
end
if t0 == 2t0
    tmax = 5e-3        # which is normalized by `τ₀`
    nstepMax = 50000
    dt_intial = 1e-4   # for initial dt which is normalized by `τ₀`
    dt_fix = 1e-2      # for dt for fix step which is normalized by `τ₀`
    is_cb = true       # = true (default, callback is on )
    Nt5 = 2            # ∈ N⁺ , for show when solving during ODE
    # ts = range(t0,stop=tmax,length=nstepMax)
    contert5 = 0       # (==0), for show when solving during ODE
    Atol = 2e-4        # (=1e-3, default) Relative tolerance for ODE solver
    Rtol = Atol        # Relative tolerance for ODE solver
    maxiter = 5     # (=10000 default) maximum iteration number of ODE solver
end

tau0 = zeros(2)
tau_fM!(tau0, ma, Zq, na, vth, Coeff_tau, nai, vthi, nMod)
tau_min_0 = deepcopy(tau0[1])
tau_max_0 = deepcopy(tau0[2])
ratio_tau_min = tau_min_0 / τ₀
ratio_tau_max = tau_max_0 / τ₀

# ratio_tau_max = 20


nτ = tdd_scale * 1e-1
is_skip_solve = false
is_ode_solver = false
is_implicit = false

is_fixed_timestep = false
nτ = 0.05                             # `tmax = nτ * tau_max_0`
if is_fixed_timestep == false
    Nstep_max = 10000
    count_tau_update = 10

    Ndtmax = 2                     # The minimum number of time steps during one characteristic time `τ₀`
else
    Ndtmax = 40                    # The minimum number of time steps during one characteristic time `τ₀`    
    Nstep_max = ceil(Int64,Nτ_fix * nτ * ratio_tau_max) 
    # Nstep_max = 5                 # For the testting of the fixed time step algorithm.
end

tmax = t_unit * nτ              # The maximal time moment.

dt_initial = t_unit / Nτ_fix    # (~ τ₀) initial time step when given and time step for fixed_timestep algorithms.
dt_ratio = dt_initial / τ₀      # which gives the arbitrary time step to be `dtk = dt_ratio * tauk[1]`

if is_fixed_timestep
    dt_initial_k = dt_initial
else
    dt_initial_k = dt_ratio * tau_min_0    # = dt_initial * ratio_tau_min # (~ tau_min_0) = 
end
dt_initial_max = dt_ratio * tau_max_0      # = dt_initial * ratio_tau_max # (~ tau_max_0)
dtmax = t_unit / Ndtmax         # () maximum time step for self-adaptive algorithms
tspan = (0.0, tmax)
@show tspan, (nτ, Nstep_max, Ndtmax), dt_initial, dt_initial_k
@show (tspan[2] - tspan[1]) / dt_initial
Nt_save = 1                     # The period number of time step to save the solution
contert5 = 0       # (==0), for show when solving during ODE

# Nspan_optim_nuTi = 2.0

maxiter_t = 50000
is_multistep = false
is_stiff = false 
is_cb = false       # = true (default, callback is on)
if is_fixed_timestep == false
    adaptive_t = true
    dt_initial = 1e0
    dtmin = 1e0
    force_dtmin = true    # (= false, defaultly)
else
    adaptive_t = false
    dtmin = 0.0
    force_dtmin = false
end
# ODE algorithms
save_everystep = true
if is_implicit
    IMEXplicit = :implicit         # [:explicit, :implicit, :IMEX]
else
    IMEXplicit = :explicit
end
order_stage = 1                # ∈ [0,1,2] for explicit method
alg, algname = solverODEAlg(;out=:both,IMEXplicit=IMEXplicit,is_multistep=is_multistep,
               is_fixed_timestep=is_fixed_timestep,is_stiff=is_stiff,order_stage=order_stage)
# 1
tstops = [tmax/2, tmax]         # `tstops` is the time which is enforced to be hit on.
Atolt = 1e-14
Rtolt = 1e-14

# alg, algname = IRKGL16(), IRKGL16
# alg, algname = IRKGL16(;autodiff=false), IRKGL16

if is_ode_solver && is_cb
    ##### DiscreteCallback   affect!(integrator) = integrator.u[1] += 1
    # Ta = (2Ka - Ia² ./ ρa) / (3na)
    function condition(u,t,int)
        Iak = int.p[10]
        Tak = (2 * int.p[11] - Iak.^2 ./ (ma .* int.p[9])) / (3int.p[9])
        RDTak = abs(Tak[1] - Tak[2]) / sum(Tak)
        Iask = sum(abs.(Iak))
        if Iask ≤ epsT1000
            RDIak = abs(Iak[1] - Iak[2])
        else
            RDIak = abs(Iak[1] - Iak[2]) / Iask
        end
        max(RDTak, RDIak) < Rtol_Ms_termination
    end
    affect!(integrator) = terminate!(integrator)
    cb1 = DiscreteCallback(condition, affect!)

    
    condition2(u,t,int) = int.p[32] == true
    affect2!(integrator) = terminate!(integrator)
    # affect2!(integrator) = reinit!(integrator)
    cb2 = DiscreteCallback(condition2, affect2!)

    cbs = cb2
    cbs = CallbackSet(cb1,cb2)

    #### ContinuousCallback: a positive target function: g(u,t) which should be interpolated at arbitrary time step.
    function condition(resid,u,t,int)

        # (Mck1[2, 1, isp] / ρk1[isp])^0.5 / vthk11[isp] - 1
        resid[1] = (u[2, 1, 1] / ρa[1])^0.5 / u[njMs+1,1,1] - 1
        resid[2] = (u[2, 1, 2] / ρa[2])^0.5 / u[njMs+1,1,2] - 1
    end
    cbs = ManifoldProjection(condition)          # for period 
    is_cb ? save_everystep = false : nothing
    #### VectorContinuousCallback:
end 
