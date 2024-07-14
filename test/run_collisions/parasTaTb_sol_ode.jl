
## Parameters: Time integration (ODE solver)
is_cb = true

is_implicit = false
order_stage = 1                          # ∈ [0,1,2] for explicit method
is_multistep = false
is_stiff = false 
save_everystep = true

if is_fixed_timestep
    adaptive_t = false
    dtmin = 1e-10
    force_dtmin = false
else
    adaptive_t = true
    dtmin = τ₀ / 1e6
    force_dtmin = true    # (= false, defaultly)
end
if is_implicit
    IMEXplicit = :implicit         # [:explicit, :implicit, :IMEX]
else
    IMEXplicit = :explicit
end

if unit_type == :PS
    tmax = t_unit * nτ * τ₀                   # The maximal time moment.
    tspan = (0.0, tmax)
    dtmax = τ₀ / Nτ_fix_TaTb                   # Number of steps during a `τ₀`
else
    tmax = t_unit * nτ * τ₀  
    tspan = (0.0, tmax)
    dtmax = τ₀ / Nτ_fix_TaTb
end
# tspanN = (0.0, nτ)
println("------------------------------")
@show unit_type,t_unit, dtmax, τ₀
alg, algname = solverODEAlg(;out=:both,IMEXplicit=IMEXplicit,is_multistep=is_multistep,
               is_fixed_timestep=is_fixed_timestep,is_stiff=is_stiff,order_stage=order_stage)
# 1
@show algname, nτ, dt_initial

