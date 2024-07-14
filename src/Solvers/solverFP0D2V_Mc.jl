
"""

  Inputs:
    is_cb: = 0 means callback is off or else is on
    tspan: (t0,tmax)
    moments: []
    dfdtnuT: dn,du,dK

  outputs:
    sol = solverFP0D2V_Mc(Mck1,tspan,psvec,cbs;dt_initial=dt_initial,
                dtmax=dtmax, dtmin=dtmin, force_dtmin=force_dtmin,Atolt= Atolt,Rtolt= Rtolt,
                IMEXplicit=IMEXplicit,is_multistep=is_multistep,is_stiff=is_stiff,
                is_fixed_timestep=is_fixed_timestep,order_stage=order_stage,
                adaptive=adaptive,save_everystep=save_everystep)

    sol: [t, ...] 

"""

function solverFP0D2V_Mc(Mck1::AbstractArray{T,N},tspan::Tuple,ps::AbstractVector{Any};
    maxiter_t::Int64=100000,IMEXplicit::Symbol=:explicit, is_multistep::Bool=true, 
    is_fixed_timestep::Bool=false,is_stiff::Bool=false, order_stage::Int64=1,
    dt_initial::Real=1e0,dtmax::Real=1e2,dtmin::Real=1e-6,
    force_dtmin::Bool=false,Atolt::T=1e-6,Rtolt::T=1e-3,
    adaptive::Bool=false,save_everystep::Bool=true) where{T,N}
  
    prob = ODEProblem(Mck1integral_ode!!,Mck1,tspan,ps)   # dnuT/dt
    alg = solverODEAlg(;IMEXplicit=IMEXplicit,is_multistep=is_multistep,is_stiff=is_stiff,
                        is_fixed_timestep=is_fixed_timestep,order_stage=order_stage)
    # sol = solve(prob,alg,tstops=tstops,force_dtmin=true,progress=true)
    if dtmax > 0.0
        sol = solve(prob,alg,dt= dt_initial,dtmax=dtmax, dtmin=dtmin, 
                   force_dtmin=force_dtmin, maxiters=maxiter_t,
                   abstol= Atolt,reltol= Rtolt,adaptive=adaptive,save_everystep=save_everystep)
    else
        sol = solve(prob,alg,dt= dt_initial,dtmax = dtmax, maxiters=maxiter_t, 
                   abstol= Atolt,reltol= Rtolt,adaptive=adaptive,save_everystep=save_everystep)
    end
    return sol
end

function solverFP0D2V_Mc(Mck1::AbstractArray{T,N},tspan::Tuple,ps::AbstractVector{Any},cbs::CallbackSet;
    maxiter_t::Int64=100000,IMEXplicit::Symbol=:explicit, is_multistep::Bool=true, 
    is_fixed_timestep::Bool=false,is_stiff::Bool=false, order_stage::Int64=1,
    dt_initial::Real=1e0,dtmax::Real=1e2,dtmin::Real=1e-6,
    force_dtmin::Bool=false,Atolt::T=1e-6,Rtolt::T=1e-3,
    adaptive::Bool=false,save_everystep::Bool=true) where{T,N}

    prob = ODEProblem(Mck1integral_ode!!,Mck1,tspan,ps)   # dnuT/dt
    alg = solverODEAlg(;IMEXplicit=IMEXplicit,is_multistep=is_multistep,is_stiff=is_stiff,
                        is_fixed_timestep=is_fixed_timestep,order_stage=order_stage)
    # sol = solve(prob,alg,tstops=tstops,force_dtmin=true,progress=true)
    if dtmax > 0.0
        sol = solve(prob,alg,dt= dt_initial,dtmax=dtmax, dtmin=dtmin, 
                   force_dtmin=force_dtmin, maxiters=maxiter_t, callback=cbs,
                   abstol= Atolt,reltol= Rtolt,adaptive=adaptive,save_everystep=save_everystep)
    else
        sol = solve(prob,alg,dt= dt_initial,dtmax = dtmax, maxiters=maxiter_t, callback=cbs,
                   abstol= Atolt,reltol= Rtolt,adaptive=adaptive,save_everystep=save_everystep)
    end
    return sol
end

function solverFP0D2V_Mc(Mck1::AbstractArray{T,N},tspan::Tuple,ps::AbstractVector{Any},cbs::DiscreteCallback;
    maxiter_t::Int64=100000,IMEXplicit::Symbol=:explicit, is_multistep::Bool=true, 
    is_fixed_timestep::Bool=false,is_stiff::Bool=false, order_stage::Int64=1,
    dt_initial::Real=1e0,dtmax::Real=1e2,dtmin::Real=1e-6,
    force_dtmin::Bool=false,Atolt::T=1e-6,Rtolt::T=1e-3,
    adaptive::Bool=false,save_everystep::Bool=true) where{T,N}

    prob = ODEProblem(Mck1integral_ode!!,Mck1,tspan,ps)   # dnuT/dt
    alg = solverODEAlg(;IMEXplicit=IMEXplicit,is_multistep=is_multistep,is_stiff=is_stiff,
                        is_fixed_timestep=is_fixed_timestep,order_stage=order_stage)
    # sol = solve(prob,alg,tstops=tstops,force_dtmin=true,progress=true)
    if dtmax > 0.0
        sol = solve(prob,alg,dt= dt_initial,dtmax=dtmax, dtmin=dtmin, 
                   force_dtmin=force_dtmin, maxiters=maxiter_t, callback=cbs,
                   abstol= Atolt,reltol= Rtolt,adaptive=adaptive,save_everystep=save_everystep)
    else
        sol = solve(prob,alg,dt= dt_initial,dtmax = dtmax, maxiters=maxiter_t, callback=cbs,
                   abstol= Atolt,reltol= Rtolt,adaptive=adaptive,save_everystep=save_everystep)
    end
    return sol
end

function solverFP0D2V_Mc(Mck1::AbstractArray{T,N},tspan::Tuple,ps::AbstractVector{Any},cbs::ContinuousCallback;
    maxiter_t::Int64=100000,IMEXplicit::Symbol=:explicit, is_multistep::Bool=true, 
    is_fixed_timestep::Bool=false,is_stiff::Bool=false, order_stage::Int64=1,
    dt_initial::Real=1e0,dtmax::Real=1e2,dtmin::Real=1e-6,
    force_dtmin::Bool=false,Atolt::T=1e-6,Rtolt::T=1e-3,
    adaptive::Bool=false,save_everystep::Bool=true) where{T,N}

    prob = ODEProblem(Mck1integral_ode!!,Mck1,tspan,ps)   # dnuT/dt
    alg = solverODEAlg(;IMEXplicit=IMEXplicit,is_multistep=is_multistep,is_stiff=is_stiff,
                        is_fixed_timestep=is_fixed_timestep,order_stage=order_stage)
    # sol = solve(prob,alg,tstops=tstops,force_dtmin=true,progress=true)
    if dtmax > 0.0
        sol = solve(prob,alg,dt= dt_initial,dtmax=dtmax, dtmin=dtmin, 
                   force_dtmin=force_dtmin, maxiters=maxiter_t, callback=cbs,
                   abstol= Atolt,reltol= Rtolt,adaptive=adaptive,save_everystep=save_everystep)
    else
        sol = solve(prob,alg,dt= dt_initial,dtmax = dtmax, maxiters=maxiter_t, callback=cbs,
                   abstol= Atolt,reltol= Rtolt,adaptive=adaptive,save_everystep=save_everystep)
    end
    return sol
end
