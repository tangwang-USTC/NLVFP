
"""

  Inputs:
    is_cb: = 0 means callback is off or else is on
    tspan: (t0,tmax)
    moments: []
    dfdtnuT: dn,du,dK

  outputs:
    sol = solverTaTb(TaTb,tspan,moments,ns,cbs;maxiter_t=maxiter_t,dt_initial=dt_initial,
                dtmax=dtmax, dtmin=dtmin, force_dtmin=force_dtmin,Atolt= Atolt,Rtolt= Rtolt,
                IMEXplicit=IMEXplicit,is_multistep=is_multistep,is_stiff=is_stiff,
                is_fixed_timestep=is_fixed_timestep,order_stage=order_stage,
                adaptive=adaptive,save_everystep=save_everystep,unit_type=unit_type)

    sol: [t, ...] 

"""

function solverTaTb(TaTb::AbstractVector{T},tspan::Tuple,moments::AbstractArray{T,2},ns::Int64;
    maxiter_t::Int64=5000,IMEXplicit::Symbol=:explicit, is_multistep::Bool=true, 
    is_fixed_timestep::Bool=false,is_stiff::Bool=false, order_stage::Int64=1,
    dt_initial::Real=1e0,dtmax::Real=1e2,dtmin::Real=1e-6,
    force_dtmin::Bool=false,Atolt::T=1e-6,Rtolt::T=1e-3,
    adaptive::Bool=false,save_everystep::Bool=true,unit_type::Symbol=:PS) where{T}
  
    # dT/dt
    if unit_type == :PS
        if ns == 2
            prob = ODEProblem(dtTaTb2,TaTb,tspan,moments)
        else
            if ns == 3
                prob = ODEProblem(dtTaTb3,TaTb,tspan,moments)
            elseif ns == 4
                prob = ODEProblem(dtTaTb4,TaTb,tspan,moments)
            else
                wedfgvb
            end
        end
    else
        if unit_type == :Tk
            if ns == 2
                prob = ODEProblem(dtTaTb2_Tk,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_Tk,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_Tk,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        elseif unit_type == :CGS
            if ns == 2
                prob = ODEProblem(dtTaTb2_CGS,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_CGS,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_CGS,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        elseif unit_type == :SI
            if ns == 2
                prob = ODEProblem(dtTaTb2_SI,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_SI,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_SI,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        end
    end
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

function solverTaTb(TaTb::AbstractVector{T},tspan::Tuple,moments::AbstractArray{T,2},ns::Int64,cbs::CallbackSet;
    maxiter_t::Int64=5000,IMEXplicit::Symbol=:explicit, is_multistep::Bool=true, 
    is_fixed_timestep::Bool=false,is_stiff::Bool=false, order_stage::Int64=1,
    dt_initial::Real=1e0,dtmax::Real=1e2,dtmin::Real=1e-6,
    force_dtmin::Bool=false,Atolt::T=1e-6,Rtolt::T=1e-3,
    adaptive::Bool=false,save_everystep::Bool=true,unit_type::Symbol=:PS) where{T}

    # dT/dt
    if unit_type == :PS
        if ns == 2
            prob = ODEProblem(dtTaTb2,TaTb,tspan,moments)
        else
            if ns == 3
                prob = ODEProblem(dtTaTb3,TaTb,tspan,moments)
            elseif ns == 4
                prob = ODEProblem(dtTaTb4,TaTb,tspan,moments)
            else
                wedfgvb
            end
        end
    else
        if unit_type == :Tk
            if ns == 2
                prob = ODEProblem(dtTaTb2_Tk,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_Tk,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_Tk,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        elseif unit_type == :CGS
            if ns == 2
                prob = ODEProblem(dtTaTb2_CGS,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_CGS,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_CGS,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        elseif unit_type == :SI
            if ns == 2
                prob = ODEProblem(dtTaTb2_SI,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_SI,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_SI,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        end
    end
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

function solverTaTb(TaTb::AbstractVector{T},tspan::Tuple,moments::AbstractArray{T,2},ns::Int64,cbs::DiscreteCallback;
    maxiter_t::Int64=5000,IMEXplicit::Symbol=:explicit, is_multistep::Bool=true, 
    is_fixed_timestep::Bool=false,is_stiff::Bool=false, order_stage::Int64=1,
    dt_initial::Real=1e0,dtmax::Real=1e2,dtmin::Real=1e-6,
    force_dtmin::Bool=false,Atolt::T=1e-6,Rtolt::T=1e-3,
    adaptive::Bool=false,save_everystep::Bool=true,unit_type::Symbol=:PS) where{T}

    # dT/dt
    if unit_type == :PS
        if ns == 2
            prob = ODEProblem(dtTaTb2,TaTb,tspan,moments)
        else
            if ns == 3
                prob = ODEProblem(dtTaTb3,TaTb,tspan,moments)
            elseif ns == 4
                prob = ODEProblem(dtTaTb4,TaTb,tspan,moments)
            else
                wedfgvb
            end
        end
    else
        if unit_type == :Tk
            if ns == 2
                prob = ODEProblem(dtTaTb2_Tk,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_Tk,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_Tk,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        elseif unit_type == :CGS
            if ns == 2
                prob = ODEProblem(dtTaTb2_CGS,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_CGS,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_CGS,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        elseif unit_type == :SI
            if ns == 2
                prob = ODEProblem(dtTaTb2_SI,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_SI,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_SI,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        end
    end
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

function solverTaTb(TaTb::AbstractVector{T},tspan::Tuple,moments::AbstractArray{T,2},ns::Int64,cbs::ContinuousCallback;
    maxiter_t::Int64=5000,IMEXplicit::Symbol=:explicit, is_multistep::Bool=true, 
    is_fixed_timestep::Bool=false,is_stiff::Bool=false, order_stage::Int64=1,
    dt_initial::Real=1e0,dtmax::Real=1e2,dtmin::Real=1e-6,
    force_dtmin::Bool=false,Atolt::T=1e-6,Rtolt::T=1e-3,
    adaptive::Bool=false,save_everystep::Bool=true,unit_type::Symbol=:PS) where{T}

    # dT/dt
    if unit_type == :PS
        if ns == 2
            prob = ODEProblem(dtTaTb2,TaTb,tspan,moments)
        else
            if ns == 3
                prob = ODEProblem(dtTaTb3,TaTb,tspan,moments)
            elseif ns == 4
                prob = ODEProblem(dtTaTb4,TaTb,tspan,moments)
            else
                wedfgvb
            end
        end
    else
        if unit_type == :Tk
            if ns == 2
                prob = ODEProblem(dtTaTb2_Tk,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_Tk,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_Tk,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        elseif unit_type == :CGS
            if ns == 2
                prob = ODEProblem(dtTaTb2_CGS,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_CGS,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_CGS,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        elseif unit_type == :SI
            if ns == 2
                prob = ODEProblem(dtTaTb2_SI,TaTb,tspan,moments)
            else
                if ns == 3
                    prob = ODEProblem(dtTaTb3_SI,TaTb,tspan,moments)
                elseif ns == 4
                    prob = ODEProblem(dtTaTb4_SI,TaTb,tspan,moments)
                else
                    wedfgvb
                end
            end
        end
    end
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
