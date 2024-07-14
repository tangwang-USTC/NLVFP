
# # Updating the parameters `ps` when `t = 0.0` 
# IMEXplicit

# unit_type = :PS                  # [:PS, :Tk, :SI, :CGS]

#### datas 
include(joinpath(pathroot,"test/run_collisions/datas/writting/data_initial_TaTb.jl")) 

## Files
include(joinpath(pathroot,"test/run_collisions/datas/files/files_namesTaTb.jl"))

idnIK = open(file_TaTb,"a+")     # truncate/appand, read, write, create
# main procedure 
if is_skip_solve == false
    CSV.write(idnIK, datas_TaTb_csv, newline='\n')                     # Over-writing existing datas in the file `idnIk`

    if is_cb
        condition(u,t,int) = sum(abs.(diff(u))) / sum(u) ≤ rtol_TiTaTb
        # function condition(u,t,int)

        #     @show t, sum(abs.(diff(u))) / sum(u)
        #     return sum(abs.(diff(u))) / sum(u) ≤ rtol_TiTaTb
        # end
        affect!(integrator) = terminate!(integrator)
        cb = DiscreteCallback(condition, affect!)
    end

    moments = zeros(4,ns)
    moments[1,:] = deepcopy(ma)
    moments[2,:] = deepcopy(Zq)
    if unit_type == :PS
        TaTb = Float64.(Ta)
        moments[3,:] = deepcopy(na)
    else
        if unit_type == :Tk
            TaTb = Float64.(T0)
            moments[3,:] = deepcopy(n0)
        else
            TaTb = Float64.(1000T0)
            if unit_type == :CGS
                moments[3,:] = deepcopy(naSI0/1e6)
            elseif unit_type == :SI
                moments[3,:] = deepcopy(naSI0)
            end
        end
    end
    is_fixed_timestepTaTb = is_fixed_timestep
    dt_initial_kTaTb = dt_initial * τ₀ * 50 / Nτ_fix
    # moments[4,:] = zeros(ns)         # dT 
    @timev solab  = solverTaTb(TaTb,tspan,moments,ns,cb;maxiter_t=maxiter_t,dt_initial=dt_initial_kTaTb,
                dtmax=dtmax, dtmin=dtmin, force_dtmin=force_dtmin,Atolt=Atolt,Rtolt=Rtolt,
                IMEXplicit=IMEXplicit,is_multistep=is_multistep,is_stiff=is_stiff,
                is_fixed_timestep=is_fixed_timestepTaTb,order_stage=order_stage,
                adaptive=adaptive_t,save_everystep=save_everystep,unit_type=unit_type)
    CSV.write(idnIK, solab, newline='\n') 
end
 
close(idnIK)
# Plotting
include(joinpath(pathroot,"test/run_collisions/datas/reading/uT_TaTb.jl"))
@show tspan,  algname
