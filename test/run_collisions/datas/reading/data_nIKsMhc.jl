
tplot_min = -1e-2                 # `τ₀`
tplot_max = 2e3                  # `τ₀`
missing_deal = :NaN        # [:nothing, :drop, :NaN, :zero]
dkivv2 = 1 * dkivv

xlabel = string(algname,", t[τ₀]")
xlabel_gridv_type = string(gridv_name)

wline = 2

# # [nh, uh, Th],              _nuTi 
include(joinpath(pathroot,"test/run_collisions/datas/reading/nuTi.jl"))

if 1 == 1
    # [u, T, I, K],              _uTIK
    if is_fvL_CP
        include(joinpath(pathroot,"test/run_collisions/datas/reading/uTIK_IK.jl"))
    else
        include(joinpath(pathroot,"test/run_collisions/datas/reading/uTIK_Ms.jl"))
        
        if is_moments_out
            is_plot_Mhc_l_1 = false
            # [Mhc, dtMhc],          _Mhcl
            include(joinpath(pathroot,"test/run_collisions/datas/reading/Mhc_Ms.jl"))
            include(joinpath(pathroot,"test/run_collisions/datas/reading/errMhcop_Ms.jl"))
            # if prod_nMod ≥ 2
            # end
        
            # # [Mhcs, dtMhcs],        _Mhcs
            if ns == 2
                if norm(ua) ≤ epsT1000
                    # include(string(pathroot,"/test/run_collisions/datas/reading/Mhcs_Ms_fM.jl"))
                    # # if prod_nMod ≥ 2
                    # # end
                else
                    include(string(pathroot,"/test/run_collisions/datas/reading/Mhcs_Ms_fDM.jl"))
                end
            else
            end
        end
    end
    
    # [edtnIKTs],                 _Cerror
    if is_Cerror_dtnIKTs
        include(joinpath(pathroot,"test/run_collisions/datas/reading/edtnIKTs.jl"))
    end
    
    # [Dt, Duab, DTab, us, Ts],   _uTs                  # `τ₀`
    include(joinpath(pathroot,"test/run_collisions/datas/reading/dt_uTs.jl"))
    
    # [sa, dtsab],                _sa
    include(joinpath(pathroot,"test/run_collisions/datas/reading/sa.jl"))
    
end