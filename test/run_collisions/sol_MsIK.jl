
# # Updating the parameters `ps` when `t = 0.0` 

if is_plot_only == false
    # Self-consistent for `Mhc`, or equivalent to `fvL` at the initial timestep
    include(joinpath(pathroot, "test/run_collisions/algorithm/dtMc_Mhc_self_consistent.jl"))

    include(joinpath(pathroot,"test/run_collisions/datas/ps_files.jl"))
    #### datas 
    if is_fvL_CP
        include(joinpath(pathroot,"test/run_collisions/datas/writting/data_initial_IK.jl"))
    else
        if is_NKCnhC
            if NKk == 1
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_Ms_namesNK.jl"))
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_initial_MsNKk.jl"))
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_errMhcop_namesNK.jl"))
            else
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_Ms_namesNK.jl"))
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_initial_MsNKmax.jl"))
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_errMhcop_namesNK.jl"))
            end
        else
            if is_NKC
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_Ms_namesNK.jl"))
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_initial_MsNKk.jl"))
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_errMhcop_namesNK.jl"))
            else
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_Ms_names_nMod.jl"))
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_initial_Ms_nMod.jl"))
                include(joinpath(pathroot,"test/run_collisions/datas/writting/data_errMhcop_names_nMod.jl"))
            end
        end
        # if is_nai_const
        #     include(joinpath(pathroot,"test/run_collisions/datas/writting/data_Ms_names_nMod.jl"))
        #     include(joinpath(pathroot,"test/run_collisions/datas/writting/data_initial_Ms_nMod.jl"))
        #     include(joinpath(pathroot,"test/run_collisions/datas/writting/data_errMhcop_names_nMod.jl"))
        # else
        #     if NKk == 1
        #         include(joinpath(pathroot,"test/run_collisions/datas/writting/data_Ms_names_nMod.jl"))
        #         include(joinpath(pathroot,"test/run_collisions/datas/writting/data_initial_Ms_nMod.jl"))
        #         include(joinpath(pathroot,"test/run_collisions/datas/writting/data_errMhcop_names_nMod.jl"))
        #     else
        #         include(joinpath(pathroot,"test/run_collisions/datas/writting/data_Ms_namesNK.jl"))
        #         include(joinpath(pathroot,"test/run_collisions/datas/writting/data_initial_MsNK.jl"))
        #         include(joinpath(pathroot,"test/run_collisions/datas/writting/data_errMhcop_namesNK.jl"))
        #     end
        # end
    end
end

## Files
include(joinpath(pathroot,"test/run_collisions/datas/files/files_names.jl"))

include(joinpath(pathroot,"test/run_collisions/datas/titles.jl"))

idsa = open(file_Ms_sa,"a+")
idnIK = open(file_Ms_nIK,"a+")     # truncate/appand, read, write, create
if is_Cerror_dtnIKTs
    idCerror = open(file_Ms_Cerror,"a+") 
end

idnModa = open(file_Ms_nModa,"a+") 
idnModb = open(file_Ms_nModb,"a+") 
if ns ≥ 3
    idnModc = open(file_Ms_nModc,"a+") 
    if ns ≥ 4
        idnModd = open(file_Ms_nModd,"a+") 
    end
end

if is_moments_out
    idMhcla = open(file_Ms_Mhcla,"a+") 
    idMhclb = open(file_Ms_Mhclb,"a+") 
    iderrMhcopla = open(file_Ms_errMhcopla,"a+") 
    iderrMhcoplb = open(file_Ms_errMhcoplb,"a+") 
    idRDMck1a = open(file_Ms_RDMck1a,"a+") 
    idRDMck1b = open(file_Ms_RDMck1b,"a+") 
    if ns ≥ 3
        idMhclc = open(file_Ms_Mhclc,"a+") 
        iderrMhcoplc = open(file_Ms_errMhcoplc,"a+") 
        idRDMck1c = open(file_RDMck1_Mhclc,"a+") 
        if ns ≥ 4
            idMhcld = open(file_Ms_Mhcld,"a+") 
            iderrMhcopld = open(file_Ms_errMhcopld,"a+") 
            idRDMck1c = open(file_RDMck1_Mhcld,"a+") 
        end
    end
end

# main procedure 

if is_skip_solve == false
    CSV.write(idsa, datas_sa_csv, newline='\n')
    CSV.write(idnIK, datas_nIK_csv, newline='\n')                     # Over-writing existing datas in the file `idnIk`
    if is_Cerror_dtnIKTs
        CSV.write(idCerror, datas_Cerror_dtnIK_csv, newline='\n')
    end
    # CSV.write(idnIK,datas_nIK_csv,newline='\n',append=true)         # Appending datas in the last line of the file `idnIk`

    CSV.write(idnModa, datas_nModa_csv, newline='\n')
    CSV.write(idnModb, datas_nModb_csv, newline='\n')
    if ns ≥ 3
        CSV.write(idnModc, datas_nModc_csv, newline='\n')
        if ns ≥ 4
            CSV.write(idnModd, datas_nModd_csv, newline='\n')
        end
    end

    if is_moments_out
        CSV.write(idMhcla, datas_Mhcla_csv, newline='\n')
        CSV.write(idMhclb, datas_Mhclb_csv, newline='\n')
        CSV.write(iderrMhcopla, datas_errMhcopla_csv, newline='\n')
        CSV.write(iderrMhcoplb, datas_errMhcoplb_csv, newline='\n')
        CSV.write(idRDMck1a, datas_Mhcla_csv, newline='\n')
        CSV.write(idRDMck1b, datas_Mhclb_csv, newline='\n')
        if ns ≥ 3
            CSV.write(idMhclc, datas_Mhclc_csv, newline='\n')
            CSV.write(iderrMhcoplc, datas_errMhcoplc_csv, newline='\n')
            CSV.write(idRDMck1c, datas_Mhclc_csv, newline='\n')
            if ns ≥ 4
                CSV.write(idMhcld, datas_Mhcld_csv, newline='\n')
                CSV.write(iderrMhcopld, datas_errMhcopld_csv, newline='\n')
                CSV.write(idRDMck1d, datas_Mhcld_csv, newline='\n')
            end
        end
    end
  
    if is_fvL_CP
        include(joinpath(pathroot, "test/run_collisions/algorithm/dtIKs_testting.jl"))   
    
        @time IKk1integralk!(dtIKk1,IKk1,pstk, Nstep_max;
                is_normal=is_normal,
                restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM,is_boundaryv0=is_boundaryv0,
                is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
            
                eps_fup=eps_fup,eps_flow=eps_flow,
                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                abstol=abstol,reltol=reltol,
                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
    
                i_iter_rs2=i_iter_rs2,is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                is_Cerror_dtnIKTs=is_Cerror_dtnIKTs,
                rtol_dtsa=rtol_dtsa,ratio_dtk1=ratio_dtk1)
    else
        if is_sol_fvLc
            fvL0k1 = deepcopy(fvL0e) 
            @time fvLck1integralk!(fvL0k1, pstk, Nstep_max;orderRK=orderRK,i_iter_rs2=i_iter_rs2, 
                    Nspan_nuTi_max=Nspan_nuTi_maxmin,
                    NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
                    restartfit=restartfit,maxIterTR=maxIterTR,maxIterKing=maxIterKing,
                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                    rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                    optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                    is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                    is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
                
                    eps_fup=eps_fup,eps_flow=eps_flow,
                    maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                    abstol=abstol,reltol=reltol,
                    vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                    is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
        
                    is_corrections=is_corrections,
                    is_extrapolate_FLn=is_extrapolate_FLn, rtol_DnIK=rtol_DnIK,
                    is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                    dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
                    is_MjMs_max=is_MjMs_max,
                    is_moments_out=is_moments_out,is_Cerror_dtnIKTs=is_Cerror_dtnIKTs,
                    ratio_dtk1=ratio_dtk1)
        else
            Mck1 = deepcopy(Mck10) 
            Rck1 = deepcopy(Rck10)
        
            Ia4 = zeros(ns)
            Ia4[1] = 1Mck1[1, 2, 1]
            Ia4[2] = 1Mck1[1, 2, 2]
            
            Ka4 = zeros(ns)
            Ka4[1] = Mck1[2, 1, 1] * CMcKa 
            Ka4[2] = Mck1[2, 1, 2] * CMcKa 
            # rfghnm
            error_na = na - Mck1[1,1,:]
            error_Ia = Ia - Mck1[2,1,:]
            error_Ka = Ka - Mck1[1,1,:] * CMcKa 
            # error_nIK = 
            @show error_na, error_Ia, error_Ka
            # wwwwww, NK_initial, NKmax
            
            @time Mck1integralk!(Rck1, Mck1, pstk, Nstep_max;
                    orderRK=orderRK, rsRK=rsRK, iterRK=iterRK, 
                    orderEmbeded=orderEmbeded, rsEmbeded=rsEmbeded, iterEmbeded=iterEmbeded, 
                    Nspan_nuTi_max=Nspan_nuTi_maxmin,
                    NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
                    restartfit=restartfit,maxIterTR=maxIterTR,maxIterKing=maxIterKing,
                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                    rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                    optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                    is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                    is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
                
                    eps_fup=eps_fup,eps_flow=eps_flow,
                    maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                    abstol=abstol,reltol=reltol,
                    vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                    is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
        
                    is_vth_ode=is_vth_ode, is_corrections=is_corrections,
                    is_extrapolate_FLn=is_extrapolate_FLn, rtol_DnIK=rtol_DnIK,
                    is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                    dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
                    is_MjMs_max=is_MjMs_max,ratio_dtk1=ratio_dtk1,
                    is_moments_out=is_moments_out,is_Cerror_dtnIKTs=is_Cerror_dtnIKTs,
                    
                    is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
                    is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
                    is_fixed_NK=is_fixed_NK,is_nMod_update_back=is_nMod_update_back)
        end
    end
end
 

close(idsa)
close(idnIK)
if is_Cerror_dtnIKTs
    close(idCerror)
end
close(idnModa)
close(idnModb)
if ns ≥ 3
    close(idnModc)
    if ns ≥ 4
        close(idnModd)
    end
end
if is_moments_out
    close(idMhcla)
    close(idMhclb)
    close(iderrMhcopla)
    close(iderrMhcoplb)
    close(idRDMck1a)
    close(idRDMck1b)
    if ns ≥ 3
        close(idMhclc)
        close(iderrMhcoplc)
        close(idRDMck1c)
        if ns ≥ 4
            close(idMhcld)
            close(iderrMhcopld)
            close(idRDMck1d)
        end
    end
end
# # Plotting
if is_nai_const == false && is_fixed_NK == false
    include(joinpath(pathroot,"test/run_collisions/datas/reading/data_nIKsMhcNKnh13.jl"))
else
    include(joinpath(pathroot,"test/run_collisions/datas/reading/data_nIKsMhc.jl"))
end

@show tspan, (nτ, Nstep_max), dt_initial
algname
