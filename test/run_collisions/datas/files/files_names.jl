if gridv_type == :uniform
    gridv_name = :uni
elseif gridv_type == :chebyshev
    gridv_name = :cheby
elseif gridv_type == :laguerre
    gridv_name = :lag
else
    asdfbh 
end

if 2 == 2
    # path_material
    if ns == 2
        path_material = string(pathdatas,"\\ns",ns,spices0[1],spices0[2],"_m",fmtf2.(mD0),"Z",Zq)
    else
        if  ns == 3
            path_material = string(pathdatas,"\\ns",ns,spices0[1],spices0[2],spices0[3],"_m",fmtf2.(mD0),"Z",Zq)
        elseif ns == 4
            path_material = string(pathdatas,"\\ns",ns,spices0[1],spices0[2],spices0[3],spices0[3],"_m",fmtf2.(mD0),"Z",Zq)
        else
            esdfgbhn 
        end
    end
    ispath(path_material) ? nothing : mkpath(path_material)
    
    # filepath_physicial
    filepath_variable = string("\\T",fmtf2.(T0),"Ek",fmtf2.(Ek0.*μEk),"na",fmtf2.(n0),"nMod",nMod0)
    filepath_mesh = string("\\",functionName,"_",nnv0,"_",ocp0,"_",vadaptlevels,"_",Int(is_vth_ode))

    if is_nai_const
        if is_fixed_NK
            Optim_type = string("n",Int(is_nai_const),"s",Int(is_nhnMod_adapt),"fC",Int(is_optim_CnIK),"_",Int(is_NKCnhC),"A",Int(is_NK_adapt_max),"B",Int(is_nMod_update_back),"Mh",Int(is_Mhc_ode))
        else
            Optim_type = string("n",Int(is_nai_const),"s",Int(is_nhnMod_adapt),"C",Int(is_optim_CnIK),"_",Int(is_NKCnhC),"A",Int(is_NK_adapt_max),"B",Int(is_nMod_update_back),"Mh",Int(is_Mhc_ode))
        end
    else
        if is_fixed_NK
            Optim_type = string("n",Int(is_nai_const),"s",Int(is_nhnMod_adapt),"NK",NKk,"fC",Int(is_optim_CnIK),"_",Int(is_NKCnhC),"A",Int(is_NK_adapt_max),"B",Int(is_nMod_update_back),"Mh",Int(is_Mhc_ode))
        else
            Optim_type = string("n",Int(is_nai_const),"s",Int(is_nhnMod_adapt),"NK",NKk,"C",Int(is_optim_CnIK),"_",Int(is_NKCnhC),"A",Int(is_NK_adapt_max),"B",Int(is_nMod_update_back),"Mh",Int(is_Mhc_ode))
        end
    end
    filepath_physicial = string(filepath_variable,filepath_mesh,"lnA",Int(is_lnA_const),
                               "C",Int(is_enforce_errdtnIKab),"E",Int(is_extrapolate_FLn),Optim_type,"M",Brag_condition_RDMhck1)

    # filepath_solver
    if orderRK ≤ 2
        if is_fvL_CP
            filepath_solver = string("\\t",nτ,"_",fmtf2.(tspan),"_",Int(Nτ_fix),",",Int(is_fixed_timestep),"_",fmtf2(rtol_DnIK),
                                    "_",fmtf2(rtol_dtsa_terminate),"_",gridv_name,"_",algname,iterRK)
        else
            filepath_solver = string("\\t",nτ,"_",fmtf2.(tspan),"_",Int(Nτ_fix),",",Int(is_fixed_timestep),"_",fmtf2(rtol_DnIK),
                                    "_",fmtf2.([rtol_DnuTi,rtol_dtsa_terminate]),"_",gridv_name,"_",algname,iterRK)
        end
    else
        if is_fvL_CP
            filepath_solver = string("\\t",nτ,"_",fmtf2.(tspan),"_",Int(Nτ_fix),",",Int(is_fixed_timestep),"_",fmtf2(rtol_DnIK),
                                    "_",fmtf2(rtol_dtsa_terminate),"_",gridv_name,"_",algname,"_",
                                    iterRK,"_", iterEmbeded)
        else
            filepath_solver = string("\\t",nτ,"_",fmtf2.(tspan),"_",Int(Nτ_fix),",",Int(is_fixed_timestep),"_",fmtf2(rtol_DnIK),
                                    "_",fmtf2.([rtol_DnuTi,rtol_dtsa_terminate]),"_",gridv_name,"_",algname,"_",
                                    iterRK,"_", iterEmbeded)
        end
    end

    filepath = string(filepath_physicial, filepath_solver)
    figpath = string(path_material,filepath)

    filename_nModa = string(filepath,"_nModa",".csv")
    filename_nModb = string(filepath,"_nModb",".csv")
    if is_moments_out
        filename_Mhcla = string(filepath,"_Mhcla",".csv")
        filename_Mhclb = string(filepath,"_Mhclb",".csv")

        filename_errMhcopla = string(filepath,"_errMhcopla",".csv")
        filename_errMhcoplb = string(filepath,"_errMhcoplb",".csv")

        filename_RDMck1a = string(filepath,"_RDMck1a",".csv")
        filename_RDMck1b = string(filepath,"_RDMck1b",".csv")
    end

    if  ns ≥ 3
        filename_nModc = string(filepath,"_nModc",".csv")
        if is_moments_out
            filename_Mhclc = string(filepath,"_Mhclc",".csv")
            filename_errMhcoplc = string(filepath,"_errMhcoplc",".csv")
            filename_RDMck1c = string(filepath,"_RDMck1c",".csv")
        end
        if  ns ≥ 4
            filename_nModd = string(filepath,"_nModd",".csv")
            if is_moments_out
                filename_Mhcld = string(filepath,"_Mhcld",".csv")
                filename_errMhcopld = string(filepath,"_errMhcopld",".csv")
                filename_RDMck1d = string(filepath,"_RDMck1d",".csv")
            end
        end
    end
    
    filename_sa = string(filepath,"_sa",".csv")
    filename_nIK = string(filepath,"_nIK",".csv")
    if is_Cerror_dtnIKTs
        filename_Cerror = string(filepath,"_Cerror",".csv")
    end

    printstyled(path_material,color=:red,"\n")
    printstyled("filename_nIK=",filename_nIK,color=:red,"\n")
    
    # file_fig = string(path_material,figname)
    # file_fig_path = normpath(file_fig)
    # file_fig_fold, file_fig_file = splitdir(file_fig_path)
    file_fig_file = figpath
    
    file_Ms_sa = string(path_material,filename_sa)
    file_Ms_sa = normpath(file_Ms_sa)
    file_Ms_sa_fold, file_Ms_sa_file = splitdir(file_Ms_sa)
    ispath(file_Ms_sa_fold) || mkpath(file_Ms_sa_fold)

    file_Ms_nIK = string(path_material,filename_nIK)
    file_Ms_nIK = normpath(file_Ms_nIK)
    file_Ms_fold, file_Ms_file = splitdir(file_Ms_nIK)
    ispath(file_Ms_fold) || mkpath(file_Ms_fold)

    if is_Cerror_dtnIKTs
        file_Ms_Cerror = string(path_material,filename_Cerror)
        file_Ms_Cerror = normpath(file_Ms_Cerror)
        file_Ms_Cerror_fold, file_Ms_Cerror_file = splitdir(file_Ms_Cerror)
        ispath(file_Ms_Cerror_fold) || mkpath(file_Ms_Cerror_fold)
    end
    
    # isfile(file_Ms_nIK)
    # is_path = ispath(file_Ms_fold) 

    file_Ms_nModa = string(path_material,filename_nModa)
    file_Ms_nModa = normpath(file_Ms_nModa)
    file_Ms_nModa_fold, file_Ms_nModa_file = splitdir(file_Ms_nModa)
    ispath(file_Ms_nModa_fold) || mkpath(file_Ms_nModa_fold)

    file_Ms_nModb = string(path_material,filename_nModb)
    file_Ms_nModb = normpath(file_Ms_nModb)
    file_Ms_nModb_fold, file_Ms_nModb_file = splitdir(file_Ms_nModb)
    ispath(file_Ms_nModb_fold) || mkpath(file_Ms_nModb_fold)
    
    if is_moments_out
        file_Ms_Mhcla = string(path_material,filename_Mhcla)
        file_Ms_Mhcla = normpath(file_Ms_Mhcla)
        file_Ms_Mhcla_fold, file_Ms_Mhcla_file = splitdir(file_Ms_Mhcla)
        ispath(file_Ms_Mhcla_fold) || mkpath(file_Ms_Mhcla_fold)

        file_Ms_Mhclb = string(path_material,filename_Mhclb)
        file_Ms_Mhclb = normpath(file_Ms_Mhclb)
        file_Ms_Mhclb_fold, file_Ms_Mhclb_file = splitdir(file_Ms_Mhclb)
        ispath(file_Ms_Mhclb_fold) || mkpath(file_Ms_Mhclb_fold)
    end
    if is_moments_out
        file_Ms_errMhcopla = string(path_material,filename_errMhcopla)
        file_Ms_errMhcopla = normpath(file_Ms_errMhcopla)
        file_Ms_errMhcopla_fold, file_Ms_errMhcopla_file = splitdir(file_Ms_errMhcopla)
        ispath(file_Ms_errMhcopla_fold) || mkpath(file_Ms_errMhcopla_fold)

        file_Ms_errMhcoplb = string(path_material,filename_errMhcoplb)
        file_Ms_errMhcoplb = normpath(file_Ms_errMhcoplb)
        file_Ms_errMhcoplb_fold, file_Ms_errMhcoplb_file = splitdir(file_Ms_errMhcoplb)
        ispath(file_Ms_errMhcoplb_fold) || mkpath(file_Ms_errMhcoplb_fold)
    end
    if is_moments_out
        file_Ms_RDMck1a = string(path_material,filename_RDMck1a)
        file_Ms_RDMck1a = normpath(file_Ms_RDMck1a)
        file_Ms_RDMck1a_fold, file_Ms_RDMck1a_file = splitdir(file_Ms_RDMck1a)
        ispath(file_Ms_RDMck1a_fold) || mkpath(file_Ms_RDMck1a_fold)

        file_Ms_RDMck1b = string(path_material,filename_RDMck1b)
        file_Ms_RDMck1b = normpath(file_Ms_RDMck1b)
        file_Ms_RDMck1b_fold, file_Ms_RDMck1b_file = splitdir(file_Ms_RDMck1b)
        ispath(file_Ms_RDMck1b_fold) || mkpath(file_Ms_RDMck1b_fold)
    end

    if  ns ≥ 3
        file_Ms_nModc = string(path_material,filename_nModc)
        file_Ms_nModc = normpath(file_Ms_nModc)
        file_Ms_nModc_fold, file_Ms_nModc_file = splitdir(file_Ms_nModc)
        ispath(file_Ms_nModc_fold) || mkpath(file_Ms_nModc_fold)

        if is_moments_out
            file_Ms_Mhclc = string(path_material,filename_Mhclc)
            file_Ms_Mhclc = normpath(file_Ms_Mhclc)
            file_Ms_Mhclc_fold, file_Ms_Mhclc_file = splitdir(file_Ms_Mhclc)
            ispath(file_Ms_Mhclc_fold) || mkpath(file_Ms_Mhclc_fold)

            file_Ms_errMhcoplc = string(path_material,filename_errMhcoplc)
            file_Ms_errMhcoplc = normpath(file_Ms_errMhcoplc)
            file_Ms_errMhcoplc_fold, file_Ms_errMhcoplc_file = splitdir(file_Ms_errMhcoplc)
            ispath(file_Ms_errMhcoplc_fold) || mkpath(file_Ms_errMhcoplc_fold)
            
            file_Ms_RDMck1c = string(path_material,filename_RDMck1c)
            file_Ms_RDMck1c = normpath(file_Ms_RDMck1c)
            file_Ms_RDMck1c_fold, file_Ms_RDMck1c_file = splitdir(file_Ms_RDMck1c)
            ispath(file_Ms_RDMck1c_fold) || mkpath(file_Ms_RDMck1c_fold)
        end

        if  ns ≥ 4
            file_Ms_nModd = string(path_material,filename_nModd)
            file_Ms_nModd = normpath(file_Ms_nModd)
            file_Ms_nModd_fold, file_Ms_nModd_file = splitdir(file_Ms_nModd)
            ispath(file_Ms_nModd_fold) || mkpath(file_Ms_nModd_fold)

            if is_moments_out
                file_Ms_Mhcld = string(path_material,filename_Mhcld)
                file_Ms_Mhcld = normpath(file_Ms_Mhcld)
                file_Ms_Mhcld_fold, file_Ms_Mhcld_file = splitdir(file_Ms_Mhcld)
                ispath(file_Ms_Mhcld_fold) || mkpath(file_Ms_Mhcld_fold)

                file_Ms_errMhcopld = string(path_material,filename_errMhcopld)
                file_Ms_errMhcopld = normpath(file_Ms_errMhcopld)
                file_Ms_errMhcopld_fold, file_Ms_errMhcopld_file = splitdir(file_Ms_errMhcopld)
                ispath(file_Ms_errMhcopld_fold) || mkpath(file_Ms_errMhcopld_fold)

                file_Ms_RDMck1d = string(path_material,filename_RDMck1d)
                file_Ms_RDMck1d = normpath(file_Ms_RDMck1d)
                file_Ms_RDMck1d_fold, file_Ms_RDMck1d_file = splitdir(file_Ms_RDMck1d)
                ispath(file_Ms_RDMck1d_fold) || mkpath(file_Ms_RDMck1d_fold)
            end
        end
    end
    
    # isfile(file_Ms_sa) 
    # isfile(file_Ms_nIK) 
    # isfile(file_Ms_Cerror) 
    # isfile(file_Ms_nModa)
    # isfile(file_Ms_nModb)
    

    cd(file_Ms_fold)

    # writedlm(file_Ms_file,datas_csv,'\t')
    # datas_csv = 2datas_csv
    # writedlm(file_Ms_file,datas_csv,'\t')
    
    # println(idnIK,datas_csv[:])
    # datas_csv = 2datas_csv
    # println(idnIK,datas_csv[:])
end