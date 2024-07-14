

    # pathTaTb_material
    if ns == 2
        pathTaTb_material = string(pathdatas,"\\ns",ns,spices0[1],spices0[2],"_m",fmtf2.(mD0),"Z",Zq)
    else
        if  ns == 3
            pathTaTb_material = string(pathdatas,"\\ns",ns,spices0[1],spices0[2],spices0[3],"_m",fmtf2.(mD0),"Z",Zq)
        elseif ns == 4
            pathTaTb_material = string(pathdatas,"\\ns",ns,spices0[1],spices0[2],spices0[3],spices0[3],"_m",fmtf2.(mD0),"Z",Zq)
        else
            esdfgbhn 
        end
    end
    ispath(pathTaTb_material) ? nothing : mkpath(pathTaTb_material)
    
    # filepathTaTb_physicial
    if is_lnA_const
        filepathTaTb_physicial = string("\\T",fmtf2.(T0),"Ek",fmtf2.(Ek0.*μEk),"na",fmtf2.(n0),"nMod",nMod0,"\\","lnA1")
    else
        filepathTaTb_physicial = string("\\T",fmtf2.(T0),"Ek",fmtf2.(Ek0.*μEk),"na",fmtf2.(n0),"nMod",nMod0,"\\","lnA0")
    end

    # filepathTaTb_solver
    filepathTaTb_solver = string("\\t",nτ,"_",fmtf2.(tspan),"_",Int(Nτ_fix_TaTb),"_",fmtf2(rtol_DnIK),"_",fmtf2.([rtol_DnuTi,rtol_dtsa_terminate]))

    filepathTaTb = string(filepathTaTb_physicial, filepathTaTb_solver)
    figpathTaTb = string(pathTaTb_material,filepathTaTb)
    
    filename_nIKTaTb = string(filepathTaTb,"_nIK",".csv")

    printstyled(pathTaTb_material,color=:red,"\n")
    printstyled("filename_nIKTaTb=",filename_nIKTaTb,color=:red,"\n")
    
    fileTaTb_fig_file = figpathTaTb
    

    file_TaTb = string(pathTaTb_material,filename_nIKTaTb)
    file_TaTb = normpath(file_TaTb)
    file_TaTb_fold, file_TaTb_file = splitdir(file_TaTb)
    ispath(file_TaTb_fold) || mkpath(file_TaTb_fold)
    
    # isfile(file_TaTb) 
    
    cd(file_TaTb_fold)
    