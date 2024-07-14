
if ns == 2
    datas_sa_csv = DataFrame(t=t0,
                            sa=sa[1]/s_unit, 
                            sb=sa[2]/s_unit,
                            dts= 1 * dtsabk0)
    
    datas_nIK_csv = DataFrame(t=t0,
            na=na[1]/n_unit, Ia=Ia[1]/I_unit, Ka=Ka[1]/K_unit, errTha=0.0,  
            nb=na[2]/n_unit, Ib=Ia[2]/I_unit, Kb=Ka[2]/K_unit, errThb=0.0)
    
    if is_Cerror_dtnIKTs
        datas_Cerror_dtnIK_csv = DataFrame(t=t0,
                                edtna=0.0, edtIa=0.0, edtKa=0.0, eRdtTa=0.0,  
                                edtnb=0.0, edtIb=0.0, edtKb=0.0, eRdtTb=0.0,
                                CRDn=0.0)
    end
else
    if ns == 3
        datas_sa_csv = DataFrame(t=t0,
                                sa=sa[1]/s_unit, 
                                sb=sa[2]/s_unit,
                                sc=sa[3]/s_unit,
                                dts= 1 * dtsabk0)
        
        datas_nIK_csv = DataFrame(t=t0,
                na=na[1]/n_unit, Ia=Ia[1]/I_unit, Ka=Ka[1]/K_unit, errTha=0.0,  
                nb=na[2]/n_unit, Ib=Ia[2]/I_unit, Kb=Ka[2]/K_unit, errThb=0.0,  
                nc=na[3]/n_unit, Ic=Ia[3]/I_unit, Kc=Ka[3]/K_unit, errThc=0.0)
        
        if is_Cerror_dtnIKTs
            datas_Cerror_dtnIK_csv = DataFrame(t=t0,
                                    edtna=0.0, edtIa=0.0, edtKa=0.0, eRdtTa=0.0,  
                                    edtnb=0.0, edtIb=0.0, edtKb=0.0, eRdtTb=0.0,  
                                    edtnc=0.0, edtIc=0.0, edtKc=0.0, eRdtTc=0.0,
                                    CRDn=0.0)
        end
    elseif ns == 4
        datas_sa_csv = DataFrame(t=t0,
                                sa=sa[1]/s_unit, 
                                sb=sa[2]/s_unit,
                                sc=sa[3]/s_unit,
                                sd=sa[4]/s_unit,
                                dts= 1 * dtsabk0)
        
        datas_nIK_csv = DataFrame(t=t0,
                na=na[1]/n_unit, Ia=Ia[1]/I_unit, Ka=Ka[1]/K_unit, errTha=0.0,  
                nb=na[2]/n_unit, Ib=Ia[2]/I_unit, Kb=Ka[2]/K_unit, errThb=0.0,  
                nc=na[3]/n_unit, Ic=Ia[3]/I_unit, Kc=Ka[3]/K_unit, errThc=0.0,  
                nd=na[3]/n_unit, Id=Ia[3]/I_unit, Kd=Ka[3]/K_unit, errThd=0.0)
        
        if is_Cerror_dtnIKTs
            datas_Cerror_dtnIK_csv = DataFrame(t=t0,
                                    edtna=0.0, edtIa=0.0, edtKa=0.0, eRdtTa=0.0,  
                                    edtnb=0.0, edtIb=0.0, edtKb=0.0, eRdtTb=0.0,  
                                    edtnc=0.0, edtIc=0.0, edtKc=0.0, eRdtTc=0.0,  
                                    edtnd=0.0, edtId=0.0, edtKd=0.0, eRdtTd=0.0,
                                    CRDn=0.0)
        end
    else
        sedfgbhn
    end
end

isp33 = 1
x0_nuTi = zeros(3nMod[isp33])
nuTi_vec!(x0_nuTi, nMod0[isp33], nai0[isp33], uai0[isp33], vthi0[isp33])
datas_nModa = zeros(1, 3nMod0[isp33] + 3)
datas_nModa[1:3] = [t0, LM[isp33], nMod0[isp33]]
datas_nModa[4:end] = x0_nuTi[:]
if nMod0[isp33] == 1
    datas_nModa_name = ["t", "LMa",  "nModa", "na1", "ua1", "vath1"]
elseif nMod0[isp33] == 2
    datas_nModa_name = ["t", "LMa",  "nModa", "na1", "ua1", "vath1", "na2", "ua2", "vath2"]
elseif nMod0[isp33] == 3
    datas_nModa_name = ["t", "LMa",  "nModa", "na1", "ua1", "vath1", "na2", "ua2", "vath2", "na3", "ua3", "vath3"]
elseif nMod0[isp33] == 4
    datas_nModa_name = ["t", "LMa",  "nModa", "na1", "ua1", "vath1", "na2", "ua2", "vath2", "na3", "ua3", "vath3", 
                                              "na4", "ua4", "vath4"]
elseif nMod0[isp33] == 5
    datas_nModa_name = ["t", "LMa",  "nModa", "na1", "ua1", "vath1", "na2", "ua2", "vath2", "na3", "ua3", "vath3", 
                                              "na4", "ua4", "vath4","na5", "ua5", "vath5"]
elseif nMod0[isp33] == 6
    datas_nModa_name = ["t", "LMa",  "nModa", "na1", "ua1", "vath1", "na2", "ua2", "vath2", "na3", "ua3", "vath3", 
                                              "na4", "ua4", "vath4","na5", "ua5", "vath5","na6", "ua6", "vath6"]
elseif nMod0[isp33] == 7
    datas_nModa_name = ["t", "LMa",  "nModa", "na1", "ua1", "vath1", "na2", "ua2", "vath2", "na3", "ua3", "vath3", 
                                              "na4", "ua4", "vath4","na5", "ua5", "vath5","na6", "ua6", "vath6", 
                                              "na7", "ua7", "vath7"]
elseif nMod0[isp33] == 8
    datas_nModa_name = ["t", "LMa",  "nModa", "na1", "ua1", "vath1", "na2", "ua2", "vath2", "na3", "ua3", "vath3", 
                                              "na4", "ua4", "vath4","na5", "ua5", "vath5","na6", "ua6", "vath6", 
                                              "na7", "ua7", "vath7","na8", "ua8", "vath8"]
elseif nMod0[isp33] == 9
    qawsdfg
else
    dfgjhjh
end

isp33 = 2
x0_nuTi = zeros(3nMod[isp33])
nuTi_vec!(x0_nuTi,nMod0[isp33], nai0[isp33], uai0[isp33], vthi0[isp33])
datas_nModb = zeros(1, 3nMod0[isp33] + 3)
datas_nModb[1:3] = [t0, LM[isp33], nMod0[isp33]]
datas_nModb[4:end] = x0_nuTi[:]
if nMod0[isp33] == 1
    datas_nModb_name = ["t", "LMb",  "nModb", "nb1", "ub1", "vbth1"]
elseif nMod0[isp33] == 2
    datas_nModb_name = ["t", "LMb",  "nModb", "nb1", "ub1", "vbth1", "nb2", "ub2", "vbth2"]
elseif nMod0[isp33] == 3
    datas_nModb_name = ["t", "LMb",  "nModb", "nb1", "ub1", "vbth1", "nb2", "ub2", "vbth2", "nb3", "ub3", "vbth3"]
elseif nMod0[isp33] == 4
    datas_nModb_name = ["t", "LMb",  "nModb", "nb1", "ub1", "vbth1", "nb2", "ub2", "vbth2", "nb3", "ub3", "vbth3", 
                                              "nb4", "ub4", "vbth4"]
elseif nMod0[isp33] == 5
    datas_nModb_nbme = ["t", "LMb",  "nModb", "nb1", "ub1", "vbth1", "nb2", "ub2", "vbth2", "nb3", "ub3", "vbth3", 
                                              "nb4", "ub4", "vbth4","nb5", "ub5", "vbth5"]
elseif nMod0[isp33] == 6
    datas_nModb_nbme = ["t", "LMb",  "nModb", "nb1", "ub1", "vbth1", "nb2", "ub2", "vbth2", "nb3", "ub3", "vbth3", 
                                              "nb4", "ub4", "vbth4","nb5", "ub5", "vbth5","nb6", "ub6", "vbth6"]
elseif nMod0[isp33] == 7
    datas_nModb_nbme = ["t", "LMb",  "nModb", "nb1", "ub1", "vbth1", "nb2", "ub2", "vbth2", "nb3", "ub3", "vbth3", 
                                              "nb4", "ub4", "vbth4","nb5", "ub5", "vbth5","nb6", "ub6", "vbth6", 
                                              "nb7", "ub7", "vbth7"]
elseif nMod0[isp33] == 8
    datas_nModb_nbme = ["t", "LMb",  "nModb", "nb1", "ub1", "vbth1", "nb2", "ub2", "vbth2", "nb3", "ub3", "vbth3", 
                                              "nb4", "ub4", "vbth4","nb5", "ub5", "vbth5","nb6", "ub6", "vbth6", 
                                              "nb7", "ub7", "vbth7","nb8", "ub8", "vbth8"]
elseif nMod0[isp33] == 9
    qawsdfg
else
    dfgjhjh
end

datas_nModa_csv = DataFrame(datas_nModa, :auto)
rename!(datas_nModa_csv, datas_nModa_name)
datas_nModb_csv = DataFrame(datas_nModb, :auto)
rename!(datas_nModb_csv, datas_nModb_name)

if ns ≥ 3
    isp33 = 3
    x0_nuTi = zeros(3nMod[isp33])
    nuTi_vec!(x0_nuTi,nMod0[isp33], nai0[isp33], uai0[isp33], vthi0[isp33])
    datas_nModc = zeros(1, 3nMod0[isp33] + 3)
    datas_nModc[1:3] = [t0, LM[isp33], nMod0[isp33]]
    datas_nModc[4:end] = x0_nuTi[:]
    if nMod0[isp33] == 1
        datas_nModc_name = ["t", "LMc",  "nModc", "nc1", "uc1", "vcth1"]
    elseif nMod0[isp33] == 2
        datas_nModc_name = ["t", "LMc",  "nModc", "nc1", "uc1", "vcth1", "nc2", "uc2", "vcth2"]
    elseif nMod0[isp33] == 3
        datas_nModc_name = ["t", "LMc",  "nModc", "nc1", "uc1", "vcth1", "nc2", "uc2", "vcth2", "nc3", "uc3", "vcth3"]
    else
        dfgjhjh
    end
    datas_nModc_csv = DataFrame(datas_nModc, :auto)
    rename!(datas_nModc_csv, datas_nModc_name)
    if ns ≥ 4
        isp33 = 4
        x0_nuTi = zeros(3nMod[isp33])
        nuTi_vec!(x0_nuTi,nMod0[isp33], nai0[isp33], uai0[isp33], vthi0[isp33])
        datas_nModd = zeros(1, 3nMod0[isp33] + 3)
        datas_nModd[1:3] = [t0, LM[isp33], nMod0[isp33]]
        datas_nModd[4:end] = x0_nuTi[:]
        if nMod0[isp33] == 1
            datas_nModd_name = ["t", "LMd",  "nModd", "nd1", "ud1", "vdth1"]
        elseif nMod0[isp33] == 2
            datas_nModd_name = ["t", "LMd",  "nModd", "nd1", "ud1", "vdth1", "nd2", "ud2", "vdth2"]
        elseif nMod0[isp33] == 3
            datas_nModd_name = ["t", "LMd",  "nModd", "nd1", "ud1", "vdth1", "nd2", "ud2", "vdth2", "nd3", "ud3", "vdth3"]
        else
            dfgjhjh
        end
        datas_nModd_csv = DataFrame(datas_nModd, :auto)
        rename!(datas_nModd_csv, datas_nModd_name)
    end
end

