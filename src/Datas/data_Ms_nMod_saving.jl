
# data_saving for version `Ms`

function data_Ms_saving(ps;is_moments_out::Bool=false,is_Cerror_dtnIKTs::Bool=false)

    if ps["ns"] == 2
        println(idnIK,fmtf4(ps["tk"]),", ",ps["nk"][1], ", ",ps["Ik"][1]/I_unit, ", ",ps["Kk"][1]/K_unit, ", ",ps["DThk"][1],
                                      ", ",ps["nk"][2], ", ",ps["Ik"][2]/I_unit, ", ",ps["Kk"][2]/K_unit, ", ",ps["DThk"][2])
        
        # 
        if is_Cerror_dtnIKTs
            println(idCerror,fmtf4(ps["tk"]),", ",fmtf4(ps["edtnIKTsk"][1,1]), ", ",fmtf4(ps["edtnIKTsk"][2,1]), ", ",fmtf4(ps["edtnIKTsk"][3,1]), ", ",fmtf4(ps["edtnIKTsk"][4,1]),
                                             ", ",fmtf4(ps["edtnIKTsk"][1,2]), ", ",fmtf4(ps["edtnIKTsk"][2,2]), ", ",fmtf4(ps["edtnIKTsk"][3,2]), ", ",fmtf4(ps["edtnIKTsk"][4,2]),
                                             ", ",fmtf4(ps["CRDnk"][1,1]))
        end
    
        println(idsa,fmtf4(ps["tk"]),", ",ps["sak"][1],
                                      ", ",ps["sak"][2],
                                      ", ",ps["dtsabk"])
    else
        if ps["ns"] == 3
            println(idnIK,fmtf4(ps["tk"]),", ",ps["nk"][1], ", ",ps["Ik"][1]/I_unit, ", ",ps["Kk"][1]/K_unit, ", ",ps["DThk"][1],
                                          ", ",ps["nk"][2], ", ",ps["Ik"][2]/I_unit, ", ",ps["Kk"][2]/K_unit, ", ",ps["DThk"][2],
                                          ", ",ps["nk"][3], ", ",ps["Ik"][3]/I_unit, ", ",ps["Kk"][3]/K_unit, ", ",ps["DThk"][3])
            
            # 
            if is_Cerror_dtnIKTs
                println(idCerror,fmtf4(ps["tk"]),", ",fmtf4(ps["edtnIKTsk"][1,1]), ", ",fmtf4(ps["edtnIKTsk"][2,1]), ", ",fmtf4(ps["edtnIKTsk"][3,1]), ", ",fmtf4(ps["edtnIKTsk"][4,1]),
                                                 ", ",fmtf4(ps["edtnIKTsk"][1,2]), ", ",fmtf4(ps["edtnIKTsk"][2,2]), ", ",fmtf4(ps["edtnIKTsk"][3,2]), ", ",fmtf4(ps["edtnIKTsk"][4,2]),
                                                 ", ",fmtf4(ps["edtnIKTsk"][1,3]), ", ",fmtf4(ps["edtnIKTsk"][2,3]), ", ",fmtf4(ps["edtnIKTsk"][3,3]), ", ",fmtf4(ps["edtnIKTsk"][4,3]),
                                                 ", ",fmtf4(ps["CRDnk"][1,1]))
            end
        
            println(idsa,fmtf4(ps["tk"]),", ",ps["sak"][1],
                                        ", ",ps["sak"][2],
                                        ", ",ps["sak"][3],
                                        ", ",ps["dtsabk"])
        elseif ps["ns"] == 4
            println(idnIK,fmtf4(ps["tk"]),", ",ps["nk"][1], ", ",ps["Ik"][1]/I_unit, ", ",ps["Kk"][1]/K_unit, ", ",ps["DThk"][1],
                                          ", ",ps["nk"][2], ", ",ps["Ik"][2]/I_unit, ", ",ps["Kk"][2]/K_unit, ", ",ps["DThk"][2],
                                          ", ",ps["nk"][3], ", ",ps["Ik"][3]/I_unit, ", ",ps["Kk"][3]/K_unit, ", ",ps["DThk"][3],
                                          ", ",ps["nk"][4], ", ",ps["Ik"][4]/I_unit, ", ",ps["Kk"][4]/K_unit, ", ",ps["DThk"][4])
            
            # 
            if is_Cerror_dtnIKTs
                println(idCerror,fmtf4(ps["tk"]),", ",fmtf4(ps["edtnIKTsk"][1,1]), ", ",fmtf4(ps["edtnIKTsk"][2,1]), ", ",fmtf4(ps["edtnIKTsk"][3,1]), ", ",fmtf4(ps["edtnIKTsk"][4,1]),
                                                 ", ",fmtf4(ps["edtnIKTsk"][1,2]), ", ",fmtf4(ps["edtnIKTsk"][2,2]), ", ",fmtf4(ps["edtnIKTsk"][3,2]), ", ",fmtf4(ps["edtnIKTsk"][4,2]),
                                                 ", ",fmtf4(ps["edtnIKTsk"][1,3]), ", ",fmtf4(ps["edtnIKTsk"][2,3]), ", ",fmtf4(ps["edtnIKTsk"][3,3]), ", ",fmtf4(ps["edtnIKTsk"][4,3]),
                                                 ", ",fmtf4(ps["edtnIKTsk"][1,4]), ", ",fmtf4(ps["edtnIKTsk"][2,4]), ", ",fmtf4(ps["edtnIKTsk"][3,4]), ", ",fmtf4(ps["edtnIKTsk"][4,4]),
                                                 ", ",fmtf4(ps["CRDnk"][1,1]))
            end
        
            println(idsa,fmtf4(ps["tk"]),", ",ps["sak"][1],
                                        ", ",ps["sak"][2],
                                        ", ",ps["sak"][3],
                                        ", ",ps["sak"][4],
                                        ", ",ps["dtsabk"])
        else
            frghj
        end
    end
    isp = 1
    if ps["nModk"][isp] == 1
        println(idnModa,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],", ",ps["naik"][isp][1],", ",ps["uaik"][isp][1],", ",ps["vthik"][isp][1])
    elseif ps["nModk"][isp] == 2
        println(idnModa,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2])
    elseif ps["nModk"][isp] == 3
        println(idnModa,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3])
    elseif ps["nModk"][isp] == 4
        println(idnModa,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3],
                                        ", ",ps["naik"][isp][4], ", ",ps["uaik"][isp][4], ", ",ps["vthik"][isp][4])
    elseif ps["nModk"][isp] == 5
        println(idnModa,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3],
                                        ", ",ps["naik"][isp][4], ", ",ps["uaik"][isp][4], ", ",ps["vthik"][isp][4],
                                        ", ",ps["naik"][isp][5], ", ",ps["uaik"][isp][5], ", ",ps["vthik"][isp][5])
    elseif ps["nModk"][isp] == 6
        println(idnModa,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3],
                                        ", ",ps["naik"][isp][4], ", ",ps["uaik"][isp][4], ", ",ps["vthik"][isp][4],
                                        ", ",ps["naik"][isp][5], ", ",ps["uaik"][isp][5], ", ",ps["vthik"][isp][5],
                                        ", ",ps["naik"][isp][6], ", ",ps["uaik"][isp][6], ", ",ps["vthik"][isp][6])
    elseif ps["nModk"][isp] == 7
        println(idnModa,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3],
                                        ", ",ps["naik"][isp][4], ", ",ps["uaik"][isp][4], ", ",ps["vthik"][isp][4],
                                        ", ",ps["naik"][isp][5], ", ",ps["uaik"][isp][5], ", ",ps["vthik"][isp][5],
                                        ", ",ps["naik"][isp][6], ", ",ps["uaik"][isp][6], ", ",ps["vthik"][isp][6],
                                        ", ",ps["naik"][isp][7], ", ",ps["uaik"][isp][7], ", ",ps["vthik"][isp][7])
    elseif ps["nModk"][isp] == 8
        println(idnModa,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3],
                                        ", ",ps["naik"][isp][4], ", ",ps["uaik"][isp][4], ", ",ps["vthik"][isp][4],
                                        ", ",ps["naik"][isp][5], ", ",ps["uaik"][isp][5], ", ",ps["vthik"][isp][5],
                                        ", ",ps["naik"][isp][6], ", ",ps["uaik"][isp][6], ", ",ps["vthik"][isp][6],
                                        ", ",ps["naik"][isp][7], ", ",ps["uaik"][isp][7], ", ",ps["vthik"][isp][7],
                                        ", ",ps["naik"][isp][8], ", ",ps["uaik"][isp][8], ", ",ps["vthik"][isp][8])
    elseif ps["nModk"][isp] == 9
        rfghhh
    else
        @show ps["nModk"]
        erbgjj
    end

    isp = 2
    if ps["nModk"][isp] == 1
        println(idnModb,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],", ",ps["naik"][isp][1],", ",ps["uaik"][isp][1],", ",ps["vthik"][isp][1])
    elseif ps["nModk"][isp] == 2
        println(idnModb,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2])
    elseif ps["nModk"][isp] == 3
        println(idnModb,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3])
    elseif ps["nModk"][isp] == 4
        println(idnModb,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3],
                                        ", ",ps["naik"][isp][4], ", ",ps["uaik"][isp][4], ", ",ps["vthik"][isp][4])
    elseif ps["nModk"][isp] == 5
        println(idnModb,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3],
                                        ", ",ps["naik"][isp][4], ", ",ps["uaik"][isp][4], ", ",ps["vthik"][isp][4],
                                        ", ",ps["naik"][isp][5], ", ",ps["uaik"][isp][5], ", ",ps["vthik"][isp][5])
    elseif ps["nModk"][isp] == 6
        println(idnModb,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3],
                                        ", ",ps["naik"][isp][4], ", ",ps["uaik"][isp][4], ", ",ps["vthik"][isp][4],
                                        ", ",ps["naik"][isp][5], ", ",ps["uaik"][isp][5], ", ",ps["vthik"][isp][5],
                                        ", ",ps["naik"][isp][6], ", ",ps["uaik"][isp][6], ", ",ps["vthik"][isp][6])
    elseif ps["nModk"][isp] == 7
        println(idnModb,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3],
                                        ", ",ps["naik"][isp][4], ", ",ps["uaik"][isp][4], ", ",ps["vthik"][isp][4],
                                        ", ",ps["naik"][isp][5], ", ",ps["uaik"][isp][5], ", ",ps["vthik"][isp][5],
                                        ", ",ps["naik"][isp][6], ", ",ps["uaik"][isp][6], ", ",ps["vthik"][isp][6],
                                        ", ",ps["naik"][isp][7], ", ",ps["uaik"][isp][7], ", ",ps["vthik"][isp][7])
    elseif ps["nModk"][isp] == 8
        println(idnModb,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3],
                                        ", ",ps["naik"][isp][4], ", ",ps["uaik"][isp][4], ", ",ps["vthik"][isp][4],
                                        ", ",ps["naik"][isp][5], ", ",ps["uaik"][isp][5], ", ",ps["vthik"][isp][5],
                                        ", ",ps["naik"][isp][6], ", ",ps["uaik"][isp][6], ", ",ps["vthik"][isp][6],
                                        ", ",ps["naik"][isp][7], ", ",ps["uaik"][isp][7], ", ",ps["vthik"][isp][7],
                                        ", ",ps["naik"][isp][8], ", ",ps["uaik"][isp][8], ", ",ps["vthik"][isp][8])
    elseif ps["nModk"][isp] == 9
        rfghhh
    else
        @show ps["nModk"]
        
        erbgjj
    end
    
    if ps["ns"] ≥ 3
        isp = 3
        if ps["nModk"][isp] == 1
            println(idnModc,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],", ",ps["naik"][isp][1],", ",ps["uaik"][isp][1],", ",ps["vthik"][isp][1])
        elseif ps["nModk"][isp] == 2
            println(idnModc,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                            ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                            ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2])
        elseif ps["nModk"][isp] == 3
            println(idnModc,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                            ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                            ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                            ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3])
        else
            erbgjj
        end
        if ps["ns"] ≥ 4
            isp = 4
            if ps["nModk"][isp] == 1
                println(idnModd,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],", ",ps["naik"][isp][1],", ",ps["uaik"][isp][1],", ",ps["vthik"][isp][1])
            elseif ps["nModk"][isp] == 2
                println(idnModd,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                                ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                                ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2])
            elseif ps["nModk"][isp] == 3
                println(idnModd,fmtf4(ps["tk"]),", ",ps["LMk"][isp],", ",ps["nModk"][isp],
                                                ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                                ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                                ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3])
            else
                erbgjj
            end
        end
    end

    # Mhc
    if is_moments_out
        if is_MjMs_max
            if norm(ps["Ik"]) ≤ epsT1000
                isp = 1
                nnjM = ps["nMjMs"][isp]                 # `nnjM ≥ nModk1` for `fM` and
                                                        # `nnjM ≥ 2nMod` for `fDM`
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1])
                    elseif nnjM == 3
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                    elseif nnjM == 4
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                    elseif nnjM == 5
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1])
                    elseif nnjM == 6
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1])
                    elseif nnjM == 7
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                    elseif nnjM == 8
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1])
                    elseif nnjM == 9
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1])
                    elseif nnjM == 10
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1])
                    elseif nnjM == 11
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1])
                    elseif nnjM == 12
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1],", ",ps["Mhck"][isp][12,1])
                    elseif nnjM == 13
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1],", ",ps["Mhck"][isp][12,1],
                                      ", ",ps["Mhck"][isp][13,1])
                    elseif nnjM == 14
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1],", ",ps["Mhck"][isp][12,1],
                                      ", ",ps["Mhck"][isp][13,1],", ",ps["Mhck"][isp][14,1])
                    elseif nnjM == 15
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1],", ",ps["Mhck"][isp][12,1],
                                      ", ",ps["Mhck"][isp][13,1],", ",ps["Mhck"][isp][14,1],", ",ps["Mhck"][isp][15,1])
                    elseif nnjM == 16
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1],", ",ps["Mhck"][isp][12,1],
                                      ", ",ps["Mhck"][isp][13,1],", ",ps["Mhck"][isp][14,1],", ",ps["Mhck"][isp][15,1],", ",ps["Mhck"][isp][16,1])
                    elseif nnjM == 17
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                         ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                         ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][6,2],", ",ps["Mhck"][isp][7,1])
                    else
                        vvbgggg
                    end
                end
        
                isp = 2
                nnjM = ps["nMjMs"][isp]
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1])
                    elseif nnjM == 3
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                    elseif nnjM == 4
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                    elseif nnjM == 5
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1])
                    elseif nnjM == 6
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1])
                    elseif nnjM == 7
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                    elseif nnjM == 8
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1])
                    elseif nnjM == 9
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1])
                    elseif nnjM == 10
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1])
                    elseif nnjM == 11
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1])
                    elseif nnjM == 12
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1],", ",ps["Mhck"][isp][12,1])
                    elseif nnjM == 13
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1],", ",ps["Mhck"][isp][12,1],
                                      ", ",ps["Mhck"][isp][13,1])
                    elseif nnjM == 14
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1],", ",ps["Mhck"][isp][12,1],
                                      ", ",ps["Mhck"][isp][13,1],", ",ps["Mhck"][isp][14,1])
                    elseif nnjM == 15
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1],", ",ps["Mhck"][isp][12,1],
                                      ", ",ps["Mhck"][isp][13,1],", ",ps["Mhck"][isp][14,1],", ",ps["Mhck"][isp][15,1])
                    elseif nnjM == 16
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1],", ",ps["Mhck"][isp][11,1],", ",ps["Mhck"][isp][12,1],
                                      ", ",ps["Mhck"][isp][13,1],", ",ps["Mhck"][isp][14,1],", ",ps["Mhck"][isp][15,1],", ",ps["Mhck"][isp][16,1])
                    elseif nnjM == 17
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                         ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                         ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][6,2],", ",ps["Mhck"][isp][7,1])
                    else
                        wsdfjgf
                    end
                end
                if ps["ns"] ≥ 3
                    isp = 3
                    nnjM = ps["nMjMs"][isp]
                    if norm(ps["uaik"][isp]) ≤ epsT10
                        if nnjM == 2
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1])
                        elseif nnjM == 3
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                        elseif nnjM == 4
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                        elseif nnjM == 5
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1])
                        elseif nnjM == 6
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1])
                        elseif nnjM == 7
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                          ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                        elseif nnjM == 8
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                          ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1])
                        else
                            vvbgggg
                        end
                    else
                        if nnjM == 2          # `nMod_max == 1`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1])
                        elseif nnjM == 4      # `nMod_max == 2`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1])
                        elseif nnjM == 6      # `nMod_max == 3`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                             ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1])
                        elseif nnjM == 8      # `nMod_max == 4`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                             ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1])
                        elseif nnjM == 10      # `nMod_max == 5`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                             ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                             ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1])
                        elseif nnjM == 12      # `nMod_max == 6`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                             ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                             ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][6,2],", ",ps["Mhck"][isp][7,1])
                        else
                            wsdfjgf
                        end
                    end
                    if ps["ns"] ≥ 4
                        isp = 4
                        nnjM = ps["nMjMs"][isp]
                        if norm(ps["uaik"][isp]) ≤ epsT10
                            if nnjM == 2
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1])
                            elseif nnjM == 3
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                            elseif nnjM == 4
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                            elseif nnjM == 5
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1])
                            elseif nnjM == 6
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1])
                            elseif nnjM == 7
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                              ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                            elseif nnjM == 8
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                              ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1])
                            else
                                vvbgggg
                            end
                        else
                            if nnjM == 2          # `nMod_max == 1`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1])
                            elseif nnjM == 4      # `nMod_max == 2`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1])
                            elseif nnjM == 6      # `nMod_max == 3`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                                 ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1])
                            elseif nnjM == 8      # `nMod_max == 4`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                                 ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1])
                            elseif nnjM == 10      # `nMod_max == 5`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                                 ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                                 ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1])
                            elseif nnjM == 12      # `nMod_max == 6`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                                 ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                                 ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][6,2],", ",ps["Mhck"][isp][7,1])
                            else
                                wsdfjgf
                            end
                        end
                    end
                end
            else
                isp = 1
                nnjM = ps["nMjMs"][isp]                 # `nnjM ≥ nModk1` for `fM` and
                                                        # `nnjM ≥ 2nMod` for `fDM`
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1])
                    elseif nnjM == 3
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                    elseif nnjM == 4
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                    elseif nnjM == 5
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1])
                    elseif nnjM == 6
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1])
                    elseif nnjM == 7
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                    elseif nnjM == 8
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1])
                    elseif nnjM == 9
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1])
                    elseif nnjM == 10
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1])
                    else
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                         ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                         ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][6,2],", ",ps["Mhck"][isp][7,1])
                    else
                        vvbgggg
                    end
                end
        
                isp = 2
                nnjM = ps["nMjMs"][isp]
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1])
                    elseif nnjM == 3
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                    elseif nnjM == 4
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                    elseif nnjM == 5
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1])
                    elseif nnjM == 6
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1])
                    elseif nnjM == 7
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                    elseif nnjM == 8
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1])
                    elseif nnjM == 9
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1])
                    elseif nnjM == 10
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                      ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1],
                                      ", ",ps["Mhck"][isp][9,1],", ",ps["Mhck"][isp][10,1])
                    else
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                         ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                         ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                         ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][6,2],", ",ps["Mhck"][isp][7,1])
                    else
                        wsdfjgf
                    end
                end

                if ps["ns"] ≥ 3
                    isp = 3
                    nnjM = ps["nMjMs"][isp]
                    if norm(ps["uaik"][isp]) ≤ epsT10
                        if nnjM == 2
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1])
                        elseif nnjM == 3
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                        elseif nnjM == 4
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                        elseif nnjM == 5
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1])
                        elseif nnjM == 6
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1])
                        elseif nnjM == 7
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                          ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                        elseif nnjM == 8
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                          ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1])
                        else
                            vvbgggg
                        end
                    else
                        if nnjM == 2          # `nMod_max == 1`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1])
                        elseif nnjM == 4      # `nMod_max == 2`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1])
                        elseif nnjM == 6      # `nMod_max == 3`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                             ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1])
                        elseif nnjM == 8      # `nMod_max == 4`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                             ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1])
                        elseif nnjM == 10      # `nMod_max == 5`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                             ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                             ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1])
                        elseif nnjM == 12      # `nMod_max == 6`
                            println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                             ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                             ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][6,2],", ",ps["Mhck"][isp][7,1])
                        else
                            wsdfjgf
                        end
                    end
                    if ps["ns"] ≥ 4
                        isp = 4
                        nnjM = ps["nMjMs"][isp]
                        if norm(ps["uaik"][isp]) ≤ epsT10
                            if nnjM == 2
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1])
                            elseif nnjM == 3
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                            elseif nnjM == 4
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                            elseif nnjM == 5
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1])
                            elseif nnjM == 6
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1])
                            elseif nnjM == 7
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                              ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                            elseif nnjM == 8
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,1],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                              ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1],", ",ps["Mhck"][isp][8,1])
                            else
                                vvbgggg
                            end
                        else
                            if nnjM == 2          # `nMod_max == 1`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1])
                            elseif nnjM == 4      # `nMod_max == 2`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1])
                            elseif nnjM == 6      # `nMod_max == 3`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                                 ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1])
                            elseif nnjM == 8      # `nMod_max == 4`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                                 ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1])
                            elseif nnjM == 10      # `nMod_max == 5`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                                 ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                                 ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1])
                            elseif nnjM == 12      # `nMod_max == 6`
                                println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][1,2],", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][2,2],", ",ps["Mhck"][isp][3,1],
                                                                 ", ",ps["Mhck"][isp][3,2],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][4,2],", ",ps["Mhck"][isp][5,1],
                                                                 ", ",ps["Mhck"][isp][5,2],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][6,2],", ",ps["Mhck"][isp][7,1])
                            else
                                wsdfjgf
                            end
                        end
                    end
                end
            end
        else
            rttttttt
            isp = 1
            if norm(ps["uaik"][isp]) ≤ epsT10
                if ps["nModk"][isp] == 1
                    println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1])
                elseif ps["nModk"][isp] == 2
                    println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                elseif ps["nModk"][isp] == 3
                    println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                else
                    vvbgggg
                end
            else
                if ps["nModk"][isp] == 1
                    println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                elseif ps["nModk"][isp] == 2
                    println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1])
                elseif ps["nModk"][isp] == 3
                    println(idMhcla,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1],
                                                     ", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                else
                    vvbgggg
                end
            end
    
            isp = 2
            if norm(ps["uaik"][isp]) ≤ epsT10
                if ps["nModk"][isp] == 1
                    println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1])
                elseif ps["nModk"][isp] == 2
                    println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                elseif ps["nModk"][isp] == 3
                    println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                else
                    vvbgggg
                end
            else
                if ps["nModk"][isp] == 1
                    println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                elseif ps["nModk"][isp] == 2
                    println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1])
                elseif ps["nModk"][isp] == 3
                    println(idMhclb,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                                     ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                else
                    vvbgggg
                end
            end

            if ps["ns"] ≥ 3
                isp = 3
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if ps["nModk"][isp] == 1
                        println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1])
                    elseif ps["nModk"][isp] == 2
                        println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                    elseif ps["nModk"][isp] == 3
                        println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                    else
                        vvbgggg
                    end
                else
                    if ps["nModk"][isp] == 1
                        println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                    elseif ps["nModk"][isp] == 2
                        println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1])
                    elseif ps["nModk"][isp] == 3
                        println(idMhclc,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                                         ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                    else
                        vvbgggg
                    end
                end
                if ps["ns"] ≥ 4
                    isp = 4
                    if norm(ps["uaik"][isp]) ≤ epsT10
                        if ps["nModk"][isp] == 1
                            println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1])
                        elseif ps["nModk"][isp] == 2
                            println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                        elseif ps["nModk"][isp] == 3
                            println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1])
                        else
                            vvbgggg
                        end
                    else
                        if ps["nModk"][isp] == 1
                            println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1])
                        elseif ps["nModk"][isp] == 2
                            println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],", ",ps["Mhck"][isp][5,1])
                        elseif ps["nModk"][isp] == 3
                            println(idMhcld,fmtf4(ps["tk"]),", ",ps["Mhck"][isp][2,1],", ",ps["Mhck"][isp][3,1],", ",ps["Mhck"][isp][4,1],
                                                             ", ",ps["Mhck"][isp][5,1],", ",ps["Mhck"][isp][6,1],", ",ps["Mhck"][isp][7,1])
                        else
                            vvbgggg
                        end
                    end
                end
            end
        end
    end

    # errMhcop
    if is_moments_out
        if is_MjMs_max
            if norm(ps["Ik"]) ≤ epsT1000
                isp = 1
                nnjM = ps["nMjMs"][isp]                 # `nnjM ≥ nModk1` for `fM` and
                                                        # `nnjM ≥ 2nMod` for `fDM`
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1])
                    elseif nnjM == 3
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                    elseif nnjM == 4
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                    elseif nnjM == 5
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1])
                    elseif nnjM == 6
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1])
                    elseif nnjM == 7
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                    elseif nnjM == 8
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1])
                    elseif nnjM == 9
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1])
                    elseif nnjM == 10
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1])
                    elseif nnjM == 11
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1])
                    elseif nnjM == 12
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1],", ",ps["errMhcop"][isp][12,1])
                    elseif nnjM == 13
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1],", ",ps["errMhcop"][isp][12,1],
                                      ", ",ps["errMhcop"][isp][13,1])
                    elseif nnjM == 14
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1],", ",ps["errMhcop"][isp][12,1],
                                      ", ",ps["errMhcop"][isp][13,1],", ",ps["errMhcop"][isp][14,1])
                    elseif nnjM == 15
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1],", ",ps["errMhcop"][isp][12,1],
                                      ", ",ps["errMhcop"][isp][13,1],", ",ps["errMhcop"][isp][14,1],", ",ps["errMhcop"][isp][15,1])
                    elseif nnjM == 16
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1],", ",ps["errMhcop"][isp][12,1],
                                      ", ",ps["errMhcop"][isp][13,1],", ",ps["errMhcop"][isp][14,1],", ",ps["errMhcop"][isp][15,1],", ",ps["errMhcop"][isp][16,1])
                    elseif nnjM == 17
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                         ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                         ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][6,2],", ",ps["errMhcop"][isp][7,1])
                    else
                        vvbgggg
                    end
                end
        
                isp = 2
                nnjM = ps["nMjMs"][isp]
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1])
                    elseif nnjM == 3
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                    elseif nnjM == 4
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                    elseif nnjM == 5
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1])
                    elseif nnjM == 6
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1])
                    elseif nnjM == 7
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                    elseif nnjM == 8
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1])
                    elseif nnjM == 9
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1])
                    elseif nnjM == 10
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1])
                    elseif nnjM == 11
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1])
                    elseif nnjM == 12
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1],", ",ps["errMhcop"][isp][12,1])
                    elseif nnjM == 13
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1],", ",ps["errMhcop"][isp][12,1],
                                      ", ",ps["errMhcop"][isp][13,1])
                    elseif nnjM == 14
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1],", ",ps["errMhcop"][isp][12,1],
                                      ", ",ps["errMhcop"][isp][13,1],", ",ps["errMhcop"][isp][14,1])
                    elseif nnjM == 15
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1],", ",ps["errMhcop"][isp][12,1],
                                      ", ",ps["errMhcop"][isp][13,1],", ",ps["errMhcop"][isp][14,1],", ",ps["errMhcop"][isp][15,1])
                    elseif nnjM == 16
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1],", ",ps["errMhcop"][isp][11,1],", ",ps["errMhcop"][isp][12,1],
                                      ", ",ps["errMhcop"][isp][13,1],", ",ps["errMhcop"][isp][14,1],", ",ps["errMhcop"][isp][15,1],", ",ps["errMhcop"][isp][16,1])
                    elseif nnjM == 17
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                         ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                         ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][6,2],", ",ps["errMhcop"][isp][7,1])
                    else
                        wsdfjgf
                    end
                end
                if ps["ns"] ≥ 3
                    isp = 3
                    nnjM = ps["nMjMs"][isp]
                    if norm(ps["uaik"][isp]) ≤ epsT10
                        if nnjM == 2
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1])
                        elseif nnjM == 3
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                        elseif nnjM == 4
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                        elseif nnjM == 5
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1])
                        elseif nnjM == 6
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1])
                        elseif nnjM == 7
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                          ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                        elseif nnjM == 8
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                          ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1])
                        else
                            vvbgggg
                        end
                    else
                        if nnjM == 2          # `nMod_max == 1`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1])
                        elseif nnjM == 4      # `nMod_max == 2`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1])
                        elseif nnjM == 6      # `nMod_max == 3`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                             ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1])
                        elseif nnjM == 8      # `nMod_max == 4`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                             ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1])
                        elseif nnjM == 10      # `nMod_max == 5`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                             ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                             ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1])
                        elseif nnjM == 12      # `nMod_max == 6`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                             ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                             ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][6,2],", ",ps["errMhcop"][isp][7,1])
                        else
                            wsdfjgf
                        end
                    end
                    if ps["ns"] ≥ 4
                        isp = 4
                        nnjM = ps["nMjMs"][isp]
                        if norm(ps["uaik"][isp]) ≤ epsT10
                            if nnjM == 2
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1])
                            elseif nnjM == 3
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                            elseif nnjM == 4
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                            elseif nnjM == 5
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1])
                            elseif nnjM == 6
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1])
                            elseif nnjM == 7
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                              ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                            elseif nnjM == 8
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                              ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1])
                            else
                                vvbgggg
                            end
                        else
                            if nnjM == 2          # `nMod_max == 1`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1])
                            elseif nnjM == 4      # `nMod_max == 2`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1])
                            elseif nnjM == 6      # `nMod_max == 3`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                                 ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1])
                            elseif nnjM == 8      # `nMod_max == 4`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                                 ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1])
                            elseif nnjM == 10      # `nMod_max == 5`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                                 ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                                 ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1])
                            elseif nnjM == 12      # `nMod_max == 6`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                                 ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                                 ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][6,2],", ",ps["errMhcop"][isp][7,1])
                            else
                                wsdfjgf
                            end
                        end
                    end
                end
            else
                isp = 1
                nnjM = ps["nMjMs"][isp]                 # `nnjM ≥ nModk1` for `fM` and
                                                        # `nnjM ≥ 2nMod` for `fDM`
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1])
                    elseif nnjM == 3
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                    elseif nnjM == 4
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                    elseif nnjM == 5
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1])
                    elseif nnjM == 6
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1])
                    elseif nnjM == 7
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                    elseif nnjM == 8
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1])
                    elseif nnjM == 9
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1])
                    elseif nnjM == 10
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1])
                    else
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                         ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                         ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][6,2],", ",ps["errMhcop"][isp][7,1])
                    else
                        vvbgggg
                    end
                end
        
                isp = 2
                nnjM = ps["nMjMs"][isp]
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1])
                    elseif nnjM == 3
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                    elseif nnjM == 4
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                    elseif nnjM == 5
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1])
                    elseif nnjM == 6
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1])
                    elseif nnjM == 7
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                    elseif nnjM == 8
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1])
                    elseif nnjM == 9
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1])
                    elseif nnjM == 10
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                      ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1],
                                      ", ",ps["errMhcop"][isp][9,1],", ",ps["errMhcop"][isp][10,1])
                    else
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                         ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                         ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                         ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][6,2],", ",ps["errMhcop"][isp][7,1])
                    else
                        wsdfjgf
                    end
                end

                if ps["ns"] ≥ 3
                    isp = 3
                    nnjM = ps["nMjMs"][isp]
                    if norm(ps["uaik"][isp]) ≤ epsT10
                        if nnjM == 2
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1])
                        elseif nnjM == 3
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                        elseif nnjM == 4
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                        elseif nnjM == 5
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1])
                        elseif nnjM == 6
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1])
                        elseif nnjM == 7
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                          ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                        elseif nnjM == 8
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                          ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1])
                        else
                            vvbgggg
                        end
                    else
                        if nnjM == 2          # `nMod_max == 1`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1])
                        elseif nnjM == 4      # `nMod_max == 2`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1])
                        elseif nnjM == 6      # `nMod_max == 3`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                             ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1])
                        elseif nnjM == 8      # `nMod_max == 4`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                             ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1])
                        elseif nnjM == 10      # `nMod_max == 5`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                             ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                             ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1])
                        elseif nnjM == 12      # `nMod_max == 6`
                            println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                             ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                             ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][6,2],", ",ps["errMhcop"][isp][7,1])
                        else
                            wsdfjgf
                        end
                    end
                    if ps["ns"] ≥ 4
                        isp = 4
                        nnjM = ps["nMjMs"][isp]
                        if norm(ps["uaik"][isp]) ≤ epsT10
                            if nnjM == 2
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1])
                            elseif nnjM == 3
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                            elseif nnjM == 4
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                            elseif nnjM == 5
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1])
                            elseif nnjM == 6
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1])
                            elseif nnjM == 7
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                              ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                            elseif nnjM == 8
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,1],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                              ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1],", ",ps["errMhcop"][isp][8,1])
                            else
                                vvbgggg
                            end
                        else
                            if nnjM == 2          # `nMod_max == 1`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1])
                            elseif nnjM == 4      # `nMod_max == 2`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1])
                            elseif nnjM == 6      # `nMod_max == 3`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                                 ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1])
                            elseif nnjM == 8      # `nMod_max == 4`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                                 ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1])
                            elseif nnjM == 10      # `nMod_max == 5`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                                 ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                                 ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1])
                            elseif nnjM == 12      # `nMod_max == 6`
                                println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][1,2],", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][2,2],", ",ps["errMhcop"][isp][3,1],
                                                                 ", ",ps["errMhcop"][isp][3,2],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][4,2],", ",ps["errMhcop"][isp][5,1],
                                                                 ", ",ps["errMhcop"][isp][5,2],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][6,2],", ",ps["errMhcop"][isp][7,1])
                            else
                                wsdfjgf
                            end
                        end
                    end
                end
            end
        else
            rttttttt
            isp = 1
            if norm(ps["uaik"][isp]) ≤ epsT10
                if ps["nModk"][isp] == 1
                    println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1])
                elseif ps["nModk"][isp] == 2
                    println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                elseif ps["nModk"][isp] == 3
                    println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                else
                    vvbgggg
                end
            else
                if ps["nModk"][isp] == 1
                    println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                elseif ps["nModk"][isp] == 2
                    println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1])
                elseif ps["nModk"][isp] == 3
                    println(iderrMhcopla,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1],
                                                     ", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                else
                    vvbgggg
                end
            end
    
            isp = 2
            if norm(ps["uaik"][isp]) ≤ epsT10
                if ps["nModk"][isp] == 1
                    println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1])
                elseif ps["nModk"][isp] == 2
                    println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                elseif ps["nModk"][isp] == 3
                    println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                else
                    vvbgggg
                end
            else
                if ps["nModk"][isp] == 1
                    println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                elseif ps["nModk"][isp] == 2
                    println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1])
                elseif ps["nModk"][isp] == 3
                    println(iderrMhcoplb,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                                     ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                else
                    vvbgggg
                end
            end

            if ps["ns"] ≥ 3
                isp = 3
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if ps["nModk"][isp] == 1
                        println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1])
                    elseif ps["nModk"][isp] == 2
                        println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                    elseif ps["nModk"][isp] == 3
                        println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                    else
                        vvbgggg
                    end
                else
                    if ps["nModk"][isp] == 1
                        println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                    elseif ps["nModk"][isp] == 2
                        println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1])
                    elseif ps["nModk"][isp] == 3
                        println(iderrMhcoplc,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                                         ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                    else
                        vvbgggg
                    end
                end
                if ps["ns"] ≥ 4
                    isp = 4
                    if norm(ps["uaik"][isp]) ≤ epsT10
                        if ps["nModk"][isp] == 1
                            println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1])
                        elseif ps["nModk"][isp] == 2
                            println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                        elseif ps["nModk"][isp] == 3
                            println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1])
                        else
                            vvbgggg
                        end
                    else
                        if ps["nModk"][isp] == 1
                            println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1])
                        elseif ps["nModk"][isp] == 2
                            println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],", ",ps["errMhcop"][isp][5,1])
                        elseif ps["nModk"][isp] == 3
                            println(iderrMhcopld,fmtf4(ps["tk"]),", ",ps["errMhcop"][isp][2,1],", ",ps["errMhcop"][isp][3,1],", ",ps["errMhcop"][isp][4,1],
                                                             ", ",ps["errMhcop"][isp][5,1],", ",ps["errMhcop"][isp][6,1],", ",ps["errMhcop"][isp][7,1])
                        else
                            vvbgggg
                        end
                    end
                end
            end
        end
    end
    
    # RDMck1
    if is_moments_out
        if is_MjMs_max
            if norm(ps["Ik"]) ≤ epsT1000
                isp = 1
                nnjM = ps["nMjMs"][isp]                 # `nnjM ≥ nModk1` for `fM` and
                                                        # `nnjM ≥ 2nMod` for `fDM`
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp])
                    elseif nnjM == 3
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                    elseif nnjM == 4
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                    elseif nnjM == 5
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp])
                    elseif nnjM == 6
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp])
                    elseif nnjM == 7
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                    elseif nnjM == 8
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp])
                    elseif nnjM == 9
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp])
                    elseif nnjM == 10
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp])
                    elseif nnjM == 11
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp])
                    elseif nnjM == 12
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp],", ",ps["RDMck1"][12,1,isp])
                    elseif nnjM == 13
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp],", ",ps["RDMck1"][12,1,isp],
                                      ", ",ps["RDMck1"][13,1,isp])
                    elseif nnjM == 14
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp],", ",ps["RDMck1"][12,1,isp],
                                      ", ",ps["RDMck1"][13,1,isp],", ",ps["RDMck1"][14,1,isp])
                    elseif nnjM == 15
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp],", ",ps["RDMck1"][12,1,isp],
                                      ", ",ps["RDMck1"][13,1,isp],", ",ps["RDMck1"][14,1,isp],", ",ps["RDMck1"][15,1,isp])
                    elseif nnjM == 16
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp],", ",ps["RDMck1"][12,1,isp],
                                      ", ",ps["RDMck1"][13,1,isp],", ",ps["RDMck1"][14,1,isp],", ",ps["RDMck1"][15,1,isp],", ",ps["RDMck1"][16,1,isp])
                    elseif nnjM == 17
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                         ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                         ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][isp][6,2],", ",ps["RDMck1"][7,1,isp])
                    else
                        vvbgggg
                    end
                end
        
                isp = 2
                nnjM = ps["nMjMs"][isp]
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp])
                    elseif nnjM == 3
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                    elseif nnjM == 4
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                    elseif nnjM == 5
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp])
                    elseif nnjM == 6
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp])
                    elseif nnjM == 7
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                    elseif nnjM == 8
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp])
                    elseif nnjM == 9
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp])
                    elseif nnjM == 10
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp])
                    elseif nnjM == 11
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp])
                    elseif nnjM == 12
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp],", ",ps["RDMck1"][12,1,isp])
                    elseif nnjM == 13
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp],", ",ps["RDMck1"][12,1,isp],
                                      ", ",ps["RDMck1"][13,1,isp])
                    elseif nnjM == 14
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp],", ",ps["RDMck1"][12,1,isp],
                                      ", ",ps["RDMck1"][13,1,isp],", ",ps["RDMck1"][14,1,isp])
                    elseif nnjM == 15
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp],", ",ps["RDMck1"][12,1,isp],
                                      ", ",ps["RDMck1"][13,1,isp],", ",ps["RDMck1"][14,1,isp],", ",ps["RDMck1"][15,1,isp])
                    elseif nnjM == 16
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp],", ",ps["RDMck1"][11,1,isp],", ",ps["RDMck1"][12,1,isp],
                                      ", ",ps["RDMck1"][13,1,isp],", ",ps["RDMck1"][14,1,isp],", ",ps["RDMck1"][15,1,isp],", ",ps["RDMck1"][16,1,isp])
                    elseif nnjM == 17
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                         ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                         ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][isp][6,2],", ",ps["RDMck1"][7,1,isp])
                    else
                        wsdfjgf
                    end
                end
                if ps["ns"] ≥ 3
                    isp = 3
                    nnjM = ps["nMjMs"][isp]
                    if norm(ps["uaik"][isp]) ≤ epsT10
                        if nnjM == 2
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp])
                        elseif nnjM == 3
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                        elseif nnjM == 4
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                        elseif nnjM == 5
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp])
                        elseif nnjM == 6
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp])
                        elseif nnjM == 7
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                          ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                        elseif nnjM == 8
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                          ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp])
                        else
                            vvbgggg
                        end
                    else
                        if nnjM == 2          # `nMod_max == 1`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp])
                        elseif nnjM == 4      # `nMod_max == 2`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp])
                        elseif nnjM == 6      # `nMod_max == 3`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                             ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp])
                        elseif nnjM == 8      # `nMod_max == 4`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                             ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp])
                        elseif nnjM == 10      # `nMod_max == 5`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                             ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                             ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp])
                        elseif nnjM == 12      # `nMod_max == 6`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                             ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                             ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][isp][6,2],", ",ps["RDMck1"][7,1,isp])
                        else
                            wsdfjgf
                        end
                    end
                    if ps["ns"] ≥ 4
                        isp = 4
                        nnjM = ps["nMjMs"][isp]
                        if norm(ps["uaik"][isp]) ≤ epsT10
                            if nnjM == 2
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp])
                            elseif nnjM == 3
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                            elseif nnjM == 4
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                            elseif nnjM == 5
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp])
                            elseif nnjM == 6
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp])
                            elseif nnjM == 7
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                              ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                            elseif nnjM == 8
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                              ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp])
                            else
                                vvbgggg
                            end
                        else
                            if nnjM == 2          # `nMod_max == 1`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp])
                            elseif nnjM == 4      # `nMod_max == 2`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp])
                            elseif nnjM == 6      # `nMod_max == 3`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                                 ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp])
                            elseif nnjM == 8      # `nMod_max == 4`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                                 ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp])
                            elseif nnjM == 10      # `nMod_max == 5`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                                 ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                                 ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp])
                            elseif nnjM == 12      # `nMod_max == 6`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                                 ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                                 ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][isp][6,2],", ",ps["RDMck1"][7,1,isp])
                            else
                                wsdfjgf
                            end
                        end
                    end
                end
            else
                isp = 1
                nnjM = ps["nMjMs"][isp]                 # `nnjM ≥ nModk1` for `fM` and
                                                        # `nnjM ≥ 2nMod` for `fDM`
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp])
                    elseif nnjM == 3
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                    elseif nnjM == 4
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                    elseif nnjM == 5
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp])
                    elseif nnjM == 6
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp])
                    elseif nnjM == 7
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                    elseif nnjM == 8
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp])
                    elseif nnjM == 9
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp])
                    elseif nnjM == 10
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp])
                    else
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                         ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                         ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][isp][6,2],", ",ps["RDMck1"][7,1,isp])
                    else
                        vvbgggg
                    end
                end
        
                isp = 2
                nnjM = ps["nMjMs"][isp]
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if nnjM == 2
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp])
                    elseif nnjM == 3
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                    elseif nnjM == 4
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                    elseif nnjM == 5
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp])
                    elseif nnjM == 6
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp])
                    elseif nnjM == 7
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                    elseif nnjM == 8
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp])
                    elseif nnjM == 9
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp])
                    elseif nnjM == 10
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                      ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp],
                                      ", ",ps["RDMck1"][9,1,isp],", ",ps["RDMck1"][10,1,isp])
                    else
                        vvbgggg
                    end
                else
                    if nnjM == 2          # `nMod_max == 1`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp])
                    elseif nnjM == 4      # `nMod_max == 2`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp])
                    elseif nnjM == 6      # `nMod_max == 3`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp])
                    elseif nnjM == 8      # `nMod_max == 4`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp])
                    elseif nnjM == 10      # `nMod_max == 5`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                         ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp])
                    elseif nnjM == 12      # `nMod_max == 6`
                        println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                         ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                         ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][isp][6,2],", ",ps["RDMck1"][7,1,isp])
                    else
                        wsdfjgf
                    end
                end

                if ps["ns"] ≥ 3
                    isp = 3
                    nnjM = ps["nMjMs"][isp]
                    if norm(ps["uaik"][isp]) ≤ epsT10
                        if nnjM == 2
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp])
                        elseif nnjM == 3
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                        elseif nnjM == 4
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                        elseif nnjM == 5
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp])
                        elseif nnjM == 6
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp])
                        elseif nnjM == 7
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                          ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                        elseif nnjM == 8
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                          ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp])
                        else
                            vvbgggg
                        end
                    else
                        if nnjM == 2          # `nMod_max == 1`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp])
                        elseif nnjM == 4      # `nMod_max == 2`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp])
                        elseif nnjM == 6      # `nMod_max == 3`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                             ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp])
                        elseif nnjM == 8      # `nMod_max == 4`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                             ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp])
                        elseif nnjM == 10      # `nMod_max == 5`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                             ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                             ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp])
                        elseif nnjM == 12      # `nMod_max == 6`
                            println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                             ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                             ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][isp][6,2],", ",ps["RDMck1"][7,1,isp])
                        else
                            wsdfjgf
                        end
                    end
                    if ps["ns"] ≥ 4
                        isp = 4
                        nnjM = ps["nMjMs"][isp]
                        if norm(ps["uaik"][isp]) ≤ epsT10
                            if nnjM == 2
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp])
                            elseif nnjM == 3
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                            elseif nnjM == 4
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                            elseif nnjM == 5
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp])
                            elseif nnjM == 6
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp])
                            elseif nnjM == 7
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                              ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                            elseif nnjM == 8
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][1,1,isp],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                              ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp],", ",ps["RDMck1"][8,1,isp])
                            else
                                vvbgggg
                            end
                        else
                            if nnjM == 2          # `nMod_max == 1`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp])
                            elseif nnjM == 4      # `nMod_max == 2`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp])
                            elseif nnjM == 6      # `nMod_max == 3`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                                 ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp])
                            elseif nnjM == 8      # `nMod_max == 4`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                                 ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp])
                            elseif nnjM == 10      # `nMod_max == 5`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                                 ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                                 ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp])
                            elseif nnjM == 12      # `nMod_max == 6`
                                println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][isp][1,2],", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][isp][2,2],", ",ps["RDMck1"][3,1,isp],
                                                                 ", ",ps["RDMck1"][isp][3,2],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][isp][4,2],", ",ps["RDMck1"][5,1,isp],
                                                                 ", ",ps["RDMck1"][isp][5,2],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][isp][6,2],", ",ps["RDMck1"][7,1,isp])
                            else
                                wsdfjgf
                            end
                        end
                    end
                end
            end
        else
            rttttttt
            isp = 1
            if norm(ps["uaik"][isp]) ≤ epsT10
                if NK == 1
                    println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp])
                elseif NK == 2
                    println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                elseif NK == 3
                    println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                else
                    vvbgggg
                end
            else
                if NK == 1
                    println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                elseif NK == 2
                    println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp])
                elseif NK == 3
                    println(idRDMck1a,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp],
                                                     ", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                else
                    vvbgggg
                end
            end
    
            isp = 2
            if norm(ps["uaik"][isp]) ≤ epsT10
                if NK == 1
                    println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp])
                elseif NK == 2
                    println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                elseif NK == 3
                    println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                else
                    vvbgggg
                end
            else
                if NK == 1
                    println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                elseif NK == 2
                    println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp])
                elseif NK == 3
                    println(idRDMck1b,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                                     ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                else
                    vvbgggg
                end
            end

            if ps["ns"] ≥ 3
                isp = 3
                if norm(ps["uaik"][isp]) ≤ epsT10
                    if NK == 1
                        println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp])
                    elseif NK == 2
                        println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                    elseif NK == 3
                        println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                    else
                        vvbgggg
                    end
                else
                    if NK == 1
                        println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                    elseif NK == 2
                        println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp])
                    elseif NK == 3
                        println(idRDMck1c,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                                         ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                    else
                        vvbgggg
                    end
                end
                if ps["ns"] ≥ 4
                    isp = 4
                    if norm(ps["uaik"][isp]) ≤ epsT10
                        if NK == 1
                            println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp])
                        elseif NK == 2
                            println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                        elseif NK == 3
                            println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp])
                        else
                            vvbgggg
                        end
                    else
                        if NK == 1
                            println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp])
                        elseif NK == 2
                            println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],", ",ps["RDMck1"][5,1,isp])
                        elseif NK == 3
                            println(idRDMck1d,fmtf4(ps["tk"]),", ",ps["RDMck1"][2,1,isp],", ",ps["RDMck1"][3,1,isp],", ",ps["RDMck1"][4,1,isp],
                                                             ", ",ps["RDMck1"][5,1,isp],", ",ps["RDMck1"][6,1,isp],", ",ps["RDMck1"][7,1,isp])
                        else
                            vvbgggg
                        end
                    end
                end
            end
        end
    end
end
