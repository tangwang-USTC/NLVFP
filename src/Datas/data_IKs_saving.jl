
# data_saving for version `IKs`

function data_IKs_saving(ps;is_Cerror_dtnIKTs::Bool=false)
 
    println(idnIK,fmtf4(ps["tk"]),", ",ps["nk"][1], ", ",ps["Ik"][1]/I_unit, ", ",ps["Kk"][1]/K_unit,
                                  ", ",ps["nk"][2], ", ",ps["Ik"][2]/I_unit, ", ",ps["Kk"][2]/K_unit)
    
    #
    if is_Cerror_dtnIKTs
        println(idCerror,fmtf4(ps["tk"]),", ",fmtf4(ps["edtnIKTsk"][1,1]), ", ",fmtf4(ps["edtnIKTsk"][2,1]), ", ",fmtf4(ps["edtnIKTsk"][3,1]), ", ",fmtf4(ps["edtnIKTsk"][4,1]),
                                      ", ",fmtf4(ps["edtnIKTsk"][1,2]), ", ",fmtf4(ps["edtnIKTsk"][2,2]), ", ",fmtf4(ps["edtnIKTsk"][3,2]), ", ",fmtf4(ps["edtnIKTsk"][4,2]))
    end

    println(idsa,fmtf4(ps["tk"]),", ",ps["sak"][1],
                                  ", ",ps["sak"][2],
                                  ", ",ps["dtsabk"])

    isp = 1
    if ps["nModk"][isp] == 1
        println(idnModa,fmtf4(ps["tk"]),", ",ps["nModk"][isp],", ",ps["naik"][isp][1],", ",ps["uaik"][isp][1],", ",ps["vthik"][isp][1])
    elseif ps["nModk"][isp] == 2
        println(idnModa,fmtf4(ps["tk"]),", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2])
    elseif ps["nModk"][isp] == 3
        println(idnModa,fmtf4(ps["tk"]),", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3])
    else
        erbgjj
    end

    isp = 2
    if ps["nModk"][isp] == 1
        println(idnModb,fmtf4(ps["tk"]),", ",ps["nModk"][isp],", ",ps["naik"][isp][1],", ",ps["uaik"][isp][1],", ",ps["vthik"][isp][1])
    elseif ps["nModk"][isp] == 2
        println(idnModb,fmtf4(ps["tk"]),", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2])
    elseif ps["nModk"][isp] == 3
        println(idnModb,fmtf4(ps["tk"]),", ",ps["nModk"][isp],
                                        ", ",ps["naik"][isp][1], ", ",ps["uaik"][isp][1], ", ",ps["vthik"][isp][1],
                                        ", ",ps["naik"][isp][2], ", ",ps["uaik"][isp][2], ", ",ps["vthik"][isp][2],
                                        ", ",ps["naik"][isp][3], ", ",ps["uaik"][isp][3], ", ",ps["vthik"][isp][3])
    else
        erbgjj
    end
end