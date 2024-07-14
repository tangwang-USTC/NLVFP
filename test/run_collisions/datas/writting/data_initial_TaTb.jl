
if ns == 2
    datas_TaTb_csv = DataFrame(t=t0,Ta=T0[1],Tb=T0[2])
    TaTb_name = [:t, :Ta, :Tb]
else
    if ns == 3
        datas_TaTb_csv = DataFrame(t=t0,Ta=T0[1],Tb=T0[2],Tc=T0[3])
        TaTb_name = [:t, :Ta, :Tb, :Tc]
    elseif ns == 4
        datas_TaTb_csv = DataFrame(t=t0,Ta=T0[1],Tb=T0[2],Tc=T0[3],Td=T0[4])
        TaTb_name = [:t, :Ta, :Tb, :Tc, :Td]
    else
        sedfgbhn
    end
end
