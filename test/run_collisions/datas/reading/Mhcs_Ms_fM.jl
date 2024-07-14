
# [Mhcs, dtMhcs]
wline3 = 3
ylabel = string("Mhcs(j)")

n000 = 4 
if is_MjMs_max
    nnjM = minimum(nMjMs)         # `nnjM ≥ nMod` for `fM` and
                                        # `nnjM ≥ 2nMod` for `fDM`
    Mhcj10 = ones(Nt,nnjM,ns)
    isp33 = 1
    if 1 == 1
        if nnjM == 2
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 3
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 4
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 5
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 6
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc10
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 7
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc10
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc12
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        else
            vvbgggg
        end
    
        isp33 = 2
        if nnjM == 2
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 3
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 4
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 5
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 6
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc10
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 7
            nj = 2
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc10
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc12
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        else
            vvbgggg
        end
    end
    
    Mhcsj = zeros(Nt,nnjM)
    if ns == 2
        Mhcsj_fM!(Mhcsj,ma,na,[vatht vbtht],Nt,nnjM,Mhcj10)
    else
        if ns == 3
            Mhcsj_fM!(Mhcsj,ma,na,[vatht vbtht vctht],Nt,nnjM,Mhcj10,ns)
        elseif ns == 4
            Mhcsj_fM!(Mhcsj,ma,na,[vatht vbtht vctht vdtht],Nt,nnjM,Mhcj10,ns)
        else
            defg
        end
    end
    
    
    Dt = diff(tplot)
    RDMhcsj = (Mhcsj[2:end,:] ./ Mhcsj[1:end-1,:] .- 1) ./ Dt
    
    label = (0:2:2(nnjM-1))'


    ylabel = string("Mhcs")
    plot(tplot[tvec],Mhcsj[tvec,:],title=title,
        xlabel=xlabel,ylabel=ylabel,
        line=(wline3,:auto),label=label)
    savefig(string(file_fig_file,"_Mhcs.png"))
    
    tvec22 = tplot_min .< tplot[1:end-1] .< tplot_max
    tvec[1:dkivv2] .= false
    ylabel = string("R∂ₜMhcs")
    plot(tplot[1:end-1][tvec22],RDMhcsj[tvec22,:],
                xlabel=xlabel,ylabel=ylabel,
                line=(wline3,:auto),label=label)
    savefig(string(file_fig_file,"_RdtMhcs.png"))

    pMhcsj = plot(tplot[tvec],Mhcsj[tvec,:],title=title,
                ylabel=ylabel,
                line=(wline3,:auto),label=label)
    pRDMhcsj = plot(tplot[1:end-1][tvec22],RDMhcsj[tvec22,:],
                    ylabel=ylabel,xlabel=xlabel,
                    line=(wline3,:auto),label=label)
    display(plot(pMhcsj,pRDMhcsj,layout=(2,1)))
end
