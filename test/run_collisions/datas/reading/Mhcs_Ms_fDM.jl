
# [Mhcs, dtMhcs]
if is_MjMs_max
    nnjM = minimum(nMjMs)         # `nnjM ≥ nMod` for `fM` and
                                        # `nnjM ≥ 2nMod` for `fDM`
    Mhcj10 = ones(Nt,nnjM,ns)
    isp33 = 1
    if 1 == 1
        if nnjM == 2
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 4
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc3
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 6
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc3
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc5
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 8
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc3
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc5
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc7
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 10
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc3
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc5
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc7
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc9
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc10
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 12
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc3
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc5
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc7
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc9
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc10
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc11
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclat.Mhc12
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        else
            vvbgggg
        end
    
        isp33 = 2
        if nnjM == 2
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 4
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc3
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 6
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc3
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc5
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 8
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc3
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc5
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc7
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 10
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc3
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc5
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc7
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc9
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc10
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        elseif nnjM == 12
            nj = 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc1
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc2
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc3
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc4
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc5
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc6
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc7
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc8
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc9
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc10
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc11
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
            nj += 1
            Mhcj10[:,nj,isp33] = Mhclbt.Mhc12
            Mhcj10[1,nj,isp33] = Mhcj10[2,nj,isp33]
        else
            vvbgggg
        end
    end
    
    Mhcsj = zeros(Nt,nnjM)
    Mhcsj_fDM!(Mhcsj,ma,na, [uat ubt],[vatht vbtht],Nt,nnjM,Mhcj10)
    
    RDMhcsj = Mhcsj[2:end,:] ./ Mhcsj[1:end-1,:] .- 1
    # RDMhcsj = Mhcsj ./ Mhcsj[1,:]' .- 1
    # RDMhcsj = Mhcsj .- Mhcsj[1,:]'

    label = (1:2:nnjM)'
    # ylabel = string("Mhcs1")
    # plot(tplot[2:end],Mhcsj[2:end,1:2:nnjM],line=(3,:auto),title=title,
    #     xlabel=xlabel,ylabel=ylabel,label=label)
    # savefig(string(file_fig_file,"_Mhcs1.png"))
    ylabel = string("Mhcs")
    pMhcsj1 = plot(tplot[2:end],Mhcsj[2:end,1:2:nnjM],line=(3,:auto),
                  xlabel=xlabel,ylabel=ylabel,label=label)
    
    ylabel = string("RDMhcs")
    pRDMhcsj1 = plot(tplot[2:end-1],RDMhcsj[2:end,1:2:nnjM],line=(3,:auto),xlabel=xlabel,ylabel=ylabel,label=label)
    # savefig(string(file_fig_file,"_RDMhcs.png"))

    2
    label = (2:2:nnjM)'
    ylabel = string("DMhcs0")
    plot(tplot[2:end],Mhcsj[2:end,2:2:nnjM] .- 1,line=(3,:auto),title=title,
        xlabel=xlabel,ylabel=ylabel,label=label)
    # savefig(string(file_fig_file,"_Mhcs0.png"))
    pMhcsj0 = plot(tplot[2:end],Mhcsj[2:end,2:2:nnjM] .- 1,line=(3,:auto),
                  xlabel=xlabel,label=label)
    
    ylabel = string("RDMhcs")
    pRDMhcsj0 = plot(tplot[2:end-1],RDMhcsj[2:end,2:2:nnjM],line=(3,:auto),xlabel=xlabel,ylabel=ylabel,label=label)
    # savefig(string(file_fig_file,"_RDMhcs.png"))

    plot(plot(pMhcsj1,pMhcsj0,layout=(1,2)))
    savefig(string(file_fig_file,"_Mhcs.png"))
    pMhcsj = plot(plot(pMhcsj1,pMhcsj0,layout=(1,2)))

    plot(plot(pRDMhcsj1,pRDMhcsj0,layout=(1,2)))
    savefig(string(file_fig_file,"_RDMhcs.png"))
    pRDMhcsj = plot(plot(pRDMhcsj1,pRDMhcsj0,layout=(1,2)))

    plot(pMhcsj,pRDMhcsj,layout=(2,1))
    savefig(string(file_fig_file,"_Mhcs.png"))

    display(plot(pMhcsj,pRDMhcsj,layout=(2,1)))
else
    if maximum(nMod) == 1
        nnjM = 4
        Mhcj10 = ones(Nt,nnjM,ns)
        isp33 = 1
        if nMod[isp33] == 2
            Mhcj10[:,2,isp33] = Mhclat.Mhc2
            Mhcj10[1,2,isp33] = Mhcj10[2,2,isp33]
        end
        isp33 = 2
        if nMod[isp33] == 2
            Mhcj10[:,2,isp33] = Mhclbt.Mhc2
            Mhcj10[1,2,isp33] = Mhcj10[2,2,isp33]
        end
        
        Mhcsj = zeros(Nt,nnjM)
        # Mhcsj_fDM!(Mhcsj,na, [0uat 0ubt], [vatht vbtht],Nt,nnjM,Mhcj10;ℓ=0)
        Mhcsj_fDM!(Mhcsj,ma,na, [uat ubt],[vatht vbtht],Nt,nnjM,Mhcj10;ℓ=0)
        
        RDMhcsj = Mhcsj[2:end,:] ./ Mhcsj[1:end-1,:] .- 1
        # RDMhcsj = Mhcsj ./ Mhcsj[1,:]' .- 1
        # RDMhcsj = Mhcsj .- Mhcsj[1,:]'
        label = (0:2:2nnjM-2)'
        plot(RDMhcsj,line=(3,:auto),label=label)
        savefig(string(file_fig_file,"_RDMhcs.png"))
    
        display(plot(pub1,pvbth1,layout=(2,1)))
        
    elseif maximum(nMod) == 2
        nnjM = 3
        Mhcj10 = ones(Nt,nnjM,ns)
        isp33 = 1
        if nMod[isp33] == 2
            Mhcj10[:,2,isp33] = Mhclat.Mhc2
            Mhcj10[:,3,isp33] = Mhclat.Mhc4
            Mhcj10[1,2,isp33] = Mhcj10[2,2,isp33]
            Mhcj10[1,3,isp33] = Mhcj10[2,3,isp33]
        end
        isp33 = 2
        if nMod[isp33] == 2
            Mhcj10[:,2,isp33] = Mhclbt.Mhc2
            Mhcj10[:,3,isp33] = Mhclbt.Mhc4
            Mhcj10[1,2,isp33] = Mhcj10[2,2,isp33]
            Mhcj10[1,3,isp33] = Mhcj10[2,3,isp33]
        end
        Mhcsj = zeros(Nt,nnjM)
        Mhcsj_fDM!(Mhcsj,na, [0uat 0ubt],  [vatht vbtht],Nt,nnjM,Mhcj10)
        # Mhcsj_fDM!(Mhcsj,ma,na, [uat ubt],[vatht vbtht],Nt,nnjM)
        RDMhcsj = zeros(Nt,nnjM)
        for jj in 1:nnjM
            if norm(Mhcsj[:,jj]) ≥ epsT1000
                RDMhcsj[1:end-1,jj] = Mhcsj[2:end,jj] ./ Mhcsj[1:end-1,jj] .- 1
            else
                RDMhcsj[1:end-1,jj] = Mhcsj[2:end,jj] .- Mhcsj[1:end-1,jj]
            end
            RDMhcsj[end,jj] = RDMhcsj[end-1,jj]
        end
        # RDMhcsj = Mhcsj ./ Mhcsj[1,:]' .- 1
        label = (0:2:2nnjM-2)'
        plot(tplot,RDMhcsj,line=(3,:auto),label=label)
        
    elseif maximum(nMod) == 3
    else
        sdf
    end
end