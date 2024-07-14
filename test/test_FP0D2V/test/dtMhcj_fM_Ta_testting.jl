
if maximum(nMod) == 1
    njMs = 5
    
    Mhcsj = zeros(Nt,njMs)
    Mhcsj_fM!(Mhcsj,na,[Tat Tbt],Nt,njMs)
    # Mhcsj_fM!(Mhcsj,ma,na,[Tat Tbt],Nt,njMs)
    
    RDMhcsj = Mhcsj[2:end,:] ./ Mhcsj[1:end-1,:] .- 1
    # RDMhcsj = Mhcsj ./ Mhcsj[1,:]' .- 1
    # RDMhcsj = Mhcsj .- Mhcsj[1,:]'
    label = (0:2:2njMs-2)'
    plot(RDMhcsj,line=(3,:auto),label=label)
    
elseif maximum(nMod) == 2

    njMs = 3
    Mhcj = ones(Nt,njMs,ns)
    isp33 = 1
    if nMod[isp33] == 2
        Mhcj[:,2,isp33] = Mhcslat.Mhcs2
        Mhcj[:,3,isp33] = Mhcslat.Mhcs4
    end
    isp33 = 2
    if nMod[isp33] == 2
        Mhcj[:,2,isp33] = Mhcslbt.Mhcs2
        Mhcj[:,3,isp33] = Mhcslbt.Mhcs4
    end
    Mhcsj = zeros(Nt,njMs)
    Mhcsj_fM!(Mhcsj,na,[Tat Tbt],Nt,njMs,Mhcj)
    # Mhcsj_fM!(Mhcsj,ma,na,[Tat Tbt],Nt,njMs)
    
    RDMhcsj = Mhcsj[2:end,:] ./ Mhcsj[1:end-1,:] .- 1
    # RDMhcsj = Mhcsj ./ Mhcsj[1,:]' .- 1
    # RDMhcsj = Mhcsj .- Mhcsj[1,:]'
    label = (0:2:2njMs-2)'
    plot(RDMhcsj[2:end,:],line=(3,:auto),label=label)
    
elseif maximum(nMod) == 3
else
    sdf
end