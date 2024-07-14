
ivvv0 = 2
# [sa, dtsab]
if isfile(file_Ms_sa)
    sadtsat = CSV.File(read(file_Ms_sa)) |> DataFrame
    if missing_deal == :nothing
    elseif missing_deal == :drop
        dropmissing!(sadtsat)
    elseif missing_deal == :NaN
        replace!.(eachcol(sadtsat), missing => NaN)
    elseif missing_deal == :zero
        replace!.(eachcol(sadtsat), missing => 0.0)
    end
    unique!(sadtsat,1)            # due to t = x[1]
    tplot = sadtsat.t * (tdd / τ₀)
    Nt = length(tplot)

    if ns == 2
        sabt = sadtsat.sa + sadtsat.sb
    else
        if ns == 3
            sabt = sadtsat.sa + sadtsat.sb + sadtsat.sc
        elseif ns == 4
            sabt = sadtsat.sa + sadtsat.sb + sadtsat.sc + sadtsat.sd
        end
    end
    sabt[1] = sabt[2]
    dtsabt = deepcopy(sadtsat.dts)
    if sabt[2] < 0
        Rdtsabt = - dtsabt ./ sabt 
    else
        Rdtsabt = dtsabt ./ sabt 
    end
    logRdtsabt = (abs.(Rdtsabt) .+ epsT)
    DRdtsabt = diff(Rdtsabt) ./ diff(tplot)
    # DRdtsabt = diff(Rdtsabt) ./ Rdtsabt[2:end]

    CRDs = deepcopy(logRdtsabt[tvec])
    if is_MultiCase
        CRDsM[iCase] = deepcopy(CRDs)
    end

    tvec = tplot_min .< tplot .≤ tplot_max
    tvec[1:dkivv2] .= false
 
    label = string("DRdts")
    a = abs.(DRdtsabt[tvec[ivvv0:end]])
    if ivvv0 == 2
        a[1] = a[2]
    end
    pRDRdt = plot(tplot[tvec],a,line=(wline,:auto),
                yscale=:log10,
                label=label)

    ylabel = string("s")
    psa = plot(tplot[tvec][ivvv0:end],sadtsat.sa[tvec][ivvv0:end],line=(wline,:auto),label="sₐ",ylabel=ylabel)
    psb = plot(tplot[tvec][ivvv0:end],sadtsat.sb[tvec][ivvv0:end],line=(wline,:auto),label="sᵦ")
    if ns == 2
        psasb = plot(psa,psb,layout=(1,2))
    else
        if ns ≥ 3
            psc = plot(tplot[tvec][ivvv0:end],sadtsat.sc[tvec][ivvv0:end],line=(wline,:auto),label="sᵪ")
            if ns == 3
                psasb = plot(psa,psb,psc,layout=(1,3))
            elseif ns == 4
            end
        end
    end
    
    ylabel = string("Entropy")
    psab = plot(tplot[tvec][ivvv0:end],sabt[tvec][ivvv0:end],line=(wline,:auto),label="s",ylabel=ylabel,title=title_nv_nMod)
    psab = plot(psab, pRDRdt,layout=(1,2))

    ylabel = string("Entropy rate")
    pdts = plot(tplot[tvec][ivvv0:end],dtsabt[tvec][ivvv0:end],line=(wline,:auto),label="∂ₜs",ylabel=ylabel,xlabel=title_model)
  
    pRdts = plot(tplot[tvec][ivvv0:end],logRdtsabt[tvec][ivvv0:end],line=(wline,:auto),
                yscale=:log10,
                label="s⁻¹∂ₜs",xlabel=title_alg)
    pRdtsdts = plot(pdts,pRdts,layout=(1,2))

    puTK = display(plot(psab,psasb,pRdtsdts,layout=(3,1)))

    plot(psab,psasb,pRdtsdts,layout=(3,1))
    savefig(string(file_fig_file,"_sa.png"))
end
