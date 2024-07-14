
missing_deal = :NaN        # [:nothing, :drop, :NaN, :zero]
tplot_minTaTb = 1e-5
tplot_maxTaTb = 7e3
# [T]
if isfile(file_TaTb)
    TaTbt = CSV.File(read(file_TaTb)) |> DataFrame
    
    if missing_deal == :nothing
    elseif missing_deal == :drop
        dropmissing!(TaTbt)
    elseif missing_deal == :NaN
        replace!.(eachcol(TaTbt), missing => NaN)
    elseif missing_deal == :zero
        replace!.(eachcol(TaTbt), missing => 0.0)
    end
    unique!(TaTbt,1)            # due to t = x[1]
    rename!(TaTbt, TaTb_name)
    
    if unit_type == :PS
        # tplotTaTb = (TaTbt.t / tdd) / τ₀
        tplotTaTb = TaTbt.t / τ₀
    else
        tplotTaTb = TaTbt.t / τ₀
    end
    tvecTaTb = tplot_minTaTb .< tplotTaTb .≤ tplot_maxTaTb
    NtTaTb = length(tplotTaTb)

    DtTaTb = diff(tvecTaTb)
    title = string(unit_type)
    pptTaTb = plot(tvecTaTb[2:end],DtTaTb,ylabel="Dt",label=string(NtTaTb),title=title)

    wlineTaTb = 3
    if ns == 2
        if unit_type == :PS
            TatTaTb = TaTbt.Ta / T_unit
            TbtTaTb = TaTbt.Tb / T_unit
        else
            if unit_type == :Tk
                TatTaTb = TaTbt.Ta
                TbtTaTb = TaTbt.Tb
            else
                TatTaTb = TaTbt.Ta / 1000
                TbtTaTb = TaTbt.Tb / 1000
                # if unit_type == :CGS
                #     TatTaTb = TaTbt.Ta / 1000
                #     TbtTaTb = TaTbt.Tb / 1000
                # elseif unit_type == :SI
                #     TatTaTb = TaTbt.Ta / 1000
                #     TbtTaTb = TaTbt.Tb / 1000
                # end
            end
        end
    
        DTabtTaTb = TatTaTb - TbtTaTb
        RDTabtTaTb = DTabtTaTb ./ (TatTaTb + TbtTaTb)
    
        title = string("TaTb")
        ylabel = string("T [keV]")
        ppTa = plot(tplotTaTb[tvecTaTb],TatTaTb[tvecTaTb],line=(wlineTaTb,:auto),label="a",ylabel=ylabel,title=title)
        ppTb = plot!(tplotTaTb[tvecTaTb],TbtTaTb[tvecTaTb],line=(wlineTaTb,:auto),label="b")
        
        ylabel = string("DT")
        ppRDTab = plot(tplotTaTb[tvecTaTb],(abs.(RDTabtTaTb[tvecTaTb])),line=(wlineTaTb,:auto),
                                ylabel=ylabel,yscale=:log10)

        ppuTK = display(plot(pptTaTb,ppTa,ppRDTab,layout=(3,1)))
    
        plot(pptTaTb,ppTa,ppRDTab,layout=(3,1))
        savefig(string(fileTaTb_fig_file,"_TaTb.png"))
    else
        if ns == 3
            TatTaTb = TaTbt.Ta
            TbtTaTb = TaTbt.Tb
            Tct = TaTbt.Tc
        
            DTabtTaTb = TatTaTb - TbtTaTb
            DTact = TatTaTb - Tct
            DTbct = TbtTaTb - Tct
            RDTabtTaTb = DTabtTaTb ./ (TatTaTb + TbtTaTb)
            RDTact = DTact ./ (TatTaTb + Tct)
            RDTbct = DTbct ./ (Tct + TbtTaTb)
        
            title = string("TaTb")
            ylabel = string("T [keV]")
            ppTa = plot(tplotTaTb[tvecTaTb],TatTaTb[tvecTaTb],line=(wlineTaTb,:auto),label="a",ylabel=ylabel,title=title)
            ppTb = plot!(tplotTaTb[tvecTaTb],TbtTaTb[tvecTaTb],line=(wlineTaTb,:auto),label="b")
            ppTc = plot!(tplotTaTb[tvecTaTb],Tct[tvecTaTb],line=(wlineTaTb,:auto),label="c")
            
            ylabel = string("ΔT")
            ppRDTab = plot(tplotTaTb[tvecTaTb],(abs.(RDTabtTaTb[tvecTaTb])),line=(wlineTaTb,:auto),label="ab",
                                    ylabel=ylabel,yscale=:log10)
            ppRDTac = plot!(tplotTaTb[tvecTaTb],(abs.(RDTact[tvecTaTb])),line=(wlineTaTb,:auto),label="ac")
            ppRDTbc = plot!(tplotTaTb[tvecTaTb],(abs.(RDTbct[tvecTaTb])),line=(wlineTaTb,:auto),label="bc")
    
            puTK = display(plot(pptTaTb,ppTa,ppRDTab,layout=(3,1)))
        
            plot(pptTaTb,ppTa,ppRDTab,layout=(3,1))
            savefig(string(fileTaTb_fig_file,"_TaTb.png"))
        elseif ns == 4
            edfghj
        end
    end
end