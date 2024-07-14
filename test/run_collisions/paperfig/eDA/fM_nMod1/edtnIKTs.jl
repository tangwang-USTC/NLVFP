
is_logx = true

# [edtn,edtI,edtK,edtT]                  # `τ₀`
if isfile(file_Ms_Cerror)
    Cerrort = CSV.File(read(file_Ms_Cerror)) |> DataFrame
    if missing_deal == :nothing
    elseif missing_deal == :drop
        dropmissing!(Cerrort)
    elseif missing_deal == :NaN
        replace!.(eachcol(Cerrort), missing => NaN)
    elseif missing_deal == :zero
        replace!.(eachcol(Cerrort), missing => 0.0)
    end
    unique!(Cerrort,1)            # due to t = x[1]
    tplot = Cerrort.t * (tdd / τ₀)
    Nt = length(tplot)


    tvec = tplot_min .< tplot .≤ tplot_max
    t09 = diff(tplot * τ₀)
    if is_enforce_errdtnIKab
        CRDn = deepcopy(Cerrort.CRDn[tvec])
    else
        CRDn = (cumsum(Cerrort.edtna)[tvec] + cumsum(Cerrort.edtnb)[tvec] + cumsum(Cerrort.edtnc)[tvec]) / ns
    end
    if is_MultiCase
        CRDnM[iCase] = deepcopy(CRDn)
    end
    ylabel = string("Error(δₜn̂)")
    if is_logx
        a = (abs.(Cerrort.edtna[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtnaM[iCase] = deepcopy(a)
        end
        ppena = plot(tplot[tvec],a,line=(wline,:auto,linetypes[1]),label="e",
                        ylabel=ylabel,yscale=:log10,xscale=:log10)
        b = (abs.(Cerrort.edtnb[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtnbM[iCase] = deepcopy(b)
        end
        ppenb = plot!(tplot[tvec],b,line=(wline,:auto,linetypes[2]),label="D")
        if ns ≥ 3
            c = (abs.(Cerrort.edtnc[tvec]) .+ epsT01)
            if is_MultiCase
                eRdtncM[iCase] = deepcopy(c)
            end
            ppenc = plot!(tplot[tvec],c,line=(wline,:auto,linetypes[3]),label="α")
        end
        # ppenab = plot(ppena,ppenb,layout=(1,2))
        
        ylabel = string("Error(δₜÎ)")
        a = (abs.(Cerrort.edtIa[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtIaM[iCase] = deepcopy(a)
        end
        ppeIha = plot(tplot[tvec],a,line=(wline,:auto,linetypes[1]),label="e",
                          ylabel=ylabel,yscale=:log10,xscale=:log10)
        b = (abs.(Cerrort.edtIb[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtIbM[iCase] = deepcopy(b)
        end
        ppeIhb = plot!(tplot[tvec],b,line=(wline,:auto,linetypes[2]),label="D")
        if ns ≥ 3
            c = (abs.(Cerrort.edtIc[tvec]) .+ epsT01)
            if is_MultiCase
                eRdtIcM[iCase] = deepcopy(c)
            end
            ppeIc = plot!(tplot[tvec],c,line=(wline,:auto,linetypes[3]),label="α")
        end
        # ppeIab = plot(ppeIha,ppeIhb,layout=(1,2))
        
        ylabel = string("Error(δₜK̂)")
        a = (abs.(Cerrort.edtKa[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtKaM[iCase] = deepcopy(a)
        end
        ppeKha = plot(tplot[tvec],a,line=(wline,:auto,linetypes[1]),label="e",
                     ylabel=ylabel,yscale=:log10,xscale=:log10,
                     xlabel="t")
        b = (abs.(Cerrort.edtKb[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtKbM[iCase] = deepcopy(b)
        end
        ppeKhb = plot!(tplot[tvec],b,line=(wline,:auto,linetypes[2]),label="D")
        if ns ≥ 3
            c = (abs.(Cerrort.edtKc[tvec]) .+ epsT01)
            if is_MultiCase
                eRdtKcM[iCase] = deepcopy(c)
            end
            ppeKhc = plot!(tplot[tvec],c,line=(wline,:auto,linetypes[3]),label="α")
        end
        # ppeKhab = plot(ppeKha,ppeKhb,layout=(1,2))
        
        ylabel = string("Error(dtK)")
        a = (abs.(Cerrort.edtKa[tvec] .* Tat[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtTaM[iCase] = deepcopy(a)
        end
        ppeKa = plot(tplot[tvec],a,line=(wline,:auto,linetypes[1]),label="e",
                         ylabel=ylabel,yscale=:log10,xscale=:log10,
                         xlabel=title_model)
        b = (abs.(Cerrort.edtKb[tvec] .* Tbt[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtTbM[iCase] = deepcopy(b)
        end
        ppeKb = plot!(tplot[tvec],b,line=(wline,:auto,linetypes[2]),label="D")
        if ns ≥ 3
            c = (abs.(Cerrort.edtKc[tvec] .* Tct[tvec]) .+ epsT01)
            if is_MultiCase
                eRdtTcM[iCase] = deepcopy(c)
            end
            ppeKc = plot!(tplot[tvec],c,line=(wline,:auto,linetypes[3]),label="α")
        end
    
        # ylabel = string("Error(dtK)")
        # NTat = length(Tat)
        # dN = 7
        # a = (abs.(Cerrort.edtKa[tvec][1:NTat-dN] .* Tat[tvec[1:NTat]][1:NTat-dN]) .+ epsT01)
        # eRdtnTM[iCase] = deepcopy(a)
        # ppeKa = plot(tplot[tvec][1:NTat-dN],a,line=(wline,:auto,linetypes[1]),label="e",
        #                  ylabel=ylabel,yscale=:log10,xscale=:log10,
        #                  xlabel=title_model)
        # b = (abs.(Cerrort.edtKb[tvec][1:NTat-dN] .* Tbt[tvec[1:NTat]][1:NTat-dN]) .+ epsT01)
        # ppeKb = plot!(tplot[tvec][1:NTat-dN],b,line=(wline,:auto,linetypes[2]),label="D")
        # if ns ≥ 3
        #     c = (abs.(Cerrort.edtKc[tvec][1:NTat-dN] .* Tct[tvec[1:NTat]][1:NTat-dN]) .+ epsT01)
        #     ppeKc = plot!(tplot[tvec][1:NTat-dN],c,line=(wline,:auto,linetypes[3]),label="α")
        # end
        # # ppeKab = plot(ppeKa,ppeKb,layout=(1,2))
        
        ylabel = string("δₜT̂")
        a = (abs.(Cerrort.eRdtTa[tvec]) .+ epsT01)
        ppeTa = plot(tplot[tvec],a,line=(wline,:auto,linetypes[1]),label="e",
                    ylabel=ylabel,
                    xlabel="t",xscale=:log10)
        b = (abs.(Cerrort.eRdtTb[tvec]) .+ epsT01)
        ppeTb = plot!(tplot[tvec],b,line=(wline,:auto,linetypes[2]),label="D")
        if ns ≥ 3
            c = (abs.(Cerrort.eRdtTc[tvec]) .+ epsT01)
            ppeTc = plot!(tplot[tvec],c,line=(wline,:auto,linetypes[3]),label="α")
        end
    else
        a = (abs.(Cerrort.edtna[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtnaM[iCase] = deepcopy(a)
        end
        ppena = plot(tplot[tvec],a,line=(wline,:auto,linetypes[1]),label="e",
                        ylabel=ylabel,yscale=:log10)
        b = (abs.(Cerrort.edtnb[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtnbM[iCase] = deepcopy(b)
        end
        ppenb = plot!(tplot[tvec],b,line=(wline,:auto,linetypes[2]),label="D")
        if ns ≥ 3
            c = (abs.(Cerrort.edtnc[tvec]) .+ epsT01)
            if is_MultiCase
                eRdtncM[iCase] = deepcopy(c)
            end
            ppenc = plot!(tplot[tvec],c,line=(wline,:auto,linetypes[3]),label="α")
        end
        # ppenab = plot(ppena,ppenb,layout=(1,2))
        
        ylabel = string("Error(δₜÎ)")
        a = (abs.(Cerrort.edtIa[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtIaM[iCase] = deepcopy(a)
        end
        ppeIha = plot(tplot[tvec],a,line=(wline,:auto,linetypes[1]),label="e",
                          ylabel=ylabel,yscale=:log10)
        b = (abs.(Cerrort.edtIb[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtIbM[iCase] = deepcopy(b)
        end
        ppeIhb = plot!(tplot[tvec],b,line=(wline,:auto,linetypes[2]),label="D")
        if ns ≥ 3
            c = (abs.(Cerrort.edtIc[tvec]) .+ epsT01)
            if is_MultiCase
                eRdtIcM[iCase] = deepcopy(c)
            end
            ppeIc = plot!(tplot[tvec],c,line=(wline,:auto,linetypes[3]),label="α")
        end
        # ppeIab = plot(ppeIha,ppeIhb,layout=(1,2))
        
        ylabel = string("Error(δₜK̂)")
        a = (abs.(Cerrort.edtKa[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtKaM[iCase] = deepcopy(a)
        end
        ppeKha = plot(tplot[tvec],a,line=(wline,:auto,linetypes[1]),label="e",
                     ylabel=ylabel,yscale=:log10,
                     xlabel="t")
        b = (abs.(Cerrort.edtKb[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtKbM[iCase] = deepcopy(b)
        end
        ppeKhb = plot!(tplot[tvec],b,line=(wline,:auto,linetypes[2]),label="D")
        if ns ≥ 3
            c = (abs.(Cerrort.edtKc[tvec]) .+ epsT01)
            if is_MultiCase
                eRdtKcM[iCase] = deepcopy(c)
            end
            ppeKhc = plot!(tplot[tvec],c,line=(wline,:auto,linetypes[3]),label="α")
        end
        # ppeKhab = plot(ppeKha,ppeKhb,layout=(1,2))
        
        ylabel = string("Error(dtK)")
        a = (abs.(Cerrort.edtKa[tvec] .* Tat[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtTaM[iCase] = deepcopy(a)
        end
        ppeKa = plot(tplot[tvec],a,line=(wline,:auto,linetypes[1]),label="e",
                         ylabel=ylabel,yscale=:log10,
                         xlabel=title_model)
        b = (abs.(Cerrort.edtKb[tvec] .* Tbt[tvec]) .+ epsT01)
        if is_MultiCase
            eRdtTbM[iCase] = deepcopy(b)
        end
        ppeKb = plot!(tplot[tvec],b,line=(wline,:auto,linetypes[2]),label="D")
        if ns ≥ 3
            c = (abs.(Cerrort.edtKc[tvec] .* Tct[tvec]) .+ epsT01)
            if is_MultiCase
                eRdtTcM[iCase] = deepcopy(c)
            end
            ppeKc = plot!(tplot[tvec],c,line=(wline,:auto,linetypes[3]),label="α")
        end
    
        # ylabel = string("Error(dtK)")
        # NTat = length(Tat)
        # dN = 7
        # a = (abs.(Cerrort.edtKa[tvec][1:NTat-dN] .* Tat[tvec[1:NTat]][1:NTat-dN]) .+ epsT01)
        # eRdtnTM[iCase] = deepcopy(a)
        # ppeKa = plot(tplot[tvec][1:NTat-dN],a,line=(wline,:auto,linetypes[1]),label="e",
        #                  ylabel=ylabel,yscale=:log10,
        #                  xlabel=title_model)
        # b = (abs.(Cerrort.edtKb[tvec][1:NTat-dN] .* Tbt[tvec[1:NTat]][1:NTat-dN]) .+ epsT01)
        # ppeKb = plot!(tplot[tvec][1:NTat-dN],b,line=(wline,:auto,linetypes[2]),label="D")
        # if ns ≥ 3
        #     c = (abs.(Cerrort.edtKc[tvec][1:NTat-dN] .* Tct[tvec[1:NTat]][1:NTat-dN]) .+ epsT01)
        #     ppeKc = plot!(tplot[tvec][1:NTat-dN],c,line=(wline,:auto,linetypes[3]),label="α")
        # end
        # # ppeKab = plot(ppeKa,ppeKb,layout=(1,2))
        
        ylabel = string("δₜT̂")
        a = (abs.(Cerrort.eRdtTa[tvec]) .+ epsT01)
        ppeTa = plot(tplot[tvec],a,line=(wline,:auto,linetypes[1]),label="e",
                    ylabel=ylabel,
                    xlabel="t")
        b = (abs.(Cerrort.eRdtTb[tvec]) .+ epsT01)
        ppeTb = plot!(tplot[tvec],b,line=(wline,:auto,linetypes[2]),label="D")
        if ns ≥ 3
            c = (abs.(Cerrort.eRdtTc[tvec]) .+ epsT01)
            ppeTc = plot!(tplot[tvec],c,line=(wline,:auto,linetypes[3]),label="α")
        end
    end
    # ppeTab = plot(ppeTa,ppeTb,layout=(1,2))

    # puTK = display(plot(ppenab,ppeIab,ppeKab,ppeTab,layout=(4,1)))
    # plot(ppenab,ppeIab,ppeKab,ppeTab,layout=(4,1))
    display(plot(ppenb,ppeIhb,ppeKhb,ppeTb,layout=(2,2)))
    
    plot(ppenb,ppeIhb,ppeKhb,ppeTb,layout=(2,2))
    if is_logx
        savefig(string(file_fig_file,"_CerrTaTblogx.png"))
    
        display(plot(ppenb,ppeKhb,layout=(2,1)))
        
        plot(ppenb,ppeKhb,layout=(2,1))
        savefig(string(file_fig_file,"_CerrnKTaTblogx.png"))
    else
        savefig(string(file_fig_file,"_CerrTaTb.png"))
    
        display(plot(ppenb,ppeKhb,layout=(2,1)))
        
        plot(ppenb,ppeKhb,layout=(2,1))
        savefig(string(file_fig_file,"_CerrnKTaTb.png"))
    end
end
