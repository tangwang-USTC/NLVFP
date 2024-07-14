# gr()

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
    tvec[1:dkivv2] .= false

    ylabel = string("Error(δₜn̂)")
    a = (abs.(Cerrort.edtna[tvec]) .+ epsT01)
    pena = plot(tplot[tvec],a,line=(wline,:auto),label="a",
                    ylabel=ylabel,yscale=:log10,
                    title=title_TEk)
    b = (abs.(Cerrort.edtnb[tvec]) .+ epsT01)
    penb = plot!(tplot[tvec],b,line=(wline,:auto),label="b")
    if ns ≥ 3
        c = (abs.(Cerrort.edtnc[tvec]) .+ epsT01)
        penc = plot!(tplot[tvec],c,line=(wline,:auto),label="c")
    end
    # penab = plot(pena,penb,layout=(1,2))
    
    ylabel = string("Error(δₜÎ)")
    a = (abs.(Cerrort.edtIa[tvec]) .+ epsT01)
    peIha = plot(tplot[tvec],a,line=(wline,:auto),label="a",
                    ylabel=ylabel,yscale=:log10,
                    title=title_nv_nMod)
    b = (abs.(Cerrort.edtIb[tvec]) .+ epsT01)
    peIhb = plot!(tplot[tvec],b,line=(wline,:auto),label="b")
    if ns ≥ 3
        c = (abs.(Cerrort.edtIc[tvec]) .+ epsT01)
        peIc = plot!(tplot[tvec],c,line=(wline,:auto),label="c")
    end
    # peIab = plot(peIha,peIhb,layout=(1,2))
    
    ylabel = string("Error(δₜK̂)")
    a = (abs.(Cerrort.edtKa[tvec]) .+ epsT01)
    peKha = plot(tplot[tvec],a,line=(wline,:auto),label="a",
                    ylabel=ylabel,yscale=:log10,
                    xlabel=title_model)
    b = (abs.(Cerrort.edtKb[tvec]) .+ epsT01)
    peKhb = plot!(tplot[tvec],b,line=(wline,:auto),label="b")
    if ns ≥ 3
        c = (abs.(Cerrort.edtKc[tvec]) .+ epsT01)
        peKhc = plot!(tplot[tvec],c,line=(wline,:auto),label="c")
    end
    # peKhab = plot(peKha,peKhb,layout=(1,2))
    
    ylabel = string("Error(dtK)")
    NTat = length(Tat[tvec])
    dN = 7
    a = (abs.(Cerrort.edtKa[tvec][1:NTat-dN] .* Tat[tvec][1:NTat-dN]) .+ epsT01)
    peKa = plot(tplot[tvec][1:NTat-dN],a,line=(wline,:auto),label="a",
                    ylabel=ylabel,yscale=:log10,
                    xlabel=title_model)
    b = (abs.(Cerrort.edtKb[tvec][1:NTat-dN] .* Tbt[tvec][1:NTat-dN]) .+ epsT01)
    peKb = plot!(tplot[tvec][1:NTat-dN],b,line=(wline,:auto),label="b")
    if ns ≥ 3
        c = (abs.(Cerrort.edtKc[tvec][1:NTat-dN] .* Tct[tvec][1:NTat-dN]) .+ epsT01)
        peKc = plot!(tplot[tvec][1:NTat-dN],c,line=(wline,:auto),label="c")
    end
    # peKab = plot(peKa,peKb,layout=(1,2))
    
    ylabel = string("Error(δₜT̂)")
    a = (abs.(Cerrort.eRdtTa[tvec]) .+ epsT01)
    peTa = plot(tplot[tvec],a,line=(wline,:auto),label="a",
                    ylabel=ylabel,yscale=:log10,
                    xlabel=title_alg)
    b = (abs.(Cerrort.eRdtTb[tvec]) .+ epsT01)
    peTb = plot!(tplot[tvec],b,line=(wline,:auto),label="b")
    if ns ≥ 3
        c = (abs.(Cerrort.eRdtTc[tvec]) .+ epsT01)
        peTc = plot!(tplot[tvec],c,line=(wline,:auto),label="c")
    end
    # peTab = plot(peTa,peTb,layout=(1,2))

    # puTK = display(plot(penab,peIab,peKab,peTab,layout=(4,1)))
    # plot(penab,peIab,peKab,peTab,layout=(4,1))
    puTK = display(plot(penb,peIhb,peKhb,peTb,layout=(2,2)))
    plot(penb,peIhb,peKhb,peTb,layout=(2,2))
    savefig(string(file_fig_file,"_Cerror.png"))
end
