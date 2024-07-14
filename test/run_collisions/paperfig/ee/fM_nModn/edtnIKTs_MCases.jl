# pyplot()
# pythonplot()
# [edtn,edtI,edtK,edtT]                  # `τ₀`
if isfile(file_Ms_Cerror)

    linetypes =  [:solid, :dot,  :dash,   :dashdot, :dashdotdot]
    linecolors = [:orange, :green,:blue,   :yellow,  :red ]
    linemakers = [:circle,:star5,:diamond,:xcross,  :rtriangle ]
    layoutab = (2,1)

    xlabel = string("t")

    title = string("Error(δₜn̂)")
    if ns == 2
        ylabel = string("a")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        Mpena = plot(tplotM[iCasep],eRdtnaM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10,
                    title=title)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        Mpena = plot!(tplotM[iCasep],eRdtnaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        Mpena = plot!(tplotM[iCasep],eRdtnaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        ylabel = string("b")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        Mpenb = plot(tplotM[iCasep],eRdtnbM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        Mpenb = plot!(tplotM[iCasep],eRdtnbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        Mpenb = plot!(tplotM[iCasep],eRdtnbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    elseif ns == 3
        ylabel = string("a")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        Mpena = plot(tplotM[iCasep],eRdtnaM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        Mpena = plot!(tplotM[iCasep],eRdtnaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        Mpena = plot!(tplotM[iCasep],eRdtnaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        ylabel = string("b")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        Mpenb = plot(tplotM[iCasep],eRdtnbM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        Mpenb = plot!(tplotM[iCasep],eRdtnbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        Mpenb = plot!(tplotM[iCasep],eRdtnbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        ylabel = string("c")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        Mpenc = plot(tplotM[iCasep],eRdtncM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    end
    Mpenab = plot(Mpena,Mpenb,layout=layoutab)
    
    ylabel = string("Error(δₜÎ)")
    if ns == 2
        ylabel = string("Error(δₜÎa)")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeIa = plot(tplotM[iCasep],eRdtIaM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeIa = plot!(tplotM[iCasep],eRdtIaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeIa = plot!(tplotM[iCasep],eRdtIaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        ylabel = string("Error(δₜÎb)")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeIb = plot(tplotM[iCasep],eRdtIbM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeIb = plot!(tplotM[iCasep],eRdtIbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeIb = plot!(tplotM[iCasep],eRdtIbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    elseif ns == 3
        ylabel = string("Error(δₜÎa)")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeIa = plot(tplotM[iCasep],eRdtIaM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeIa = plot!(tplotM[iCasep],eRdtIaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeIa = plot!(tplotM[iCasep],eRdtIaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        ylabel = string("Error(δₜÎb)")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeIb = plot(tplotM[iCasep],eRdtIbM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeIb = plot!(tplotM[iCasep],eRdtIbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeIb = plot!(tplotM[iCasep],eRdtIbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        ylabel = string("Error(δₜÎc)")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        Mpenc = plot(tplotM[iCasep],eRdtncM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    end
    
    title = string("Error(δₜK̂)")
    if ns == 2
        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeKa = plot(tplotM[iCasep],eRdtKaM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    yscale=:log10,
                    title=title)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeKa = plot!(tplotM[iCasep],eRdtKaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeKa = plot!(tplotM[iCasep],eRdtKaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeKb = plot(tplotM[iCasep],eRdtKbM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    yscale=:log10,
                    xlabel=xlabel)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeKb = plot!(tplotM[iCasep],eRdtKbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeKb = plot!(tplotM[iCasep],eRdtKbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    elseif ns == 3
        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeKa = plot(tplotM[iCasep],eRdtKaM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    yscale=:log10,
                    title=title)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeKa = plot!(tplotM[iCasep],eRdtKaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeKa = plot!(tplotM[iCasep],eRdtKaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeKb = plot(tplotM[iCasep],eRdtKbM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    yscale=:log10,
                    xlabel=xlabel)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeKb = plot!(tplotM[iCasep],eRdtKbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeKb = plot!(tplotM[iCasep],eRdtKbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        iCasep = 1
        label = string(nnvocpM[iCasep])
        Mpenc = plot(tplotM[iCasep],eRdtncM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    end
    MpeKab = plot(MpeKa,MpeKb,layout=layoutab)
    
    ylabel = string("δₜT̂")
    if ns == 2
        ylabel = string("δₜT̂a")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeTa = plot(tplotM[iCasep],eRdtTaM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeTa = plot!(tplotM[iCasep],eRdtTaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeTa = plot!(tplotM[iCasep],eRdtTaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        ylabel = string("δₜT̂b")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeTb = plot(tplotM[iCasep],eRdtTbM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeTb = plot!(tplotM[iCasep],eRdtTbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeTb = plot!(tplotM[iCasep],eRdtTbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    elseif ns == 3
        ylabel = string("δₜT̂a")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeTa = plot(tplotM[iCasep],eRdtTaM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeTa = plot!(tplotM[iCasep],eRdtTaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeTa = plot!(tplotM[iCasep],eRdtTaM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        ylabel = string("δₜT̂b")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        MpeTb = plot(tplotM[iCasep],eRdtTbM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        MpeTb = plot!(tplotM[iCasep],eRdtTbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        MpeTb = plot!(tplotM[iCasep],eRdtTbM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

        ylabel = string("δₜT̂c")
        iCasep = 1
        label = string(nnvocpM[iCasep])
        Mpenc = plot(tplotM[iCasep],eRdtncM[iCasep],
                    line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
        iCasep = 2
        label = string(nnvocpM[iCasep])
        Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
        iCasep = 3
        label = string(nnvocpM[iCasep])
        Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    end

    display(plot(Mpena,MpeIa,MpeKa,MpeTa,layout=(2,2)))
    plot(Mpena,MpeIa,MpeKa,MpeTa,layout=(2,2))
    savefig(string(file_fig_file,"_CerrTaM.png"))

    display(plot(Mpenb,MpeIb,MpeKb,MpeTb,layout=(2,2)))
    plot(Mpenb,MpeIb,MpeKb,MpeTb,layout=(2,2))
    savefig(string(file_fig_file,"_CerrTbM.png"))

    display(plot(Mpenab,MpeKab,layout=(1,2)))
    plot(Mpenab,MpeKab,layout=(1,2))
    savefig(string(file_fig_file,"_CerrTaTbM.png"))
end
