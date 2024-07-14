
# [edtn,edtI,edtK,edtT]                  # `τ₀`
if isfile(file_Ms_Cerror)
    layoutab = (2,1)

    xlabel = string("t")

    title = string("Error(δₜn̂)")
    if ns == 2
        if is_name_ab
            ylabel = string("a")
        else
            ylabel = string(spices0[1])
        end
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        Mpena = plot(tplotM[iCasep],eRdtnaM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10,
                    title=title)
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            Mpena = plot!(tplotM[iCasep],eRdtnaM[iCasep],line=(wline,:auto),label=label)
        end

        if is_name_ab
            ylabel = string("b")
        else
            ylabel = string(spices0[2])
        end
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        Mpenb = plot(tplotM[iCasep],eRdtnbM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            Mpenb = plot!(tplotM[iCasep],eRdtnbM[iCasep],line=(wline,:auto),label=label)
        end
    elseif ns == 3
        if is_name_ab
            ylabel = string("a")
        else
            ylabel = string(spices0[1])
        end
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        Mpena = plot(tplotM[iCasep],eRdtnaM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            Mpena = plot!(tplotM[iCasep],eRdtnaM[iCasep],line=(wline,:auto),label=label)
        end

        if is_name_ab
            ylabel = string("b")
        else
            ylabel = string(spices0[2])
        end
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        Mpenb = plot(tplotM[iCasep],eRdtnbM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            Mpenb = plot!(tplotM[iCasep],eRdtnbM[iCasep],line=(wline,:auto),label=label)
        end

        if is_name_ab
            ylabel = string("c")
        else
            ylabel = string(spices0[3])
        end
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        Mpenc = plot(tplotM[iCasep],eRdtncM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,:auto),label=label)
        end
    end
    Mpenab = plot(Mpena,Mpenb,layout=layoutab)
    
    ylabel = string("error_δₜÎ")
    if ns == 2
        ylabel = string("Error(δₜÎa)")
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeIa = plot(tplotM[iCasep],eRdtIaM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeIa = plot!(tplotM[iCasep],eRdtIaM[iCasep],line=(wline,:auto),label=label)
        end

        ylabel = string("Error(δₜÎb)")
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeIb = plot(tplotM[iCasep],eRdtIbM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeIb = plot!(tplotM[iCasep],eRdtIbM[iCasep],line=(wline,:auto),label=label)
        end
    elseif ns == 3
        ylabel = string("Error(δₜÎa)")
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeIa = plot(tplotM[iCasep],eRdtIaM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeIa = plot!(tplotM[iCasep],eRdtIaM[iCasep],line=(wline,:auto),label=label)
        end

        ylabel = string("Error(δₜÎb)")
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeIb = plot(tplotM[iCasep],eRdtIbM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeIb = plot!(tplotM[iCasep],eRdtIbM[iCasep],line=(wline,:auto),label=label)
        end

        ylabel = string("Error(δₜÎc)")
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        Mpenc = plot(tplotM[iCasep],eRdtncM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,:auto),label=label)
        end
    end
    
    title = string("Error(δₜK̂)")
    if ns == 2
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeKa = plot(tplotM[iCasep],eRdtKaM[iCasep],
                    line=(wline,:auto),label=label,
                    yscale=:log10,
                    title=title)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeKa = plot!(tplotM[iCasep],eRdtKaM[iCasep],line=(wline,:auto),label=label)
        end

        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeKb = plot(tplotM[iCasep],eRdtKbM[iCasep],
                    line=(wline,:auto),label=label,
                    yscale=:log10,
                    xlabel=xlabel)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeKb = plot!(tplotM[iCasep],eRdtKbM[iCasep],line=(wline,:auto),label=label)
        end
    elseif ns == 3
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeKa = plot(tplotM[iCasep],eRdtKaM[iCasep],
                    line=(wline,:auto),label=label,
                    yscale=:log10,
                    title=title)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeKa = plot!(tplotM[iCasep],eRdtKaM[iCasep],line=(wline,:auto),label=label)
        end

        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeKb = plot(tplotM[iCasep],eRdtKbM[iCasep],
                    line=(wline,:auto),label=label,
                    yscale=:log10,
                    xlabel=xlabel)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeKb = plot!(tplotM[iCasep],eRdtKbM[iCasep],line=(wline,:auto),label=label)
        end

        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        Mpenc = plot(tplotM[iCasep],eRdtncM[iCasep],
                    line=(wline,:auto),label=label,
                    yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,:auto),label=label)
        end
    end
    MpeKab = plot(MpeKa,MpeKb,layout=layoutab)
    
    ylabel = string("error_δₜT̂")
    if ns == 2
        ylabel = string("error_δₜT̂a")
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeTa = plot(tplotM[iCasep],eRdtTaM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeTa = plot!(tplotM[iCasep],eRdtTaM[iCasep],line=(wline,:auto),label=label)
        end

        ylabel = string("error_δₜT̂b")
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeTb = plot(tplotM[iCasep],eRdtTbM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeTb = plot!(tplotM[iCasep],eRdtTbM[iCasep],line=(wline,:auto),label=label)
        end
    elseif ns == 3
        ylabel = string("error_δₜT̂a")
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeTa = plot(tplotM[iCasep],eRdtTaM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeTa = plot!(tplotM[iCasep],eRdtTaM[iCasep],line=(wline,:auto),label=label)
        end

        ylabel = string("error_δₜT̂b")
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        MpeTb = plot(tplotM[iCasep],eRdtTbM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            MpeTb = plot!(tplotM[iCasep],eRdtTbM[iCasep],line=(wline,:auto),label=label)
        end

        ylabel = string("error_δₜT̂c")
        iCasep = 1
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        Mpenc = plot(tplotM[iCasep],eRdtncM[iCasep],
                    line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
        
        for iCasep in 2:NCasep
            label = string(L"1 / \Delta t=",vec_sweep[iCasep])
            Mpenc = plot!(tplotM[iCasep],eRdtncM[iCasep],line=(wline,:auto),label=label)
        end
    end

    display(plot(Mpena,MpeIa,MpeKa,MpeTa,layout=(2,2)))
    plot(Mpena,MpeIa,MpeKa,MpeTa,layout=(2,2))
    savefig(string(file_fig_file,"_CerrTaMdt.png"))

    display(plot(Mpenb,MpeIb,MpeKb,MpeTb,layout=(2,2)))
    plot(Mpenb,MpeIb,MpeKb,MpeTb,layout=(2,2))
    savefig(string(file_fig_file,"_CerrTbMdt.png"))

    display(plot(Mpenab,MpeKab,layout=(1,2)))
    plot(Mpenab,MpeKab,layout=(1,2))
    savefig(string(file_fig_file,"_CerrTaTbMdt.png"))
end
