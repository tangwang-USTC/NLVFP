
N_Brag = 20
wlineT2 = 2
is_scatter = false
xlabel = L"t"
tscale = 1.4
tvecp = tplot_min .< tplot / tscale .< tplot_max
tvecTaTbp = tplotTaTb .< tplot_max

ylabel = L"T"
if is_name_ab
    if NKk == 1
        label = L"a,L01jd2"
    else
        label = string(L"a,N_K=",NKk)
    end
    pTa = plot(tplot[tvecp]/ tscale,Tat[tvecp],line=line=(wlineT2,:auto),label=label,ylabel=ylabel)
    if NKk == 1
        label = L"b,L01jd2"
    else
        label = string(L"b,N_K=",NKk)
    end
    pTb = plot!(tplot[tvecp]/ tscale,Tbt[tvecp],line=(wlineT2,:auto),label=label)
else
    if NKk == 1
        label = string(spices0[1],L",L01jd2")
    else
        label = string(spices0[1],L",N_K=",NKk)
    end
    pTa = plot(tplot[tvecp]/ tscale,Tat[tvecp],line=line=(wlineT2,:auto),label=label,ylabel=ylabel)
    if NKk == 1
        label = string(spices0[2],L",L01jd2")
    else
        label = string(spices0[2],L",N_K=",NKk)
    end
    pTb = plot!(tplot[tvecp]/ tscale,Tbt[tvecp],line=(wlineT2,:auto),label=label)
end

if MultiType == :NK
    if NK_vec[1] == 1
        if iCase == 1
            TabNK1_vec[1] = deepcopy(tplot[tvecp])
            TabNK1_vec[2] = deepcopy(Tat[tvecp])
            TabNK1_vec[3] = deepcopy(Tbt[tvecp])
            TabNK1_vec[4] = deepcopy(RDTabt[tvecp])
        elseif iCase == NCase
            if is_name_ab
                label = L"a,L01jd2"
                pTa = plot!(TabNK1_vec[1],TabNK1_vec[2],line=line=(wlineT2,:auto),label=label,ylabel=ylabel)
                label = L"b,L01jd2"
                pTb = plot!(TabNK1_vec[1],TabNK1_vec[3],line=(wlineT2,:auto),label=label)
            else
                label = string(spices0[1],L",L01jd2")
                pTa = plot!(TabNK1_vec[1],TabNK1_vec[2],line=line=(wlineT2,:auto),label=label,ylabel=ylabel)
                label = string(spices0[2],L",L01jd2")
                pTb = plot!(TabNK1_vec[1],TabNK1_vec[3],line=(wlineT2,:auto),label=label)
            end
        end
    end
end


if is_sol_Brag
    NtTaTb2 = length(tplotTaTb[tvecTaTbp])
    if is_scatter
        Nt10 = max(1,round(Int64,NtTaTb2 / N_Brag))
        vec10 = 1:Nt10:NtTaTb2
        if is_name_ab
            ppTa = scatter!(tplotTaTb[tvecTaTbp][vec10],TatTaTb[tvecTaTbp][vec10],m=:star5,label=L"a,Brg.")
            ppTb = scatter!(tplotTaTb[tvecTaTbp][vec10],TbtTaTb[tvecTaTbp][vec10],m=:circle,label=L"b,Brg.")
        else
            label = string(spices0[1],L",Brg.")
            ppTa = scatter!(tplotTaTb[tvecTaTbp][vec10],TatTaTb[tvecTaTbp][vec10],m=:star5,label=label)
            label = string(spices0[2],L",Brg.")
            ppTb = scatter!(tplotTaTb[tvecTaTbp][vec10],TbtTaTb[tvecTaTbp][vec10],m=:circle,label=label)
        end
    else
        if is_name_ab
            ppTa = plot!(tplotTaTb[tvecTaTbp],TatTaTb[tvecTaTbp],line=line=(wlineT2,:auto,:gray),label=L"a,Brg.")
            ppTb = plot!(tplotTaTb[tvecTaTbp],TbtTaTb[tvecTaTbp],line=line=(wlineT2,:auto,:black),label=L"b,Brg.")
        else
            label = string(spices0[1],L",Brg.")
            ppTa = plot!(tplotTaTb[tvecTaTbp],TatTaTb[tvecTaTbp],line=line=(wlineT2,:auto,:gray),label=label)
            label = string(spices0[2],L",Brg.")
            ppTb = plot!(tplotTaTb[tvecTaTbp],TbtTaTb[tvecTaTbp],line=line=(wlineT2,:auto,:black),label=label)
        end
    end
end

# lens!([x0, x9], [y0, y9], inset = (1, bbox(0.5, 0.0, 0.4, 0.4)))
################################################################################# (TD,TT)= (10,20)
Tmax = max(T0[1],T0[2]) - 1.45
# lens!([0.02, 0.05], [Tmax-0.2, Tmax], inset = (1, bbox(0.2, 0.44, 0.25, 0.44)))
dtlens = 0.12                     
dylocal = 0.5
lens!([0.02, 0.1] .+ dtlens, [Tmax-0.2, Tmax], inset = (1, bbox(0.34, 0.05, 0.22, 0.25)))


# ################################################################################## (TD,TT)= (10,100)
# Tmax = max(T0[1],T0[2]) - 11.5
# # lens!([0.02, 0.05], [Tmax-0.2, Tmax], inset = (1, bbox(0.2, 0.44, 0.25, 0.44)))
# dtlens = 0.12                     
# lens!([0.02, 0.1] .+ dtlens, [Tmax-0.2, Tmax], inset = (1, bbox(0.55, 0.5, 0.25, 0.35)))
# # lens!([0.03, 0.05], [Tmax-0.28, Tmax-0.1-0.2], inset = (1, bbox(0.5, 0.01, 0.25, 0.4)))

ylabel = L"\Delta T"
if NKk == 1
    label = L"L01jd2"
else
    label = string(L"N_K=",NKk)
end
pRDTabt = plot(tplot[tvecp]/ tscale,abs.(RDTabt[tvecp]),line=line=(wlineT2,:auto),label=label,
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel)

if MultiType == :NK
    if NK_vec[1] == 1
        if iCase == NCase
            label = L"L01jd2"
            pRDTabt = plot!(TabNK1_vec[1],abs.(TabNK1_vec[4]),line=line=(wlineT2,:auto),label=label)
        end
    end
end

if is_sol_Brag
    if is_scatter
        ppRDTab = scatter!(tplotTaTb[tvecTaTbp][vec10],(abs.(RDTabtTaTb[tvecTaTbp][vec10])),label=L"Brag.",
                            ylabel=ylabel,yscale=:log10)
        # lens!([4.75,4.7501], [1e-5, 7e-6], inset = (1, bbox(0.5,0,0.3,0.5)))
    else
        ppRDTab = plot!(tplotTaTb[tvecTaTbp],(abs.(RDTabtTaTb[tvecTaTbp])),line=line=(wlineT2,:auto,:black),label=L"Brag.",
                            ylabel=ylabel,yscale=:log10)
    end
end
# lens!([0.02, 0.05], [0.001, 0.01], inset = (1, bbox(0.1, 0.44, 0.25, 0.44)))

display(plot(pTa,pRDTabt,layout=(2,1)))
plot(pTa,pRDTabt,layout=(2,1))
if MultiType == :NK
    savefig(string(file_fig_file,"_TaTb2_NK.png"))
elseif MultiType == :nnv
    savefig(string(file_fig_file,"_TaTb2_nnv.png"))
else
    savefig(string(file_fig_file,"_TaTb2.png"))
end