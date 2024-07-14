

wlineT2 = 3
xlabel = string("t")
tvecp = tplot .< 50

ylabel = string("T")
pTa = plot(tplot[tvecp],Tat[tvecp],line=(wlineT2),label="a,Kin.",ylabel=ylabel)
pTb = plot!(tplot[tvecp],Tbt[tvecp],line=(wlineT2,:auto),label="b,Kin.")

NtTaTb2 = length(tplotTaTb[tvecTaTb])
Nt10 = max(1,round(Int64,NtTaTb2 / 10))
vec10 = 1:Nt10:NtTaTb2
ppTa = scatter!(tplotTaTb[tvecTaTb][vec10],TatTaTb[tvecTaTb][vec10],line=(wlineT2,:auto),m=:star5,label="a,Brg.")
ppTb = scatter!(tplotTaTb[tvecTaTb][vec10],TbtTaTb[tvecTaTb][vec10],line=(wlineT2,:auto),m=:circle,label="b,Brg.")

ylabel = string("ΔT")
pRDTabt = plot(tplot[tvecp],abs.(RDTabt[tvecp]),line=(wlineT2),label="Kin.",
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel)

ppRDTab = scatter!(tplotTaTb[tvecTaTb][vec10],(abs.(RDTabtTaTb[tvecTaTb][vec10])),line=(wlineTaTb,:auto),label="Brag.",
                    ylabel=ylabel,yscale=:log10)
# lens!([4.75,4.7501], [1e-5, 7e-6], inset = (1, bbox(0.5,0,0.3,0.5)))

display(plot(ppTa,pRDTabt,layout=(2,1)))
plot(ppTa,pRDTabt,layout=(2,1))
savefig(string(file_fig_file,"_test_TaTb2.png"))