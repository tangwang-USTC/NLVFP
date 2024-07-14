
tvecu = tplot .< 1e-2
wlineu = 2

ylabel = string("u [Mms]")
puab = plot(tplot[tvecu],uat[tvecu],line=(wlineu,:solid),label="e",ylabel=ylabel)
puab = plot!(tplot[tvecu],ubt[tvecu],line=(wlineu,:dashdot),label="D")

ylabel = string("T")
pTab = plot(tplot[tvec],Tat[tvec],line=(wlineu,:solid),label="e",ylabel=ylabel)
pTab = plot!(tplot[tvec],Tbt[tvec],line=(wlineu,:dashdot),label="D",xlabel="t")

puTab = plot(puab,pTab,layout=(2,1))
display(puTab)

plot(puab,pTab,layout=(2,1))
savefig(string(file_fig_file,"_uTab.png"))


