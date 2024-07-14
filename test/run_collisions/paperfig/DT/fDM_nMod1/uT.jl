
tvecu = tvec
wlineu = 2

ylabel = string("u [Mms]")
puab = plot(tplot[tvecu],uat[tvecu],line=(wlineu,:solid),label="D",ylabel=ylabel)
puab = plot!(tplot[tvecu],ubt[tvecu],line=(wlineu,:dashdot),label="Tri")

ylabel = string("T")
pTab = plot(tplot[tvec],Tat[tvec],line=(wlineu,:solid),label="D",ylabel=ylabel)
pTab = plot!(tplot[tvec],Tbt[tvec],line=(wlineu,:dashdot),label="Tri",xlabel="t")

puTab = plot(puab,pTab,layout=(2,1))
display(puTab)

plot(puab,pTab,layout=(2,1))
savefig(string(file_fig_file,"_uTab.png"))


