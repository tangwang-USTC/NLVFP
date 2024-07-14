# plotly()
# pyplot()
# pythonplot()
# pgfplotsx()
# plotlyjs()
# gr()
# inspectdr()
tvec3 = tplot .< 3000
wlineT3 = 2
ylabel = string("T")
xlabel = "t"
il = 1
pTeDA = plot(tplot[tvec3],Tat[tvec3],line=(wlineT3,linetypes[il],linecolors[il]),label="e",
                ylabel=ylabel)
il = 2
pTeDA = plot!(tplot[tvec3],Tbt[tvec3],line=(wlineT3,linetypes[il],linecolors[il]),label="D")
il = 3
pTeDA = plot!(tplot[tvec3],Tct[tvec3],line=(wlineT3,linetypes[il],linecolors[il]),label="α")

tvec3 = tplot .< 30
il = 1
pTeD = plot(tplot[tvec3],Tat[tvec3],line=(wlineT3,linetypes[il],linecolors[il]),label="e",
                ylabel=ylabel,
                xlabel=xlabel)
il = 2
pTeD = plot!(tplot[tvec3],Tbt[tvec3],line=(wlineT3,linetypes[il],linecolors[il]),label="D")


display(plot(pTeDA,pTeD,layout=(2,1)))

plot(pTeDA,pTeD,layout=(2,1))
savefig(string(file_fig_file,"_TeTDTA.png"))
