
# [Cn, CI, CK, Cs]
is_out_CnIKs = false

is_CK_logy = true
is_CI_logy = true  
is_Cn_logy = true          
xlabel = "t"               

title = "Conservations"
ylabel = string("Δnₛ")
if is_Cn_logy
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDnM = plot(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    yscale=:log10,
                    xlabel=xlabel)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDnM = plot!(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDnM = plot!(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
else
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDnM = plot(tplotM[iCasep],CRDnM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    xlabel=xlabel)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDnM = plot!(tplotM[iCasep],CRDnM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDnM = plot!(tplotM[iCasep],CRDnM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    
end

ylabel = string("ΔIₛ")
if is_CI_logy
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDIM = plot(tplotM[iCasep],abs.(CRDIM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    yscale=:log10,
                    xlabel=xlabel)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDIM = plot!(tplotM[iCasep],abs.(CRDIM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDIM = plot!(tplotM[iCasep],abs.(CRDIM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
else
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDIM = plot(tplotM[iCasep],CRDIM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    xlabel=xlabel)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDIM = plot!(tplotM[iCasep],CRDIM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDIM = plot!(tplotM[iCasep],CRDIM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    
end

ylabel = string("ΔKₛ")
if is_CK_logy
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDKM = plot(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label,ylabel=ylabel,yscale=:log,
                    title=title)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDKM = plot!(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDKM = plot!(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
else
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDKM = plot(tplotM[iCasep],CRDKM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label,ylabel=ylabel,
                    title=title)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDKM = plot!(tplotM[iCasep],CRDKM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDKM = plot!(tplotM[iCasep],CRDKM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
end

title = "No conservations"
ylabel = string("Δnₛ")
if is_Cn_logy
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDnMC0 = plot(tplotMC0[iCasep],abs.(CRDnMC0[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel,legend=legendbR)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDnMC0 = plot!(tplotMC0[iCasep],abs.(CRDnMC0[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDnMC0 = plot!(tplotMC0[iCasep],abs.(CRDnMC0[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
else
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDnMC0 = plot(tplotMC0[iCasep],CRDnMC0[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,
                    xlabel=xlabel)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDnMC0 = plot!(tplotMC0[iCasep],CRDnMC0[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDnMC0 = plot!(tplotMC0[iCasep],CRDnMC0[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    
end

ylabel = string("ΔIₛ")
if is_Cn_logy
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDIMC0 = plot(tplotMC0[iCasep],abs.(CRDIMC0[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel,legend=legendbR)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDIMC0 = plot!(tplotMC0[iCasep],abs.(CRDIMC0[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDIMC0 = plot!(tplotMC0[iCasep],abs.(CRDIMC0[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
else
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDIMC0 = plot(tplotMC0[iCasep],CRDIMC0[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,
                    xlabel=xlabel)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDIMC0 = plot!(tplotMC0[iCasep],CRDIMC0[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDIMC0 = plot!(tplotMC0[iCasep],CRDIMC0[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    
end

ylabel = string("ΔKₛ")
if is_CK_logy
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDKMC0 = plot(tplotMC0[iCasep],abs.(CRDKMC0[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label,ylabel=ylabel,yscale=:log,
                    title=title)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDKMC0 = plot!(tplotMC0[iCasep],abs.(CRDKMC0[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDKMC0 = plot!(tplotMC0[iCasep],abs.(CRDKMC0[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label)
else
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDKMC0 = plot(tplotMC0[iCasep],CRDKMC0[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label,ylabel=ylabel,
                    title=title)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDKMC0 = plot!(tplotMC0[iCasep],CRDKMC0[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDKMC0 = plot!(tplotMC0[iCasep],CRDKMC0[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
end

pCnKC01M = display(plot(pCRDKMC0,pCRDKM,pCRDIMC0,pCRDIM,pCRDnMC0,pCRDnM,layout=(3,2)))
display(pCnKC01M)


plot(pCRDKMC0,pCRDKM,pCRDIMC0,pCRDIM,pCRDnMC0,pCRDnM,layout=(3,2))
savefig(string(file_fig_file,"_CnIKC01M.png"))
  