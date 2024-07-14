
# [Cn, CI, CK, Cs]

is_logy = true
xlabel = "t"               

ylabel = string("Δnₛ")
if is_logy == false
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDnM = plot(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDnM = plot!(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label,yscale=:log10)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDnM = plot!(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label,yscale=:log10)
else
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDnM = plot(tplotM[iCasep],CRDnM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDnM = plot!(tplotM[iCasep],CRDnM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDnM = plot!(tplotM[iCasep],CRDnM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    
end

ylabel = string("ΔIₛ")
iCasep = 1
label = string(nnvocpM[iCasep])
pCRDIM = plot(tplotM[iCasep],CRDIM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                ylabel=ylabel)
iCasep = 2
label = string(nnvocpM[iCasep])
pCRDIM = plot!(tplotM[iCasep],CRDIM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
iCasep = 3
label = string(nnvocpM[iCasep])
pCRDIM = plot!(tplotM[iCasep],CRDIM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

ylabel = string("ΔKₛ")
if is_logy
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDKM = plot(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDKM = plot!(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,yscale=:log10)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDKM = plot!(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,yscale=:log10)
else
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDKM = plot(tplotM[iCasep],CRDKM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,
                    xlabel=xlabel)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDKM = plot!(tplotM[iCasep],CRDKM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDKM = plot!(tplotM[iCasep],CRDKM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
end
                
ylabel = string("Δsₛ")        # ("sₛ⁻¹∂ₜsₛ")
iCasep = 1
label = string(nnvocpM[iCasep])
pCRDsM = plot(tplotM[iCasep],CRDsM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                ylabel=ylabel,
                xlabel=xlabel)
iCasep = 2
label = string(nnvocpM[iCasep])
pCRDsM = plot!(tplotM[iCasep],CRDsM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
iCasep = 3
label = string(nnvocpM[iCasep])
pCRDsM = plot!(tplotM[iCasep],CRDsM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)

pCnIKs = display(plot(pCRDnM,pCRDIM,pCRDKM,pCRDsM,layout=(2,2)))
display(pCnIKs)

plot(pCRDnM,pCRDIM,pCRDKM,pCRDsM,layout=(2,2))
savefig(string(file_fig_file,"_CnIKsM.png"))
