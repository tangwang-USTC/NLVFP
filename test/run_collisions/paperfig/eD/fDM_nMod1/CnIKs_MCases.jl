
# [Cn, CI, CK, Cs]
is_out_CnIKs = true

is_CK_logy = false
is_Cn_logy = true
xlabel = "t"               

ylabel = string("Δnₛ")
if is_Cn_logy
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDnM = plot(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10)
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
                    ylabel=ylabel)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDnM = plot!(tplotM[iCasep],CRDnM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDnM = plot!(tplotM[iCasep],CRDnM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    
end

ylabel = string("ΔKₛ")
if is_CK_logy
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDKM = plot(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,linetypes[iCasep],linecolors[iCasep]),
                    label=label,ylabel=ylabel,yscale=:log,
                    xlabel=xlabel,legend=legendtR)
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
                    xlabel=xlabel,legend=legendtR)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDKM = plot!(tplotM[iCasep],CRDKM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDKM = plot!(tplotM[iCasep],CRDKM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
end

if is_Case_C01 && is_enforce_errdtnIKab == false
    tplotMC0 = deepcopy(tplotM)
    CRDnMC0 = deepcopy(CRDnM)
    CRDKMC0 = deepcopy(CRDKM)
end

pCnKM = display(plot(pCRDnM,pCRDKM,layout=(2,1)))
display(pCnKM)

plot(pCRDnM,pCRDKM,layout=(2,1))
savefig(string(file_fig_file,"_CnKM.png"))

pCnIKM = display(plot(pCRDIM,pCRDKM,layout=(1,2)))
display(pCnIKM)

plot(pCRDIM,pCRDKM,layout=(1,2))
savefig(string(file_fig_file,"_CIKM.png"))

if is_out_CnIKs
    
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
    
    ylabel = string("Δsₛ")        # ("sₛ⁻¹∂ₜsₛ")
    iCasep = 1
    label = string(nnvocpM[iCasep])
    pCRDsM = plot(tplotM[iCasep],CRDsM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label,
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel)
    iCasep = 2
    label = string(nnvocpM[iCasep])
    pCRDsM = plot!(tplotM[iCasep],CRDsM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),label=label)
    iCasep = 3
    label = string(nnvocpM[iCasep])
    pCRDsM = plot!(tplotM[iCasep],CRDsM[iCasep],line=(wline,linetypes[iCasep],linecolors[iCasep]),
                  label=label)

    pCnIKsM = display(plot(pCRDnM,pCRDIM,pCRDKM,pCRDsM,layout=(2,2)))
    display(pCnIKsM)
    
    plot(pCRDnM,pCRDIM,pCRDKM,pCRDsM,layout=(2,2))
    savefig(string(file_fig_file,"_CnIKsM.png"))

    pCnIKM = display(plot(pCRDnM,pCRDIM,pCRDKM,layout=(3,1)))
    display(pCnIKM)
    
    plot(pCRDnM,pCRDIM,pCRDKM,layout=(3,1))
    savefig(string(file_fig_file,"_CnIKM.png"))

    if is_Case_C01 && is_enforce_errdtnIKab == false
        CRDIMC0 = deepcopy(CRDIM)
        CRDsMC0 = deepcopy(CRDsM)
    end
end
