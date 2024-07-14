
# [Cn, CI, CK, Cs]
is_out_CnIKs = true

is_CK_logy = true
is_Cn_logy = true
xlabel = "t"               

ylabel = string("Δnₛ")
if is_Cn_logy
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDnM = plot(tplotM[iCasep],abs.(CRDnM[iCasep]),line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10,legend=legendbR)
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDnM = plot!(tplotM[iCasep],abs.(CRDnM[iCasep]),line=(wline,:auto),
                        label=label)
    end
else
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDnM = plot(tplotM[iCasep],CRDnM[iCasep],line=(wline,:auto),label=label,
                    ylabel=ylabel)
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDnM = plot!(tplotM[iCasep],CRDnM[iCasep],line=(wline,:auto),label=label)
    end
    
end

ylabel = string("ΔKₛ")
if is_CK_logy
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDKM = plot(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,:auto),
                    label=label,ylabel=ylabel,yscale=:log,legend=legendbR,
                    xlabel=xlabel)
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDKM = plot!(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,:auto),
                        label=label)
    end
else
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDKM = plot(tplotM[iCasep],CRDKM[iCasep],line=(wline,:auto),
                    label=label,ylabel=ylabel,
                    xlabel=xlabel)
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDKM = plot!(tplotM[iCasep],CRDKM[iCasep],line=(wline,:auto),label=label)
    end
end

if is_Case_C01 && is_enforce_errdtnIKab == false
    tplotMC0 = deepcopy(tplotM)
    CRDnMC0 = deepcopy(CRDnM)
    CRDKMC0 = deepcopy(CRDKM)
end

pCnKM = display(plot(pCRDnM,pCRDKM,layout=(2,1)))
display(pCnKM)

plot(pCRDnM,pCRDKM,layout=(2,1))
savefig(string(file_fig_file,"_CnKMdt.png"))
  
if is_out_CnIKs
    
    ylabel = string("ΔIₛ")
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDIM = plot(tplotM[iCasep],CRDIM[iCasep],line=(wline,:auto),label=label,
                    ylabel=ylabel)
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDIM = plot!(tplotM[iCasep],CRDIM[iCasep],line=(wline,:auto),label=label)
    end
    
    ylabel = string("Δsₛ")        # ("sₛ⁻¹∂ₜsₛ")
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDsM = plot(tplotM[iCasep],CRDsM[iCasep],line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10,legend=legendbL,
                    xlabel=xlabel)
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDsM = plot!(tplotM[iCasep],CRDsM[iCasep],line=(wline,:auto),label=label)
    end

    pCnIKsM = display(plot(pCRDnM,pCRDIM,pCRDKM,pCRDsM,layout=(2,2)))
    display(pCnIKsM)
    
    plot(pCRDnM,pCRDIM,pCRDKM,pCRDsM,layout=(2,2))
    savefig(string(file_fig_file,"_CnIKsMdt.png"))

    if is_Case_C01 && is_enforce_errdtnIKab == false
        CRDIMC0 = deepcopy(CRDIM)
        CRDsMC0 = deepcopy(CRDsM)
    end
end
