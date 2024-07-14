
# [Cn, CI, CK, Cs]
is_out_CnIKs = false

is_CK_logy = true
is_Cn_logy = true        
xlabel = "t"               

title = "Conservations"
ylabel = string("Δnₛ")
if is_Cn_logy
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDnM = plot(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,:auto),label=label,
                    yscale=:log10,legend=legendbR,
                    xlabel=xlabel,
                    title=title)
    iCasep = 2
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDnM = plot!(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,:auto),
                        label=label)
    end
else
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDnM = plot(tplotM[iCasep],CRDnM[iCasep],line=(wline,:auto),label=label,
                    xlabel=xlabel,
                    title=title)
    iCasep = 2
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
                    label=label,yscale=:log,legend=legendbR)
    iCasep = 2
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDKM = plot!(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,:auto),
                        label=label)
    end
else
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDKM = plot(tplotM[iCasep],CRDKM[iCasep],line=(wline,:auto),
                    label=label)
    iCasep = 2
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDKM = plot!(tplotM[iCasep],CRDKM[iCasep],line=(wline,:auto),label=label)
    end
end

title = "No conservations"
ylabel = string("Δnₛ")
if is_Cn_logy
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDnMC0 = plot(tplotMC0[iCasep],abs.(CRDnMC0[iCasep]).+epsT,line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10,legend=legendbR,
                    xlabel=xlabel,
                    title=title)
    iCasep = 2
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDnMC0 = plot!(tplotMC0[iCasep],abs.(CRDnMC0[iCasep]).+epsT,line=(wline,:auto),
                        label=label)
    end
else
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDnMC0 = plot(tplotMC0[iCasep],CRDnMC0[iCasep],line=(wline,:auto),label=label,
                    ylabel=ylabel,
                    xlabel=xlabel,
                    title=title)
    iCasep = 2
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDnMC0 = plot!(tplotMC0[iCasep],CRDnMC0[iCasep],line=(wline,:auto),label=label)
    end
    
end

ylabel = string("ΔKₛ")
if is_CK_logy
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDKMC0 = plot(tplotMC0[iCasep],abs.(CRDKMC0[iCasep]).+epsT,line=(wline,:auto),
                    label=label,ylabel=ylabel,yscale=:log,legend=legendbR)
    iCasep = 2
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDKMC0 = plot!(tplotMC0[iCasep],abs.(CRDKMC0[iCasep]).+epsT,line=(wline,:auto),
                        label=label)
    end
else
    iCasep = 1
    label = string(L"1 / \Delta t=",vec_sweep[iCasep])
    pCRDKMC0 = plot(tplotMC0[iCasep],CRDKMC0[iCasep],line=(wline,:auto),
                    label=label,ylabel=ylabel)
    iCasep = 2
    for iCasep in 2:NCasep
        label = string(L"1 / \Delta t=",vec_sweep[iCasep])
        pCRDKMC0 = plot!(tplotMC0[iCasep],CRDKMC0[iCasep],line=(wline,:auto),label=label)
    end
end

pCnKC01M = display(plot(pCRDnMC0,pCRDnM,pCRDKMC0,pCRDKM,layout=(2,2)))
display(pCnKC01M)


plot(pCRDnMC0,pCRDnM,pCRDKMC0,pCRDKM,layout=(2,2))
savefig(string(file_fig_file,"_CnKC01Mdt.png"))
  