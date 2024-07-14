
# [Cn, CI, CK, Cs]
is_out_CnIKs = false

is_CK_logy = true
is_Cn_logy = true        
xlabel = "t"               

title = "Conservations"
ylabel = string("Δnₛ")
if is_Cn_logy
    iCasep = 1
    label = string(vec_sweep[iCasep])
    pCRDnM = plot(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,:auto),label=label,
                    yscale=:log10,
                    xlabel=xlabel)
    iCasep = 2
    label = string(vec_sweep[iCasep])
    pCRDnM = plot!(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,:auto),
                    label=label)
    iCasep = 3
    label = string(vec_sweep[iCasep])
    pCRDnM = plot!(tplotM[iCasep],abs.(CRDnM[iCasep]).+epsT,line=(wline,:auto),
                    label=label)
else
    iCasep = 1
    label = string(vec_sweep[iCasep])
    pCRDnM = plot(tplotM[iCasep],CRDnM[iCasep],line=(wline,:auto),label=label,
                    xlabel=xlabel)
    iCasep = 2
    label = string(vec_sweep[iCasep])
    pCRDnM = plot!(tplotM[iCasep],CRDnM[iCasep],line=(wline,:auto),label=label)
    iCasep = 3
    label = string(vec_sweep[iCasep])
    pCRDnM = plot!(tplotM[iCasep],CRDnM[iCasep],line=(wline,:auto),label=label)
    
end

ylabel = string("ΔKₛ")
if is_CK_logy
    iCasep = 1
    label = string(vec_sweep[iCasep])
    pCRDKM = plot(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,:auto),
                    label=label,yscale=:log,
                    title=title)
    iCasep = 2
    label = string(vec_sweep[iCasep])
    pCRDKM = plot!(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,:auto),
                    label=label)
    iCasep = 3
    label = string(vec_sweep[iCasep])
    pCRDKM = plot!(tplotM[iCasep],abs.(CRDKM[iCasep]).+epsT,line=(wline,:auto),
                    label=label)
else
    iCasep = 1
    label = string(vec_sweep[iCasep])
    pCRDKM = plot(tplotM[iCasep],CRDKM[iCasep],line=(wline,:auto),
                    label=label,
                    title=title)
    iCasep = 2
    label = string(vec_sweep[iCasep])
    pCRDKM = plot!(tplotM[iCasep],CRDKM[iCasep],line=(wline,:auto),label=label)
    iCasep = 3
    label = string(vec_sweep[iCasep])
    pCRDKM = plot!(tplotM[iCasep],CRDKM[iCasep],line=(wline,:auto),label=label)
end

title = "No conservations"
ylabel = string("Δnₛ")
if is_Cn_logy
    iCasep = 1
    label = string(vec_sweep[iCasep])
    pCRDnMC0 = plot(tplotMC0[iCasep],abs.(CRDnMC0[iCasep]).+epsT,line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10,
                    xlabel=xlabel)
    iCasep = 2
    label = string(vec_sweep[iCasep])
    pCRDnMC0 = plot!(tplotMC0[iCasep],abs.(CRDnMC0[iCasep]).+epsT,line=(wline,:auto),
                    label=label)
    iCasep = 3
    label = string(vec_sweep[iCasep])
    pCRDnMC0 = plot!(tplotMC0[iCasep],abs.(CRDnMC0[iCasep]).+epsT,line=(wline,:auto),
                    label=label)
else
    iCasep = 1
    label = string(vec_sweep[iCasep])
    pCRDnMC0 = plot(tplotMC0[iCasep],CRDnMC0[iCasep],line=(wline,:auto),label=label,
                    ylabel=ylabel,
                    xlabel=xlabel)
    iCasep = 2
    label = string(vec_sweep[iCasep])
    pCRDnMC0 = plot!(tplotMC0[iCasep],CRDnMC0[iCasep],line=(wline,:auto),label=label)
    iCasep = 3
    label = string(vec_sweep[iCasep])
    pCRDnMC0 = plot!(tplotMC0[iCasep],CRDnMC0[iCasep],line=(wline,:auto),label=label)
    
end

ylabel = string("ΔKₛ")
if is_CK_logy
    iCasep = 1
    label = string(vec_sweep[iCasep])
    pCRDKMC0 = plot(tplotMC0[iCasep],abs.(CRDKMC0[iCasep]).+epsT,line=(wline,:auto),
                    label=label,ylabel=ylabel,yscale=:log,
                    title=title)
    iCasep = 2
    label = string(vec_sweep[iCasep])
    pCRDKMC0 = plot!(tplotMC0[iCasep],abs.(CRDKMC0[iCasep]).+epsT,line=(wline,:auto),
                    label=label)
    iCasep = 3
    label = string(vec_sweep[iCasep])
    pCRDKMC0 = plot!(tplotMC0[iCasep],abs.(CRDKMC0[iCasep]).+epsT,line=(wline,:auto),
                    label=label)
else
    iCasep = 1
    label = string(vec_sweep[iCasep])
    pCRDKMC0 = plot(tplotMC0[iCasep],CRDKMC0[iCasep],line=(wline,:auto),
                    label=label,ylabel=ylabel,
                    title=title)
    iCasep = 2
    label = string(vec_sweep[iCasep])
    pCRDKMC0 = plot!(tplotMC0[iCasep],CRDKMC0[iCasep],line=(wline,:auto),label=label)
    iCasep = 3
    label = string(vec_sweep[iCasep])
    pCRDKMC0 = plot!(tplotMC0[iCasep],CRDKMC0[iCasep],line=(wline,:auto),label=label)
end

pCnKC01M = display(plot(pCRDKMC0,pCRDKM,pCRDnMC0,pCRDnM,layout=(2,2)))
display(pCnKC01M)


plot(pCRDKMC0,pCRDKM,pCRDnMC0,pCRDnM,layout=(2,2))
savefig(string(file_fig_file,"_CnKC01M.png"))
  