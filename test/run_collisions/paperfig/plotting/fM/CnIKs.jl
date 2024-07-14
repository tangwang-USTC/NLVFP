
# [Cn, CI, CK, Cs]

label = string("n₂,N₀=",(nnv0,ocp0))
ylabel = string("Δnₛ")
xlabel = "t"                  # *neps
pDnab = plot(tplot[tvec],abs.(CRDn),line=(wline,:auto),label=label,
                ylabel=ylabel)
ylabel = string("ΔIₛ")
pDIab = plot(tplot[tvec],CRDI,line=(wline,:auto),label=label,
                ylabel=ylabel)
ylabel = string("ΔKₛ")
pDKab = plot(tplot[tvec],CRDK,line=(wline,:auto),label=label,
                ylabel=ylabel,xlabel=xlabel)
ylabel = string("Δsₛ")        # ("sₛ⁻¹∂ₜsₛ")
pRdts = plot(tplot[tvec],CRDs,line=(wline,:auto),label=label,
                ylabel=ylabel,yscale=:log10,
                xlabel=xlabel)

pCnIKs = display(plot(pDnab,pDIab,pDKab,pRdts,layout=(2,2)))
display(pCnIKs)

plot(pDnab,pDIab,pDKab,pRdts,layout=(2,2))
savefig(string(file_fig_file,"_CnIKs.png"))
