
# [Cn, CI, CK, Cs]
is_CRDn_log = true

label = string("n₂,N₀=",(nnv0,ocp0))
xlabel = "t"                  # *neps

ylabel = string("Δnₛ")
if is_CRDn_log
    pDnab = plot(tplot[tvec],CRDn,line=(wline,:auto),label=label,
                    ylabel=ylabel,yscale=:log10)
else
    pDnab = plot(tplot[tvec],CRDn,line=(wline,:auto),label=label,
                    ylabel=ylabel)
end
ylabel = string("ΔIₛ")
pDIab = plot(tplot[tvec],(CRDI .+ epsT),line=(wline,:auto),label=label,
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
