## δₜf, the Fokker-Planck collisions
println()
fvLclog0 = deepcopy(fvLclog)
vs0 = deepcopy(vs)
if 11 == 1
    @time dtfc = dtfvLcg(fvLc,fvLclog,HvLcv0,GvLcv0,vs,vc,nc,Dc,mu,DP,Mμ,Mun,LM1,LM,ns,maD,Zq,na,vth,orders,domain;k=k,isvboundary=isvboundary)

    # @time dtfc = dtfvLc8g(fvLc,fvLclog,HvLcv0,GvLcv0,vs0,vc,nc,Dc,mu,DP,Mμ,Mun,LM1,LM,ns,maD,Zq,na0,vth0,orders,domain,Ks05;
    #                isdtfcon=isdtfcon,atol_dMs=atol_dMs,iter_MsMax=iter_MsMax,k=k,isvboundary=isvboundary,nSdtf=nSdtf)
    # Δdtfc = (dtf0 - dtfc)
    # Rdtf = Δdtfc ./ dtfc
    # @show fmtf4.([norm(Δdtfc) norm(Rdtf[:,1,:])])
    # #### Plotting
    dtfLnc = dtfc[:,L1,isp3]
    c0 = 4π * (- (vs[1,isp3] - vs[end,isp3]) / 2)  # c0 = 4π/π^(3/2) * (-(a-b)/2)
    dna1 = (wc * (vs[1,isp3].^2 .* dtfLnc))[1] * c0
    Rdna1 = dna1 / na0[isp3]
    Rdna12 = (wc * (vs[1,isp3].^2 .* (dtfLnc / na0[isp3])))[1] * c0
    if is_plotdtf == 1
        L1 = 1
        xlabel = string("log10(v)")
        label = string("fₐ,L=",L1-1)
        pfL = plot(vlog[nvaplot],fvLc[nvaplot,L1,isp3],label=label)
        label = string("fᵦ,L=",L1-1)
        pfL = plot!(vlog[nvaplot],fvLc[nvaplot,L1,iFv3],label=label,xlabel=xlabel,line=(3,:auto))
        label = string("δₜfₐ")
        pdtf = plot(vlog[nvaplot],dtfc[nvaplot,L1,isp3],label=label)
        label = string("δₜfᵦ")
        pdtf = plot!(vlog[nvaplot],dtfc[nvaplot,L1,iFv3],label=label,xlabel=xlabel,line=(3,:auto))
        display(plot(pfL,pdtf,layout=(2,1)))
    end
    dna2, dKa2 = momentscGauss(ns,ma,vs,wc,dtfc[:,1,:],1)
    dna, dIa, dKa = momentscGauss(ns,ma,vs,wc,dtfc)
    printstyled("f0",",Rdtn=",fmtf2.(dna./na0),",dtI=",fmtf2.(Ia - Ia0),",dtK=",fmtf2.(dKa);color=:yellow)
    println()
    printstyled("f0",",δₜn̂=",fmtf4.(dna./na0),",ΔₜI=",fmtf4.(sum(dIa)),",ΔₜK=",fmtf4.(sum(dKa)/sum(abs.(dKa)));color=:green)
    println()
end
nak0, Iak0, Kak0 = momentscGauss(ns,ma,vs,wc,fvLc)
printstyled("fc",",Rdn=",fmtf2.(nak0./na0.-1),",dI=",fmtf2.(Iak0 - Ia0),",RdK=",fmtf2.(Kak0./Ka0 .- 1);color=:yellow)
println()
RdK = Kak0 ./ Ka0 .- 1
@show RdK ./ reverse(vth)
