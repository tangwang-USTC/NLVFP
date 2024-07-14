## δₜf,aa: the Fokker-Planck collisions
println("//////////")
# is_optimδtfvL = true
nSδtf = 1
# # @show 1, size(fvL0)
# is_δtfvLaa = 0          # [-1,    0,     1   ]
#                         # [δtfaa, δtfab, δtfa]
if is_normδtf == false
    fvL4 = deepcopy(fvLc0e)
else
    fvL4 = deepcopy(fvL0e)
end

Mhcsd2l = Vector{Matrix{datatype}}(undef,ns+1)
# w3k0 = dtnIKsc[4,:]           # `w3k = Rdtvth = 𝒲 / 3`
if is_normδtf
    if is_moments_out 
        @time dtfvL0, w3k0, fvL4, err_dtnIK1, DThk, Mhcsd2l  = dtfvLSplineab(Mhcsd2l, vhk, vhe, 
                nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2, 
                CΓ, εᵣ, ma, Zq, spices0, na, vth, nai, uai, vthi, LM, LM1, ns, nMod;
                is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR, 
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, 
                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                is_δtfvLaa=is_δtfvLaa,is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0, 
                is_check_conservation_dtM=is_check_conservation_dtM,
                is_moments_out=is_moments_out,is_MjMs_max=is_MjMs_max,
                is_fit_f=is_fit_f,is_extrapolate_FLn=is_extrapolate_FLn_initial)
    else
        @time dtfvL0, w3k0, fvL4, err_dtnIK1, DThk = dtfvLSplineab2(Mhcsd2l, vhk, vhe, 
                nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2, 
                CΓ, εᵣ, ma, Zq, spices0, na, vth, nai, uai, vthi, LM, LM1, ns, nMod;
                is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR, 
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, 
                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                is_δtfvLaa=is_δtfvLaa,is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0, 
                is_check_conservation_dtM=is_check_conservation_dtM,
                is_moments_out=is_moments_out,is_MjMs_max=is_MjMs_max,
                is_fit_f=is_fit_f,is_extrapolate_FLn=is_extrapolate_FLn_initial)
    end
    dtfvLc0 = deepcopy(dtfvL0)
    for isp33 in 1:ns
        dtfvLc0[isp33] *= (na[isp33] / sqrtpi3 / vth[isp33]^3)
    end
else
    @time dtfvLc0, w3k0, fvL4, err_dtnIK1, DThk = dtfvLSplineab2(Mhcsd2l, vhk, vhe, 
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2, 
            CΓ, εᵣ, ma, Zq, spices0, na, vth, nai, uai, vthi, LM, LM1, ns, nMod;
            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR, 
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, 
            rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
            is_δtfvLaa=is_δtfvLaa,is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0, 
            is_check_conservation_dtM=is_check_conservation_dtM,
            is_fit_f=is_fit_f)
    dtfvL0 = deepcopy(dtfvLc0)
    for isp33 in 1:ns
        dtfvL0[isp33] *= (sqrtpi3 * vth[isp33]^3 / na[isp33])
    end
end

dtn̂aE, dtIaE, dtKaE = nIKs(dtfvL0,vhe,ma,na,vth,ns)

dtnIKsc = zeros(3,ns)
nIKsc!(dtnIKsc,dtfvLc0,vhe,ma,vth,ns;atol_nIK=atol_nIK)
@show sum(dtnIKsc[3,:]), dtnIKsc[3,1];

#########  Optimizations according to `y(v̂→0) → Cₗᵐ` and `Rd1y(v̂→0) → 0`
isp33 = iFv3
# isp33 = isp3
nsp_vec = 1:ns
iFv33 = nsp_vec[nsp_vec .≠ isp33][1] 
nvc3 = zeros(Int64, 2(order_smooth + 1), LM1, ns)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
if is_optimδtfvL
    δtfvL0e3D = deepcopy(δtfvL0e)
    optimdtfvL0e!(nvc3, δtfvL0e3D, fvL4, vGe, nvG, ns, LM;
        orders=order_dvδtf, is_boundv0=is_boundv0,
        Nsmooth=Nsmooth, order_smooth=order_smooth, abstol_Rdy=abstol_Rdy,
        k=k_δtf, Nitp=Nitp, order_smooth_itp=order_smooth_itp, order_nvc_itp=order_nvc_itp,
        nvc0_limit=nvc0_limit, L1nvc_limit=L1nvc_limit)
    dtMsnnE33D = zeros(datatype, njMs, LM1, ns)
    dtMsnnE33D = MsnnEvens(dtMsnnE33D, δtfvL0e3D, vGe, njMs, LM, LM1, ns; is_renorm=is_renorm)
end

if 21 == 1
    LLL1 = 1
    LLL1 ≤ LM1 || (LLL1 = 1)

    # dtf
    isp33 = 1
    label = string(spices0[isp33],",L=",LLL1-1)
    ylabel = L"\partial_t f_l^0"
    pppfa = plot(vhe[isp33],dtfvLc0[isp33][:,LLL1],
                ylabel=ylabel,label=label,line=(2,:auto),title=title_TEk)
    isp33 = 2
    label = string(spices0[isp33],",L=",LLL1-1)
    pppfb = plot(vhe[isp33],dtfvLc0[isp33][:,LLL1],
                ylabel=ylabel,label=label,line=(2,:auto))

    # dtf / f
    isp33 = 1
    label = string(spices0[isp33],",L=",LLL1-1)
    ylabel = L"\partial_t f_l^0 / f_l^0"
    pppRfa = plot(vhe[isp33],dtfvLc0[isp33][:,LLL1] ./ fvL4[isp33][:,LLL1],
                ylabel=ylabel,label=label,line=(2,:auto),title=title_TEk)
    isp33 = 2
    label = string(spices0[isp33],",L=",LLL1-1)
    pppRfb = plot(vhe[isp33],dtfvLc0[isp33][:,LLL1] ./ fvL4[isp33][:,LLL1],
                ylabel=ylabel,label=label,line=(2,:auto))
    display(plot(pppfa,pppfb,pppRfa,pppRfb,layout=(2,2)))
end

# println("//////////")

pldtfL0(isp3,l1) = display(
    plot(vGe,dtfvL0[:,l1,isp3],
        label=string("isp=",isp3,",ℓ = ",l1 - 1),
        xlabel=string("v̂"),
        ylabel=string("∂ₜf̂ₗ⁰")
        )
    )
    
is_dt_theory = false
if is_dt_theory
    println("-----------------------------------------")
    if sum(abs.(u0)) == 0
        Zqab = Zq[isp3] * Zq[iFv3]
        if nSδtf == 1
            # if prod(nMod) == 1
            #     δtfvLt = zeros(datatype,nvG,LM1,ns)
            #     δtfvLt[:,L1,:] = dtfMab(δtfvLt[:,L1,:],ma,Zq,na,vth,ns,vGe,εᵣ)
            #     # δtfvLt[:,L1,:] = dtfMab(δtfvLt[:,L1,:],ma,Zq,na,vth,ns,vGe,εᵣ,nai,vthi,nMod)
            #     # δtfvLt[:,L1,isp3] = dtfMab(δtfvLt[:,L1,isp3],ma[isp3],mM,Zqab,na[iFv3],vth[iFv3],vabth,vGe,εᵣ)
            #     # δtfvLt[:,L1,iFv3] = dtfMab(δtfvLt[:,L1,iFv3],ma[iFv3],1/mM,Zqab,na[isp3],vth[isp3],1/vabth,vGe,εᵣ)
            # else
                δtfvLt = zeros(datatype,nvG,LM1,ns)
                δtfvLt[:,L1,:] = dtfMab(δtfvLt[:,L1,:],ma,Zq,na,vth,ns,vGe,εᵣ,nai,vthi,nMod)
            # end
        elseif nSδtf == 7
            δtfvL7t = zeros(nvG,nSδtf,ns)
            δtfvL7t[:,:,isp3] = dtfMab(δtfvL7t[:,:,isp3],ma[isp3],mM,Zqab,na[iFv3],vth[iFv3],vabth,vGe,εᵣ)
            δtfvL7t[:,:,iFv3] = dtfMab(δtfvL7t[:,:,iFv3],ma[iFv3],1/mM,Zqab,na[isp3],vth[isp3],1/vabth,vGe,εᵣ)
            δtfvLt7 = zeros(datatype,nvG,LM1,ns)
            δtfvLt7[:,1,:] = sum(δtfvL7t;dims=2)[:,1,:]
            δtfvL7at = δtfvL7t[:,:,isp3]
            δtfvL7bt = δtfvL7t[:,:,iFv3]
            a = δtfvLt7[:,1,:] ./ δtfvLt[:,1,:] .- 1
        end
    end
end
is_plot_dtf = false
## plotting
if is_plot_dtf
    v0log = log.(vGe)
    if sum(abs.(u0)) == 0
        if nSδtf == 1
            errδtfvL0 = dtfvL0[:, L1, :] - δtfvLt[:, L1, :]
        else
            errδtfvL0 = dtfvL0[:, L1, :] - δtfvLt7[:, L1, :]
        end
        RerrδtfvL0 = errδtfvL0 ./ maximum(abs.(fvL[:, L1, :]); dims=1)
        ## Plotting
        nvaplot = v0log .> -6.0
        if is_plotdtf == 1 && sum(u0) == 0
            xlabel = string("log10(v), RTab,=", fmtf2.(Float64(T0[isp33] / T0[iFv33])))
            label = string("fₐ,L=", L1 - 1)
            pfL = plot(v0log[nvaplot], fvL0[nvaplot, L1, isp33], legend=legendtR, label=label)
            label = string("fᵦ,L=", L1 - 1)
            pfL = plot!(v0log[nvaplot], fvL0[nvaplot, L1, iFv33], label=label, line=(3, :auto))

            label = string("RΔδₜfₐ")
            pRDdtf = plot(v0log[nvaplot], RerrδtfvL0[nvaplot, isp33] * neps, label=label, legend=legendtR, line=(3, :auto))
            label = string("RΔδₜfᵦ")
            pRDdtf = plot!(v0log[nvaplot], RerrδtfvL0[nvaplot, iFv33] * neps, label=label, line=(3, :auto))
            pfLs = plot(pfL, pRDdtf, layout=(1, 2))
            ###########################
            vabth == 1.0 ? nepsc = neps : nepsc = 1
            label = string("δₜfₐ")
            pdtf = plot(v0log[nvaplot], dtfvL0[nvaplot, L1, isp33] * nepsc, legend=legendtR, label=label, line=(2, :auto))
            label = string("δₜfᵦ")
            pdtf = plot!(v0log[nvaplot], dtfvL0[nvaplot, L1, iFv33] * nepsc, label=label, xlabel=xlabel, line=(2, :auto))
            #
            label = string("δₜfₐ_t")
            pdtf = plot!(v0log[nvaplot], δtfvLt[nvaplot, L1, isp33] * nepsc, legend=legendbR, label=label, line=(3, :auto))
            label = string("δₜFᵦ_t")
            pdtf = plot!(v0log[nvaplot], δtfvLt[nvaplot, L1, iFv33] * nepsc, label=label, xlabel=xlabel, line=(3, :auto))

            xlabel = string("nG0,nvG,nGk,=", (nvG, nc0, nck))
            label = string("Δδₜfₐ[eps]")
            pDdtf = plot(v0log[nvaplot], errδtfvL0[nvaplot, isp33] * nepsc, label=label, legend=legendtR, line=(3, :auto))
            label = string("Δδₜfᵦ")
            pDdtf = plot!(v0log[nvaplot], errδtfvL0[nvaplot, iFv33] * nepsc, label=label, xlabel=xlabel, line=(3, :auto))
            pDdtfs = plot(pdtf, pDdtf, layout=(1, 2))
            display(plot(pfLs, pDdtfs, layout=(2, 1)))
        end
    else
        nvaplot = v0log .> -6.0
        if is_plotdtf == 1 && sum(u0) == 0
            xlabel = string("log10(v), RTab,=", fmtf2.(Float64(T0[isp33] / T0[iFv33])))
            label = string("fₐ,L=", L1 - 1)
            pfL = plot(v0log[nvaplot], fvL0[nvaplot, L1, isp33], legend=legendtR, label=label)
            label = string("fᵦ,L=", L1 - 1)
            pfL = plot!(v0log[nvaplot], fvL0[nvaplot, L1, iFv33], label=label, line=(3, :auto))
            pfLs = plot(pfL)
            ###########################
            vabth == 1.0 ? nepsc = neps : nepsc = 1
            label = string("δₜfₐ")
            pdtf = plot(v0log[nvaplot], dtfvL0[nvaplot, L1, isp33] * nepsc, legend=legendtR, label=label, line=(2, :auto))
            label = string("δₜfᵦ")
            pdtf = plot!(v0log[nvaplot], dtfvL0[nvaplot, L1, iFv33] * nepsc, label=label, xlabel=xlabel, line=(2, :auto))
            pDdtfs = plot(pdtf)
            display(plot(pfLs, pDdtfs, layout=(2, 1)))
        end
        log10(7)
        nvaplot = -6 .< v0log .< 0.7
        if is_plotdtf == 1
            δtfvL_max = maximum(abs.(dtfvL0[nvaplot, :, :]); dims=1)[1, :, :]
            xlabel = string("L_limit=", L_limit)
            title = string("(nnv,ocp,nvG,nc0,nck)=", (nnv, ocp, nvG, nc0, nck))
            pdtfai(i) = plot(vGe[nvaplot], dtfvL0[nvaplot, i, 1], label=string("L=", i - 1))
            pdtfbi(i) = plot(vGe[nvaplot], dtfvL0[nvaplot, i, 2], label=string("L=", i - 1))
            pfai(i) = plot(vGe[nvaplot], fvL30[nvaplot, i, 1], label=string("fa,L=", i - 1))
            pfbi(i) = plot(vGe[nvaplot], fvL30[nvaplot, i, 2], label=string("fb,L=", i - 1))
            pfa = plot(vGe[nvaplot], fvL0[nvaplot, :, 1], line=(2, :auto), label=string("fa"))
            pfb = plot(vGe[nvaplot], fvL0[nvaplot, :, 2], line=(2, :auto), label=string("fb"))
            pdtfa = plot(vGe[nvaplot], dtfvL0[nvaplot, :, 1], line=(2, :auto), label=string("dtfa"))
            title!(title)
            pdtfb = plot(vGe[nvaplot], dtfvL0[nvaplot, :, 2], line=(2, :auto), label=string("dtfb"))
            pdtfnn(nn) = plot(vGe[nn:end], δtfvLa[nn:end, :], line=(2, :auto))
            # display(plot(pdtfa,pdtfb,layout=(2,1)))
            pdtfa_max = plot(δtfvL_max[:, 1], line=(2, :auto), label=string("δtfa_max"), xlabel=xlabel)
            pdtfb_max = plot(δtfvL_max[:, 2], line=(2, :auto), label=string("δtfb_max"))
            # display(plot(pdtfa_max,pdtfb_max,layout=(2,1)))
            display(plot(pdtfa, pdtfb, pdtfa_max, pdtfb_max, layout=(2, 2)))
        end
    end

    nvaplot2 = -6 .< v0log .< 1.5
    pδtfvLa(L1) = plot(vGe[nvaplot2], δtfvLa[nvaplot2, L1], line=(2, :auto), label=string("dtfa,L=", L1 - 1))

    pδtfvLb(L1) = plot(vGe[nvaplot2], δtfvLb[nvaplot2, L1], line=(2, :auto), label=string("dtfb,L=", L1 - 1))
    pδtfvLa(L1, nn) = plot(vGe[nn:end], δtfvLa[nn:end, L1], line=(2, :auto), label=string("dtfa,L=", L1 - 1))

    pδtfvLb(L1, nn) = plot(vGe[nn:end], δtfvLb[nn:end, L1], line=(2, :auto), label=string("dtfb,L=", L1 - 1))

    pδtfvLaL2(L1) = display(plot(vGe, δtfvLa[1:end, L1] .* vGe .^ (L1 + 1), label=string("dtfa*vL2,L=", L1 - 1)))
    pδtfvLaL2(L1, j) = display(plot(vGe, δtfvLa[1:end, L1] .* vGe .^ (L1 + 1 + j), label=string("dtfa*vL2j,L=", L1 - 1, ",j=", j)))
    pδtfvLbL2(L1) = display(plot(vGe, δtfvLb[1:end, L1] .* vGe .^ (L1 + 1), label=string("dtfb*vL2,L=", L1 - 1)))
    pδtfvLbL2(L1, j) = display(plot(vGe, δtfvLb[1:end, L1] .* vGe .^ (L1 + 1 + j), label=string("dtfb*vL2j,L=", L1 - 1, ",j=", j)))
    δtfvLaL2(L1, j) = δtfvLa[end, L1] .* vGe[end] .^ (L1 + 1 + j)
    δtfvLbL2(L1, j) = δtfvLb[end, L1] .* vGe[end] .^ (L1 + 1 + j)
end