is_xscale_log10 = true

# [Dt, Duab, DTab, us, Ts]                  # `τ₀`
if isfile(file_Ms_nModa) && isfile(file_Ms_nIK)

    if is_xscale_log10
        xscale0 = :log10
        xscale0 = :norm
    else
        xscale0 = :norm
    end
    Dt = diff(tplot[tvec])
    ylabel="Δt"
    label = string("Dt,Nt=",Nt * Nt_save)
    pDt = plot(tplot[tvec][2:end],Dt/Nt_save,ylabel=ylabel,label=label,title=title_TEk)

    RDt = (Dt/Nt_save ./ tplot[tvec][2:end])
    if is_fixed_timestep
        label = string("RDt,Nτ=",Nτ_fix)
    else
        label = string("RDt")
    end
    pRDt = plot(tplot[tvec][2:end],RDt,label=label,yscale=:log10,title=title_nv_nMod)

    pt = plot(pDt,pRDt,layout=(1,2))

    # ylabel = string("log10(Du)")
    # pDu = plot(tplot[tvec],log10.(abs.(Duabt[tvec])),line=(wline,:auto),ylabel=ylabel)
    # ylabel = string("log10(DT)")
    # pDT = plot(tplot[tvec],log10.(abs.(DTabt[tvec])),line=(wline,:auto),ylabel=ylabel)

    # pDuDT = plot(pDu,pDT,layout=(1,2))
    
    ylabel = string("u [Mms]")

    if nModat == 1
        uati1 = (nuTat.ua1 .* vatht)
        uati = nuTat.na1 .* uati1
        pua1 = plot(tplot[tvec],uati1[tvec],line=(wline,:auto),label="a",xlabel=title_model)
    elseif nModat == 2
        uati1 = (nuTat.ua1 .* vatht)
        uati2 = (nuTat.ua2 .* vatht)
        uati = nuTat.na1 .* uati1 + nuTat.na2 .* uati2
        pua1 = plot(tplot[tvec],uati1[tvec],line=(wline,:auto),label="a₁",xlabel=title_model)
        pua2 = plot!(tplot[tvec],uati2[tvec],line=(wline,:auto),label="a₂")
    elseif nModat == 3
        uati1 = (nuTat.ua1 .* vatht)
        uati2 = (nuTat.ua2 .* vatht)
        uati3 = (nuTat.ua3 .* vatht)
        uati = nuTat.na1 .* uati1 + nuTat.na2 .* uati2 + nuTat.na3 .* uati3
        pua1 = plot(tplot[tvec],uati1[tvec],line=(wline,:auto),label="a₁",xlabel=title_model)
        pua2 = plot!(tplot[tvec],uati2[tvec],line=(wline,:auto),label="a₂")
        pua3 = plot!(tplot[tvec],uati3[tvec],line=(wline,:auto),label="a₃")
    elseif nModat == 4
        uati1 = (nuTat.ua1 .* vatht)
        uati2 = (nuTat.ua2 .* vatht)
        uati3 = (nuTat.ua3 .* vatht)
        uati4 = (nuTat.ua4 .* vatht)
        uati = nuTat.na1 .* uati1 + nuTat.na2 .* uati2 + nuTat.na3 .* uati3 + nuTat.na4 .* uati4
        pua1 = plot(tplot[tvec],uati1[tvec],line=(wline,:auto),label="a₁",xlabel=title_model)
        pua2 = plot!(tplot[tvec],uati2[tvec],line=(wline,:auto),label="a₂")
        pua3 = plot!(tplot[tvec],uati3[tvec],line=(wline,:auto),label="a₃")
        pua4 = plot!(tplot[tvec],uati4[tvec],line=(wline,:auto),label="a₄")
    elseif nModat == 5
        uati1 = (nuTat.ua1 .* vatht)
        uati2 = (nuTat.ua2 .* vatht)
        uati3 = (nuTat.ua3 .* vatht)
        uati4 = (nuTat.ua4 .* vatht)
        uati5 = (nuTat.ua5 .* vatht)
        uati = nuTat.na1 .* uati1 + nuTat.na2 .* uati2 + nuTat.na3 .* uati3 + nuTat.na4 .* uati4 + nuTat.na5 .* uati5
        pua1 = plot(tplot[tvec],uati1[tvec],line=(wline,:auto),label="a₁",xlabel=title_model)
        pua2 = plot!(tplot[tvec],uati2[tvec],line=(wline,:auto),label="a₂")
        pua3 = plot!(tplot[tvec],uati3[tvec],line=(wline,:auto),label="a₃")
        pua4 = plot!(tplot[tvec],uati4[tvec],line=(wline,:auto),label="a₄")
        pua5 = plot!(tplot[tvec],uati5[tvec],line=(wline,:auto),label="a₅")
    else
        sdbfgn
    end
    # pua0 = plot!(tplot[tvec],uat[tvec],line=(wline,:auto),label="aₛ")

    if nModbt == 1
        ubti1 = (nuTbt.ub1 .* vbtht)
        ubti = nuTbt.nb1 .* ubti1
        # pub1 = plot!(tplot[tvec],ubti1[tvec] .* abs.(ubt[tvec]),line=(wline,:auto),label="ub1",legend=legendtL)
        pub1 = plot!(tplot[tvec],ubti[tvec],line=(wline,:auto),label="b",ylabel=ylabel,legend=legendtR)
    elseif nModbt == 2
        ubti1 = (nuTbt.ub1 .* vbtht)
        ubti2 = (nuTbt.ub2 .* vbtht)
        ubti = nuTbt.nb1 .* ubti1 + nuTbt.nb2 .* ubti2
        pub1 = plot!(tplot[tvec],ubti1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
        pub2 = plot!(tplot[tvec],ubti2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
    elseif nModbt == 3
        ubti1 = (nuTbt.ub1 .* vbtht)
        ubti2 = (nuTbt.ub2 .* vbtht)
        ubti3 = (nuTbt.ub3) .* vbtht
        ubti = nuTbt.nb1 .* ubti1 + nuTbt.nb2 .* ubti2 + nuTbt.nb3 .* ubti3
        pub1 = plot!(tplot[tvec],ubti1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
        pub2 = plot!(tplot[tvec],ubti2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
        pub3 = plot!(tplot[tvec],ubti3[tvec],line=(wline,:auto),label="b₃")
    elseif nModbt == 4
        ubti1 = (nuTbt.ub1 .* vbtht)
        ubti2 = (nuTbt.ub2 .* vbtht)
        ubti3 = (nuTbt.ub3 .* vbtht)
        ubti4 = (nuTbt.ub4 .* vbtht)
        ubti = nuTbt.nb1 .* ubti1 + nuTbt.nb2 .* ubti2 + nuTbt.nb3 .* ubti3 + nuTbt.nb4 .* ubti4
        pub1 = plot(tplot[tvec],ubti1[tvec],line=(wline,:auto),label="a₁",xlabel=title_model)
        pub2 = plot!(tplot[tvec],ubti2[tvec],line=(wline,:auto),label="a₂")
        pub3 = plot!(tplot[tvec],ubti3[tvec],line=(wline,:auto),label="a₃")
        pub4 = plot!(tplot[tvec],ubti4[tvec],line=(wline,:auto),label="a₄")
    elseif nModbt == 5
        ubti1 = (nuTbt.ub1 .* vbtht)
        ubti2 = (nuTbt.ub2 .* vbtht)
        ubti3 = (nuTbt.ub3 .* vbtht)
        ubti4 = (nuTbt.ub4 .* vbtht)
        ubti5 = (nuTbt.ub5 .* vbtht)
        ubti = nuTbt.nb1 .* ubti1 + nuTbt.nb2 .* ubti2 + nuTbt.nb3 .* ubti3 + nuTbt.nb4 .* ubti4 + nuTbt.nb5 .* ubti5
        pub1 = plot(tplot[tvec],ubti1[tvec],line=(wline,:auto),label="a₁",xlabel=title_model)
        pub2 = plot!(tplot[tvec],ubti2[tvec],line=(wline,:auto),label="a₂")
        pub3 = plot!(tplot[tvec],ubti3[tvec],line=(wline,:auto),label="a₃")
        pub4 = plot!(tplot[tvec],ubti4[tvec],line=(wline,:auto),label="a₄")
        pub5 = plot!(tplot[tvec],ubti5[tvec],line=(wline,:auto),label="a₅")
    else
        sdbfgn
    end
    # pub0 = plot!(tplot[tvec],ubt[tvec],line=(wline,:auto),label="bₛ")

    if ns == 2
        Duuait = uati - uat
        label = "a"
        pDuui = plot(tplot[tvec],log.(abs.(Duuait[tvec]) .+ epsT),label=label,line=(2,:auto),
                        ylabel="log(ua-∑uai*vathi)")
        
        Duubit = ubti - ubt
        label = "b"
        pDuui = plot!(tplot[tvec],log.(abs.(Duubit[tvec]) .+ epsT),label=label,line=(2,:auto),
                        xlabel="t[τ₀]",xscale=:log10)
        display(pDuui)
    end

    if ns ≥ 3
        if nModct == 1
            ucti1 = (nuTct.uc1 .* vctht)
            ucti = nuTct.nc1 .* ucti1
            # puc1 = plot!(tplot[tvec],nuTct.uc1[tvec] .* abs.(uct[tvec]),line=(wline,:auto),label="uc1",legend=legendtL)
            puc1 = plot!(tplot[tvec],ucti1[tvec],line=(wline,:auto),label="c",ylabel=ylabel,legend=legendtR)
        elseif nModct == 2
            ucti1 = (nuTct.uc1 .* vctht)
            ucti2 = (nuTct.uc2 .* vctht)
            ucti = nuTct.nc1 .* ucti1 + nuTct.nc2 .* ucti2
            puc1 = plot!(tplot[tvec],ucti1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
            puc2 = plot!(tplot[tvec],ucti2[tvec],line=(wline,:auto),label="c₂",legend=legendtR)
        elseif nModct == 3
            ucti1 = (nuTct.uc1 .* vctht)
            ucti2 = (nuTct.uc2 .* vctht)
            ucti3 = (nuTct.uc3) .* vctht
            ucti = nuTct.nc1 .* ucti1 + nuTct.nc2 .* ucti2 + nuTct.nc3 .* ucti3
            puc1 = plot!(tplot[tvec],ucti1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
            puc2 = plot!(tplot[tvec],ucti2[tvec],line=(wline,:auto),label="c₂",legend=legendtR)
            puc3 = plot!(tplot[tvec],ucti3[tvec],line=(wline,:auto),label="c₃")
        else
            sdbfgn
        end
        puc0 = plot!(tplot[tvec],uct[tvec],line=(wline,:auto),label="cₛ")

        if ns == 3
            Duuait = uati - uat
            label = "a"
            pDuui = plot(tplot,log.(abs.(Duuait) .+ epsT),label=label,line=(2,:auto),ylabel="log(ua-∑uai*vathi)")
            
            Duubit = ubti - ubt
            label = "b"
            pDuui = plot!(tplot,log.(abs.(Duubit) .+ epsT),label=label,line=(2,:auto),xlabel="t[τ₀]")

            Duucit = ucti - uct
            label = "c"
            pDuui = plot!(tplot,log.(abs.(Duucit) .+ epsT),label=label,line=(2,:auto))
            display(pDuui)
        end
    end


    ylabel = string("T [keV]")
    T_DMmsTk = (Dₐ * Mms^2 / Tk)

    isp3 = 1
    if nModat == 1
        pTa1 = plot(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath1[tvec] .* vatht[tvec]).^2,
                line=(wline,:auto),label="a",
                xscale=xscale0)
    elseif nModat == 2
        pTa1 = plot(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath1[tvec] .* vatht[tvec]).^2,line=(wline,:auto),label="a₁")
        pTa2 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath2[tvec] .* vatht[tvec]).^2,
                line=(wline,:auto),label="a₂",
                xscale=xscale0)
    elseif nModat == 3
        pTa1 = plot(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath1[tvec] .* vatht[tvec]).^2,
                line=(wline,:auto),label="a₁",
                xscale=xscale0)
        pTa2 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath2[tvec] .* vatht[tvec]).^2,line=(wline,:auto),label="a₂")
        pTa3 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath3[tvec] .* vatht[tvec]).^2,line=(wline,:auto),label="a₃")
    elseif nModat == 4
        pTa1 = plot(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath1[tvec] .* vatht[tvec]).^2,
                line=(wline,:auto),label="a₁",
                xscale=xscale0)
        pTa2 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath2[tvec] .* vatht[tvec]).^2,line=(wline,:auto),label="a₂")
        pTa3 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath3[tvec] .* vatht[tvec]).^2,line=(wline,:auto),label="a₃")
        pTa4 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath4[tvec] .* vatht[tvec]).^2,line=(wline,:auto),label="a₄")
    elseif nModat == 5
        pTa1 = plot(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath1[tvec] .* vatht[tvec]).^2,
                line=(wline,:auto),label="a₁",
                xscale=xscale0)
        pTa2 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath2[tvec] .* vatht[tvec]).^2,line=(wline,:auto),label="a₂")
        pTa3 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath3[tvec] .* vatht[tvec]).^2,line=(wline,:auto),label="a₃")
        pTa4 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath4[tvec] .* vatht[tvec]).^2,line=(wline,:auto),label="a₄")
        pTa5 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTat.vath5[tvec] .* vatht[tvec]).^2,line=(wline,:auto),label="a₅")
    else
        sdbfgn
    end

    isp3 = 2
    if nModbt == 1
        # pTb1 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth1[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₁")
        pTb1 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth1[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b",xlabel=title_alg,ylabel=ylabel,legend=legendtR)
    elseif nModbt == 2
        pTb1 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth1[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₁",xlabel=title_alg,ylabel=ylabel)
        pTb2 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth2[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₂",legend=legendtR)
    elseif nModbt == 3
        pTb1 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth1[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₁",xlabel=title_alg,ylabel=ylabel)
        pTb2 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth2[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₂",legend=legendtR)
        pTb3 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth3[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₃")
    elseif nModbt == 4
        pTb1 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth1[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₁",xlabel=title_alg,ylabel=ylabel)
        pTb2 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth2[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₂",legend=legendtR)
        pTb3 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth3[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₃")
        pTb4 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth4[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₄")
    elseif nModbt == 5
        pTb1 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth1[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₁",xlabel=title_alg,ylabel=ylabel)
        pTb2 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth2[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₂",legend=legendtR)
        pTb3 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth3[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₃")
        pTb4 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth4[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₄")
        pTb5 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTbt.vbth5[tvec] .* vbtht[tvec]).^2,line=(wline,:auto),label="b₅")
    else
        sdbfgn
    end

    if ns ≥ 3
        isp3 = 3
        if nModct == 1
            # pTc1 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTct.vcth1[tvec] .* vctht[tvec]).^2,line=(wline,:auto),label="c₁")
            pTc1 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTct.vcth1[tvec] .* vctht[tvec]).^2,line=(wline,:auto),label="c",xlabel=title_alg,ylabel=ylabel,legend=legendtR)
        elseif nModct == 2
            pTc1 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTct.vcth1[tvec] .* vctht[tvec]).^2,line=(wline,:auto),label="c₁",xlabel=title_alg,ylabel=ylabel)
            pTc2 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTct.vcth2[tvec] .* vctht[tvec]).^2,line=(wline,:auto),label="c₂",legend=legendtR)
        elseif nModct == 3
            pTc1 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTct.vcth1[tvec] .* vctht[tvec]).^2,line=(wline,:auto),label="c₁",xlabel=title_alg,ylabel=ylabel)
            pTc2 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTct.vcth2[tvec] .* vctht[tvec]).^2,line=(wline,:auto),label="c₂",legend=legendtR)
            pTc3 = plot!(tplot[tvec],0.5T_DMmsTk * ma[isp3]*(nuTct.vcth3[tvec] .* vctht[tvec]).^2,line=(wline,:auto),label="c₃")
        else
            sdbfgn
        end
    end

    puT = plot(pub1,pTb1,layout=(1,2))

    plot(pt,puT,layout=(2,1))
    savefig(string(file_fig_file,"_uTs.png"))

    display(plot(pt,puT,layout=(2,1)))
end
