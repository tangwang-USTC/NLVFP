
# [u, T, I, K]
if isfile(file_Ms_nIK)
    nIKt = CSV.File(read(file_Ms_nIK)) |> DataFrame
    
    if missing_deal == :nothing
    elseif missing_deal == :drop
        dropmissing!(nIKt)
    elseif missing_deal == :NaN
        replace!.(eachcol(nIKt), missing => NaN)
    elseif missing_deal == :zero
        replace!.(eachcol(nIKt), missing => 0.0)
    end
    unique!(nIKt,1)            # due to t = x[1]
    tplot = nIKt.t * (tdd / τ₀)
    Nt = length(tplot)

    if ns == 2
        Iabt = nIKt.Ia + nIKt.Ib
        Kabt = nIKt.Ka + nIKt.Kb
        DIabt = Iabt .- Iabt[1]
        DKabt = Kabt .- Kabt[1]
    
        tvec = tplot_min .< tplot .≤ tplot_max
        tvec[1:dkivv2] .= false
    
        # pIa = plot(tplot[tvec],nIKt.Ia[tvec],line=(wline,:auto),label="Iₐ")
        # pIb = plot(tplot[tvec],nIKt.Ib[tvec],line=(wline,:auto),label="Iᵦ")
        # pKa = plot(tplot[tvec],nIKt.Ka[tvec],line=(wline,:auto),label="Kₐ")
        # pKb = plot(tplot[tvec],nIKt.Kb[tvec],line=(wline,:auto),label="Kᵦ")
    
        CRDI = abs.(Iabt[tvec].-Iabt[1])
        if is_MultiCase
            CRDIM[iCase] = deepcopy(CRDI)
        end
        pIab = plot(tplot[tvec],CRDI*neps,line=(wline,:auto),label="ΔIₐᵦ[ε]",xlabel=title_model)

        ylabel = string("Kₛ/Kₛ₀ - 1")
        CRDK = abs.(Kabt[tvec]/Kabt[1].-1)
        if is_MultiCase
            CRDKM[iCase] = deepcopy(CRDK)
        end
        pKab = plot(tplot[tvec],CRDK*neps,line=(wline,:auto),label="RΔKₐᵦ[ε]",ylabel=ylabel,xlabel=title_alg)
    
        # display(plot(pIa,pKa,pIb,pKb,pIab,pKab,layout=(3,2)))
    
        Ka000 = 3/4 * ρa .* vth.^2 + 1/2 * Ia.^2 ./ ρa
        ρat = ma .* na
        Kat = nIKt.Ka * K_unit
        Kbt = nIKt.Kb * K_unit
        Iat = nIKt.Ia * I_unit
        Ibt = nIKt.Ib * I_unit
        uhat = 1.5^0.5 * Iat ./ (2ρat[1] .* Kat - Iat .^2).^0.5    # uha = 1.5^0.5 * Ia ./ (2ρa .* Ka - Ia .^2).^0.5
        uhbt = 1.5^0.5 * Ibt ./ (2ρat[2] .* Kbt - Ibt .^2).^0.5
        vatht = (2/3 * (2Kat ./ ρat[1] - (Iat ./ ρat[1]).^2)).^0.5    # vath = √(2/3 * (2Ka ./ ρa - (Ia ./ ρa).^2))
        vbtht = (2/3 * (2Kbt ./ ρat[2] - (Ibt ./ ρat[2]).^2)).^0.5
        uat = uhat .* vatht / v_unit
        ubt = uhbt .* vbtht / v_unit
        vatht /= v_unit                  # [Mms]
        vbtht /= v_unit
    
        # @show vth * (vd/Mms)
        # @show ua * (vd/Mms)
        
        # Tbtt = ma[2] * (2Kbt ./ ρat[2] - (Ibt ./ ρat[2]).^2) / 3
        Tat = (2Kat - Iat.^2 ./ (ma[1] * na[1])) ./ (3na[1])
        Tbt = (2Kbt - Ibt.^2 ./ (ma[2] * na[2])) ./ (3na[2])
        
        Tat /= T_unit
        Tbt /= T_unit
    
        # RDuhat = zeros(Nt)
        # RDuhat[1:end-1] = diff(uhat) ./ uhat[2:end]
        # RDuhat[end] = RDuhat[end-1]
        # RDuhbt = zeros(Nt)
        # RDuhbt[1:end-1] = diff(uhbt) ./ uhbt[2:end]
        # RDuhbt[end] = RDuhbt[end-1]
    
        RDuat = zeros(Nt)
        RDuat[1:end-1] = diff(uat) ./ uat[2:end]
        RDuat[end] = RDuat[end-1]
        RDubt = zeros(Nt)
        RDubt[1:end-1] = diff(ubt) ./ ubt[2:end]
        RDubt[end] = RDubt[end-1]
     
        DTabt = Tat - Tbt
        DKabt = nIKt.Ka - nIKt.Kb
        DIabt = nIKt.Ia - nIKt.Ib
        Duabt = uat - ubt
        Duhabt = uhat - uhbt
        D0DTabt = DTabt .- DTabt[1]
        RDTabt = DTabt ./ (Tat + Tbt)
    
        ylabel = string("u [Mms]")
        pua = plot(tplot[tvec],uat[tvec],line=(wline,:auto),label="a",ylabel=ylabel,title=title_spices)
        pub = plot!(tplot[tvec],ubt[tvec],line=(wline,:auto),label="b")
        ylabel = string("uᵏ⁺¹/uᵏ-1")
        pRDua = plot(tplot[tvec],RDuat[tvec],line=(wline,:auto),label="a",ylabel=ylabel)
        pRDub = plot(tplot[tvec],RDubt[tvec],line=(wline,:auto),label="b")
        ylabel = string("T [keV]")
        pTa = plot(tplot[tvec],Tat[tvec],line=(wline,:auto),label="a",ylabel=ylabel,title=title_nv_nMod)
        pTb = plot!(tplot[tvec],Tbt[tvec],line=(wline,:auto),label="b")
        ylabel = string("T̂ - 1 [ε]")
        errThat = nIKt.errTha[tvec]*neps
        errThat[abs.(errThat) .> 1e8] .= 0.0
        perrTha = plot(tplot[tvec],errThat,line=(wline,:auto),label="a",ylabel=ylabel)
        errThbt = nIKt.errThb[tvec]*neps
        errThbt[abs.(errThbt) .> 1e8] .= 0.0
        perrThb = plot(tplot[tvec],errThbt,line=(wline,:auto),label="b")
        
        puTK = display(plot(pua,pTa,perrTha,perrThb,pRDua,pRDub,pIab,pKab,layout=(4,2)))
    
        plot(pua,pTa,perrTha,perrThb,pRDua,pRDub,pIab,pKab,layout=(4,2))
        savefig(string(file_fig_file,"_nIK.png"))
    else
        if ns == 3
            Iabt = nIKt.Ia + nIKt.Ib + nIKt.Ic
            Kabt = nIKt.Ka + nIKt.Kb + nIKt.Kc
            DIabt = Iabt .- Iabt[1]
            DKabt = Kabt .- Kabt[1]
        
            tvec = tplot_min .< tplot .≤ tplot_max
            tvec[1:dkivv2] .= false
        
            # pIa = plot(tplot[tvec],nIKt.Ia[tvec],line=(wline,:auto),label="Iₐ")
            # pIb = plot(tplot[tvec],nIKt.Ib[tvec],line=(wline,:auto),label="Iᵦ")
            # pIc = plot(tplot[tvec],nIKt.Ic[tvec],line=(wline,:auto),label="Iᵪ")
            # pKa = plot(tplot[tvec],nIKt.Ka[tvec],line=(wline,:auto),label="Kₐ")
            # pKb = plot(tplot[tvec],nIKt.Kb[tvec],line=(wline,:auto),label="Kᵦ")
            # pKc = plot(tplot[tvec],nIKt.Kc[tvec],line=(wline,:auto),label="Kᵪ")
        
            CRDI = abs.(Iabt[tvec].-Iabt[1])
            if is_MultiCase
                CRDIM[iCase] = deepcopy(CRDI)
            end
            pIab = plot(tplot[tvec],CRDI*neps,line=(wline,:auto),label="ΔIₐᵦ[ε]",xlabel=title_model)
            
            ylabel = string("Kₛ/Kₛ₀ - 1")
            CRDK = abs.(Kabt[tvec]/Kabt[1].-1)
            if is_MultiCase
                CRDKM[iCase] = deepcopy(CRDK)
            end
            pKab = plot(tplot[tvec],CRDK*neps,line=(wline,:auto),label="RΔKₐᵦ[ε]",
                            ylabel=ylabel,xlabel=title_alg)
        
            # display(plot(pIa,pKa,pIb,pKb,pIab,pKab,layout=(3,2)))
        
            Ka000 = 3/4 * ρa .* vth.^2 + 1/2 * Ia.^2 ./ ρa
            ρat = ma .* na
            Kat = nIKt.Ka * K_unit
            Kbt = nIKt.Kb * K_unit
            Kct = nIKt.Kc * K_unit

            Iat = nIKt.Ia * I_unit
            Ibt = nIKt.Ib * I_unit
            Ict = nIKt.Ic * I_unit

            uhat = 1.5^0.5 * Iat ./ (2ρat[1] .* Kat - Iat .^2).^0.5    # uha = 1.5^0.5 * Ia ./ (2ρa .* Ka - Ia .^2).^0.5
            uhbt = 1.5^0.5 * Ibt ./ (2ρat[2] .* Kbt - Ibt .^2).^0.5
            uhct = 1.5^0.5 * Ict ./ (2ρat[3] .* Kct - Ict .^2).^0.5

            vatht = (2/3 * (2Kat ./ ρat[1] - (Iat ./ ρat[1]).^2)).^0.5    # vath = √(2/3 * (2Ka ./ ρa - (Ia ./ ρa).^2))
            vbtht = (2/3 * (2Kbt ./ ρat[2] - (Ibt ./ ρat[2]).^2)).^0.5
            vctht = (2/3 * (2Kct ./ ρat[3] - (Ict ./ ρat[3]).^2)).^0.5

            uat = uhat .* vatht / v_unit
            ubt = uhbt .* vbtht / v_unit
            uct = uhct .* vctht / v_unit

            vatht /= v_unit                  # [Mms]
            vbtht /= v_unit
            vctht /= v_unit
        
            # @show vth * (vd/Mms)
            # @show ua * (vd/Mms)
            
            # Tbtt = ma[2] * (2Kbt ./ ρat[2] - (Ibt ./ ρat[2]).^2) / 3
            Tat = (2Kat - Iat.^2 ./ (ma[1] * na[1])) ./ (3na[1])
            Tbt = (2Kbt - Ibt.^2 ./ (ma[2] * na[2])) ./ (3na[2])
            Tct = (2Kct - Ict.^2 ./ (ma[3] * na[3])) ./ (3na[3])
            
            Tat /= T_unit
            Tbt /= T_unit
            Tct /= T_unit
        
            RDuat = zeros(Nt)
            RDuat[1:end-1] = diff(uat) ./ uat[2:end]
            RDuat[end] = RDuat[end-1]

            RDubt = zeros(Nt)
            RDubt[1:end-1] = diff(ubt) ./ ubt[2:end]
            RDubt[end] = RDubt[end-1]

            RDuct = zeros(Nt)
            RDuct[1:end-1] = diff(uct) ./ uct[2:end]
            RDuct[end] = RDuct[end-1]
        
            ylabel = string("u [Mms]")
            pua = plot(tplot[tvec],uat[tvec],line=(wline,:auto),label="a",ylabel=ylabel,title=title_TEk)
            pub = plot!(tplot[tvec],ubt[tvec],line=(wline,:auto),label="b")
            puc = plot!(tplot[tvec],uct[tvec],line=(wline,:auto),label="c")

            ylabel = string("uᵏ⁺¹/uᵏ-1")
            pRDua = plot(tplot[tvec],RDuat[tvec],line=(wline,:auto),label="a",ylabel=ylabel)
            pRDub = plot(tplot[tvec],RDubt[tvec],line=(wline,:auto),label="b")
            pRDuc = plot(tplot[tvec],RDuct[tvec],line=(wline,:auto),label="c")

            ylabel = string("T [keV]")
            pTa = plot(tplot[tvec],Tat[tvec],line=(wline,:auto),label="a",ylabel=ylabel,title=title_nv_nMod)
            pTb = plot!(tplot[tvec],Tbt[tvec],line=(wline,:auto),label="b")
            pTc = plot!(tplot[tvec],Tct[tvec],line=(wline,:auto),label="c")

            ylabel = string("T̂ - 1 [ε]")
            errThat = nIKt.errTha[tvec]*neps
            errThat[abs.(errThat) .> 1e8] .= 0.0
            perrTha = plot(tplot[tvec],errThat,line=(wline,:auto),label="a",ylabel=ylabel)
            errThbt = nIKt.errThb[tvec]*neps
            errThbt[abs.(errThbt) .> 1e8] .= 0.0
            perrThb = plot(tplot[tvec],errThbt,line=(wline,:auto),label="b")
            errThct = nIKt.errThc[tvec]*neps
            errThct[abs.(errThct) .> 1e8] .= 0.0
            perrThc = plot(tplot[tvec],errThct,line=(wline,:auto),label="c")
            
            puT = plot(pua,pTa,layout=(1,2))
            pIK = plot(pIab,pKab,layout=(1,2))
            pTh = plot(perrTha,perrThb,perrThc,layout=(1,3))
            pRDu = plot(pRDua,pRDub,pRDuc,layout=(1,3))
            puTK = display(plot(puT,pIK,pTh,pRDu,layout=(4,1)))
        
            plot(puT,pIK,pTh,pRDu,layout=(4,1))
            savefig(string(file_fig_file,"_nIK.png"))
            if spices0[3] == :α
                ylabel = string("T [keV]")
                pTa22 = plot(tplot[tvec],Tat[tvec],line=(wline,:auto),label="a",ylabel=ylabel,title=title_nv_nMod)
                pTb = plot!(tplot[tvec],Tbt[tvec],line=(wline,:auto),label="b")
                
                savefig(string(file_fig_file,"_uTs12.png"))
            
                display(plot(pTa22))
            end
        elseif ns == 4
            edfghj
        end
    end

end