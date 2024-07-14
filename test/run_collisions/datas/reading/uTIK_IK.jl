
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

    pIab = plot(tplot[tvec],(Iabt[tvec].-Iabt[1])*neps,line=(wline,:auto),
               label="ΔIₐᵦ[ε]",xlabel=title_model)
    ylabel = string("Kₛ/Kₛ₀ - 1")
    pKab = plot(tplot[tvec],(Kabt[tvec]/Kabt[1].-1)*neps,line=(wline,:auto),
               label="RΔKₐᵦ[ε]",ylabel=ylabel,xlabel=title_alg)

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
    Tat = (2Kat - Iat.^2 ./ (ma[2] * na[1])) ./ (3na[1])
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
    pua = plot(tplot[tvec],uat[tvec],line=(wline,:auto),label="a",ylabel=ylabel)
    pub = plot!(tplot[tvec],ubt[tvec],line=(wline,:auto),label="b",title=title_spices)
    ylabel = string("uᵏ⁺¹/uᵏ-1")
    pRDua = plot(tplot[tvec],RDuat[tvec],line=(wline,:auto),label="a",ylabel=ylabel)
    pRDub = plot(tplot[tvec],RDubt[tvec],line=(wline,:auto),label="b")
    ylabel = string("T [keV]")
    pTa = plot(tplot[tvec],Tat[tvec],line=(wline,:auto),label="a",ylabel=ylabel)
    pTb = plot!(tplot[tvec],Tbt[tvec],line=(wline,:auto),label="b",title=title_nv_nMod)

    ylabel = string("Du")
    pDu = plot(tplot[tvec],(abs.(Duabt[tvec])),line=(wline,:auto),
                    ylabel=ylabel,yscale=:log10)
    ylabel = string("DT")
    pDT = plot(tplot[tvec],(abs.(DTabt[tvec])),line=(wline,:auto),
                    ylabel=ylabel,yscale=:log10)
    
    puTK = display(plot(pua,pTa,pDu,pDT,pRDua,pRDub,pIab,pKab,layout=(4,2)))

    plot(pua,pTa,pDu,pDT,pRDua,pRDub,pIab,pKab,layout=(4,2))
    savefig(string(file_fig_file,"_nIK.png"))
end
