
# [Mhc, dtMhc]
if is_moments_out
    if isfile(file_Ms_Mhcla)
        Mhclat = CSV.File(read(file_Ms_Mhcla)) |> DataFrame
        if 1 == 1
            if missing_deal == :nothing
            elseif missing_deal == :drop
                dropmissing!(Mhclat)
            elseif missing_deal == :NaN
                replace!.(eachcol(Mhclat), missing => NaN)
            elseif missing_deal == :zero
                replace!.(eachcol(Mhclat), missing => 0.0)
            end
            unique!(Mhclat,1)            # due to t = x[1]
            Mhclat[1,2:end] = Mhclat[2,2:end]
            tplot = Mhclat.t * (tdd / τ₀)
            Nt = length(tplot)
        
            tvec = tplot_min .< tplot .≤ tplot_max
            tvec[1:dkivv2] .= false
            Dt = diff(tplot[tvec])
            # tvec[1:9] .= false
            isp33 = 1
            nnjM = nMjMs[isp33]
            title_nMod = string("nMod=",nMod0)
            ylabel = string("Mhca")
            ylabelD = string("∂ₜMhca")
            if is_MjMs_max
                if norm(ua[isp33]) ≤ epsT10
                    if nnjM == 2
                        pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                    elseif nnjM == 3
                        pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                        a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                    elseif nnjM == 4
                        pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                        a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                        a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆",ylabel=ylabelD)
                    elseif nnjM == 5
                        pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="a₈")
                        a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                        a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆",ylabel=ylabelD)
                        a = Mhclat.Mhc8[tvec][2:end] ./ Mhclat.Mhc8[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₈",ylabel=ylabelD)
                    elseif nnjM == 6
                        pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="a₈")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="a₁₀")
                        a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                        a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆",ylabel=ylabelD)
                        a = Mhclat.Mhc8[tvec][2:end] ./ Mhclat.Mhc8[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₈",ylabel=ylabelD)
                        a = Mhclat.Mhc10[tvec][2:end] ./ Mhclat.Mhc10[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₀",ylabel=ylabelD)
                    elseif nnjM == 7
                        pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="a₈")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="a₁₀")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="a₁₂")
                        a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                        a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆",ylabel=ylabelD)
                        a = Mhclat.Mhc8[tvec][2:end] ./ Mhclat.Mhc8[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₈",ylabel=ylabelD)
                        a = Mhclat.Mhc10[tvec][2:end] ./ Mhclat.Mhc10[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₀",ylabel=ylabelD)
                        a = Mhclat.Mhc12[tvec][2:end] ./ Mhclat.Mhc12[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₂",ylabel=ylabelD)
                    elseif nnjM == 8
                        pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="a₈")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="a₁₀")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="a₁₂")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc14[tvec].- 1.0 ,line=(wline,:auto),label="a₁₄")
                        a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                        a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆",ylabel=ylabelD)
                        a = Mhclat.Mhc8[tvec][2:end] ./ Mhclat.Mhc8[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₈",ylabel=ylabelD)
                        a = Mhclat.Mhc10[tvec][2:end] ./ Mhclat.Mhc10[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₀",ylabel=ylabelD)
                        a = Mhclat.Mhc12[tvec][2:end] ./ Mhclat.Mhc12[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₂",ylabel=ylabelD)
                        a = Mhclat.Mhc14[tvec][2:end] ./ Mhclat.Mhc14[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₄",ylabel=ylabelD)
                    else
                        pMhca = plot()
                    end
                else
                    if is_plot_Mhc_l_1
                        if nnjM == 2
                            pMhca = plot(tplot[tvec],Mhclat.Mhc1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0,line=(wline,:auto),label="a₂")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        elseif nnjM == 4
                            pMhca = plot(tplot[tvec],Mhclat.Mhc1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc3[tvec],line=(wline,:auto),label="a₃")
                
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄")
                        elseif nnjM == 6
                            pMhca = plot(tplot[tvec],Mhclat.Mhc1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc3[tvec] ,line=(wline,:auto),label="a₃")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc5[tvec],line=(wline,:auto),label="a₅")
                
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄")
                            a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆")
                        elseif nnjM == 8
                            pMhca = plot(tplot[tvec],Mhclat.Mhc1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc3[tvec] ,line=(wline,:auto),label="a₃")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc5[tvec],line=(wline,:auto),label="a₅")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc7[tvec],line=(wline,:auto),label="a₇")
                
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="a₈")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄")
                            a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆")
                            a = Mhclat.Mhc8[tvec][2:end] ./ Mhclat.Mhc8[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₈")
                        elseif nnjM == 10
                            pMhca = plot(tplot[tvec],Mhclat.Mhc1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc3[tvec] ,line=(wline,:auto),label="a₃")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc5[tvec],line=(wline,:auto),label="a₅")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc7[tvec],line=(wline,:auto),label="a₇")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc9[tvec],line=(wline,:auto),label="a₉")
                
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="a₈")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="a₁₀")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄")
                            a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆")
                            a = Mhclat.Mhc8[tvec][2:end] ./ Mhclat.Mhc8[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₈")
                            a = Mhclat.Mhc10[tvec][2:end] ./ Mhclat.Mhc10[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₀")
                        elseif nnjM == 12
                            pMhca = plot(tplot[tvec],Mhclat.Mhc1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc3[tvec] ,line=(wline,:auto),label="a₃")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc5[tvec],line=(wline,:auto),label="a₅")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc7[tvec],line=(wline,:auto),label="a₇")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc9[tvec],line=(wline,:auto),label="a₉")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc11[tvec] ,line=(wline,:auto),label="a₁₁")
                
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="a₈")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="a₁₀")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="a₁₂")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄")
                            a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆")
                            a = Mhclat.Mhc8[tvec][2:end] ./ Mhclat.Mhc8[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₈")
                            a = Mhclat.Mhc10[tvec][2:end] ./ Mhclat.Mhc10[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₀")
                            a = Mhclat.Mhc12[tvec][2:end] ./ Mhclat.Mhc12[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₂")
                        else
                            pMhca = plot()
                            pMhcD = plot()
                        end
                    else
                        if nnjM == 2
                            pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        elseif nnjM == 4
                            pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄")
                        elseif nnjM == 6
                            pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄")
                            a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆")
                        elseif nnjM == 8
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="a₈")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄")
                            a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆")
                            a = Mhclat.Mhc8[tvec][2:end] ./ Mhclat.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₈")
                        elseif nnjM == 10
                            pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="a₈")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="a₁₀")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄")
                            a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆")
                            a = Mhclat.Mhc8[tvec][2:end] ./ Mhclat.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₈")
                            a = Mhclat.Mhc10[tvec][2:end] ./ Mhclat.Mhc10[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₀")
                        elseif nnjM == 12
                            pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="a₈")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="a₁₀")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="a₁₂")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄")
                            a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆")
                            a = Mhclat.Mhc8[tvec][2:end] ./ Mhclat.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₈")
                            a = Mhclat.Mhc10[tvec][2:end] ./ Mhclat.Mhc10[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₀")
                            a = Mhclat.Mhc12[tvec][2:end] ./ Mhclat.Mhc12[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₁₂")
                        else
                            pMhca = plot()
                        end
                    end
                end
            else
                if norm(ua[isp33]) ≤ epsT10
                    if nMod[isp33] == 1
                        pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
    
                        a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                        pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                    elseif nMod[isp33] == 2
                        pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
    
                        a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                        pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                    elseif nMod[isp33] == 3
                        pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                        pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
                        a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                        pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                        a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                        pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆",ylabel=ylabelD)
                    else
                        pMhca = plot()
                        pMhcD = plot()
                    end
                else
                    if is_plot_Mhc_l_1
                        if nMod[isp33] == 1
                            pMhca = plot(tplot[tvec],Mhclat.Mhc1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                            
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        elseif nMod[isp33] == 2
                            pMhca = plot(tplot[tvec],Mhclat.Mhc1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc3[tvec],line=(wline,:auto),label="a₃")
                
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                        elseif nMod[isp33] == 3
                            pMhca = plot(tplot[tvec],Mhclat.Mhc1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc3[tvec] ,line=(wline,:auto),label="a₃")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc5[tvec],line=(wline,:auto),label="a₅")
                
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                            a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆",ylabel=ylabelD)
                        else
                            pMhca = plot()
                            pMhcD = plot()
                        end
                    else
                        if nMod[isp33] == 1
                            pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                        elseif nMod[isp33] == 2
                            pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                        elseif nMod[isp33] == 3
                            pMhca = plot(tplot[tvec],Mhclat.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="a₂",ylabel=ylabel)
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="a₄")
                            pMhca = plot!(tplot[tvec],Mhclat.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="a₆")
    
                            a = Mhclat.Mhc2[tvec][2:end] ./ Mhclat.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcD = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₂",ylabel=ylabelD)
                            a = Mhclat.Mhc4[tvec][2:end] ./ Mhclat.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₄",ylabel=ylabelD)
                            a = Mhclat.Mhc6[tvec][2:end] ./ Mhclat.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcD = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="a₆",ylabel=ylabelD)
                        else
                            pMhca = plot()
                            pMhcD = plot()
                        end
                    end
                end
            end
        end

        Mhclbt = CSV.File(read(file_Ms_Mhclb)) |> DataFrame
        if 1 == 1
            if missing_deal == :nothing
            elseif missing_deal == :drop
                dropmissing!(Mhclbt)
            elseif missing_deal == :NaN
                replace!.(eachcol(Mhclbt), missing => NaN)
            elseif missing_deal == :zero
                replace!.(eachcol(Mhclbt), missing => 0.0)
            end
            unique!(Mhclbt,1)            # due to t = x[1]
            Mhclbt[1,2:end] = Mhclbt[2,2:end]
            tplot = Mhclbt.t * (tdd / τ₀)
            Nt = length(tplot)
        
            tvec = tplot_min .< tplot .≤ tplot_max
            tvec[1:dkivv2] .= false
            Dt = diff(tplot[tvec])
            isp33 = 2
            nnjM = nMjMs[isp33]
            ylabel = string("Mhcb")
            ylabelD = string("∂ₜMhcb")
            if is_MjMs_max
                if norm(ua[isp33]) ≤ epsT10
                    if nnjM == 2
                        pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                    elseif nnjM == 3
                        pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                        a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                        a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                    elseif nnjM == 4
                        pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                        a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                        a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                        a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                    elseif nnjM == 5
                        pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                        a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                        a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                        a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                        a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈",ylabel=ylabelD)
                    elseif nnjM == 6
                        pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="b₁₀")
                        a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                        a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                        a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                        a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈",ylabel=ylabelD)
                        a = Mhclbt.Mhc10[tvec][2:end] ./ Mhclbt.Mhc10[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₀",ylabel=ylabelD)
                    elseif nnjM == 7
                        pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="b₁₀")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="b₁₂")
                        a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                        a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                        a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                        a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈",ylabel=ylabelD)
                        a = Mhclbt.Mhc10[tvec][2:end] ./ Mhclbt.Mhc10[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₀",ylabel=ylabelD)
                        a = Mhclbt.Mhc12[tvec][2:end] ./ Mhclbt.Mhc12[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₂",ylabel=ylabelD)
                    elseif nnjM == 8
                        pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="b₁₀")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="b₁₂")
                        pMhcb = plot!(tplot[tvec],Mhclbt.Mhc14[tvec].- 1.0 ,line=(wline,:auto),label="b₁₄")
                        a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                        a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                        a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                        a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈",ylabel=ylabelD)
                        a = Mhclbt.Mhc10[tvec][2:end] ./ Mhclbt.Mhc10[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₀",ylabel=ylabelD)
                        a = Mhclbt.Mhc12[tvec][2:end] ./ Mhclbt.Mhc12[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₂",ylabel=ylabelD)
                        a = Mhclbt.Mhc14[tvec][2:end] ./ Mhclbt.Mhc14[tvec][1:end-1] .- 1.0
                        a ./= Dt
                        pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₄",ylabel=ylabelD)
                    else
                        pMhcb = plot()
                    end
                else
                    if is_plot_Mhc_l_1
                        if nnjM == 2
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                        elseif nnjM == 4
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc3[tvec],line=(wline,:auto),label="b₃")
                
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄")
                        elseif nnjM == 6
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc3[tvec] ,line=(wline,:auto),label="b₃")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc5[tvec],line=(wline,:auto),label="b₅")
                
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄")
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆")
                        elseif nnjM == 8
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc3[tvec] ,line=(wline,:auto),label="b₃")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc5[tvec],line=(wline,:auto),label="b₅")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc7[tvec],line=(wline,:auto),label="b₇")
                
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄")
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆")
                            a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈")
                        elseif nnjM == 10
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc3[tvec] ,line=(wline,:auto),label="b₃")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc5[tvec],line=(wline,:auto),label="b₅")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc7[tvec],line=(wline,:auto),label="b₇")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc9[tvec],line=(wline,:auto),label="b₉")
                
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="b₁₀")
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄")
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆")
                            a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈")
                            a = Mhclbt.Mhc10[tvec][2:end] ./ Mhclbt.Mhc10[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₀")
                        elseif nnjM == 12
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc3[tvec] ,line=(wline,:auto),label="b₃")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc5[tvec],line=(wline,:auto),label="b₅")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc7[tvec],line=(wline,:auto),label="b₇")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc9[tvec],line=(wline,:auto),label="b₉")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc11[tvec] ,line=(wline,:auto),label="b₁₁")
                
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="b₁₀")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="b₁₂")
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄")
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆")
                            a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈")
                            a = Mhclbt.Mhc10[tvec][2:end] ./ Mhclbt.Mhc10[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₀")
                            a = Mhclbt.Mhc12[tvec][2:end] ./ Mhclbt.Mhc12[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₂")
                        else
                            pMhcb = plot()
                            pMhcDb = plot()
                        end
                    else
                        if nnjM == 2
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                        elseif nnjM == 4
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄")
                        elseif nnjM == 6
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄")
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆")
                        elseif nnjM == 8
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄")
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆")
                            a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈")
                        elseif nnjM == 10
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="b₁₀")
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄")
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆")
                            a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈")
                            a = Mhclbt.Mhc10[tvec][2:end] ./ Mhclbt.Mhc10[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₀")
                        elseif nnjM == 12
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="b₁₀")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="b₁₂")
    
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄")
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆")
                            a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈")
                            a = Mhclbt.Mhc10[tvec][2:end] ./ Mhclbt.Mhc10[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₀")
                            a = Mhclbt.Mhc12[tvec][2:end] ./ Mhclbt.Mhc12[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₂")
                        else
                            pMhcb = plot()
                            pMhcDb = plot()
                        end
                    end
                end
            else
                if norm(ua[isp33]) ≤ epsT10
                    if 1 == 1
                        if nnjM == 2
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                        elseif nnjM == 3
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                        elseif nnjM == 4
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                        elseif nnjM == 5
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                            a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈",ylabel=ylabelD)
                        elseif nnjM == 6
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="b₁₀")
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                            a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈",ylabel=ylabelD)
                            a = Mhclbt.Mhc10[tvec][2:end] ./ Mhclbt.Mhc10[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₀",ylabel=ylabelD)
                        elseif nnjM == 7
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="b₁₀")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="b₁₂")
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                            a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈",ylabel=ylabelD)
                            a = Mhclbt.Mhc10[tvec][2:end] ./ Mhclbt.Mhc10[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₀",ylabel=ylabelD)
                            a = Mhclbt.Mhc12[tvec][2:end] ./ Mhclbt.Mhc12[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₂",ylabel=ylabelD)
                        elseif nnjM == 8
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="b₈")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="b₁₀")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="b₁₂")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc14[tvec].- 1.0 ,line=(wline,:auto),label="b₁₄")
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                            a = Mhclbt.Mhc8[tvec][2:end] ./ Mhclbt.Mhc8[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₈",ylabel=ylabelD)
                            a = Mhclbt.Mhc10[tvec][2:end] ./ Mhclbt.Mhc10[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₀",ylabel=ylabelD)
                            a = Mhclbt.Mhc12[tvec][2:end] ./ Mhclbt.Mhc12[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc14[tvec][2:end] ./ Mhclbt.Mhc14[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₁₄",ylabel=ylabelD)
                        else
                            pMhcb = plot()
                            pMhcDb = plot()
                        end
                    end
                else
                    if is_plot_Mhc_l_1
                        if nMod[isp33] == 1
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc1[tvec] ,line=(wline,:auto),label="b₁",ylabel=ylabel,xlabel=xlabel)
            
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂")
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                        elseif nMod[isp33] == 2
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc1[tvec] ,line=(wline,:auto),label="b₁",ylabel=ylabel,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc3[tvec] ,line=(wline,:auto),label="b₃")
            
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                        elseif nMod[isp33] == 3
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc1[tvec] ,line=(wline,:auto),label="b₁",ylabel=ylabel,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc3[tvec] ,line=(wline,:auto),label="b₃")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc5[tvec] ,line=(wline,:auto),label="b₅")
            
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                        else
                            pMhcb = plot()
                            pMhcDb = plot()
                        end
                    else
                        if nMod[isp33] == 1
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                        elseif nMod[isp33] == 2
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                        elseif nMod[isp33] == 3
                            pMhcb = plot(tplot[tvec],Mhclbt.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="b₄")
                            pMhcb = plot!(tplot[tvec],Mhclbt.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="b₆")
                            a = Mhclbt.Mhc2[tvec][2:end] ./ Mhclbt.Mhc2[tvec][1:end-1] .- 1.0
                            pMhcDb = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₂",ylabel=ylabelD)
                            a = Mhclbt.Mhc4[tvec][2:end] ./ Mhclbt.Mhc4[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₄",ylabel=ylabelD)
                            a = Mhclbt.Mhc6[tvec][2:end] ./ Mhclbt.Mhc6[tvec][1:end-1] .- 1.0
                            pMhcDb = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="b₆",ylabel=ylabelD)
                        else
                            pMhcb = plot()
                            pMhcDb = plot()
                        end
                    end
                end
            end
        end

        if ns ≥ 3
            Mhclct = CSV.File(read(file_Ms_Mhclc)) |> DataFrame
            if 1 == 1
                if missing_deal == :nothing
                elseif missing_deal == :drop
                    dropmissing!(Mhclct)
                elseif missing_deal == :NaN
                    replace!.(eachcol(Mhclct), missing => NaN)
                elseif missing_deal == :zero
                    replace!.(eachcol(Mhclct), missing => 0.0)
                end
                unique!(Mhclct,1)            # due to t = x[1]
                Mhclct[1,2:end] = Mhclct[2,2:end]
                tplot = Mhclct.t * (tdd / τ₀)
                Nt = length(tplot)
            
                tvec = tplot_min .< tplot .≤ tplot_max
                tvec[1:dkivv2] .= false
                Dt = diff(tplot[tvec])
                isp33 = 2
                nnjM = nMjMs[isp33]
                ylabel = string("Mhcc")
                ylabelD = string("∂ₜMhcc")
                if is_MjMs_max
                    if norm(ua[isp33]) ≤ epsT10
                        if nnjM == 2
                            pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                        elseif nnjM == 3
                            pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                            a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                            a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                        elseif nnjM == 4
                            pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                            a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                            a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                            a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                        elseif nnjM == 5
                            pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                            a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                            a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                            a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                            a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈",ylabel=ylabelD)
                        elseif nnjM == 6
                            pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="c₁₀")
                            a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                            a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                            a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                            a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈",ylabel=ylabelD)
                            a = Mhclct.Mhc10[tvec][2:end] ./ Mhclct.Mhc10[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₀",ylabel=ylabelD)
                        elseif nnjM == 7
                            pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="c₁₀")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="c₁₂")
                            a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                            a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                            a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                            a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈",ylabel=ylabelD)
                            a = Mhclct.Mhc10[tvec][2:end] ./ Mhclct.Mhc10[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₀",ylabel=ylabelD)
                            a = Mhclct.Mhc12[tvec][2:end] ./ Mhclct.Mhc12[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₂",ylabel=ylabelD)
                        elseif nnjM == 8
                            pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="c₁₀")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="c₁₂")
                            pMhcc = plot!(tplot[tvec],Mhclct.Mhc14[tvec].- 1.0 ,line=(wline,:auto),label="c₁₄")
                            a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                            a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                            a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                            a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈",ylabel=ylabelD)
                            a = Mhclct.Mhc10[tvec][2:end] ./ Mhclct.Mhc10[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₀",ylabel=ylabelD)
                            a = Mhclct.Mhc12[tvec][2:end] ./ Mhclct.Mhc12[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₂",ylabel=ylabelD)
                            a = Mhclct.Mhc14[tvec][2:end] ./ Mhclct.Mhc14[tvec][1:end-1] .- 1.0
                            a ./= Dt
                            pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₄",ylabel=ylabelD)
                        else
                            pMhcc = plot()
                        end
                    else
                        if is_plot_Mhc_l_1
                            if nnjM == 2
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                    
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                            elseif nnjM == 4
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc3[tvec],line=(wline,:auto),label="c₃")
                    
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄")
                            elseif nnjM == 6
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc3[tvec] ,line=(wline,:auto),label="c₃")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc5[tvec],line=(wline,:auto),label="c₅")
                    
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄")
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆")
                            elseif nnjM == 8
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc3[tvec] ,line=(wline,:auto),label="c₃")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc5[tvec],line=(wline,:auto),label="c₅")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc7[tvec],line=(wline,:auto),label="c₇")
                    
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄")
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆")
                                a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈")
                            elseif nnjM == 10
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc3[tvec] ,line=(wline,:auto),label="c₃")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc5[tvec],line=(wline,:auto),label="c₅")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc7[tvec],line=(wline,:auto),label="c₇")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc9[tvec],line=(wline,:auto),label="c₉")
                    
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="c₁₀")
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄")
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆")
                                a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈")
                                a = Mhclct.Mhc10[tvec][2:end] ./ Mhclct.Mhc10[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₀")
                            elseif nnjM == 12
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc3[tvec] ,line=(wline,:auto),label="c₃")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc5[tvec],line=(wline,:auto),label="c₅")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc7[tvec],line=(wline,:auto),label="c₇")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc9[tvec],line=(wline,:auto),label="c₉")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc11[tvec] ,line=(wline,:auto),label="c₁₁")
                    
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="c₁₀")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="c₁₂")
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄")
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆")
                                a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈")
                                a = Mhclct.Mhc10[tvec][2:end] ./ Mhclct.Mhc10[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₀")
                                a = Mhclct.Mhc12[tvec][2:end] ./ Mhclct.Mhc12[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₂")
                            else
                                pMhcc = plot()
                                pMhcDc = plot()
                            end
                        else
                            if nnjM == 2
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                            elseif nnjM == 4
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄")
                            elseif nnjM == 6
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄")
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆")
                            elseif nnjM == 8
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄")
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆")
                                a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈")
                            elseif nnjM == 10
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="c₁₀")
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄")
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆")
                                a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈")
                                a = Mhclct.Mhc10[tvec][2:end] ./ Mhclct.Mhc10[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₀")
                            elseif nnjM == 12
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="c₁₀")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="c₁₂")
            
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄")
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆")
                                a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈")
                                a = Mhclct.Mhc10[tvec][2:end] ./ Mhclct.Mhc10[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₀")
                                a = Mhclct.Mhc12[tvec][2:end] ./ Mhclct.Mhc12[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₂")
                            else
                                pMhcc = plot()
                                pMhcDc = plot()
                            end
                        end
                    end
                else
                    if norm(ua[isp33]) ≤ epsT10
                        if 1 == 1
                            if nnjM == 2
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                            elseif nnjM == 3
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                            elseif nnjM == 4
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                            elseif nnjM == 5
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                                a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈",ylabel=ylabelD)
                            elseif nnjM == 6
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="c₁₀")
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                                a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈",ylabel=ylabelD)
                                a = Mhclct.Mhc10[tvec][2:end] ./ Mhclct.Mhc10[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₀",ylabel=ylabelD)
                            elseif nnjM == 7
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="c₁₀")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="c₁₂")
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                                a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈",ylabel=ylabelD)
                                a = Mhclct.Mhc10[tvec][2:end] ./ Mhclct.Mhc10[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₀",ylabel=ylabelD)
                                a = Mhclct.Mhc12[tvec][2:end] ./ Mhclct.Mhc12[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₂",ylabel=ylabelD)
                            elseif nnjM == 8
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc8[tvec].- 1.0 ,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc10[tvec].- 1.0 ,line=(wline,:auto),label="c₁₀")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc12[tvec].- 1.0 ,line=(wline,:auto),label="c₁₂")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc14[tvec].- 1.0 ,line=(wline,:auto),label="c₁₄")
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                                a = Mhclct.Mhc8[tvec][2:end] ./ Mhclct.Mhc8[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₈",ylabel=ylabelD)
                                a = Mhclct.Mhc10[tvec][2:end] ./ Mhclct.Mhc10[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₀",ylabel=ylabelD)
                                a = Mhclct.Mhc12[tvec][2:end] ./ Mhclct.Mhc12[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₂",ylabel=ylabelD)
                                a = Mhclct.Mhc14[tvec][2:end] ./ Mhclct.Mhc14[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₁₄",ylabel=ylabelD)
                            else
                                pMhcc = plot()
                                pMhcDc = plot()
                            end
                        end
                    else
                        if is_plot_Mhc_l_1
                            if nMod[isp33] == 1
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc1[tvec] ,line=(wline,:auto),label="c₁",ylabel=ylabel,xlabel=xlabel)
                
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂")
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                            elseif nMod[isp33] == 2
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc1[tvec] ,line=(wline,:auto),label="c₁",ylabel=ylabel,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc3[tvec] ,line=(wline,:auto),label="c₃")
                
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                            elseif nMod[isp33] == 3
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc1[tvec] ,line=(wline,:auto),label="c₁",ylabel=ylabel,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc3[tvec] ,line=(wline,:auto),label="c₃")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc5[tvec] ,line=(wline,:auto),label="c₅")
                
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                            else
                                pMhcc = plot()
                                pMhcDc = plot()
                            end
                        else
                            if nMod[isp33] == 1
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                            elseif nMod[isp33] == 2
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                            elseif nMod[isp33] == 3
                                pMhcc = plot(tplot[tvec],Mhclct.Mhc2[tvec].- 1.0 ,line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc4[tvec].- 1.0 ,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],Mhclct.Mhc6[tvec].- 1.0 ,line=(wline,:auto),label="c₆")
                                a = Mhclct.Mhc2[tvec][2:end] ./ Mhclct.Mhc2[tvec][1:end-1] .- 1.0
                                pMhcDc = plot(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₂",ylabel=ylabelD)
                                a = Mhclct.Mhc4[tvec][2:end] ./ Mhclct.Mhc4[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₄",ylabel=ylabelD)
                                a = Mhclct.Mhc6[tvec][2:end] ./ Mhclct.Mhc6[tvec][1:end-1] .- 1.0
                                pMhcDc = plot!(tplot[tvec][2:end], a ,line=(wline,:auto),label="c₆",ylabel=ylabelD)
                            else
                                pMhcc = plot()
                                pMhcDc = plot()
                            end
                        end
                    end
                end
            end
            if ns ≥ 4
            end
        end

        pMhc = display(plot(pMhca,pMhcD,pMhcb,pMhcDb,layout=(2,2)))
    
        plot(pMhca,pMhcD,pMhcb,pMhcDb,layout=(2,2))
        savefig(string(file_fig_file,"_Mhcl.png"))
    end
end
