
if nModat == 1
    yscale0 = :identity
else
    yscale0 = :log10
end
xscale0 = :log10 

# [Mhc]
if is_moments_out
    if isfile(file_Ms_RDMck1a) 
        RDMck1at = CSV.File(read(file_Ms_RDMck1a)) |> DataFrame
        if 1 == 1
            if missing_deal == :nothing
            elseif missing_deal == :drop
                dropmissing!(RDMck1at)
            elseif missing_deal == :NaN
                replace!.(eachcol(RDMck1at), missing => NaN)
            elseif missing_deal == :zero
                replace!.(eachcol(RDMck1at), missing => 0.0)
            end
            unique!(RDMck1at,1)            # due to t = x[1]
            RDMck1at[1,2:end] = RDMck1at[2,2:end]
            tplot = RDMck1at.t * (tdd / τ₀)
            Nt = length(tplot)
        
            tvec = tplot_min .≤ tplot .≤ tplot_max
            tvec[1:dkivv] .= false
            Dt = diff(tplot[tvec])
            # tvec[1:9] .= false
            isp33 = 1
            if is_nai_const
                nnjM = nMjMs[isp33] - 2
            else
                nnjM = 2NKk
                nnjM = nMjMs[isp33] - 2
            end
            title_nMod = string("nMod=",nMod0)
            ylabel = string(L"\Delta {\scrM}_{j,l}^0")
            if is_MjMs_max
                if norm(ua[isp33]) ≤ epsT10
                    if nnjM == 0
                    elseif nnjM == 1
                    elseif nnjM == 2
                        pMhca = plot()
                        # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                    elseif nnjM == 3
                        # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT.+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                    elseif nnjM == 4
                        # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                    elseif nnjM == 5
                        # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"a_8")
                    elseif nnjM == 6
                        # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"a_8")
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"a_{10}")
                    elseif nnjM == 7
                        # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"a_8")
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"a_{10}")
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc12[tvec]).+epsT,line=(wline,:auto),label=L"a_{12}")
                    elseif nnjM == 8
                        # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"a_8")
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"a_{10}")
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc12[tvec]).+epsT,line=(wline,:auto),label=L"a_{12}")
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc14[tvec]).+epsT,line=(wline,:auto),label=L"a_{14}")
                    else
                        pMhca = plot()
                    end
                else
                    if is_plot_Mhc_l_1
                        if nnjM == 0
                        elseif nnjM == 2
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc1[tvec]).+epsT,line=(wline,:auto),label="a₁",ylabel=ylabel,yscale=yscale0)
                
                            # pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2")
                        elseif nnjM == 4
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc1[tvec]).+epsT,line=(wline,:auto),label="a₁",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc3[tvec]).+epsT,line=(wline,:auto),label="a₃")
                
                            # pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4")
                        elseif nnjM == 6
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc1[tvec]).+epsT,line=(wline,:auto),label="a₁",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc3[tvec]).+epsT,line=(wline,:auto),label="a₃")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc5[tvec]).+epsT,line=(wline,:auto),label="a₅")
                
                            # pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                        elseif nnjM == 8
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc1[tvec]).+epsT,line=(wline,:auto),label="a₁",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc3[tvec]).+epsT,line=(wline,:auto),label="a₃")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc5[tvec]).+epsT,line=(wline,:auto),label="a₅")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc7[tvec]).+epsT,line=(wline,:auto),label="a₇")
                
                            # pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"a_8")
                        elseif nnjM == 10
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc1[tvec]).+epsT,line=(wline,:auto),label="a₁",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc3[tvec]).+epsT,line=(wline,:auto),label="a₃")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc5[tvec]).+epsT,line=(wline,:auto),label="a₅")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc7[tvec]).+epsT,line=(wline,:auto),label="a₇")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc9[tvec]).+epsT,line=(wline,:auto),label="a₉")
                
                            # pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"a_8")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"a_{10}")
                        elseif nnjM == 12
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc1[tvec]).+epsT,line=(wline,:auto),label="a₁",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc3[tvec]).+epsT,line=(wline,:auto),label="a₃")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc5[tvec]).+epsT,line=(wline,:auto),label="a₅")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc7[tvec]).+epsT,line=(wline,:auto),label="a₇")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc9[tvec]).+epsT,line=(wline,:auto),label="a₉")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc11[tvec]).+epsT,line=(wline,:auto),label="a₁₁")
                
                            # pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"a_8")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"a_{10}")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc12[tvec]).+epsT,line=(wline,:auto),label=L"a_{12}")
                        else
                            pMhca = plot()
                        end
                    else
                        if nnjM == 0
                        elseif nnjM == 2
                            pMhca = plot()
                            # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                        elseif nnjM == 4
                            # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                        elseif nnjM == 6
                            # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                        elseif nnjM == 8
                            # pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"a_8")
                        elseif nnjM == 10
                            # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"a_8")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"a_{10}")
                        elseif nnjM == 12
                            # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"a_8")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"a_{10}")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc12[tvec]).+epsT,line=(wline,:auto),label=L"a_{12}")
                        else
                            pMhca = plot()
                        end
                    end
                end
            else
                if norm(ua[isp33]) ≤ epsT10
                    if nMod[isp33] == 1
                        # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                    elseif nMod[isp33] == 2
                        # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                    elseif nMod[isp33] == 3
                        # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                        pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                    else
                        pMhca = plot()
                    end
                else
                    if is_plot_Mhc_l_1
                        if nMod[isp33] == 1
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc1[tvec]).+epsT,line=(wline,:auto),label="a₁",ylabel=ylabel,yscale=yscale0)
                
                            # pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                        elseif nMod[isp33] == 2
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc1[tvec]).+epsT,line=(wline,:auto),label="a₁",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc3[tvec]).+epsT,line=(wline,:auto),label="a₃")
                
                            # pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                        elseif nMod[isp33] == 3
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc1[tvec]).+epsT,line=(wline,:auto),label="a₁",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc3[tvec]).+epsT,line=(wline,:auto),label="a₃")
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc5[tvec]).+epsT,line=(wline,:auto),label="a₅")
                
                            # pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                        else
                            pMhca = plot()
                        end
                    else
                        if nMod[isp33] == 1
                            # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                        elseif nMod[isp33] == 2
                            # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                        elseif nMod[isp33] == 3
                            # pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"a_2",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot(tplot[tvec],abs.(RDMck1at.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"a_4",ylabel=ylabel,yscale=yscale0)
                            pMhca = plot!(tplot[tvec],abs.(RDMck1at.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"a_6")
                        else
                            pMhca = plot()
                        end
                    end
                end
            end
        end

        if nModbt == 1
            yscale0 = :identity
        else
            yscale0 = :log10
        end
        RDMck1bt = CSV.File(read(file_Ms_RDMck1b)) |> DataFrame
        if 1 == 1
            if missing_deal == :nothing
            elseif missing_deal == :drop
                dropmissing!(RDMck1bt)
            elseif missing_deal == :NaN
                replace!.(eachcol(RDMck1bt), missing => NaN)
            elseif missing_deal == :zero
                replace!.(eachcol(RDMck1bt), missing => 0.0)
            end
            unique!(RDMck1bt,1)            # due to t = x[1]
            RDMck1bt[1,2:end] = RDMck1bt[2,2:end]
            tplot = RDMck1bt.t * (tdd / τ₀)
            Nt = length(tplot)
        
            tvec = tplot_min .≤ tplot .≤ tplot_max
            tvec[1:dkivv] .= false
            Dt = diff(tplot[tvec])
            isp33 = 2
            if is_nai_const
                nnjM = nMjMs[isp33] - 2
            else
                nnjM = 2NKk
                nnjM = nMjMs[isp33] - 2
            end
            # ylabel = string(L"\Delta {\scrM}_{j,l}^0")
            if is_MjMs_max
                if norm(ua[isp33]) ≤ epsT10
                    if nnjM == 0
                    elseif nnjM == 1
                    elseif nnjM == 2
                        pMhcb = plot()
                        # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                    elseif nnjM == 3
                        # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                    elseif nnjM == 4
                        # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                    elseif nnjM == 5
                        # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                    elseif nnjM == 6
                        # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"b_{10}")
                    elseif nnjM == 7
                        # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"b_{10}")
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc12[tvec]).+epsT,line=(wline,:auto),label=L"b_{12}")
                    elseif nnjM == 8
                        # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"b_{10}")
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc12[tvec]).+epsT,line=(wline,:auto),label=L"b_{12}")
                        pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc14[tvec]).+epsT,line=(wline,:auto),label=L"b_{14}")
                    else
                        pMhcb = plot()
                    end
                else
                    if is_plot_Mhc_l_1
                        if nnjM == 0
                        elseif nnjM == 2
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc1[tvec]).+epsT,line=(wline,:auto),label="b₁",ylabel=ylabel,yscale=yscale0)
                
                            # pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                        elseif nnjM == 4
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc1[tvec]).+epsT,line=(wline,:auto),label="b₁",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc3[tvec]).+epsT,line=(wline,:auto),label="b₃")
                
                            # pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4")
                        elseif nnjM == 6
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc1[tvec]).+epsT,line=(wline,:auto),label="b₁",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc3[tvec]).+epsT,line=(wline,:auto),label="b₃")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc5[tvec]).+epsT,line=(wline,:auto),label="b₅")
                
                            # pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                        elseif nnjM == 8
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc1[tvec]).+epsT,line=(wline,:auto),label="b₁",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc3[tvec]).+epsT,line=(wline,:auto),label="b₃")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc5[tvec]).+epsT,line=(wline,:auto),label="b₅")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc7[tvec]).+epsT,line=(wline,:auto),label="b₇")
                
                            # pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                        elseif nnjM == 10
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc1[tvec]).+epsT,line=(wline,:auto),label="b₁",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc3[tvec]).+epsT,line=(wline,:auto),label="b₃")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc5[tvec]).+epsT,line=(wline,:auto),label="b₅")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc7[tvec]).+epsT,line=(wline,:auto),label="b₇")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc9[tvec]).+epsT,line=(wline,:auto),label="b₉")
                
                            # pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"b_{10}")
                        elseif nnjM == 12
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc1[tvec]).+epsT,line=(wline,:auto),label="b₁",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc3[tvec]).+epsT,line=(wline,:auto),label="b₃")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc5[tvec]).+epsT,line=(wline,:auto),label="b₅")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc7[tvec]).+epsT,line=(wline,:auto),label="b₇")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc9[tvec]).+epsT,line=(wline,:auto),label="b₉")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc11[tvec]).+epsT,line=(wline,:auto),label="b₁₁")
                
                            # pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"b_{10}")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc12[tvec]).+epsT,line=(wline,:auto),label=L"b_{12}")
                        else
                            pMhcb = plot()
                        end
                    else
                        if nnjM == 0
                        elseif nnjM == 2
                            pMhcb = plot()
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                        elseif nnjM == 4
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0)
                        elseif nnjM == 6
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                        elseif nnjM == 8
                            # pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                        elseif nnjM == 10
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"b_{10}")
                        elseif nnjM == 12
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"b_{10}")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc12[tvec]).+epsT,line=(wline,:auto),label=L"b_{12}")
                        else
                            pMhcb = plot()
                        end
                    end
                end
            else
                if norm(ua[isp33]) ≤ epsT10
                    if 1 == 1
                        if nnjM == 0
                        elseif nnjM == 2
                            pMhcb = plot()
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        elseif nnjM == 3
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        elseif nnjM == 4
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                        elseif nnjM == 5
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                        elseif nnjM == 6
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"b_{10}")
                        elseif nnjM == 7
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"b_{10}")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc12[tvec]).+epsT,line=(wline,:auto),label=L"b_{12}")
                        elseif nnjM == 8
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc8[tvec]).+epsT,line=(wline,:auto),label=L"b_8")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc10[tvec]).+epsT,line=(wline,:auto),label=L"b_{10}")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc12[tvec]).+epsT,line=(wline,:auto),label=L"b_{12}")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc14[tvec]).+epsT,line=(wline,:auto),label=L"b_{14}")
                        else
                            pMhcb = plot()
                        end
                    end
                else
                    if is_plot_Mhc_l_1
                        if nMod[isp33] == 1
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc1[tvec]).+epsT,line=(wline,:auto),label="b₁",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        elseif nMod[isp33] == 2
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc1[tvec]).+epsT,line=(wline,:auto),label="b₁",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc3[tvec]).+epsT,line=(wline,:auto),label="b₃")
            
                            # pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2")
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4")
                        elseif nMod[isp33] == 3
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc1[tvec]).+epsT,line=(wline,:auto),label="b₁",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc3[tvec]).+epsT,line=(wline,:auto),label="b₃")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc5[tvec]).+epsT,line=(wline,:auto),label="b₅")
            
                            # pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2")
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4")
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                        else
                            pMhcb = plot()
                        end
                    else
                        if nMod[isp33] == 1
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        elseif nMod[isp33] == 2
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        elseif nMod[isp33] == 3
                            # pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc2[tvec]).+epsT,line=(wline,:auto),label=L"b_2",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot(tplot[tvec],abs.(RDMck1bt.Mhc4[tvec]).+epsT,line=(wline,:auto),label=L"b_4",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcb = plot!(tplot[tvec],abs.(RDMck1bt.Mhc6[tvec]).+epsT,line=(wline,:auto),label=L"b_6")
                        else
                            pMhcb = plot()
                        end
                    end
                end
            end
        end

        if ns ≥ 3
            ergfbn
            RDMck1ct = CSV.File(read(file_Ms_RDMck1c)) |> DataFrame
            if 1 == 1
                if missing_deal == :nothing
                elseif missing_deal == :drop
                    dropmissing!(RDMck1ct)
                elseif missing_deal == :NaN
                    replace!.(eachcol(RDMck1ct), missing => NaN)
                elseif missing_deal == :zero
                    replace!.(eachcol(RDMck1ct), missing => 0.0)
                end
                unique!(RDMck1ct,1)            # due to t = x[1]
                RDMck1ct[1,2:end] = RDMck1ct[2,2:end]
                tplot = RDMck1ct.t * (tdd / τ₀)
                Nt = length(tplot)
            
                tvec = tplot_min .≤ tplot .≤ tplot_max
                tvec[1:dkivv] .= false
                Dt = diff(tplot[tvec])
                isp33 = 2
                if is_nai_const
                    nnjM = nMjMs[isp33] - 2
                else
                    nnjM = 2NKk
                end
                # ylabel = string(L"\Delta {\scrM}_{j,l}^0")
                if is_MjMs_max
                    if norm(ua[isp33]) ≤ epsT10
                        if nnjM == 0
                        elseif nnjM == 1
                        elseif nnjM == 2
                            pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                        elseif nnjM == 3
                            pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                        elseif nnjM == 4
                            pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                        elseif nnjM == 5
                            pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                        elseif nnjM == 6
                            pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc10[tvec].+epsT,line=(wline,:auto),label="c₁₀")
                        elseif nnjM == 7
                            pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc10[tvec].+epsT,line=(wline,:auto),label="c₁₀")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc12[tvec].+epsT,line=(wline,:auto),label="c₁₂")
                        elseif nnjM == 8
                            pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc10[tvec].+epsT,line=(wline,:auto),label="c₁₀")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc12[tvec].+epsT,line=(wline,:auto),label="c₁₂")
                            pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc14[tvec].+epsT,line=(wline,:auto),label="c₁₄")
                        else
                            pMhcc = plot()
                        end
                    else
                        if is_plot_Mhc_l_1
                            if nnjM == 0
                            elseif nnjM == 1
                            elseif nnjM == 2
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc1[tvec].+epsT,line=(wline,:auto),label="c₁",ylabel=ylabel,yscale=yscale0)
                    
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                            elseif nnjM == 4
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc1[tvec].+epsT,line=(wline,:auto),label="c₁",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc3[tvec].+epsT,line=(wline,:auto),label="c₃")
                    
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                            elseif nnjM == 6
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc1[tvec].+epsT,line=(wline,:auto),label="c₁",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc3[tvec].+epsT,line=(wline,:auto),label="c₃")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc5[tvec].+epsT,line=(wline,:auto),label="c₅")
                    
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                            elseif nnjM == 8
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc1[tvec].+epsT,line=(wline,:auto),label="c₁",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc3[tvec].+epsT,line=(wline,:auto),label="c₃")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc5[tvec].+epsT,line=(wline,:auto),label="c₅")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc7[tvec].+epsT,line=(wline,:auto),label="c₇")
                    
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                            elseif nnjM == 10
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc1[tvec].+epsT,line=(wline,:auto),label="c₁",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc3[tvec].+epsT,line=(wline,:auto),label="c₃")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc5[tvec].+epsT,line=(wline,:auto),label="c₅")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc7[tvec].+epsT,line=(wline,:auto),label="c₇")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc9[tvec].+epsT,line=(wline,:auto),label="c₉")
                    
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc10[tvec].+epsT,line=(wline,:auto),label="c₁₀")
                            elseif nnjM == 12
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc1[tvec].+epsT,line=(wline,:auto),label="c₁",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc3[tvec].+epsT,line=(wline,:auto),label="c₃")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc5[tvec].+epsT,line=(wline,:auto),label="c₅")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc7[tvec].+epsT,line=(wline,:auto),label="c₇")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc9[tvec].+epsT,line=(wline,:auto),label="c₉")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc11[tvec].+epsT,line=(wline,:auto),label="c₁₁")
                    
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc10[tvec].+epsT,line=(wline,:auto),label="c₁₀")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc12[tvec].+epsT,line=(wline,:auto),label="c₁₂")
                            else
                                pMhcc = plot()
                                pMhcDc = plot()
                            end
                        else
                            if nnjM == 0
                            elseif nnjM == 2
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                            elseif nnjM == 4
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                            elseif nnjM == 6
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                            elseif nnjM == 8
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                            elseif nnjM == 10
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc10[tvec].+epsT,line=(wline,:auto),label="c₁₀")
                            elseif nnjM == 12
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc10[tvec].+epsT,line=(wline,:auto),label="c₁₀")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc12[tvec].+epsT,line=(wline,:auto),label="c₁₂")
                            else
                                pMhcc = plot()
                                pMhcDc = plot()
                            end
                        end
                    end
                else
                    if norm(ua[isp33]) ≤ epsT10
                        if 1 == 1
                            if nnjM == 0
                            elseif nnjM == 1
                            elseif nnjM == 2
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            elseif nnjM == 3
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                            elseif nnjM == 4
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                            elseif nnjM == 5
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                            elseif nnjM == 6
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc10[tvec].+epsT,line=(wline,:auto),label="c₁₀")
                            elseif nnjM == 7
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc10[tvec].+epsT,line=(wline,:auto),label="c₁₀")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc12[tvec].+epsT,line=(wline,:auto),label="c₁₂")
                            elseif nnjM == 8
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc8[tvec].+epsT,line=(wline,:auto),label="c₈")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc10[tvec].+epsT,line=(wline,:auto),label="c₁₀")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc12[tvec].+epsT,line=(wline,:auto),label="c₁₂")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc14[tvec].+epsT,line=(wline,:auto),label="c₁₄")
                            else
                                pMhcc = plot()
                                pMhcDc = plot()
                            end
                        end
                    else
                        if is_plot_Mhc_l_1
                            if nMod[isp33] == 1
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc1[tvec].+epsT,line=(wline,:auto),label="c₁",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂")
                            elseif nMod[isp33] == 2
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc1[tvec].+epsT,line=(wline,:auto),label="c₁",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc3[tvec].+epsT,line=(wline,:auto),label="c₃")
                
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                            elseif nMod[isp33] == 3
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc1[tvec].+epsT,line=(wline,:auto),label="c₁",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc3[tvec].+epsT,line=(wline,:auto),label="c₃")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc5[tvec].+epsT,line=(wline,:auto),label="c₅")
                
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
                            else
                                pMhcc = plot()
                                pMhcDc = plot()
                            end
                        else
                            if nMod[isp33] == 1
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                            elseif nMod[isp33] == 2
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                            elseif nMod[isp33] == 3
                                pMhcc = plot(tplot[tvec],RDMck1ct.Mhc2[tvec].+epsT,line=(wline,:auto),label="c₂",ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc4[tvec].+epsT,line=(wline,:auto),label="c₄")
                                pMhcc = plot!(tplot[tvec],RDMck1ct.Mhc6[tvec].+epsT,line=(wline,:auto),label="c₆")
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

        pMhc = display(plot(pMhca,pMhcb,layout=(2,1)))
    
        plot(pMhca,pMhcb,layout=(2,1))
        savefig(string(file_fig_file,"_RDMck1.png"))
        
        if iCase == 1
            RDMck100vec = zeros(NCase,njMs+1,ns)
        end
        ivv = Nτ_fix / 2 + 1 - dkivv |> Int64
        if ivv ≤ length(tplot)
            RDMck100vec[iCase,njMs+1,:] .= tplot[ivv]
            if njMs ≥ 2
                ii = 1
                isp30 = 1
                RDMck100vec[iCase,ii,isp30] = RDMck1at.Mhc0[ivv]
                isp30 = 2
                RDMck100vec[iCase,ii,isp30] = RDMck1bt.Mhc0[ivv]
    
                ii = 2
                isp30 = 1
                RDMck100vec[iCase,ii,isp30] = RDMck1at.Mhc2[ivv]
                isp30 = 2
                RDMck100vec[iCase,ii,isp30] = RDMck1bt.Mhc2[ivv]
                if njMs ≥ 4
                    ii = 3
                    isp30 = 1
                    RDMck100vec[iCase,ii,isp30] = RDMck1at.Mhc4[ivv]
                    isp30 = 2
                    RDMck100vec[iCase,ii,isp30] = RDMck1bt.Mhc4[ivv]
        
                    ii = 4
                    isp30 = 1
                    RDMck100vec[iCase,ii,isp30] = RDMck1at.Mhc6[ivv]
                    isp30 = 2
                    RDMck100vec[iCase,ii,isp30] = RDMck1bt.Mhc6[ivv]
                    if njMs ≥ 6
                        ii = 5
                        isp30 = 1
                        RDMck100vec[iCase,ii,isp30] = RDMck1at.Mhc8[ivv]
                        isp30 = 2
                        RDMck100vec[iCase,ii,isp30] = RDMck1bt.Mhc8[ivv]
            
                        ii = 6
                        isp30 = 1
                        RDMck100vec[iCase,ii,isp30] = RDMck1at.Mhc10[ivv]
                        isp30 = 2
                        RDMck100vec[iCase,ii,isp30] = RDMck1bt.Mhc10[ivv]
                        if njMs ≥ 8
                            ii = 7
                            isp30 = 1
                            RDMck100vec[iCase,ii,isp30] = RDMck1at.Mhc12[ivv]
                            isp30 = 2
                            RDMck100vec[iCase,ii,isp30] = RDMck1bt.Mhc12[ivv]
    
                            ii = 8
                            isp30 = 1
                            RDMck100vec[iCase,ii,isp30] = RDMck1at.Mhc14[ivv]
                            isp30 = 2
                            RDMck100vec[iCase,ii,isp30] = RDMck1bt.Mhc14[ivv]
                            if njMs ≥ 10
                                wsdfdfb
                            end
                        end
                    end
                end
            end
            if iCase == NCase
                isp30 = 1
    
                if njMs ≥ 5
                    label = [L"a_4" L"a_6" L"a_8" L"a_{10}" ]
                else
                    edfbfgf
                end
                if MultiType == :dt
                    ylabel = string(L"\Delta {\scrM}_{j,l}^0 (t=",tplot[ivv],")")
                    xticks = vec_sweep
                    pRDMck1a = plot(1 ./ vec_sweep, abs.(RDMck100vec[1:NCase,3:6,isp30]).+epsT,
                                    ylabel=ylabel,yscale=yscale0,
                                    xscale=xscale0,xticks=xticks,
                                    label=label,line=(wline,:auto))
                else
                    ylabel = string(L"\Delta {\scrM}_{j,l}^0 (t=",tplot[ivv],")")
                    xticks = vec_sweep
                    pRDMck1a = plot(vec_sweep, abs.(RDMck100vec[1:NCase,3:6,isp30]).+epsT,
                                    ylabel=ylabel,yscale=yscale0,
                                    xticks=xticks,
                                    label=label,line=(wline,:auto))
                end
    
                isp30 = 2
                if njMs ≥ 5
                    label = [L"b_4" L"b_6" L"b_8" L"b_{10}" ]
                else
                    edfbfgf
                end
                if MultiType == :dt
                    ylabel = string(L"\Delta {\scrM}_{j,l}^0 (t=",tplot[ivv],")")
                    xlabel = L"\Delta t"
                    pRDMck1b = plot(1 ./ vec_sweep, abs.(RDMck100vec[1:NCase,3:6,isp30]).+epsT,
                                    ylabel=ylabel,yscale=yscale0,
                                    xlabel=xlabel,label=label,line=(wline,:auto))
                    pRDMck1 = display(plot(pRDMck1a,pRDMck1b,layout=(2,1)))
            
                    plot(pRDMck1a,pRDMck1b,layout=(2,1))
                    savefig(string(file_fig_file,"_RDMck1M4_dt.png"))
                else
                    ylabel = string(L"\Delta {\scrM}_{j,l}^0 (t=",tplot[ivv],")")
                    if MultiType == :nnv
                        xlabel = L"n_2"
                        xticks = vec_sweep
                        pRDMck1b = plot(vec_sweep, abs.(RDMck100vec[1:NCase,3:6,isp30]).+epsT,
                                        ylabel=ylabel,yscale=yscale0,xticks=xticks,
                                        xlabel=xlabel,label=label,line=(wline,:auto))
                        pRDMck1 = display(plot(pRDMck1a,pRDMck1b,layout=(2,1)))
            
                        plot(pRDMck1a,pRDMck1b,layout=(2,1))
                        savefig(string(file_fig_file,"_RDMck1M4_nnv.png"))
                    elseif MultiType == :NK
                        xlabel = L"N_K"
                        xticks = vec_sweep
                        pRDMck1b = plot(vec_sweep, abs.(RDMck100vec[1:NCase,3:6,isp30]).+epsT,
                                        ylabel=ylabel,yscale=yscale0,
                                        xlabel=xlabel,xticks=xticks,
                                        label=label,line=(wline,:auto))
                        pRDMck1 = display(plot(pRDMck1a,pRDMck1b,layout=(2,1)))
            
                        plot(pRDMck1a,pRDMck1b,layout=(2,1))
                        savefig(string(file_fig_file,"_RDMck1M4_NK.png"))
                    else
                        edfg
                    end
                end
            end
        end
    end
end
