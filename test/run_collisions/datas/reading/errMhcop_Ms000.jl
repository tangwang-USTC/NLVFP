
# [errMhcop]
if is_moments_out
    if isfile(file_Ms_errMhcopla)
        errMhcoplat = CSV.File(read(file_Ms_errMhcopla)) |> DataFrame
        if 1 == 1
            if missing_deal == :nothing
            elseif missing_deal == :drop
                dropmissing!(errMhcoplat)
            elseif missing_deal == :NaN
                replace!.(eachcol(errMhcoplat), missing => NaN)
            elseif missing_deal == :zero
                replace!.(eachcol(errMhcoplat), missing => 0.0)
            end
            unique!(errMhcoplat,1)            # due to t = x[1]
            errMhcoplat[1,2:end] = errMhcoplat[2,2:end]
            tplot = errMhcoplat.t * (tdd / τ₀)
            Nt = length(tplot)
        
            tvec = tplot_min .< tplot .≤ tplot_max
            tvec[1:dkivv2] .= false
            Dt = diff(tplot[tvec])
            # tvec[1:9] .= false
            isp33 = 1
            nnjM = nMjMs[isp33]
            title_nMod = string("nMod=",nMod0)
            ylabel = string("errMhcopa")
            if is_MjMs_max
                if norm(ua[isp33]) ≤ epsT10
                    if nnjM == 2
                        yscale=:log10
                        # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                    elseif nnjM == 3
                        # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel,title=title_nMod)
                    elseif nnjM == 4
                        # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel,title=title_nMod)
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                    elseif nnjM == 5
                        # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel,title=title_nMod)
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop8[tvec],line=(wline,:auto),label="a₈")
                    elseif nnjM == 6
                        # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel,title=title_nMod)
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop8[tvec],line=(wline,:auto),label="a₈")
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop10[tvec],line=(wline,:auto),label="a₁₀")
                    elseif nnjM == 7
                        # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel,title=title_nMod)
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop8[tvec],line=(wline,:auto),label="a₈")
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop10[tvec],line=(wline,:auto),label="a₁₀")
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop12[tvec],line=(wline,:auto),label="a₁₂")
                    elseif nnjM == 8
                        # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel,title=title_nMod)
                        perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel,title=title_nMod)
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop8[tvec],line=(wline,:auto),label="a₈")
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop10[tvec],line=(wline,:auto),label="a₁₀")
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop12[tvec],line=(wline,:auto),label="a₁₂")
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop14[tvec],line=(wline,:auto),label="a₁₄")
                    else
                        perrMhcopa = plot()
                    end
                else
                    if is_plot_errMhcop_l_1
                        if nnjM == 2
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                
                            # perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂")
                        elseif nnjM == 4
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop3[tvec],line=(wline,:auto),label="a₃")
                
                            # perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄")
                        elseif nnjM == 6
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop3[tvec],line=(wline,:auto),label="a₃")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop5[tvec],line=(wline,:auto),label="a₅")
                
                            # perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                        elseif nnjM == 8
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop3[tvec],line=(wline,:auto),label="a₃")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop5[tvec],line=(wline,:auto),label="a₅")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop7[tvec],line=(wline,:auto),label="a₇")
                
                            # perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop8[tvec],line=(wline,:auto),label="a₈")
                        elseif nnjM == 10
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop3[tvec],line=(wline,:auto),label="a₃")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop5[tvec],line=(wline,:auto),label="a₅")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop7[tvec],line=(wline,:auto),label="a₇")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop9[tvec],line=(wline,:auto),label="a₉")
                
                            # perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop8[tvec],line=(wline,:auto),label="a₈")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop10[tvec],line=(wline,:auto),label="a₁₀")
                        elseif nnjM == 12
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop3[tvec],line=(wline,:auto),label="a₃")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop5[tvec],line=(wline,:auto),label="a₅")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop7[tvec],line=(wline,:auto),label="a₇")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop9[tvec],line=(wline,:auto),label="a₉")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop11[tvec],line=(wline,:auto),label="a₁₁")
                
                            # perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop8[tvec],line=(wline,:auto),label="a₈")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop10[tvec],line=(wline,:auto),label="a₁₀")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop12[tvec],line=(wline,:auto),label="a₁₂")
                        else
                            perrMhcopa = plot()
                        end
                    else
                        if nnjM == 2
                            # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                        elseif nnjM == 4
                            # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel)
                        elseif nnjM == 6
                            # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                        elseif nnjM == 8
                            # perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop8[tvec],line=(wline,:auto),label="a₈")
                        elseif nnjM == 10
                            # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop8[tvec],line=(wline,:auto),label="a₈")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop10[tvec],line=(wline,:auto),label="a₁₀")
                        elseif nnjM == 12
                            # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop8[tvec],line=(wline,:auto),label="a₈")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop10[tvec],line=(wline,:auto),label="a₁₀")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop12[tvec],line=(wline,:auto),label="a₁₂")
                        else
                            perrMhcopa = plot()
                        end
                    end
                end
            else
                if norm(ua[isp33]) ≤ epsT10
                    if nMod[isp33] == 1
                        # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                    elseif nMod[isp33] == 2
                        # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                        perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel)
                    elseif nMod[isp33] == 3
                        # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                        perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel)
                        perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                    else
                        perrMhcopa = plot()
                    end
                else
                    if is_plot_errMhcop_l_1
                        if nMod[isp33] == 1
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                
                            # perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                        elseif nMod[isp33] == 2
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop3[tvec],line=(wline,:auto),label="a₃")
                
                            # perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel)
                        elseif nMod[isp33] == 3
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop3[tvec],line=(wline,:auto),label="a₃")
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop5[tvec],line=(wline,:auto),label="a₅")
                
                            # perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                        else
                            perrMhcopa = plot()
                        end
                    else
                        if nMod[isp33] == 1
                            # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                        elseif nMod[isp33] == 2
                            # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel)
                        elseif nMod[isp33] == 3
                            # perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop2[tvec],line=(wline,:auto),label="a₂",ylabel=ylabel)
                            perrMhcopa = plot(tplot[tvec],errMhcoplat.errMhcop4[tvec],line=(wline,:auto),label="a₄",ylabel=ylabel)
                            perrMhcopa = plot!(tplot[tvec],errMhcoplat.errMhcop6[tvec],line=(wline,:auto),label="a₆")
                        else
                            perrMhcopa = plot()
                        end
                    end
                end
            end
        end

        errMhcoplbt = CSV.File(read(file_Ms_errMhcoplb)) |> DataFrame
        if 1 == 1
            if missing_deal == :nothing
            elseif missing_deal == :drop
                dropmissing!(errMhcoplbt)
            elseif missing_deal == :NaN
                replace!.(eachcol(errMhcoplbt), missing => NaN)
            elseif missing_deal == :zero
                replace!.(eachcol(errMhcoplbt), missing => 0.0)
            end
            unique!(errMhcoplbt,1)            # due to t = x[1]
            errMhcoplbt[1,2:end] = errMhcoplbt[2,2:end]
            tplot = errMhcoplbt.t * (tdd / τ₀)
            Nt = length(tplot)
        
            tvec = tplot_min .< tplot .≤ tplot_max
            tvec[1:dkivv2] .= false
            Dt = diff(tplot[tvec])
            isp33 = 2
            nnjM = nMjMs[isp33]
            ylabel = string("errMhcopb")
            if is_MjMs_max
                if norm(ua[isp33]) ≤ epsT10
                    if nnjM == 2
                        # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                    elseif nnjM == 3
                        # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                    elseif nnjM == 4
                        # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                    elseif nnjM == 5
                        # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                    elseif nnjM == 6
                        # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop10[tvec],line=(wline,:auto),label="b₁₀")
                    elseif nnjM == 7
                        # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop10[tvec],line=(wline,:auto),label="b₁₀")
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop12[tvec],line=(wline,:auto),label="b₁₂")
                    elseif nnjM == 8
                        # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop10[tvec],line=(wline,:auto),label="b₁₀")
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop12[tvec],line=(wline,:auto),label="b₁₂")
                        perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop14[tvec],line=(wline,:auto),label="b₁₄")
                    else
                        perrMhcopb = plot()
                    end
                else
                    if is_plot_errMhcop_l_1
                        if nnjM == 2
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                
                            # perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                        elseif nnjM == 4
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop3[tvec],line=(wline,:auto),label="b₃")
                
                            # perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄")
                        elseif nnjM == 6
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop3[tvec],line=(wline,:auto),label="b₃")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop5[tvec],line=(wline,:auto),label="b₅")
                
                            # perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                        elseif nnjM == 8
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop3[tvec],line=(wline,:auto),label="b₃")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop5[tvec],line=(wline,:auto),label="b₅")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop7[tvec],line=(wline,:auto),label="b₇")
                
                            # perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                        elseif nnjM == 10
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop3[tvec],line=(wline,:auto),label="b₃")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop5[tvec],line=(wline,:auto),label="b₅")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop7[tvec],line=(wline,:auto),label="b₇")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop9[tvec],line=(wline,:auto),label="b₉")
                
                            # perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop10[tvec],line=(wline,:auto),label="b₁₀")
                        elseif nnjM == 12
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop3[tvec],line=(wline,:auto),label="b₃")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop5[tvec],line=(wline,:auto),label="b₅")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop7[tvec],line=(wline,:auto),label="b₇")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop9[tvec],line=(wline,:auto),label="b₉")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop11[tvec],line=(wline,:auto),label="b₁₁")
                
                            # perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop10[tvec],line=(wline,:auto),label="b₁₀")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop12[tvec],line=(wline,:auto),label="b₁₂")
                        else
                            perrMhcopb = plot()
                        end
                    else
                        if nnjM == 2
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                        elseif nnjM == 4
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel)
                        elseif nnjM == 6
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                        elseif nnjM == 8
                            # perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                        elseif nnjM == 10
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop10[tvec],line=(wline,:auto),label="b₁₀")
                        elseif nnjM == 12
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop10[tvec],line=(wline,:auto),label="b₁₀")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop12[tvec],line=(wline,:auto),label="b₁₂")
                        else
                            perrMhcopb = plot()
                        end
                    end
                end
            else
                if norm(ua[isp33]) ≤ epsT10
                    if 1 == 1
                        if nnjM == 2
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        elseif nnjM == 3
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                        elseif nnjM == 4
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                        elseif nnjM == 5
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                        elseif nnjM == 6
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop10[tvec],line=(wline,:auto),label="b₁₀")
                        elseif nnjM == 7
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop10[tvec],line=(wline,:auto),label="b₁₀")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop12[tvec],line=(wline,:auto),label="b₁₂")
                        elseif nnjM == 8
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop8[tvec],line=(wline,:auto),label="b₈")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop10[tvec],line=(wline,:auto),label="b₁₀")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop12[tvec],line=(wline,:auto),label="b₁₂")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop14[tvec],line=(wline,:auto),label="b₁₄")
                        else
                            perrMhcopb = plot()
                        end
                    end
                else
                    if is_plot_errMhcop_l_1
                        if nMod[isp33] == 1
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,xlabel=xlabel)
                        elseif nMod[isp33] == 2
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop3[tvec],line=(wline,:auto),label="b₃")
            
                            # perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂")
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄")
                        elseif nMod[isp33] == 3
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop3[tvec],line=(wline,:auto),label="b₃")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop5[tvec],line=(wline,:auto),label="b₅")
            
                            # perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂")
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄")
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                        else
                            perrMhcopb = plot()
                        end
                    else
                        if nMod[isp33] == 1
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                        elseif nMod[isp33] == 2
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                        elseif nMod[isp33] == 3
                            # perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop2[tvec],line=(wline,:auto),label="b₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot(tplot[tvec],errMhcoplbt.errMhcop4[tvec],line=(wline,:auto),label="b₄",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopb = plot!(tplot[tvec],errMhcoplbt.errMhcop6[tvec],line=(wline,:auto),label="b₆")
                        else
                            perrMhcopb = plot()
                        end
                    end
                end
            end
        end

        if ns ≥ 3
            errMhcoplct = CSV.File(read(file_Ms_errMhcoplc)) |> DataFrame
            if 1 == 1
                if missing_deal == :nothing
                elseif missing_deal == :drop
                    dropmissing!(errMhcoplct)
                elseif missing_deal == :NaN
                    replace!.(eachcol(errMhcoplct), missing => NaN)
                elseif missing_deal == :zero
                    replace!.(eachcol(errMhcoplct), missing => 0.0)
                end
                unique!(errMhcoplct,1)            # due to t = x[1]
                errMhcoplct[1,2:end] = errMhcoplct[2,2:end]
                tplot = errMhcoplct.t * (tdd / τ₀)
                Nt = length(tplot)
            
                tvec = tplot_min .< tplot .≤ tplot_max
                tvec[1:dkivv2] .= false
                Dt = diff(tplot[tvec])
                isp33 = 2
                nnjM = nMjMs[isp33]
                ylabel = string("errMhcopc")
                if is_MjMs_max
                    if norm(ua[isp33]) ≤ epsT10
                        if nnjM == 2
                            perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                        elseif nnjM == 3
                            perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                        elseif nnjM == 4
                            perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                        elseif nnjM == 5
                            perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                        elseif nnjM == 6
                            perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop10[tvec],line=(wline,:auto),label="c₁₀")
                        elseif nnjM == 7
                            perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop10[tvec],line=(wline,:auto),label="c₁₀")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop12[tvec],line=(wline,:auto),label="c₁₂")
                        elseif nnjM == 8
                            perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop10[tvec],line=(wline,:auto),label="c₁₀")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop12[tvec],line=(wline,:auto),label="c₁₂")
                            perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop14[tvec],line=(wline,:auto),label="c₁₄")
                        else
                            perrMhcopc = plot()
                        end
                    else
                        if is_plot_errMhcop_l_1
                            if nnjM == 2
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                    
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                            elseif nnjM == 4
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop3[tvec],line=(wline,:auto),label="c₃")
                    
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                            elseif nnjM == 6
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop3[tvec],line=(wline,:auto),label="c₃")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop5[tvec],line=(wline,:auto),label="c₅")
                    
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                            elseif nnjM == 8
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop3[tvec],line=(wline,:auto),label="c₃")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop5[tvec],line=(wline,:auto),label="c₅")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop7[tvec],line=(wline,:auto),label="c₇")
                    
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                            elseif nnjM == 10
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop3[tvec],line=(wline,:auto),label="c₃")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop5[tvec],line=(wline,:auto),label="c₅")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop7[tvec],line=(wline,:auto),label="c₇")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop9[tvec],line=(wline,:auto),label="c₉")
                    
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop10[tvec],line=(wline,:auto),label="c₁₀")
                            elseif nnjM == 12
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop3[tvec],line=(wline,:auto),label="c₃")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop5[tvec],line=(wline,:auto),label="c₅")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop7[tvec],line=(wline,:auto),label="c₇")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop9[tvec],line=(wline,:auto),label="c₉")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop11[tvec],line=(wline,:auto),label="c₁₁")
                    
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop10[tvec],line=(wline,:auto),label="c₁₀")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop12[tvec],line=(wline,:auto),label="c₁₂")
                            else
                                perrMhcopc = plot()
                                perrMhcopDc = plot()
                            end
                        else
                            if nnjM == 2
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                            elseif nnjM == 4
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                            elseif nnjM == 6
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                            elseif nnjM == 8
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                            elseif nnjM == 10
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop10[tvec],line=(wline,:auto),label="c₁₀")
                            elseif nnjM == 12
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop10[tvec],line=(wline,:auto),label="c₁₀")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop12[tvec],line=(wline,:auto),label="c₁₂")
                            else
                                perrMhcopc = plot()
                                perrMhcopDc = plot()
                            end
                        end
                    end
                else
                    if norm(ua[isp33]) ≤ epsT10
                        if 1 == 1
                            if nnjM == 2
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            elseif nnjM == 3
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                            elseif nnjM == 4
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                            elseif nnjM == 5
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                            elseif nnjM == 6
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop10[tvec],line=(wline,:auto),label="c₁₀")
                            elseif nnjM == 7
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop10[tvec],line=(wline,:auto),label="c₁₀")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop12[tvec],line=(wline,:auto),label="c₁₂")
                            elseif nnjM == 8
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop8[tvec],line=(wline,:auto),label="c₈")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop10[tvec],line=(wline,:auto),label="c₁₀")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop12[tvec],line=(wline,:auto),label="c₁₂")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop14[tvec],line=(wline,:auto),label="c₁₄")
                            else
                                perrMhcopc = plot()
                                perrMhcopDc = plot()
                            end
                        end
                    else
                        if is_plot_errMhcop_l_1
                            if nMod[isp33] == 1
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel,xlabel=xlabel)
                
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂")
                            elseif nMod[isp33] == 2
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel,xlabel=xlabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop3[tvec],line=(wline,:auto),label="c₃")
                
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                            elseif nMod[isp33] == 3
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel,xlabel=xlabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop3[tvec],line=(wline,:auto),label="c₃")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop5[tvec],line=(wline,:auto),label="c₅")
                
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                            else
                                perrMhcopc = plot()
                                perrMhcopDc = plot()
                            end
                        else
                            if nMod[isp33] == 1
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                            elseif nMod[isp33] == 2
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                            elseif nMod[isp33] == 3
                                perrMhcopc = plot(tplot[tvec],errMhcoplct.errMhcop2[tvec],line=(wline,:auto),label="c₂",ylabel=ylabel,xlabel=xlabel)
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop4[tvec],line=(wline,:auto),label="c₄")
                                perrMhcopc = plot!(tplot[tvec],errMhcoplct.errMhcop6[tvec],line=(wline,:auto),label="c₆")
                            else
                                perrMhcopc = plot()
                                perrMhcopDc = plot()
                            end
                        end
                    end
                end
            end
            if ns ≥ 4
            end
        end

        perrMhcop = display(plot(perrMhcopa,perrMhcopb,layout=(2,1)))
    
        plot(perrMhcopa,perrMhcopb,layout=(2,1))
        savefig(string(file_fig_file,"_errMhcopl.png"))
    end
end
