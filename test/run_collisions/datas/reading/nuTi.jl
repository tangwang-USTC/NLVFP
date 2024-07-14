
# [nh, uh, Th]
if isfile(file_Ms_nModa)
    nuTat = CSV.File(read(file_Ms_nModa)) |> DataFrame
    if 1 == 1
        if missing_deal == :nothing
        elseif missing_deal == :drop
            dropmissing!(nuTat)
        elseif missing_deal == :NaN
            replace!.(eachcol(nuTat), missing => NaN)
        elseif missing_deal == :zero
            replace!.(eachcol(nuTat), missing => 0.0)
        end
        unique!(nuTat,1)            # due to t = x[1]
        tplot = nuTat.t * (tdd / τ₀)
        Nt = length(tplot)
        tvec = tplot_min .< tplot .≤ tplot_max
        tvec[1:dkivv2] .= false
    
        nModat = ceil(Int64,sum(nuTat.nModa) / length(nuTat.nModa))
        if nModat == 1
            Khat = nuTat.na1 .* nuTat.vath1 .^2
            ylabel = string("n̂ - 1 [ε]")
            pna1 = plot(tplot[tvec],nuTat.na1[tvec],line=(wline,:auto),label="a",ylabel=ylabel,title=title_TEk)
            ylabel = string("û")
            puha1 = plot(tplot[tvec],nuTat.ua1[tvec],line=(wline,:auto),label="a",ylabel=ylabel)
            ylabel = string("v̂ₜₕ-1 [ε]")
            pTha1 = plot(tplot[tvec],(nuTat.vath1[tvec] .- 1) *neps,line=(wline,:auto),label="a",ylabel=ylabel,xlabel=title_model)
        elseif nModat == 2
            Khat = nuTat.na1 .* nuTat.vath1 .^2
            Khat += nuTat.na2 .* nuTat.vath2 .^2
            ylabel = string("n̂")
            pna1 = plot(tplot[tvec],nuTat.na1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel,title=title_TEk)
            pna2 = plot!(tplot[tvec],nuTat.na2[tvec],line=(wline,:auto),label="a₂")
            ylabel = string("û")
            puha1 = plot(tplot[tvec],nuTat.ua1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
            puha2 = plot!(tplot[tvec],nuTat.ua2[tvec],line=(wline,:auto),label="a₂")
            ylabel = string("v̂ₜₕ")
            pTha1 = plot(tplot[tvec],nuTat.vath1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel,xlabel=title_model)
            if nuTat.nModa[end] == 1
                vath2t = nuTat.vath2
                is_0_vath2 = vath2t .== 0
                vath2t[is_0_vath2] = nuTat.vath1[is_0_vath2]
                pTha2 = plot!(tplot[tvec],vath2t[tvec],line=(wline,:auto),label="a₂",legend=legendtR)
            else
                vath2t = nuTat.vath2
                pTha2 = plot!(tplot[tvec],vath2t[tvec],line=(wline,:auto),label="a₂",legend=legendtR)
                # pTha2 = plot!(tplot[tvec],vath2t[tvec],line=(wline,:auto),label="a₂",legend=legendtL)
            end
        elseif nModat == 3
            Khat = nuTat.na1 .* nuTat.vath1 .^2
            Khat += nuTat.na2 .* nuTat.vath2 .^2
            Khat += nuTat.na3 .* nuTat.vath3 .^2
            ylabel = string("n̂")
            pna1 = plot(tplot[tvec],nuTat.na1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel,title=title_TEk)
            pna2 = plot!(tplot[tvec],nuTat.na2[tvec],line=(wline,:auto),label="a₂",legend=legendtR)
            pna3 = plot!(tplot[tvec],nuTat.na3[tvec],line=(wline,:auto),label="a₃")
            ylabel = string("û")
            puha1 = plot(tplot[tvec],nuTat.ua1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
            puha2 = plot!(tplot[tvec],nuTat.ua2[tvec],line=(wline,:auto),label="a₂")
            puha3 = plot!(tplot[tvec],nuTat.ua3[tvec],line=(wline,:auto),label="a₃")
            ylabel = string("v̂ₜₕ")
            pTha1 = plot(tplot[tvec],nuTat.vath1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel,xlabel=title_model)
            pTha2 = plot!(tplot[tvec],nuTat.vath2[tvec],line=(wline,:auto),label="a₂",legend=legendtR)
            pTha3 = plot!(tplot[tvec],nuTat.vath3[tvec],line=(wline,:auto),label="a₃")
        elseif nModat == 4
            Khat = nuTat.na1 .* nuTat.vath1 .^2
            Khat += nuTat.na2 .* nuTat.vath2 .^2
            Khat += nuTat.na3 .* nuTat.vath3 .^2
            Khat += nuTat.na4 .* nuTat.vath4 .^2
            ylabel = string("n̂")
            pna1 = plot(tplot[tvec],nuTat.na1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel,title=title_TEk)
            pna2 = plot!(tplot[tvec],nuTat.na2[tvec],line=(wline,:auto),label="a₂",legend=legendtR)
            pna3 = plot!(tplot[tvec],nuTat.na3[tvec],line=(wline,:auto),label="a₃")
            pna4 = plot!(tplot[tvec],nuTat.na4[tvec],line=(wline,:auto),label="a₄")
            ylabel = string("û")
            puha1 = plot(tplot[tvec],nuTat.ua1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
            puha2 = plot!(tplot[tvec],nuTat.ua2[tvec],line=(wline,:auto),label="a₂")
            puha3 = plot!(tplot[tvec],nuTat.ua3[tvec],line=(wline,:auto),label="a₃")
            puha4 = plot!(tplot[tvec],nuTat.ua4[tvec],line=(wline,:auto),label="a₄")
            ylabel = string("v̂ₜₕ")
            pTha1 = plot(tplot[tvec],nuTat.vath1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel,xlabel=title_model)
            pTha2 = plot!(tplot[tvec],nuTat.vath2[tvec],line=(wline,:auto),label="a₂",legend=legendtR)
            pTha3 = plot!(tplot[tvec],nuTat.vath3[tvec],line=(wline,:auto),label="a₃")
            pTha4 = plot!(tplot[tvec],nuTat.vath4[tvec],line=(wline,:auto),label="a₄")
        elseif nModat == 5
            Khat = nuTat.na1 .* nuTat.vath1 .^2
            Khat += nuTat.na2 .* nuTat.vath2 .^2
            Khat += nuTat.na3 .* nuTat.vath3 .^2
            Khat += nuTat.na4 .* nuTat.vath4 .^2
            Khat += nuTat.na5 .* nuTat.vath5 .^2
            ylabel = string("n̂")
            pna1 = plot(tplot[tvec],nuTat.na1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel,title=title_TEk)
            pna2 = plot!(tplot[tvec],nuTat.na2[tvec],line=(wline,:auto),label="a₂",legend=legendtR)
            pna3 = plot!(tplot[tvec],nuTat.na3[tvec],line=(wline,:auto),label="a₃")
            pna4 = plot!(tplot[tvec],nuTat.na4[tvec],line=(wline,:auto),label="a₄")
            pna5 = plot!(tplot[tvec],nuTat.na5[tvec],line=(wline,:auto),label="a₅")
            ylabel = string("û")
            puha1 = plot(tplot[tvec],nuTat.ua1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel)
            puha2 = plot!(tplot[tvec],nuTat.ua2[tvec],line=(wline,:auto),label="a₂")
            puha3 = plot!(tplot[tvec],nuTat.ua3[tvec],line=(wline,:auto),label="a₃")
            puha4 = plot!(tplot[tvec],nuTat.ua4[tvec],line=(wline,:auto),label="a₄")
            puha5 = plot!(tplot[tvec],nuTat.ua5[tvec],line=(wline,:auto),label="a₅")
            ylabel = string("v̂ₜₕ")
            pTha1 = plot(tplot[tvec],nuTat.vath1[tvec],line=(wline,:auto),label="a₁",ylabel=ylabel,xlabel=title_model)
            pTha2 = plot!(tplot[tvec],nuTat.vath2[tvec],line=(wline,:auto),label="a₂",legend=legendtR)
            pTha3 = plot!(tplot[tvec],nuTat.vath3[tvec],line=(wline,:auto),label="a₃")
            pTha4 = plot!(tplot[tvec],nuTat.vath4[tvec],line=(wline,:auto),label="a₄")
            pTha5 = plot!(tplot[tvec],nuTat.vath5[tvec],line=(wline,:auto),label="a₅")
        else
            sdbfgn
        end
    end

    if ns == 2
        nuTbt = CSV.File(read(file_Ms_nModb)) |> DataFrame
        if 1 == 1
            if missing_deal == :nothing
            elseif missing_deal == :drop
                dropmissing!(nuTbt)
            elseif missing_deal == :NaN
                replace!.(eachcol(nuTbt), missing => NaN)
            elseif missing_deal == :zero
                replace!.(eachcol(nuTbt), missing => 0.0)
            end
            unique!(nuTbt,1)            # due to t = x[1]
            tplot = nuTbt.t * (tdd / τ₀)
            Nt = length(tplot)
        
            tvec = tplot_min .< tplot .≤ tplot_max
            tvec[1:dkivv2] .= false
            
            nModbt = ceil(Int64,sum(nuTbt.nModb) / length(nuTbt.nModb))
            if nModbt == 1
                Khbt = nuTbt.nb1 .* nuTbt.vbth1 .^2
                # ylabel = string("n̂ - 1 [ε]")
                pnb1 = plot(tplot[tvec],nuTbt.nb1[tvec],line=(wline,:auto),label="b",title=title_nv_nMod)
                puhb1 = plot(tplot[tvec],nuTbt.ub1[tvec],line=(wline,:auto),label="b")
                # ylabel = string("v̂ₜₕ-1 [ε]")
                pThb1 = plot(tplot[tvec],(nuTbt.vbth1[tvec] .- 1) *neps,line=(wline,:auto),label="b",xlabel=title_alg)
            elseif nModbt == 2
                Khbt = nuTbt.nb1 .* nuTbt.vbth1 .^2
                Khbt += nuTbt.nb2 .* nuTbt.vbth2 .^2
                ylabel = string("n̂")
                pnb1 = plot(tplot[tvec],nuTbt.nb1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,title=title_nv_nMod)
                pnb2 = plot!(tplot[tvec],nuTbt.nb2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
                puhb1 = plot(tplot[tvec],nuTbt.ub1[tvec],line=(wline,:auto),label="b₁",legend=legendtR)
                puhb2 = plot!(tplot[tvec],nuTbt.ub2[tvec],line=(wline,:auto),label="b₂")
                ylabel = string("v̂ₜₕ")
                pThb1 = plot(tplot[tvec],nuTbt.vbth1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,xlabel=title_alg,legend=legendtR)
                if nuTbt.nModb[end] == 1
                    vbth2t = nuTbt.vbth2
                    is_0_vbth2 = vbth2t .== 0.0
                    vbth2t[is_0_vbth2] = nuTbt.vbth1[is_0_vbth2]
                    pThb2 = plot!(tplot[tvec],vbth2t[tvec],line=(wline,:auto),label="b₂")
                else
                    vbth2t = nuTbt.vbth2
                    pThb2 = plot!(tplot[tvec],vbth2t[tvec],line=(wline,:auto),label="b₂")
                    # pThb2 = plot!(tplot[tvec],vbth2t[tvec],line=(wline,:auto),label="b₂",legend=legendtL)
                end
            elseif nModbt == 3
                Khbt = nuTbt.nb1 .* nuTbt.vbth1 .^2
                Khbt += nuTbt.nb2 .* nuTbt.vbth2 .^2
                Khbt += nuTbt.nb3 .* nuTbt.vbth3 .^2
                ylabel = string("n̂")
                pnb1 = plot(tplot[tvec],nuTbt.nb1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,title=title_nv_nMod)
                pnb2 = plot!(tplot[tvec],nuTbt.nb2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
                pnb3 = plot!(tplot[tvec],nuTbt.nb3[tvec],line=(wline,:auto),label="b₃")
                puhb1 = plot(tplot[tvec],nuTbt.ub1[tvec],line=(wline,:auto),label="b₁",legend=legendtR)
                puhb2 = plot!(tplot[tvec],nuTbt.ub2[tvec],line=(wline,:auto),label="b₂")
                puhb3 = plot!(tplot[tvec],nuTbt.ub3[tvec],line=(wline,:auto),label="b₃")
                ylabel = string("v̂ₜₕ")
                pThb1 = plot(tplot[tvec],nuTbt.vbth1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,xlabel=title_alg)
                pThb2 = plot!(tplot[tvec],nuTbt.vbth2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
                pThb3 = plot!(tplot[tvec],nuTbt.vbth3[tvec],line=(wline,:auto),label="b₃")
            elseif nModbt == 4
                Khbt = nuTbt.nb1 .* nuTbt.vbth1 .^2
                Khbt += nuTbt.nb2 .* nuTbt.vbth2 .^2
                Khbt += nuTbt.nb3 .* nuTbt.vbth3 .^2
                Khbt += nuTbt.nb4 .* nuTbt.vbth4 .^2
                ylabel = string("n̂")
                pnb1 = plot(tplot[tvec],nuTbt.nb1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,title=title_nv_nMod)
                pnb2 = plot!(tplot[tvec],nuTbt.nb2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
                pnb3 = plot!(tplot[tvec],nuTbt.nb3[tvec],line=(wline,:auto),label="b₃")
                pnb4 = plot!(tplot[tvec],nuTbt.nb4[tvec],line=(wline,:auto),label="b₄")

                puhb1 = plot(tplot[tvec],nuTbt.ub1[tvec],line=(wline,:auto),label="b₁",legend=legendtR)
                puhb2 = plot!(tplot[tvec],nuTbt.ub2[tvec],line=(wline,:auto),label="b₂")
                puhb3 = plot!(tplot[tvec],nuTbt.ub3[tvec],line=(wline,:auto),label="b₃")
                puhb4 = plot!(tplot[tvec],nuTbt.ub4[tvec],line=(wline,:auto),label="b₄")
                ylabel = string("v̂ₜₕ")
                pThb1 = plot(tplot[tvec],nuTbt.vbth1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,xlabel=title_alg)
                pThb2 = plot!(tplot[tvec],nuTbt.vbth2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
                pThb3 = plot!(tplot[tvec],nuTbt.vbth3[tvec],line=(wline,:auto),label="b₃")
                pThb4 = plot!(tplot[tvec],nuTbt.vbth4[tvec],line=(wline,:auto),label="b₄")
            elseif nModbt == 5
                Khbt = nuTbt.nb1 .* nuTbt.vbth1 .^2
                Khbt += nuTbt.nb2 .* nuTbt.vbth2 .^2
                Khbt += nuTbt.nb3 .* nuTbt.vbth3 .^2
                Khbt += nuTbt.nb4 .* nuTbt.vbth4 .^2
                Khbt += nuTbt.nb5 .* nuTbt.vbth5 .^2
                ylabel = string("n̂")
                pnb1 = plot(tplot[tvec],nuTbt.nb1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,title=title_nv_nMod)
                pnb2 = plot!(tplot[tvec],nuTbt.nb2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
                pnb3 = plot!(tplot[tvec],nuTbt.nb3[tvec],line=(wline,:auto),label="b₃")
                pnb4 = plot!(tplot[tvec],nuTbt.nb4[tvec],line=(wline,:auto),label="b₄")
                pnb5 = plot!(tplot[tvec],nuTbt.nb5[tvec],line=(wline,:auto),label="b₅")

                puhb1 = plot(tplot[tvec],nuTbt.ub1[tvec],line=(wline,:auto),label="b₁",legend=legendtR)
                puhb2 = plot!(tplot[tvec],nuTbt.ub2[tvec],line=(wline,:auto),label="b₂")
                puhb3 = plot!(tplot[tvec],nuTbt.ub3[tvec],line=(wline,:auto),label="b₃")
                puhb4 = plot!(tplot[tvec],nuTbt.ub4[tvec],line=(wline,:auto),label="b₄")
                puhb5 = plot!(tplot[tvec],nuTbt.ub5[tvec],line=(wline,:auto),label="b₅")
                ylabel = string("v̂ₜₕ")
                pThb1 = plot(tplot[tvec],nuTbt.vbth1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,xlabel=title_alg)
                pThb2 = plot!(tplot[tvec],nuTbt.vbth2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
                pThb3 = plot!(tplot[tvec],nuTbt.vbth3[tvec],line=(wline,:auto),label="b₃")
                pThb4 = plot!(tplot[tvec],nuTbt.vbth4[tvec],line=(wline,:auto),label="b₄")
                pThb5 = plot!(tplot[tvec],nuTbt.vbth5[tvec],line=(wline,:auto),label="b₅")
            else
                sdbfgn
            end
        end
    else
        nuTbt = CSV.File(read(file_Ms_nModb)) |> DataFrame
        if 1 == 1
            if missing_deal == :nothing
            elseif missing_deal == :drop
                dropmissing!(nuTbt)
            elseif missing_deal == :NaN
                replace!.(eachcol(nuTbt), missing => NaN)
            elseif missing_deal == :zero
                replace!.(eachcol(nuTbt), missing => 0.0)
            end
            unique!(nuTbt,1)            # due to t = x[1]
            tplot = nuTbt.t * (tdd / τ₀)
            Nt = length(tplot)
        
            tvec = tplot_min .< tplot .≤ tplot_max
            tvec[1:dkivv2] .= false
            
            nModbt = nuTbt.nModb[1]
            if nModbt == 1
                # ylabel = string("n̂ - 1 [ε]")
                pnb1 = plot(tplot[tvec],nuTbt.nb1[tvec],line=(wline,:auto),label="b",title=title_nv)
                puhb1 = plot(tplot[tvec],nuTbt.ub1[tvec],line=(wline,:auto),label="b")
                # ylabel = string("v̂ₜₕ-1 [ε]")
                pThb1 = plot(tplot[tvec],(nuTbt.vbth1[tvec] .- 1) *neps,line=(wline,:auto),label="b")
            elseif nModbt == 2
                ylabel = string("n̂")
                pnb1 = plot(tplot[tvec],nuTbt.nb1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,title=title_nv)
                pnb2 = plot!(tplot[tvec],nuTbt.nb2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
                puhb1 = plot(tplot[tvec],nuTbt.ub1[tvec],line=(wline,:auto),label="b₁",legend=legendtR)
                puhb2 = plot!(tplot[tvec],nuTbt.ub2[tvec],line=(wline,:auto),label="b₂")
                ylabel = string("v̂ₜₕ")
                pThb1 = plot(tplot[tvec],nuTbt.vbth1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,legend=legendtR)
                if nuTbt.nModb[end] == 1
                    vbth2t = nuTbt.vbth2
                    is_0_vbth2 = vbth2t .== 0.0
                    vbth2t[is_0_vbth2] = nuTbt.vbth1[is_0_vbth2]
                    pThb2 = plot!(tplot[tvec],vbth2t[tvec],line=(wline,:auto),label="b₂")
                else
                    vbth2t = nuTbt.vbth2
                    pThb2 = plot!(tplot[tvec],vbth2t[tvec],line=(wline,:auto),label="b₂")
                    # pThb2 = plot!(tplot[tvec],vbth2t[tvec],line=(wline,:auto),label="b₂",legend=legendtL)
                end
            elseif nModbt == 3
                ylabel = string("n̂")
                pnb1 = plot(tplot[tvec],nuTbt.nb1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel,title=title_nv)
                pnb2 = plot!(tplot[tvec],nuTbt.nb2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
                pnb3 = plot!(tplot[tvec],nuTbt.nb3[tvec],line=(wline,:auto),label="b₃")
                puhb1 = plot(tplot[tvec],nuTbt.ub1[tvec],line=(wline,:auto),label="b₁",legend=legendtR)
                puhb2 = plot!(tplot[tvec],nuTbt.ub2[tvec],line=(wline,:auto),label="b₂")
                puhb3 = plot!(tplot[tvec],nuTbt.ub3[tvec],line=(wline,:auto),label="b₃")
                ylabel = string("v̂ₜₕ")
                pThb1 = plot(tplot[tvec],nuTbt.vbth1[tvec],line=(wline,:auto),label="b₁",ylabel=ylabel)
                pThb2 = plot!(tplot[tvec],nuTbt.vbth2[tvec],line=(wline,:auto),label="b₂",legend=legendtR)
                pThb3 = plot!(tplot[tvec],nuTbt.vbth3[tvec],line=(wline,:auto),label="b₃")
            else
                sdbfgn
            end
        end
        if ns == 3
            nuTct = CSV.File(read(file_Ms_nModc)) |> DataFrame
            if 1 == 1
                if missing_deal == :nothing
                elseif missing_deal == :drop
                    dropmissing!(nuTct)
                elseif missing_deal == :NaN
                    replace!.(eachcol(nuTct), missing => NaN)
                elseif missing_deal == :zero
                    replace!.(eachcol(nuTct), missing => 0.0)
                end
                unique!(nuTct,1)            # due to t = x[1]
                tplot = nuTct.t * (tdd / τ₀)
                Nt = length(tplot)
            
                tvec = tplot_min .< tplot .≤ tplot_max
                tvec[1:dkivv2] .= false
                
                nModct = nuTct.nModc[1]
                if nModct == 1
                    # ylabel = string("n̂ - 1 [ε]")
                    pnc1 = plot(tplot[tvec],nuTct.nc1[tvec],line=(wline,:auto),label="c",title=title_nMod)
                    puhc1 = plot(tplot[tvec],nuTct.uc1[tvec],line=(wline,:auto),label="c")
                    # ylabel = string("v̂ₜₕ-1 [ε]")
                    pThc1 = plot(tplot[tvec],(nuTct.vcth1[tvec] .- 1) *neps,line=(wline,:auto),label="c",xlabel=title_alg)
                elseif nModct == 2
                    ylabel = string("n̂")
                    pnc1 = plot(tplot[tvec],nuTct.nc1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel,title=title_nMod)
                    pnc2 = plot!(tplot[tvec],nuTct.nc2[tvec],line=(wline,:auto),label="c₂",legend=legendtR)
                    puhc1 = plot(tplot[tvec],nuTct.uc1[tvec],line=(wline,:auto),label="c₁",legend=legendtR)
                    puhc2 = plot!(tplot[tvec],nuTct.uc2[tvec],line=(wline,:auto),label="c₂")
                    ylabel = string("v̂ₜₕ")
                    pThc1 = plot(tplot[tvec],nuTct.vcth1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel,xlabel=title_alg,legend=legendtR)
                    if nuTct.nModc[end] == 1
                        vcth2t = nuTct.vcth2
                        is_0_vcth2 = vcth2t .== 0.0
                        vcth2t[is_0_vcth2] = nuTct.vcth1[is_0_vcth2]
                        pThc2 = plot!(tplot[tvec],vcth2t[tvec],line=(wline,:auto),label="c₂")
                    else
                        vcth2t = nuTct.vcth2
                        pThc2 = plot!(tplot[tvec],vcth2t[tvec],line=(wline,:auto),label="c₂")
                        # pThc2 = plot!(tplot[tvec],vcth2t[tvec],line=(wline,:auto),label="c₂",legend=legendtL)
                    end
                elseif nModct == 3
                    ylabel = string("n̂")
                    pnc1 = plot(tplot[tvec],nuTct.nc1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel,title=title_nMod)
                    pnc2 = plot!(tplot[tvec],nuTct.nc2[tvec],line=(wline,:auto),label="c₂",legend=legendtR)
                    pnc3 = plot!(tplot[tvec],nuTct.nc3[tvec],line=(wline,:auto),label="c₃")
                    puhc1 = plot(tplot[tvec],nuTct.uc1[tvec],line=(wline,:auto),label="c₁",legend=legendtR)
                    puhc2 = plot!(tplot[tvec],nuTct.uc2[tvec],line=(wline,:auto),label="c₂")
                    puhc3 = plot!(tplot[tvec],nuTct.uc3[tvec],line=(wline,:auto),label="c₃")
                    ylabel = string("v̂ₜₕ")
                    pThc1 = plot(tplot[tvec],nuTct.vcth1[tvec],line=(wline,:auto),label="c₁",ylabel=ylabel,xlabel=title_alg)
                    pThc2 = plot!(tplot[tvec],nuTct.vcth2[tvec],line=(wline,:auto),label="c₂",legend=legendtR)
                    pThc3 = plot!(tplot[tvec],nuTct.vcth3[tvec],line=(wline,:auto),label="c₃")
                else
                    sdbfgn
                end
            end
    
            if ns ≥ 4
                nuTdt = CSV.File(read(file_Ms_nModd)) |> DataFrame
                if 1 == 1
                    if missing_deal == :nothing
                    elseif missing_deal == :drop
                        dropmissing!(nuTdt)
                    elseif missing_deal == :NaN
                        replace!.(eachcol(nuTdt), missing => NaN)
                    elseif missing_deal == :zero
                        replace!.(eachcol(nuTdt), missing => 0.0)
                    end
                    unique!(nuTdt,1)            # due to t = x[1]
                    tplot = nuTdt.t * (tdd / τ₀)
                    Nt = length(tplot)
                
                    tvec = tplot_min .< tplot .≤ tplot_max
                    tvec[1:dkivv2] .= false
                    
                    nModdt = nuTdt.nModd[1]
                    if nModdt == 1
                        # ylabel = string("n̂ - 1 [ε]")
                        pnd1 = plot(tplot[tvec],nuTdt.nd1[tvec],line=(wline,:auto),label="d",title=title_nMod)
                        puhd1 = plot(tplot[tvec],nuTdt.ud1[tvec],line=(wline,:auto),label="d")
                        # ylabel = string("v̂ₜₕ-1 [ε]")
                        pThd1 = plot(tplot[tvec],(nuTdt.vdth1[tvec] .- 1) *neps,line=(wline,:auto),label="d",xlabel=title_alg)
                    elseif nModdt == 2
                        ylabel = string("n̂")
                        pnd1 = plot(tplot[tvec],nuTdt.nd1[tvec],line=(wline,:auto),label="d₁",ylabel=ylabel,title=title_nMod)
                        pnd2 = plot!(tplot[tvec],nuTdt.nd2[tvec],line=(wline,:auto),label="d₂",legend=legendtR)
                        puhd1 = plot(tplot[tvec],nuTdt.ud1[tvec],line=(wline,:auto),label="d₁",legend=legendtR)
                        puhd2 = plot!(tplot[tvec],nuTdt.ud2[tvec],line=(wline,:auto),label="d₂")
                        ylabel = string("v̂ₜₕ")
                        pThd1 = plot(tplot[tvec],nuTdt.vdth1[tvec],line=(wline,:auto),label="d₁",ylabel=ylabel,xlabel=title_alg,legend=legendtR)
                        if nuTdt.nModd[end] == 1
                            vdth2t = nuTdt.vdth2
                            is_0_vdth2 = vdth2t .== 0.0
                            vdth2t[is_0_vdth2] = nuTdt.vdth1[is_0_vdth2]
                            pThd2 = plot!(tplot[tvec],vdth2t[tvec],line=(wline,:auto),label="d₂")
                        else
                            vdth2t = nuTdt.vdth2
                            pThd2 = plot!(tplot[tvec],vdth2t[tvec],line=(wline,:auto),label="d₂")
                            # pThd2 = plot!(tplot[tvec],vdth2t[tvec],line=(wline,:auto),label="d₂",legend=legendtL)
                        end
                    elseif nModdt == 3
                        ylabel = string("n̂")
                        pnd1 = plot(tplot[tvec],nuTdt.nd1[tvec],line=(wline,:auto),label="d₁",ylabel=ylabel,title=title_nMod)
                        pnd2 = plot!(tplot[tvec],nuTdt.nd2[tvec],line=(wline,:auto),label="d₂",legend=legendtR)
                        pnd3 = plot!(tplot[tvec],nuTdt.nd3[tvec],line=(wline,:auto),label="d₃")
                        puhd1 = plot(tplot[tvec],nuTdt.ud1[tvec],line=(wline,:auto),label="d₁",legend=legendtR)
                        puhd2 = plot!(tplot[tvec],nuTdt.ud2[tvec],line=(wline,:auto),label="d₂")
                        puhd3 = plot!(tplot[tvec],nuTdt.ud3[tvec],line=(wline,:auto),label="d₃")
                        ylabel = string("v̂ₜₕ")
                        pThd1 = plot(tplot[tvec],nuTdt.vdth1[tvec],line=(wline,:auto),label="d₁",ylabel=ylabel,xlabel=title_alg)
                        pThd2 = plot!(tplot[tvec],nuTdt.vdth2[tvec],line=(wline,:auto),label="d₂",legend=legendtR)
                        pThd3 = plot!(tplot[tvec],nuTdt.vdth3[tvec],line=(wline,:auto),label="d₃")
                    else
                        sdbfgn
                    end
                end
            end
        else
            sdfghjnm
        end
    end

    ylabel = string("DK̂ [ε]")
    pKha = plot(tplot[tvec],(Khat[tvec] .- 1)/epsT,line=(wline,:auto),label="a",ylabel=ylabel)
    pKhb = plot(tplot[tvec],(Khbt[tvec] .- 1)/epsT,line=(wline,:auto),label="b")

    if ns == 2
        pnMod = display(plot(pna1,pnb1,puha1,puhb1,pTha1,pThb1,pKha,pKhb,layout=(4,2)))
        if is_fvL_CP == false
            LMat = nuTat.LMa
            LMbt = nuTbt.LMb
        end
        plot(pna1,pnb1,puha1,puhb1,pTha1,pThb1,pKha,pKhb,layout=(4,2))
    else
        if ns == 3
            pnMod = display(plot(pna1,pnb1,pnc1,puha1,puhb1,puhc1,pTha1,pThb1,pThc1,layout=(3,3)))
            if is_fvL_CP == false
                LMat = nuTat.LMa
                LMbt = nuTbt.LMb
            end
            plot(pna1,pnb1,pnc1,puha1,puhb1,puhc1,pTha1,pThb1,pThc1,layout=(3,3))
        elseif ns == 4
            sdfghj
        end
    end
    savefig(string(file_fig_file,"_nuTi.png"))
end
