

is_plot_Convergence = false

if is_fixed_timestep

    ivh = 112
    isp = 1
    iFv = 2
    title = string(L"\hat{v}=", fmtf2(vhe[1][ivh]))
    if length(Nτ_fixvec) ≥ 2

        wlinecl = 2

        ######################################################## Convergence
        if is_plot_Convergence
            methodivh = 3
            if methodivh == 1
                RDKvec0 = abs.(DKingvec[ivh,1:end-1,:] .- DKingvec[ivh,end,:]')
            elseif methodivh == 2
                RDKvec0 = abs.(abs.(DKingvec[ivh,1:end-1,:] ./ DKingvec[ivh,end,:]') .- 1)
            else
                ivh = 1
                title = ""
                RDKvec0 = abs.(abs.(DKingvec[ivh,1:end-1,:] ./ DKingvec[ivh,end,:]') .- 1)
                for ivh in 2:nvG[1]
                    RDKvec0[:,:] += abs.(abs.(DKingvec[ivh,1:end-1,:] ./ DKingvec[ivh,end,:]') .- 1) / nvG[1]
                end
            end

            RDKvec1 = deepcopy(RDKvec0[:,1:2])
            RDKvec2 = deepcopy(RDKvec0[:,1:2])
            RDKvec3 = deepcopy(RDKvec0[:,1:2])
            RDKvec4 = deepcopy(RDKvec0[:,1:2])
            for iii in 1:NCase-1
                RDKvec1[iii,:] = RDKvec1[1,:] / (2^1) .^ (iii-1)
                RDKvec2[iii,:] = RDKvec1[1,:] / (2^2) .^ (iii-1)
                RDKvec3[iii,:] = RDKvec1[1,:] / (2^3) .^ (iii-1)
                RDKvec4[iii,:] = RDKvec1[1,:] / (2^4) .^ (iii-1)
            end
            # orderDKavec = order_converg(RDKvec[1:end-1,1])
            # orderDKbvec = order_converg(RDKvec[1:end-1,2])
            orderDKavec = order_converg(RDKvec0[:,1])
            orderDKbvec = order_converg(RDKvec0[:,2])
            orderDKa = sum(orderDKavec) / length(orderDKavec)
            orderDKb = sum(orderDKbvec) / length(orderDKbvec)

            ylabel = L"C ̂\hat{f}_l^0"
            xlabel = L"Δ t"
            il1 = 1
            il2 = 2
            ic1 = 1
            ic2type = 2
            if ic2type == 1
                ic2 = ic1 + 1
            else
                ic2 = ic1 + 6
            end
            vecdt = 1:length(Nτ_fixvec)-1
            dtvec = 1 ./ Nτ_fixvec[vecdt]
            label = L"Numerical"
            # label = L"a, Numerical"
            ppa = plot(dtvec, RDKvec0[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                        ylabel=ylabel,yscale=:log10,title=title,
                        xlabel=xlabel,xscale=:log10,legend=legendbR)
            # label = L"b, Numerical"
            # ppa = plot!(dtvec, RDKvec0[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
            #             ylabel=ylabel,yscale=:log10,
            #             xlabel=xlabel,xscale=:log10,legend=legendbR)

            if orderDKa ≤ 2
                ic1 += 1
                if ic2type == 1
                    ic2 = ic1 + 1
                else
                    ic2 = ic1 + 6
                end
                il1 += 1
                il2 += 1
                label = L"1^{st} order"
                # label = L"a, 1^{st} order"
                plot!(dtvec, RDKvec1[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                if 1 ≤ orderDKa
                    label = L"a, 2^{nd} order"
                    plot!(dtvec, RDKvec2[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                end
            else
                if 2 ≤ orderDKa ≤ 3
                    ic1 += 1
                    if ic2type == 1
                        ic2 = ic1 + 1
                    else
                        ic2 = ic1 + 6
                    end
                    il1 += 1
                    il2 += 1
                    label = L"2^{nd} order"
                    # label = L"a, 2^{nd} order"
                    plot!(dtvec, RDKvec2[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                    label = L"3^{rd} order"
                    plot!(dtvec, RDKvec3[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                else
                    if 3 ≤ orderDKa ≤ 4
                        ic1 += 1
                        if ic2type == 1
                            ic2 = ic1 + 1
                        else
                            ic2 = ic1 + 6
                        end
                        il1 += 1
                        il2 += 1
                        label = L"a, 3^{rd} order"
                        plot!(dtvec, RDKvec3[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                        label = L"a, 4^{th} order"
                        plot!(dtvec, RDKvec4[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                    else
                        sdfgbn
                    end
                end
            end
            # b
            if 1 == 2
                if orderDKb ≤ 2
                    ic1 += 1
                    if ic2type == 1
                        ic2 = ic1 + 1
                    else
                        ic2 = ic1 + 6
                    end
                    il1 += 1
                    il2 += 1
                    label = L"b, 1^{st} order"
                    ppa = plot!(dtvec, RDKvec1[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                ylabel=ylabel,yscale=:log10,
                                xlabel=xlabel)
                    if 1 ≤ orderDKb
                        label = L"b, 2^{nd} order"
                        ppa = plot!(dtvec, RDKvec2[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                    ylabel=ylabel,yscale=:log10,
                                    xlabel=xlabel)
                    end
                else
                    if 2 ≤ orderDKb ≤ 3
                        ic1 += 1
                        if ic2type == 1
                            ic2 = ic1 + 1
                        else
                            ic2 = ic1 + 6
                        end
                        il1 += 1
                        il2 += 1
                        label = L"b, 2^{nd} order"
                        ppa = plot!(dtvec, RDKvec2[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                    ylabel=ylabel,yscale=:log10,
                                    xlabel=xlabel)
                        # ic2 += 1
                        label = L"b, 3^{rd} order"
                        ppa = plot!(dtvec, RDKvec3[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                    ylabel=ylabel,yscale=:log10,
                                    xlabel=xlabel)
                    else
                        if 3 ≤ orderDKb ≤ 4
                            ic1 += 1
                            if ic2type == 1
                                ic2 = ic1 + 1
                            else
                                ic2 = ic1 + 6
                            end
                            il1 += 1
                            il2 += 1
                            label = L"b, 3^{rd} order"
                            ppa = plot!(dtvec, RDKvec3[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel)
                            1
                            label = L"b, 4^{th} order"
                            ppa = plot!(dtvec, RDKvec4[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel)
                        else
                            sdfgbn
                        end
                    end
                end
            end

            # display(plot(ppa))
            # savefig(string(file_fig_file,"_RDKing2th.png"))
            display(plot(ppa))
            title = string(NCase)
            savefig(string(file_fig_file,"_RDKing123th",title,".png"))
        else
            methodvv = 1
            if methodvv == 1
                # RDKing33vec[:,isp] = reverse(abs.(RDKingvec[ivh,:,isp]))
                RDKing33vec[:,isp] = (abs.(RDKingvec[ivh,:,isp]))
                orderavec = order_converg(RDKing33vec[:,isp])
            
                # RDKing33vec[:,iFv] = reverse(abs.(RDKingvec[ivh,:,iFv]))
                RDKing33vec[:,iFv] = (abs.(RDKingvec[ivh,:,iFv]))
                orderbvec = order_converg(RDKing33vec[:,iFv])
                ylabel = string(L"\delta \hat{f}_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
            elseif methodvv == 2
                RDKing33vec[:,isp] = reverse(abs.(RDKfvec[:,isp]))
                orderavec = order_converg(RDKing33vec[:,isp])
            
                RDKing33vec[:,iFv] = reverse(abs.(RDKfvec[:,iFv]))
                orderbvec = order_converg(RDKing33vec[:,iFv])
                ylabel = L"\Delta \hat{f}_l^0"
            elseif methodvv == 3
                RDKing33vec[:,isp] = abs.(RDKingvec[ivh,1:end,isp] .- RDKingvec[ivh,end,isp])
                orderavec = order_converg(RDKing33vec[1:end-1,isp])
            
                RDKing33vec[:,iFv] = abs.(RDKingvec[ivh,1:end,iFv] .- RDKingvec[ivh,end,iFv])
                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                ylabel = string(L"\delta \hat{f}_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
            else
                RDKing33vec[:,isp] = abs.(RDKingvec[ivh,1:end,isp] .- RDKingvec[ivh,end,isp]) ./ (Nτ_fixvec / Nτ_fixvec[1])
                orderavec = order_converg(RDKing33vec[1:end-1,isp])
            
                RDKing33vec[:,iFv] = abs.(RDKingvec[ivh,1:end,iFv] .- RDKingvec[ivh,end,iFv]) ./ (Nτ_fixvec / Nτ_fixvec[1])
                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                ylabel = string(L"\delta \hat{f}_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
            end
        
            ordera = sum(orderavec) / length(orderavec)
            orderb = sum(orderbvec) / length(orderbvec)
        
            RDKing33vec1 = deepcopy(RDKing33vec)
            RDKing33vec2 = deepcopy(RDKing33vec)
            RDKing33vec3 = deepcopy(RDKing33vec)
            RDKing33vec4 = deepcopy(RDKing33vec)
            for iii in 1:NCase-1
                RDKing33vec1[iii,:] = RDKing33vec1[1,:] / (2^1) .^ (iii-1)
                RDKing33vec2[iii,:] = RDKing33vec1[1,:] / (2^2) .^ (iii-1)
                RDKing33vec3[iii,:] = RDKing33vec1[1,:] / (2^3) .^ (iii-1)
                RDKing33vec4[iii,:] = RDKing33vec1[1,:] / (2^4) .^ (iii-1)
            end
            
            if 1 == 1
                xlabel = L"\Delta t"
                il1 = 1
                il2 = 1
                ic1 = 1
                ic2type = 2
                if ic2type == 1
                    ic2 = ic1 + 1
                else
                    ic2 = ic1 + 6
                end
                vecdt = 1:length(Nτ_fixvec)-1
                dtvec = 1 ./ Nτ_fixvec[vecdt]
            
                label = L"a, Numerical"
                # ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,linetypes[il1]),
                ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,:auto),
                            ylabel=ylabel,yscale=:log10)
                label = L"b, Numerical"
                # ppb = plot!(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,linetypes[il2]),
                ppb = plot!(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,:auto),
                            ylabel=ylabel,yscale=:log10,
                            xlabel=xlabel,xscale=:log10,legend=legendbR)
            else
                xlabel = L"\Delta t"
                il1 = 1
                il2 = 1
                ic1 = 1
                ic2type = 2
                if ic2type == 1
                    ic2 = ic1 + 1
                else
                    ic2 = ic1 + 6
                end
                vecdt = 1:length(Nτ_fixvec)-1
                dtvec = 1 ./ Nτ_fixvec[vecdt]
            
                label = L"a, Numerical"
                plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,linetypes[il1]))
                label = L"b, Numerical"
                ppa = plot!(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,linetypes[il2]),
                            ylabel=ylabel,yscale=:log10,
                            xlabel=xlabel,xscale=:log10,legend=legendbR)
                1
        
                if ordera ≤ 2
                    ic1 += 1
                    if ic2type == 1
                        ic2 = ic1 + 1
                    else
                        ic2 = ic1 + 6
                    end
                    il1 += 1
                    il2 += 1
                    label = L"a, 1^{th} order"
                    plot!(dtvec, RDKing33vec1[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                    if 1 ≤ ordera
                        label = L"a, 2^{nd} order"
                        plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                    end
                else
                    if 2 ≤ ordera ≤ 3
                        ic1 += 1
                        if ic2type == 1
                            ic2 = ic1 + 1
                        else
                            ic2 = ic1 + 6
                        end
                        il1 += 1
                        il2 += 1
                        label = L"a, 2^{nd} order"
                        plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                        label = L"a, 3^{th} order"
                        plot!(dtvec, RDKing33vec3[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                    else
                        if 3 ≤ ordera ≤ 4
                            ic1 += 1
                            if ic2type == 1
                                ic2 = ic1 + 1
                            else
                                ic2 = ic1 + 6
                            end
                            il1 += 1
                            il2 += 1
                            label = L"a, 3^{th} order"
                            plot!(dtvec, RDKing33vec3[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                            label = L"a, 4^{th} order"
                            plot!(dtvec, RDKing33vec4[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                        else
                            @show ordera
                            sdfgbn
                        end
                    end
                end
                
                if orderb ≤ 2
                    ic1 += 1
                    if ic2type == 1
                        ic2 = ic1 + 1
                    else
                        ic2 = ic1 + 6
                    end
                    il1 += 1
                    il2 += 1
                    label = L"b, 1^{th} order"
                    ppa = plot!(dtvec, RDKing33vec1[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                ylabel=ylabel,yscale=:log10,
                                xlabel=xlabel)
                    if 1 ≤ ordera
                        label = L"b, 2^{nd} order"
                        ppa = plot!(dtvec, RDKing33vec2[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                    ylabel=ylabel,yscale=:log10,
                                    xlabel=xlabel)
                    end
                else
                    if 2 ≤ ordera ≤ 3
                        ic1 += 1
                        if ic2type == 1
                            ic2 = ic1 + 1
                        else
                            ic2 = ic1 + 6
                        end
                        il1 += 1
                        il2 += 1
                        label = L"b, 2^{nd} order"
                        ppa = plot!(dtvec, RDKing33vec2[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                    ylabel=ylabel,yscale=:log10,
                                    xlabel=xlabel)
                        # ic2 += 1
                        label = L"b, 3^{th} order"
                        ppa = plot!(dtvec, RDKing33vec3[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                    ylabel=ylabel,yscale=:log10,
                                    xlabel=xlabel)
                    else
                        if 3 ≤ ordera ≤ 4
                            ic1 += 1
                            if ic2type == 1
                                ic2 = ic1 + 1
                            else
                                ic2 = ic1 + 6
                            end
                            il1 += 1
                            il2 += 1
                            label = L"b, 3^{th} order"
                            ppa = plot!(dtvec, RDKing33vec3[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel)
                            1
                            label = L"b, 4^{th} order"
                            ppa = plot!(dtvec, RDKing33vec4[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel)
                        else
                            sdfgbn
                        end
                    end
                end
            end
            display(plot(ppa,ppb,layout=(2,1)))
            # # savefig(string(file_fig_file,"_RDT2th.png"))
            # display(plot(ppa))
            # title = string("nτ",nτ)
            # savefig(string(file_fig_file,"_RDKing2order",title,".png"))
        end

        
    end


end

