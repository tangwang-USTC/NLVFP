
if nModat == 1
    yscale0 = :identity
else
    yscale0 = :log10
end
wline = 2

# [errMhcop]
if is_moments_out
    if isfile(file_Ms_errMhcopla) && iCase == NCase
        ylabel = string(L"\delta \hat{\scrM}_{j,l}^0")
        ############## plot(tplot,errMhcopM4)
        iCasep = 1
        label = string(L"a_4,(n_2,N_0)=",nnvocpM[iCasep])
        perrMhcopM4a = plot(errMhcop_vec4[iCasep][1],abs.(errMhcop_vec4[iCasep][2][:,1]).+epsT,line=(wline,:auto),label=label,ylabel=ylabel,yscale=yscale0)
        for iCasep in 2:NCase
            label = string(L"a_4,(n_2,N_0)=",nnvocpM[iCasep])
            perrMhcopM4a = plot!(errMhcop_vec4[iCasep][1],abs.(errMhcop_vec4[iCasep][2][:,1]).+epsT,line=(wline,:auto),label=label,ylabel=ylabel,yscale=yscale0)
        end
        
        xlabel = L"t"
        iCasep = 1
        label = string(L"b_4,(n_2,N_0)=",nnvocpM[iCasep])
        perrMhcopM4b = plot(errMhcop_vec4[iCasep][1],abs.(errMhcop_vec4[iCasep][2][:,2]).+epsT,line=(wline,:auto),label=label,ylabel=ylabel,yscale=yscale0,xlabel=xlabel)
        for iCasep in 2:NCase
            label = string(L"b_4,(n_2,N_0)=",nnvocpM[iCasep])
            perrMhcopM4b = plot!(errMhcop_vec4[iCasep][1],abs.(errMhcop_vec4[iCasep][2][:,2]).+epsT,line=(wline,:auto),label=label,ylabel=ylabel,yscale=yscale0)
        end

        perrMhcopM4 = display(plot(perrMhcopM4a,perrMhcopM4b,layout=(2,1)))
    
        plot(perrMhcopM4a,perrMhcopM4b,layout=(2,1))
        savefig(string(file_fig_file,"_errMhcopM4nnv.png"))
        
    end
end
