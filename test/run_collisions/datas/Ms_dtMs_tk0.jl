
        @show is_normδtf
        if 1 == 1
            if is_normδtf
                n̂aE, IaE, KaE = nIKs(fvL0e, vhe, ma, na, vth, ns)

                nIKs2 = zeros(3, ns)
                nIKs!(nIKs2, fvL0e, vhe, ma, na, vth, ns)

                nIKs1 = zeros(3, ns)
                nIKs!(nIKs1, ve, fvL0e, ma, na, vth, ns)

                nIKhs = zeros(3, ns)
                nIKhs!(nIKhs, fvL0e, vhe, ns)

                nIKThs = zeros(4, ns)
                nIKThs!(nIKThs, fvL0e, vhe, ns; errnIKTh=errnIKTh, reltol_DMs=reltol_DMs)
            else
                nIKs2 = zeros(3, ns)
                nIKsc!(nIKs2, fvLc0e, vhe, ma, vth, ns; atol_nIK=atol_nIK)

                nIKs1 = zeros(3, ns)
                nIKsc!(nIKs1, fvLc0e, ve, ma, ns; atol_nIK=atol_nIK)
            end

            if is_normδtf
                dtn̂aE, dtIaE, dtKaE = nIKs(dtfvL0, vhe, ma, na, vth, ns)

                dtnIKs2 = zeros(3, ns)
                nIKs!(dtnIKs2, dtfvL0, vhe, ma, na, vth, ns)

                dtnIKs1 = zeros(3, ns)
                nIKs!(dtnIKs1, ve, dtfvL0, ma, na, vth, ns)
            else
                dtnIKs2 = zeros(3, ns)
                nIKsc!(dtnIKs2, dtfvLc0, vhe, ma, vth, ns; atol_nIK=atol_nIK)

                dtnIKs1 = zeros(3, ns)
                nIKsc!(dtnIKs1, dtfvLc0, ve, ma, ns; atol_nIK=atol_nIK)

                # plotting
                if 1 == 2
                    if ns == 2
                        ylabel = "∂ₜfl0"
                        
                        xlabel = "v̂"
                        pvhdfla = plot(vhe[1], dtfvLc0[1][:,1],label="a",ylabel=ylabel)
                        pvhdflb = plot(vhe[2], dtfvLc0[2][:,1],label="b",xlabel=xlabel)
                        pdtflvhe = plot(pvhdfla,pvhdflb,layout=(2,1))
    
                        xlabel = "v"
                        vemax = min(ve[1][end],ve[2][end])
                        vvvec = ve[1] .< vemax
                        pdfla = plot(ve[1][vvvec], dtfvLc0[1][vvvec,1],label="a")
                        vvvec = ve[2] .< vemax
                        pdflb = plot(ve[2][vvvec], dtfvLc0[2][vvvec,1],label="b",xlabel=xlabel)
                        pdtflve = plot(pdfla,pdflb,layout=(2,1))
    
                        display(plot(pdtflvhe,pdtflve,layout=(1,2)))
                    else
                        if ns == 3
                            ylabel = "∂ₜfl0"
                            
                            xlabel = "v̂"
                            pvhdfla = plot(vhe[1], dtfvLc0[1][:,1],label="a",ylabel=ylabel)
                            pvhdflb = plot(vhe[2], dtfvLc0[2][:,1],label="b")
                            pvhdflc = plot(vhe[3], dtfvLc0[3][:,1],label="c",xlabel=xlabel)
                            pdtflvhe = plot(pvhdfla,pvhdflb,pvhdflc,layout=(3,1))
    
                            xlabel = "v"
                            vemax = 0.06
                            vvvec = ve[1] .< vemax
                            pdfla = plot(ve[1][vvvec], dtfvLc0[1][vvvec,1],label="a")
                            vvvec = ve[2] .< vemax
                            pdflb = plot(ve[2][vvvec], dtfvLc0[2][vvvec,1],label="b")
                            vvvec = ve[3] .< vemax
                            pdflc = plot(ve[3][vvvec], dtfvLc0[3][vvvec,1],label="c",xlabel=xlabel)
                            pdtflve = plot(pdfla,pdflb,pdflc,layout=(3,1))
    
                            display(plot(pdtflvhe,pdtflve,layout=(1,2)))
                        else
                        end
                    end
                end

                err_dtnIKs2 = sum(dtnIKs2[3,:]) / (sum(abs.(dtnIKs2[3,:])) / 3 + epsT1000)
                err_dtnIKs1 = sum(dtnIKs1[3,:]) / (sum(abs.(dtnIKs1[3,:])) / 3 + epsT1000)
                abs(err_dtnIKs2) ≤ rtol_dtnIKs || error("Energy conservation is falure! Checking the algorithm please! err_dtnIKs = ",err_dtnIKs2)
                abs(err_dtnIKs1) ≤ rtol_dtnIKs || error("Energy conservation is falure! Checking the algorithm please! err_dtnIKs = ",err_dtnIKs1)

            end
            
            if is_normδtf
                dtnIK2 = zeros(3, ns)
                dtnIKs!(dtnIK2, dtfvL0, vhe, ma, na, vth, ns; atol_nIK=atol_nIK)

                dtnIK1 = zeros(3, ns)
                dtnIKs!(dtnIK1, ve, dtfvL0, ma, na, vth, ns; atol_nIK=atol_nIK)

                dtnIKTs = zeros(4, ns)
                RdtnIKTs!(dtnIKTs, dtfvL0, vhe, uh, ma, na, vth, ns)

                dtnIKThs = zeros(4, ns)
                RdtnIKTs!(dtnIKThs, dtfvL0, vhe, uh, ns)

                dtIa, dtKa = dtnIK1[2, :], dtnIK1[3, :]
            else
                dtnIKsc2 = zeros(3, ns)
                dtnIKsc!(dtnIKsc2, dtfvLc0, vhe, ma, vth, ns; atol_nIK=atol_nIK)

                dtnIKsc1 = zeros(3, ns)
                dtnIKsc!(dtnIKsc1, dtfvLc0, ve, ma, ns; atol_nIK=atol_nIK)

                dtnIKTsc = zeros(4, ns)
                RdtnIKTcs!(dtnIKTsc, dtfvLc0, vhe, uh, ma, vth, ns)

                RdtnIKTsc = zeros(4, ns)
                RdtnIKTcs!(RdtnIKTsc, vhe, dtfvLc0, uh, na, vth, ns)

                dtIa, dtKa = dtnIKsc2[2, :], dtnIKsc2[3, :]

                err_dtnIKsc2 = sum(dtnIKsc2[3,:]) / (sum(abs.(dtnIKsc2[3,:])) / 3 + epsT1000)
                abs(err_dtnIKsc2) ≤ rtol_dtnIKs || error("Energy conservation is falure! Checking the algorithm please! err_dtnIKsc2 = ",err_dtnIKsc2)
                err_dtnIKsc1 = sum(dtnIKsc1[3,:]) / (sum(abs.(dtnIKsc1[3,:])) / 3 + epsT1000)
                abs(err_dtnIKsc1) ≤ rtol_dtnIKs || error("Energy conservation is falure! Checking the algorithm please! err_dtnIKsc1 = ",err_dtnIKsc1)
                err_dtnIKTsc = sum(dtnIKTsc[3,:]) / (sum(abs.(dtnIKsc[3,:])) / 3 + epsT1000)
                abs(err_dtnIKTsc) ≤ rtol_dtnIKs || error("Energy conservation is falure! Checking the algorithm please! err_dtnIKTsc = ",err_dtnIKTsc)
                @show err_dtnIKTsc
            end
        end
        if 1 == 1
            nas = sum_kbn(na)
            nhS = na / nas
            ρs = sum_kbn(ma .* na)
            ms = ρs / nas
            ms0 = ms / m_unit
            mhS = ma / ms
            ρhS = ρa / ρs
            Is = sum_kbn(Ia)
            us = Is ./ ρs
            Ks = sum_kbn(Ka0)
            Ks0 = Ks / K_unit
            Ts = (2 / 3 * (Ks ./ nas - us^2))
            Ts0 = Ts / T_unit
            vSth = (2Ts / ms)^0.5
            vSth0 = vSth / v_unit
            dtvSth = 2 / 3 / (vSth * ρs) * (sum_kbn(dtKa) - Is / ρs * sum_kbn(dtIa))
            RdtvSth = dtvSth / vSth
            @show dtvSth, RdtvSth
            @show ms0, Ks0, Ts0

            Rc20S = zeros(datatype, njMs, LM1)
            ussign = sign(us)
            for L1 in 1:LM1
                # if iseven(L1)
                #     Rc20S[:,L1] = sum(Rc2[:,L1,:];dims=2)[:,1] * ussign
                # else
                #     Rc20S[:,L1] = sum(Rc2[:,L1,:];dims=2)[:,1]
                # end
                Rc20S[:, L1] = sum(Rc2[:, L1, :]; dims=2)[:, 1]
            end
            RRc20S = deepcopy(Rc20S)    # = Rc20Sh = Rc20S / ρs * vSth^j
            for nj in 1:njMs
                for LL1 in 1:LM1
                    LL = LL1 - 1
                    # RRc20S[nj,LL1] /=  ρs * vSth^(nj)
                    RRc20S[nj, LL1] /= ρs * vSth^(nj + LL)
                    # RRc20S[nj,LL1] /=  ρs * vSth^(nj + LL/2)
                    # RRc20S[nj,LL1] /=  ρs * vSth^(nj + LL)

                    # j = 2(nj - 1)

                    # RRc20S[nj,LL1] /=  ρs * vSth^((j + LL))
                    # RRc20S[nj,LL1] /=  ρs * vSth^((j + LL)/2)
                    # RRc20S[nj,LL1] /=  ρs * vSth^(j/2)
                    # RRc20S[nj,LL1] /=  ρs * vSth^(j/2 + LL)
                    # RRc20S[nj,LL1] /=  ρs * vSth^(j)


                    # RRc20S[:,LL1] ./= abs(RRc20S[end,LL1])
                end
            end
            # label = string("ℓ=0")
            # xlabel = string("j")
            # if norm(ua) ≤ epsT10
            #     ylabel = string("Rₛ")
            #     label = (0:2:LM1-1)'
            #     pRsl0 = plot(RRc20S[:,1:2:end], ylabel=ylabel, label=label, line=(2, :auto), xlabel=xlabel)
            #     label = (1:2:LM1-1)'
            #     pRsl1 = plot(RRc20S[:,2:2:end], ylabel=ylabel, label=label, line=(2, :auto))
            #     pRst0 = plot(pRsl0,pRsl1,layout=(1,2))
    
            #     ylabel = string("R̂₀")
            #     label = ["a" "b"]
            #     pdtMsnnE = plot(dtMsnnE3[:, 1, :], ylabel=ylabel, label=label, line=(2, :auto), title=title)
            # else
            #     ylabel = string("Rₛ")
            #     label = (0:2:LM1-1)'
            #     pRsl0 = plot(RRc20S[:,1:2:end], ylabel=ylabel, label=label, line=(2, :auto), xlabel=xlabel)
            #     label = (1:2:LM1-1)'
            #     pRsl1 = plot(RRc20S[:,2:2:end], ylabel=ylabel, label=label, line=(2, :auto))
            #     pRst0 = plot(pRsl0,pRsl1,layout=(1,2))
    
            #     ylabel = string("R̂₀")
            #     label = ["a" "b"]
            #     pdtMsnnE = plot(dtMsnnE3[:, 1, :], ylabel=ylabel, label=label, line=(2, :auto), title=title)
    
            # end
            # # display(plot(pdtMsnnE, pRst0, layout=(2, 1)))
        end
        