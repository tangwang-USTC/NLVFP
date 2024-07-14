

"""
  When `is_nai_const = ?`
       `∑(nai) = nh = Const`
       `∑(nai .* uai) = uh = Const`
       `∑(nai .* vthi^2) = Th = Const`

  `nModel1 = nModel - 1`

  strategy: nh9 ≤ nhi, ∀i

  Optimization of the amplitude function `fhl0` by using the `King` functions 
    to finding the optimized parameters `(n̂ₛ,ûₛ,v̂ₜₕₛ) = (nai,uai,vthi)`. 

  The general moments is renormalized as: 
    
    `M̂ⱼₗᵐ*`= M̂ⱼₗᵐ / CjLL2(j,L)`.
 
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments. 
  
  Outputs:
    naik1,vthik1,nModelk1 = fl0king01_fMNKCnhCNKC!(naik1,vthik1,nModelk1,
                LMk,DnuTh,Mhck1,errMhcop,njMs,edtnIKTs,
                Rdtsabk1,Rdtsabk,NL_solve,rtol_DnuTi,NK,NKmax,kt,tk,dtk;
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi,
                is_nhnMod_adapt=is_nhnMod_adapt,is_NK_adapt_max=is_NK_adapt_max,
                is_NK_nai_const=is_NK_nai_const,is_nuTi_initial=is_nuTi_initial,
                is_fixed_NK=is_fixed_NK,
                is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
  
"""

# [ite,nMod], is_optim_CnIK=true
function fl0king01_fMNKCnhCNKC!(naik1::AbstractVector{T}, vthik1::AbstractVector{T}, nModelk1::Int64,
    LMk::Int64, DnuTh::AbstractVector{T}, Mhck1::Matrix{T}, errMhcop::Matrix{T}, 
    njMs::Int64, edtnIKTs::AbstractVector{T},
    Rdtsabk1::T, Rdtsabk::T, NL_solve::Symbol, rtol_DnuTi::T,
    NK::Int64,NKmax::Int64,kt::Int64,tk::T,dtk::T; 
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT, 
    Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1],
    is_nhnMod_adapt::Bool=false,is_NK_adapt_max::Bool=false,
    is_NK_nai_const::Bool=false,is_nuTi_initial::Bool=true,
    is_fixed_NK::Bool=false,
    is_nMod_update_back::Bool=false,is_nMod_update_advance::Bool=false) where {T}
    
    # @show is_nhnMod_adapt, is_NK_adapt_max
    # Mode recognition
    if LMk ≤ 1
        # # Tendency according to `Rdtsabk1` and `Rdtsabk`
        # if Rdtsabk1 ≤ atol_fM_Rdtsab
        #     if Rdtsabk ≤ Rdtsabk1               # Deviation from equilibrium,`Mhj0c → ∞`
        #         Mode_Rdtsab = :KM9
        #     else                                # Tending to be in equilibrium,`Mhj0c → 1`
        #         Mode_Rdtsab = :KM1
        #     end
        # else
        #     if Rdtsabk ≤ Rdtsabk1               # Deviation from equilibrium,`Mhj0c → ∞`
        #         Mode_Rdtsab = :KMM9
        #     else                                # Tending to be in equilibrium,`Mhj0c → 1`
        #         Mode_Rdtsab = :KMM1
        #     end
        # end
        # # @show Mode_Rdtsab
        # 
        if is_fixed_NK || NKmax == 1
            nModelk1 = 1NKmax
            if nModelk1 == 1
                yfitNK = zeros(T,2nModelk1)
                fl0king01_fMNK!(naik1,vthik1,nModelk1,DnuTh,yfitNK,Mhck1,NL_solve;
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                
                errMhcop[1:2nModelk1,1] = yfitNK ./ Mhck1[1:2nModelk1,1]
            else
                yfitNK = zeros(T,2nModelk1-2)
                fl0king01_fMNKC!(naik1,vthik1,nModelk1,DnuTh,yfitNK,Mhck1,NL_solve;
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                        Nspan_optim_nuTi=Nspan_optim_nuTi,is_nuTi_initial=is_nuTi_initial)
                # errMhcop[1:2,1] .= 0.0
                errMhcop[3:2nModelk1,1] = yfitNK ./ Mhck1[3:2nModelk1,1]
            end

            yfitNKM = maximum(abs.(yfitNK))
            errNormNK = norm([DnuTh[1], DnuTh[3], yfitNKM])
            if errNormNK ≤ atol_nuTi_optim
                printstyled("21NK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfitNKM), color=:green,"\n")
            else
                if yfitNKM ≤ atol_nuTi_optim
                    printstyled("21NK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                    printstyled("21NK, yfitNKM=", fmtf2.(yfitNKM), color=:green,"\n")
                    if yfitNKM ≥ atol_nuTi_optim_err
                        @error("21NK, Optimization is falure, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]))
                    end
                else
                    if norm([DnuTh[1], DnuTh[3]]) ≤ atol_nuTi_optim
                        printstyled("21NK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                        printstyled("21NK, yfitNK=", fmtf2.(yfitNK), color=:red,"\n")
                    else
                        printstyled("21NK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3],yfitNKM), color=:red,"\n")
                        if yfitNKM ≥ atol_nuTi_optim_err
                            @error("21NK, Optimization is falure, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]))
                        end
                    end
                end
        
                @error("21NK: Underfitting: Optimization process for `naik,vthik` is not converged when",nModelk1)
            end
        else
            if nModelk1 == 1
                if is_NK_adapt_max
                    nModelk1 = 1NKmax
                else
                    nModelk1 = min(nModelk1 + 1, NKmax)
                end
                @show 558, nModelk1, vthik1
                yfitNK = zeros(T,2nModelk1-2)
                fl0king01_fMNKC!(naik1,vthik1,nModelk1,DnuTh,yfitNK,Mhck1,NL_solve;
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                        Nspan_optim_nuTi=Nspan_optim_nuTi,is_nuTi_initial=is_nuTi_initial)
                errMhcop[3:2nModelk1,1] = yfitNK ./ Mhck1[3:2nModelk1,1]
                # @show yfitNK
                # @show 559, nModelk1, naik1,vthik1
                yfitNKM = maximum(abs.(yfitNK))
                errNormNK = norm([DnuTh[1], DnuTh[3], yfitNKM])
                if errNormNK ≤ atol_nuTi_optim
                    printstyled("21NK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfitNKM), color=:green,"\n")
                else
                    if yfitNKM ≤ atol_nuTi_optim
                        printstyled("21NK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                        printstyled("21NK, yfitNKM=", fmtf2.(yfitNKM), color=:green,"\n")
                        if yfitNKM ≥ atol_nuTi_optim_err
                            @error("21NK, Optimization is falure, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]))
                        end
                    else
                        if norm([DnuTh[1], DnuTh[3]]) ≤ atol_nuTi_optim
                            printstyled("21NK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                            printstyled("21NK, yfitNK=", fmtf2.(yfitNK), color=:red,"\n")
                        else
                            printstyled("21NK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3],yfitNKM), color=:red,"\n")
                            if yfitNKM ≥ atol_nuTi_optim_err
                                @error("21NK, Optimization is falure, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]))
                            end
                        end
                    end
                    @error("21NK: Underfitting: Optimization process for `naik,vthik` is not converged when",nModelk1)
                end
            else
                # @show 550,is_nhnMod_adapt,is_NK_adapt_max, is_nMod_update_back
                # @show 550, nModelk1, vthik1
                # if is_NK_nai_const
                # if tk ≤ 1111111
                if kt ≥ 1
                    naik1,vthik1,nModelk1 = fl0king01_fMnhCNKCNKC!(naik1,vthik1,nModelk1,
                                DnuTh,Mhck1,errMhcop,NL_solve,rtol_DnuTi,NK,NKmax,kt,tk,dtk;
                                optimizer=optimizer,factor=factor,autodiff=autodiff,
                                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                                Nspan_optim_nuTi=Nspan_optim_nuTi,
                                is_nhnMod_adapt=is_nhnMod_adapt,is_NK_adapt_max=is_NK_adapt_max,
                                is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
                    # @show 5511, nModelk1, vthik1
                # else
                #     yfitNK = zeros(T,2nModelk1-2)
                #     fl0king01_fMNKC!(naik1,vthik1,nModelk1,DnuTh,yfitNK,Mhck1,NL_solve;
                #             optimizer=optimizer,factor=factor,autodiff=autodiff,
                #             is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                #             p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                #             Nspan_optim_nuTi=Nspan_optim_nuTi,is_nuTi_initial=true)
                #     errMhcop[3:2nModelk1,1] = yfitNK ./ Mhck1[3:2nModelk1,1]
                #     @show 5512, nModelk1, vthik1
                #     dfhgnm
                end
            end
        end
    else
        efrgbhn
    end
    
    # @show 5510, nModelk1, vthik1
    return naik1, vthik1, nModelk1
end

# [ite,nMod≥2],    # _nhC → _NKC
function fl0king01_fMnhCNKCNKC!(naik1::AbstractVector{T}, vthik1::AbstractVector{T}, nModelk1::Int64,
    DnuTh::AbstractVector{T}, Mhck1::Matrix{T}, errMhcop::Matrix{T}, 
    NL_solve::Symbol, rtol_DnuTi::T,NK::Int64,NKmax::Int64,kt::Int64,tk::T,dtk::T; 
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT, 
    Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1],
    is_nhnMod_adapt::Bool=false,is_NK_adapt_max::Bool=false,
    is_nMod_update_back::Bool=false,is_nMod_update_advance::Bool=false) where {T}
    
    # @show atol_nuTi_optim
    # @show is_nhnMod_adapt, rtol_DnuTi
    # @show "........................................."
    # ertgfh
    yfit = zeros(nModelk1-1)
    fl0king01_fMnhC!(naik1,vthik1,nModelk1,DnuTh,yfit,Mhck1,NL_solve;
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi,
            is_nuTi_initial=false)

    yfitM = maximum(abs.(yfit))
    errMhcop[3:nModelk1+1,1] = yfit ./ Mhck1[3:nModelk1+1,1]
    is_converged_nMod = norm([DnuTh[1], DnuTh[3], yfitM]) ≤ atol_nuTi_optim
    # @show 6661, nModelk1, vthik1

    # Reduce `nMod`
    # if is_converged_nMod
    #     printstyled("221nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfitM), color=:green,"\n")
    # else
        if yfitM ≤ atol_nuTi_optim
            printstyled("222nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
            printstyled("222nh, yfitM=", fmtf2.(yfitM), color=:green,"\n")
        else
            if norm([DnuTh[1], DnuTh[3]]) ≤ atol_nuTi_optim 
                if is_nMod_update_advance
                    if kt ≥ 1
                        # if is_NK_adapt_max
                        #     nModelk1 = 1NKmax
                        # else
                        #     nModelk1 = min(nModelk1 + 1, NKmax)
                        # end
                        nModelk1 = 2
                    else
                        if is_NK_adapt_max
                            nModelk1 = 1NK
                        else
                            nModelk1 = min(nModelk1 + 1, NK)
                        end
                    end
                    yfit = zeros(T,2nModelk1-2)
                    fl0king01_fMNKC!(naik1,vthik1,nModelk1,DnuTh,yfit,Mhck1,NL_solve;
                            optimizer=optimizer,factor=factor,autodiff=autodiff,
                            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                            Nspan_optim_nuTi=Nspan_optim_nuTi,is_nuTi_initial=true)
                    # @show 22, nModelk1, vthik1
                    # @show "++++++++++++++++++++++++++++++"
                    errMhcop[3:2nModelk1,1] = yfit ./ Mhck1[3:2nModelk1,1]
                    yfitM = maximum(abs.(yfit))
                        
                    is_converged_nMod = norm([DnuTh[1], DnuTh[3], yfitM]) ≤ atol_nuTi_optim
                    if is_converged_nMod
                        printstyled("2232nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfitM), color=:green,"\n")
                    else
                        if yfitM ≤ atol_nuTi_optim
                            printstyled("2231nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                            printstyled("2231, yfitM=", fmtf2.(yfitM), color=:green,"\n")
                        else
                            if norm([DnuTh[1], DnuTh[3]]) ≤ atol_nuTi_optim
                                printstyled("2231nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                                printstyled("2231, yfit=", fmtf2.(yfit), color=:red,"\n")
                            else
                                printstyled("2231nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3],yfitM), color=:red,"\n")
                                if yfitM ≥ atol_nuTi_optim_err
                                    @error("2231nh, Optimization is falure, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]))
                                end
                            end
                        end
                        @error("2231nh: Optimization process for `naik1,vthik1` is not converged with iteration for `nMod_update`.")
                    end
                else
                    printstyled("223nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                    printstyled("223nh, yfit=", fmtf2.(yfit), color=:red,"\n")
                end
            else
                printstyled("224nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3],yfitM), color=:red,"\n")
                if yfitM ≥ atol_nuTi_optim_err
                    @error("224nh, Optimization is falure, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]))
                end
            end
        end

        ## Checking whether overdetermined, `nModelk1!
        if is_nhnMod_adapt && nModelk1 ≥ 3
            is_nMod_renew, naik1,vthik1,nModelk1 = nMod_update(naik1,vthik1,nModelk1;rtol_DnuTi=rtol_DnuTi)
            # @show 66632, is_nMod_renew, nModelk1, vthik1
        else
            is_nMod_renew = false
        end
        # @show is_nhnMod_adapt, is_nMod_renew, is_nMod_update_back, is_NK_adapt_max,(nModelk1, NK)
        # @show 66632, nModelk1, vthik1
        # wsedfv

        # Updating `naik1,vthik1` according to `Mhck1` with the new `nModelk1`
        if is_nMod_renew && 2 == 1
            if is_nMod_update_back
                if is_NK_adapt_max
                    nModelk1 = 1NKmax
                else
                    nModelk1 = min(nModelk1 + 1, NKmax)
                end
                fgthnm
            end
            # @show 6662, nModelk1, vthik1
            # @show "----------------+++++++++"
            if nModelk1 == 1
                yfitNK = zeros(T,2nModelk1)
                fl0king01_fMNK!(naik1,vthik1,nModelk1,DnuTh,yfitNK,Mhck1,NL_solve;
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                errMhcop[1:2nModelk1,1] = yfitNK ./ Mhck1[1:2nModelk1,1]
            else
                yfit = zeros(T,2nModelk1-2)
                fl0king01_fMNKC!(naik1,vthik1,nModelk1,DnuTh,yfit,Mhck1,NL_solve;
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                        Nspan_optim_nuTi=Nspan_optim_nuTi,is_nuTi_initial=true)
                errMhcop[3:2nModelk1,1] = yfit ./ Mhck1[3:2nModelk1,1]
            end
            # @show nModelk1, vthik1
            # efdgbn
            # @show "++++++++++++++++++++++++++++++"
            yfitM = maximum(abs.(yfit))
                
            is_converged_nMod = norm([DnuTh[1], DnuTh[3], yfitM]) ≤ atol_nuTi_optim
            if is_converged_nMod
                printstyled("31nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfitM), color=:green,"\n")
            else
                if yfitM ≤ atol_nuTi_optim
                    printstyled("31nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                    printstyled("31, yfitM=", fmtf2.(yfitM), color=:green,"\n")
                else
                    if norm([DnuTh[1], DnuTh[3]]) ≤ atol_nuTi_optim
                        printstyled("31nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                        printstyled("31, yfit=", fmtf2.(yfit), color=:red,"\n")
                    else
                        printstyled("31nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3],yfitM), color=:red,"\n")
                        if yfitM ≥ atol_nuTi_optim_err
                            @error("31nh, Optimization is falure, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]))
                        end
                    end
                end
                @error("31nh: Optimization process for `naik1,vthik1` is not converged with iteration for `nMod_update`.")
            end
        else
            # @error("32nh: Optimization process for `naik1,vthik1` is not converged but `Dvthi ≤ rtol_DnuTi`.")
        end
    # end
    return naik1, vthik1, nModelk1
end
