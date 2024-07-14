

"""
  When `is_nai_const = false`
       `∑(nai) = nh`
       `∑(nai .* uai) = uh`
       `∑(nai .* vthi^2) = Th`

  Optimization of the amplitude function `fhl0` by using the `King` functions 
    to finding the optimized parameters `(n̂ₛ,ûₛ,v̂ₜₕₛ) = (nai,uai,vthi)`. 

  The general moments is renormalized as: 
    
    `M̂ⱼₗᵐ*`= M̂ⱼₗᵐ / CjLL2(j,L)`.
 
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
    naik,vthik,nModel = fl0king01_fMNK!(naik1,vthik1,nModelk1,
                naik,vthik,nModelk,DnuTh,Mhck1,njMs,NL_solve,rtol_DnuTi;
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
  
"""

function fl0king01_fMNKrrrrrrr!(naik1::AbstractVector{T}, vthik1::AbstractVector{T}, nModelk1::Int64,
    naik::AbstractVector{T}, vthik::AbstractVector{T}, nModelk::Int64,
    DnuTh::AbstractVector{T}, Mhck1::Matrix{T}, njMs::Int64, NL_solve::Symbol, rtol_DnuTi::T; 
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT, 
    Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1]) where {T}
    
    # Mode recognition
    if sum(abs.(Mhck1[:,1] .- 1)) / njMs ≤ atol_d1Mhj0
        Mode_Mhck1 = :KM
    else
        Mode_Mhck1 = :KMM
    end
    if Mode_Mhck1 == :KMM
        @show Mhck1[:,1] .- 1
    else
        @show Mhck1[:,1] .- 1
    end
    
    # 
    if Mode_Mhck1 == :KM
        vec = 1:nModelk1
        if nModelk < NK
            if nModelk == 1
                yfit0 = zeros(T,2nModelk1)
                fl0king01_fMNK!(naik1[vec],vthik1[vec],nModelk1,DnuTh,yfit0,Mhck1,NL_solve;
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                yfit0M = maximum(abs.(yfit0))
                if yfit0M == NaN && yfit0M == Inf
                    @show yfit0M
                    edrf11111
                end
                errNorm = norm([DnuTh[1], DnuTh[3], yfit0M])
                @show errNorm
                if errNorm ≤ atol_nuTi_optim
                    printstyled("11KM, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                    if yfit0M ≤ atol_nuTi_optim
                        printstyled("11KM, yfit0M=", fmtf2.(yfit0M), color=:green,"\n")
                    else
                        printstyled("11KM, yfit0=", fmtf2.(yfit0), color=:blue,"\n")
                    end
                else
                    printstyled("12KM, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfit0M), color=:red,"\n")
                    if yfit0M ≤ atol_nuTi_optim
                        printstyled("12KM, yfit0M=", fmtf2.(yfit0M), color=:green,"\n")
                    else
                        printstyled("12KM, yfit0=", fmtf2.(yfit0), color=:blue,"\n")
                    end
            
                    @error("11KM: Underfitting: Optimization process for `naik,vthik` is not converged when",nModelk1)
                    @show fmtf2.(vthik1[1:nModelk1])
                end
            else
                yfit0 = zeros(T,2nModelk1)
                fl0king01_fMNK!(naik1[vec],vthik1[vec],nModelk1,DnuTh,yfit0,Mhck1,NL_solve;
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                yfit0M = maximum(abs.(yfit0))
                if yfit0M == NaN && yfit0M == Inf
                    @show yfit0M
                    edrf22222
                end
                errNorm0 = norm([DnuTh[1], DnuTh[3], yfit0M])
                if errNorm0 ≤ atol_nuTi_optim
                    printstyled("21KM, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                    if yfit0M ≤ atol_nuTi_optim
                        printstyled("21KM, yfit0M=", fmtf2.(yfit0M), color=:green,"\n")
                    else
                        printstyled("21KM, yfit0=", fmtf2.(yfit0), color=:blue,"\n")
                    end
                else
                    printstyled("22KM, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                    if yfit0M ≤ atol_nuTi_optim
                        printstyled("22KM, yfit0M=", fmtf2.(yfit0M), color=:green,"\n")
                    else
                        printstyled("22KM, yfit0=", fmtf2.(yfit0), color=:blue,"\n")
                    end
    
                    ## Checking whether overdetermined, `nModel!
                    if is_nhnMod_adapt
                        is_nMod_renew, naik1, vthik1, nModelk1 = nMod_update(naik1, vthik1, nModelk1;rtol_DnuTi=rtol_DnuTi)
                    else
                        is_nMod_renew = false
                    end
        
                    # Updating `naik,vthik` according to `Mhck1` with the new `nModel`
                    if is_nMod_renew
                        yfitn = zeros(T,2nModelk1)
                        fl0king01_fMNK!(naik1,vthik1,nModelk1,DnuTh,yfitn,Mhck1,NL_solve;
                                optimizer=optimizer,factor=factor,autodiff=autodiff,
                                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                            
                        yfitnM = maximum(abs.(yfitn))
                        if yfitnM == NaN && yfitnM == Inf
                            @show yfitnM
                            edrf33333
                        end
                        errNormn = norm([DnuTh[1], DnuTh[3], yfitnM])
                        if errNormn ≤ atol_nuTi_optim
                            printstyled("23KM, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                        else
                            printstyled("23KM, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                            if yfitnM ≤ atol_nuTi_optim
                                printstyled("23KM, yfitnM=", fmtf2.(yfitnM), color=:green,"\n")
                            else
                                printstyled("23KM, yfitn=", fmtf2.(yfitn), color=:blue,"\n")
                            end
                            @error("22KM: Overfitting: Optimization process for `naik,vthik` is not converged with iteration for `nMod_update`",nModelk1)
                            @show  fmtf2.(vthik1[1:nModelk1])
                        end
                    else
                        @warn("22: Check `rtol_DnuTi` in optimization process  for `naik,vthik`!",nModelk1)
                        @show fmtf2.(vthik1[1:nModelk1])
                        
                        nModelk1p = nModelk - 1
                        naik1p,vthik1p = [1.0], [1.0]
                        yfitn = zeros(T,2nModelk1p)
                        fl0king01_fMNK!(naik1p,vthik1p,nModelk1p,DnuTh,yfitn,Mhck1,NL_solve;
                                optimizer=optimizer,factor=factor,autodiff=autodiff,
                                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                            
                        yfitnM = maximum(abs.(yfitn))
                        if yfitnM == NaN && yfitnM == Inf
                            @show yfitnM
                            edrf33333
                        end
                        errNormn = norm([DnuTh[1], DnuTh[3], yfitnM])
                        if errNormn > errNorm0
                            printstyled("24KM, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                            if yfitnM ≤ atol_nuTi_optim
                                printstyled("24KM, yfitnM=", fmtf2.(yfitnM), color=:green,"\n")
                            else
                                printstyled("24KM, yfitn=", fmtf2.(yfitn), color=:blue,"\n")
                            end
                            @error("22KM: Overfitting: Optimization process for `naik,vthik` is not converged with iteration for `nMod_update`",nModelk1p)
                            @show fmtf2.(vthik1p[1:nModelk1p])
                        else
                            printstyled("24KM, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfitn), color=:green,"\n")
                            nModelk1 = nModelk1p
                            naik1[1],vthik1[1] = [1.0], [1.0]
                            naik1[2:end] .= 0.0
                            vthik1[2:end] .= 0.1
                        end
                    end
                end
            # else
            #     @error("Checking the timestep for optimization process!")
            #     ertfgfg
            end
        end
        # @show NK, nModelk, size(naik1), size(Mhck1)
        nModelk1NK = 1NK
        yfitNK = zeros(T,2nModelk1NK)
        naik1NK,vthik1NK = deepcopy(naik1),deepcopy(vthik1)
        fl0king01_fMNK!(naik1NK,vthik1NK,nModelk1NK,DnuTh,yfitNK,Mhck1,NL_solve;
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
        yfitNKM = maximum(abs.(yfitNK))
        if yfitNKM == NaN && yfitNKM == Inf
            @show yfitNKM
            edrf11111
        end
        errNormNK = norm([DnuTh[1], DnuTh[3], yfitNKM])
        @show errNormNK
        if errNormNK ≤ atol_nuTi_optim
            printstyled("11KMNK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
            if yfitNKM ≤ atol_nuTi_optim
                printstyled("11KMNK, yfitNKM=", fmtf2.(yfitNKM), color=:green,"\n")
            else
                printstyled("11KMNK, yfitNK=", fmtf2.(yfitNK), color=:blue,"\n")
            end
        else
            printstyled("12KMNK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfitNKM), color=:red,"\n")
            if yfitNKM ≤ atol_nuTi_optim
                printstyled("12KMNK, yfitNKM=", fmtf2.(yfitNKM), color=:green,"\n")
            else
                printstyled("12KMNK, yfitNK=", fmtf2.(yfitNK), color=:blue,"\n")
            end
    
            @error("11KMNK: Underfitting: Optimization process for `naik,vthik` is not converged when",nModelk1NK)
            @show fmtf2.(vthik1NK[1:nModelk1NK])
        end
    else
        if nModelk < NK
            nModelk1 = deepcopy(NK)
            # nModelk1 = nModelk + 1
            yfitp = zeros(T,2nModelk1)
            naik1p,vthik1p = deepcopy(naik1),deepcopy(vthik1)
            fl0king01_fMNK!(naik1p,vthik1p,nModelk1,DnuTh,yfitp,Mhck1,NL_solve;
                    optimizer=optimizer,factor=factor,autodiff=autodiff,
                    is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
            yfitpM = maximum(abs.(yfitp))
            if yfitpM == NaN && yfitpM == Inf
                @show yfitp
                edrf444
            end
            errNormp = norm([DnuTh[1], DnuTh[3], yfitpM])
            if errNormp ≤ atol_nuTi_optim
                printstyled("51, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
            else
                printstyled("51, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                if yfitpM ≤ atol_nuTi_optim
                    printstyled("51, yfitnM=", fmtf2.(yfitpM), color=:green,"\n")
                else
                    printstyled("51, yfitn=", fmtf2.(yfitp), color=:blue,"\n")
                end
                @error("51: underfitting: Optimization process for `naik,vthik` is not converged with iteration for `nMod_update`",nModelk1)
                @show fmtf2.(vthik1p[1:nModelk1])
            end
            if nModelk < NK - 1
                nModelk1p1 = nModelk + 2
                yfitp1 =  zeros(T,2nModelk1p1)
                naik1p1,vthik1p1 = deepcopy(naik1),deepcopy(vthik1)
                fl0king01_fMNK!(naik1p1,vthik1p1,nModelk1p1,DnuTh,yfitp1,Mhck1,NL_solve;
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                yfitp1M = maximum(abs.(yfitp1))
                if yfitp1M == NaN && yfitp1M == Inf
                    @show yfitp1
                    edrf555
                end
                errNormp1 = norm([DnuTh[1], DnuTh[3], yfitp1M])
                if errNormp1 ≤ atol_nuTi_optim
                    printstyled("52, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                else
                    printstyled("52, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                    if yfitp1M ≤ atol_nuTi_optim
                        printstyled("52, yfitnM=", fmtf2.(yfitp1M), color=:green,"\n")
                    else
                        printstyled("52, yfitn=", fmtf2.(yfitp1), color=:blue,"\n")
                    end
                    @error("52: Underfitting: Optimization process for `naik,vthik` is not converged with iteration for `nMod_update`",nModelk1p1)
                    @show fmtf2.(vthik1p1[1:nModelk1p1])
                end
                if errNormp > errNormp1
                    nModelk1 = nModelk1p1
                    vecout = 1:nModelk1
                    naik1[vecout],vthik1[vecout] = deepcopy(naik1p1[vecout]),deepcopy(vthik1p1[vecout])
                else
                    vecout = 1:nModelk1
                    naik1[vecout],vthik1[vecout] = deepcopy(naik1p[vecout]),deepcopy(vthik1p[vecout])
                end
            else
                vecout = 1:nModelk1
                naik1[vecout],vthik1[vecout] = deepcopy(naik1p[vecout]),deepcopy(vthik1p[vecout])
            end
            if nModelk1 < NK
                naik1[nModelk1:end] .= 0.0
                vthik1[nModelk1:end] .= 0.1
            end
        else
            yfit0 = zeros(T,2nModelk1)
            fl0king01_fMNK!(naik1,vthik1,nModelk1,DnuTh,yfit0,Mhck1,NL_solve;
                    optimizer=optimizer,factor=factor,autodiff=autodiff,
                    is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
            yfit0M = maximum(abs.(yfit0))
            if yfit0M == NaN && yfit0M == Inf
                @show yfit0M
                edrf999999
            end
            errNorm = norm([DnuTh[1], DnuTh[3], yfit0M])
            if errNorm ≤ atol_nuTi_optim
                printstyled("991, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                if yfit0M ≤ atol_nuTi_optim
                    printstyled("991, yfit0M=", fmtf2.(yfit0M), color=:green,"\n")
                else
                    printstyled("991, yfit0=", fmtf2.(yfit0), color=:blue,"\n")
                end
            else
                printstyled("992, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfit0M), color=:red,"\n")
                if yfit0M ≤ atol_nuTi_optim
                    printstyled("992, yfit0M=", fmtf2.(yfit0M), color=:green,"\n")
                else
                    printstyled("992, yfit0=", fmtf2.(yfit0), color=:blue,"\n")
                end
                @error("991: Underfitting: Optimization process for `naik,vthik` is not converged when `nModel = NK`",nModelk1)
                @show fmtf2.(vthik1[1:nModelk1])
            end
            @show vthik1
        end
    end
    return naik1, vthik1, nModelk1
end

"""
  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
    fl0king01_fMNK!(naik,vthik,nModel,DnuTh,yfit,Mhck1,NL_solve;
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
  
"""

# [nMod]
function fl0king01_fMNK!(naik::AbstractVector{T}, vthik::AbstractVector{T}, nModel::Int64,
    DnuTh::AbstractVector{T}, yfit::AbstractVector{T}, Mhck1::Matrix{T}, NL_solve::Symbol; 
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT, 
    Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1]) where {T}

    # The parameter limits for MCF plasma.
    nModel2 = 2nModel
    x0 = zeros(nModel2)
    lbs = zeros(nModel2)
    ubs = zeros(nModel2)

    vecMod = 1:nModel
        # nai
        lbs[1:2:nModel2] .= 0.0
        ubs[1:2:nModel2] .= 1.0
        ################################################### I
        # x0[1:2:nModel12] = deepcopy(naik[vecMod1])
        ################################################### II
        # for i in vecMod1
        #     x0[2i-1] = deepcopy(naik[i])
        # end
        ################################################### III
        # x0[1] = deepcopy(naik[1])
        # x0[3:2:nModel12] .= 1e-3
    
        ################################################### IV
        x0[1:2:nModel2] .= 0.1

    # vthi
    for i in vecMod
        lbs[2i] = max(vhthMin, vthik[i] / Nspan_optim_nuTi[3])
        ubs[2i] = min(vhthMax, Nspan_optim_nuTi[3] * vthik[i])
    end
    # ubs[2:2:nModel2] .= vhthMax
    # lbs[2:2:nModel2] .= vhthMin
        if is_NKC_vhthInitial
            for i in vecMod
                x0[2i] = max(vhthInitial,lbs[2i])
            end
        else
            for i in vecMod
                x0[2i] = max(vthik[i] * vhthRatio,lbs[2i])
            end
        end

    res = fl0king01_fMNK(deepcopy(x0), naik[vecMod], vthik[vecMod], nModel, Mhck1[1:nModel2,1], NL_solve; 
        lbs=lbs, ubs=ubs, 
        optimizer=optimizer, factor=factor, autodiff=autodiff,
        is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)

    if NL_solve == :LeastSquaresOptim
        xfit = res.minimizer         # the vector of best model1 parameters
        naik[vecMod] = xfit[1:2:nModel2]          # the vector of best model1 parameters
        vthik[vecMod] = xfit[2:2:nModel2]         # the vector of best model1 parameters
        niter = res.iterations
        is_converged = res.converged
        xssr = res.ssr                         # sum(abs2, fcur)
    elseif NL_solve == :NLsolve
        xfit = res.zero         # the vector of best model1 parameters
        naik[vecMod] = xfit[1:2:nModel2]          # the vector of best model1 parameters
        vthik[vecMod] = xfit[2:2:nModel2]
        niter = res.iterations
        is_converged = res.f_converged
        xssr = res.residual_norm                         # sum(abs2, fcur)
    elseif NL_solve == :JuMP
        fgfgg
    end

    DnuTh[1] = sum(naik[vecMod]) - 1
    # # T̂ = ∑ₖ (n̂ₖv̂ₜₕₖ²) - 1.
    DnuTh[3] = sum(naik[vecMod] .* vthik[vecMod] .^ 2) - 1                                   # Thfit .- 1

    king_fMNK!(yfit, xfit, naik[vecMod];vhth=vthik[vecMod], nModel=nModel, Mhck1=Mhck1[1:nModel2,1])
end

function fl0king01_fMNK(x0::AbstractVector{T}, 
    nai::AbstractVector{T}, vthi::AbstractVector{T}, nModel::Int64,
    Mhck1::AbstractVector{T}, NL_solve::Symbol;
    lbs::AbstractVector{T}=[0.5], ubs::AbstractVector{T}=[2.0],
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT,
    NL_solve_method::Symbol=:newton) where {T}

    king01_fMNK!(out, x) = king_fMNK!(out, x, nai;vhth=vthi, nModel=nModel, Mhck1=Mhck1)
    if NL_solve == :LeastSquaresOptim
        if is_Jacobian
            J!(J, x) = king_fMNK_g!(J, x, nai; vhth=vthi, nModel=nModel)
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fMNK!, (g!)=J!, output_length=length(x0), autodiff=autodiff)
        else
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fMNK!, output_length=length(x0), autodiff=autodiff)
        end
        res = optimize!(nls, optimizer(factor), iterations=maxIterKing, show_trace=show_trace,
            x_tol=p_tol, f_tol=f_tol, g_tol=g_tol, lower=lbs, upper=ubs)
    elseif NL_solve == :NLsolve
        if is_Jacobian
            Js!(J, x) = king_fMNK_g!(J, x, nai; vhth=vthi, nModel=nModel)
            nls = OnceDifferentiable(king01_fMNK!, Js!, x0, similar(x0))
            if NL_solve_method == :trust_region
                res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            elseif NL_solve_method == :newton
                res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            end
        else
            nls = OnceDifferentiable(king01_fMNK!, x0, similar(x0))
            if NL_solve_method == :trust_region
                res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace, autodiff=:forward)
            elseif NL_solve_method == :newton
                res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace, autodiff=:forward)
            end
        end
    elseif NL_solve == :JuMP
        gyhhjjmj
    else
        esfgroifrtg
    end
    return res
end

# When `is_nai_const == true`
"""
  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.

  Outputs:
    king_fMNK!(out, x, nh;vhth=vhth, nMod=nMod, Mhck1=Mhck1)

"""

function king_fMNK!(out::AbstractVector{T}, x::AbstractVector{T}, nh::AbstractVector{T};
    vhth::AbstractVector{T}=[1.0,1.0], nModel::Int64=2, Mhck1::AbstractVector{T}=[1.0, 1.0]) where{T}

    nh[:] = x[1:2:end]
    vhth[:] = x[2:2:end]
    if nModel == 1
        nj = 1
        # (l,j) = (0,0)
        out[nj] = sum(nh) - Mhck1[nj]

        nj += 1
        # (l,j) = (0,2)
        out[nj] = sum(nh .* vhth .^ 2) - Mhck1[nj]
    else
        vhth2 = vhth .^ 2
        nj = 1
        # (l,j) = (0,0)
        out[nj] = sum(nh) - Mhck1[nj]

        nj += 1
        # (l,j) = (0,2)
        out[nj] = sum(nh .* vhth2) - Mhck1[nj]

        nj += 1
        # (l,j) = (0,4) = (0,2(nj))
        out[nj] = sum(nh .* vhth2 .^ 2) - Mhck1[nj]

        nj += 1
        # (l,j) = (0,6)
        out[nj] = sum(nh .* vhth2 .^ (nj - 1)) - Mhck1[nj]

        for k in 3:nModel
            nj += 1
            out[nj] = sum(nh .* vhth2 .^ nj) - Mhck1[nj]

            nj += 1
            out[nj] = sum(nh .* vhth2 .^ nj) - Mhck1[nj]
        end
    end
end

function king_fMNK_g!(J, x::AbstractVector{T}, nh::AbstractVector{T};
    vhth::AbstractVector{T}=[1.0,1.0], nModel::Int64=2) where{T}

    nh[:] = x[1:2:end]
    vhth[:] = x[2:2:end]
    fill!(J, 0.0)
    if nModel == 1
        nj = 1
        # (l,j) = (0,0)
        for s in 1:nModel
          J[nj, 2(s-1)+1] = 1.0
        #   J[nj, 2(s-1)+2] = 0.0
        end

        nj += 1
        # (l, j) = (0, 2)
        for s in 1:nModel
          s2 = 2(s - 1)
          J[nj, s2+1] = vhth[s]^2
          J[nj, s2+2] = 2nh[s] * vhth[s]
        end
    else
        nj = 1
        # j = 2(nj - 1)
        # (l,j) = (0,0)
        for s in 1:nModel
          J[nj, 2(s-1)+1] = 1.0
          #   J[nj, 2(s-1)+2] = 0.0
        end

        nj += 1
        # j = 2(nj - 1)
        # (l, j) = (0, 2)
        for s in 1:nModel
          s2 = 2(s - 1)
          J[nj, s2+1] = vhth[s]^2
          J[nj, s2+2] = 2nh[s] * vhth[s]
        end

        nj += 1
        j = 2(nj - 1)
        # (l, j) = (0, 4)   # nj = 3
        for s in 1:nModel
          s2 = 2(s - 1)
          J[nj, s2+1] = vhth[s]^4
          J[nj, s2+2] = 4nh[s] * vhth[s]^3
        end

        nj += 1
        j = 2(nj - 1)
        # (l, j) = (0, 6)   # nj = 4
        for s in 1:nModel
          s2 = 2(s - 1)
          J[nj, s2+1] = vhth[s]^j
          J[nj, s2+2] = j * nh[s] * vhth[s]^(j - 1)
        end

        for k in 3:nModel
            nj += 1
            j = 2(nj - 1)
            for s in 1:nModel
                s2 = 2(s - 1)
                J[nj, s2+1] = vhth[s] .^ j
                J[nj, s2+2] = j * nh[s] * vhth[s] .^ (j - 1)
            end
      
            nj += 1
            j = 2(nj - 1)
            for s in 1:nModel
                s2 = 2(s - 1)
                J[nj, s2+1] = vhth[s] .^ j
                J[nj, s2+2] = j * nh[s] * vhth[s] .^ (j - 1)
            end
        end
        # ~,R = qr(J)
        # dataJ = DataFrame(J,:auto)
        # dataQ = DataFrame(Q,:auto)
        # dataR = DataFrame(R,:auto)
    end
end
