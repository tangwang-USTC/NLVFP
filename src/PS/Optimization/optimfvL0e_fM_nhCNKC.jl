

"""

  When `is_nai_const = true`
       `∑(nai) = nh = Const `
       `∑(nai .* uai) = uh = Const `
       `∑(nai .* vthi^2) = Th = Const `

  Optimization of the amplitude function `fhl0` by using the `King` functions 
    to finding the optimized parameters `(n̂ₛ,ûₛ,v̂ₜₕₛ) = (nai,uai,vthi)`. 

  The general moments is renormalized as: 
    
    `M̂ⱼₗᵐ*`= M̂ⱼₗᵐ / CjLL2(j,L)`.
 
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
    naik,vthik,nModel = fl0king01_fMnhC!(naik1,vthik1,nModelk1,
                DnuTh,Mhck1,errMhcop,RDMhck1max,NL_solve,rtol_DnuTi,tk,dtk;
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi,
                is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt)
  
"""
# [ite,nMod],    # Suitable for the situation tends to be in equilibrium,`Mhj0c → 1`
function fl0king01_fMnhC!(naik1::AbstractVector{T}, vthik1::AbstractVector{T}, nModelk1::Int64,
    DnuTh::AbstractVector{T}, Mhck1::Matrix{T}, errMhcop::Matrix{T}, 
    RDMhck1max::T, NL_solve::Symbol, rtol_DnuTi::T,tk::T,dtk::T; 

    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT, 
    Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1],
    is_optim_CnIK::Bool=false,is_nhnMod_adapt::Bool=false) where {T}
    
    if nModelk1 == 1
        yfit = zeros(nModelk1)
        fl0king01_fMnh!(naik1,vthik1,nModelk1,DnuTh,yfit,Mhck1,NL_solve;
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
    
        yfitM = maximum(abs.(yfit))
        errMhcop[2,1] = deepcopy(yfit[1])
        errNorm = norm([DnuTh[1], DnuTh[3], yfitM])
        if errNorm ≤ atol_nuTi_optim
            printstyled("12nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfitM), color=:green,"\n")
        else
            if yfitM ≤ atol_nuTi_optim
                printstyled("12nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                printstyled("12nh, yfitM=", fmtf2.(yfitM), color=:green,"\n")
                if is_optim_CnIK && yfitM ≥ atol_nuTi_optim_err
                    error("12nh, Optimization is falure, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]))
                end
            else
                if norm([DnuTh[1], DnuTh[3]]) ≤ atol_nuTi_optim
                    printstyled("12nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                    printstyled("12nh, yfit=", fmtf2.(yfit), color=:red,"\n")
                else
                    printstyled("12nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3],yfitM), color=:red,"\n")
                    if is_optim_CnIK && yfitM ≥ atol_nuTi_optim_err
                        error("12nh, Optimization is falure, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]))
                    end
                end
            end
    
            @error("11KM: Underfitting: Optimization process for `naik,vthik` is not converged when",nModelk1)
        end
    else
        if is_optim_CnIK
            yfit = zeros(nModelk1-1)
            fl0king01_fMnhC!(naik1,vthik1,nModelk1,DnuTh,yfit,Mhck1,NL_solve;
                    optimizer=optimizer,factor=factor,autodiff=autodiff,
                    is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
        
            yfitM = maximum(abs.(yfit))
            errMhcop[3:nModelk1+1,1] = deepcopy(yfit)
        else
            yfit = zeros(nModelk1)
            fl0king01_fMnh!(naik1,vthik1,nModelk1,DnuTh,yfit,Mhck1,NL_solve;
                    optimizer=optimizer,factor=factor,autodiff=autodiff,
                    is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
        
            yfitM = maximum(abs.(yfit))
            errMhcop[3:nModelk1+2,1] = deepcopy(yfit)
        end
        is_converged_nMod = norm([DnuTh[1], DnuTh[3], yfitM]) ≤ atol_nuTi_optim

        # Reduce `nMod`
        if is_converged_nMod
            printstyled("221nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfitM), color=:green,"\n")
        else
            if yfitM ≤ atol_nuTi_optim
                printstyled("222nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                printstyled("22nh, yfitM=", fmtf2.(yfitM), color=:green,"\n")
            else
                if norm([DnuTh[1], DnuTh[3]]) ≤ atol_nuTi_optim
                    printstyled("223nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                    printstyled("22nh, yfit=", fmtf2.(yfit), color=:red,"\n")
                else
                    printstyled("224nh, (Dnh,DTh)=", (DnuTh[1], DnuTh[3],yfitM), color=:red,"\n")
                    if is_optim_CnIK && yfitM ≥ atol_nuTi_optim_err
                        error("22nh, Optimization is falure, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]))
                    end
                end
            end
    
            ## Checking whether overdetermined, `nModelk1!
            if is_nhnMod_adapt
                is_nMod_renew, naik1,vthik1,nModelk1 = nMod_update(naik1,vthik1,nModelk1;rtol_DnuTi=rtol_DnuTi)
            else
                is_nMod_renew = false
            end

            # Updating `naik1,vthik1` according to `Mhck1` with the new `nModelk1`
            if is_nMod_renew
                if nModelk1 == 1
                    yfit = zeros(nModelk1)
                    fl0king01_fMnh!(naik1,vthik1,nModelk1,DnuTh,yfit,Mhck1,NL_solve;
                            optimizer=optimizer,factor=factor,autodiff=autodiff,
                            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                else
                    if is_optim_CnIK
                        yfit = zeros(nModelk1-1)
                        fl0king01_fMnhC!(naik1,vthik1,nModelk1,DnuTh,yfit,Mhck1,NL_solve;
                                optimizer=optimizer,factor=factor,autodiff=autodiff,
                                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                    else
                        yfit = zeros(nModelk1)
                        fl0king01_fMnh!(naik1,vthik1,nModelk1,DnuTh,yfit,Mhck1,NL_solve;
                                optimizer=optimizer,factor=factor,autodiff=autodiff,
                                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                    end
                end
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
                            if is_optim_CnIK && yfitM ≥ atol_nuTi_optim_err
                                error("31nh, Optimization is falure, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]))
                            end
                        end
                    end
                    @error("31nh: Optimization process for `naik1,vthik1` is not converged with iteration for `nMod_update`.")
                end
            else
                @error("32nh: Optimization process for `naik1,vthik1` is not converged but `Dvthi ≤ rtol_DnuTi`.")
            end
        end
    end
    return naik1, vthik1, nModelk1
end

"""
  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
    fl0king01_fMnhC!(naik,vthik,nModel,DnuTh,yfit,Mhck1,NL_solve;
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
  
"""
# [nMod]
function fl0king01_fMnhC!(naik::AbstractVector{T}, vthik::AbstractVector{T}, nModel::Int64, 
    DnuTh::AbstractVector{T}, yfit::AbstractVector{T}, Mhck1::Matrix{T}, NL_solve::Symbol; 
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT, 
    Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1]) where {T}

    # The parameter limits for MCF plasma.
    # naifit = naik
    nModel1 = nModel - 1
    x0 = zeros(nModel1)
    lbs = zeros(nModel1)
    ubs = zeros(nModel1)

    vecMod = 1:nModel
    vecMod1 = 1:nModel1

    # vthi
    # lbs[vecMod1] .= vhthMin
    # ubs[vecMod1] .= vhthMax
    for i in vecMod1
        lbs[i] = max(vhthMin, vthik[i] / Nspan_optim_nuTi[3])
        ubs[i] = min(vhthMax, Nspan_optim_nuTi[3] * vthik[i])
        if is_NKC_vhthInitial
            x0[i] = max(vhthInitial, lbs[i])
        else
            x0[i] = max(vthik[i] * vhthRatio,lbs[i])
        end
    end

    vhth,vhth2 = zeros(T,nModel),zeros(T,nModel)
    res = fl0king01_fMnhC(deepcopy(x0), naik[vecMod], vthik[vecMod], 
        nModel1, Mhck1[1:nModel+1,1], NL_solve; 
        vhth2=vhth2, lbs=lbs, ubs=ubs, 
        optimizer=optimizer, factor=factor, autodiff=autodiff,
        is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)

    if NL_solve == :LeastSquaresOptim
        vthik[vecMod1] = res.minimizer         # the vector of best model1 parameters
        niter = res.iterations
        is_converged = res.converged
        xssr = res.ssr                         # sum(abs2, fcur)
    elseif NL_solve == :NLsolve
        vthik[vecMod1] = res.zero         # the vector of best model1 parameters
        niter = res.iterations
        is_converged = res.f_converged
        xssr = res.residual_norm                         # sum(abs2, fcur)
    elseif NL_solve == :JuMP
        fgfgg
    end

    if nModel == 2
        vthik[nModel] = king_fMnhC2!(yfit, vthik[vecMod1], naik[vecMod], Mhck1[1:nModel+1,1];vhth=vhth,vhth2=vhth2)
    else
        vthik[nModel] = king_fMnhC3!(yfit, vthik[vecMod1], naik[vecMod], nModel1, Mhck1[1:nModel+1,1];vhth=vhth,vhth2=vhth2)
    end

    DnuTh[1] = sum(naik[vecMod]) - 1
    # # T̂ = ∑ₖ (n̂ₖv̂ₜₕₖ²) - 1.
    DnuTh[3] = sum(naik[vecMod] .* vthik[vecMod] .^ 2) - 1                                   # Thfit .- 1
end

"""
  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
    res = fl0king01_fMnhC(x0, naik, nModel, Mhck1, NL_solve; 
                    vhth=vhth,vhth2=vhth2,lbs=lbs, ubs=ubs,
                    optimizer=optimizer, factor=factor, autodiff=autodiff,
                    is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)
  
"""

# [nMod ≥ 2], Optimization
function fl0king01_fMnhC(x0::AbstractVector{T}, nai::AbstractVector{T}, vhth::AbstractVector{T}, nModel1::Int64, 
    Mhck1::AbstractVector{T}, NL_solve::Symbol; vhth2::AbstractVector{T}=[0.1, 1.0], 
    lbs::AbstractVector{T}=[-uhMax, 0.8], ubs::AbstractVector{T}=[uhMax, 1.2],
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT,
    NL_solve_method::Symbol=:newton) where {T}

    king01_fMnhC!(out, x) = king_fMnhC3!(out, x, nai;vhth=vhth,vhth2=vhth2,nModel1=nModel1,Mhck1=Mhck1)
    if NL_solve == :LeastSquaresOptim
        if is_Jacobian
            J!(J, x) = king_fMnhC3_g!(J, x, nai;vhth=vhth,vhth2=vhth2,nModel1=nModel1,Mhck1=Mhck1)
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fMnhC!, (g!)=J!, output_length=length(x0), autodiff=autodiff)
        else
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fMnhC!, output_length=length(x0), autodiff=autodiff)
        end
        res = optimize!(nls, optimizer(factor), iterations=maxIterKing, show_trace=show_trace,
            x_tol=p_tol, f_tol=f_tol, g_tol=g_tol, lower=lbs, upper=ubs)
    elseif NL_solve == :NLsolve
        if is_Jacobian
            Js!(J, x) = king_fMnhC3_g!(J, x, nai;vhth=vhth,vhth2=vhth2,nModel1=nModel1,Mhck1=Mhck1)
            nls = OnceDifferentiable(king01_fMnhC!, Js!, x0, similar(x0))
            if NL_solve_method == :trust_region
                res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            elseif NL_solve_method == :newton
                res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            end
        else
            nls = OnceDifferentiable(king01_fMnhC!, x0, similar(x0))
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

"""
  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.

  Outputs:
    king_fMnh2C!(out, x, nh; Mhck1=Mhck1, nMod=nMod)

"""

# nMode = 2
function king_fMnhC2!(out::AbstractVector{T}, x::AbstractVector{T}, nh::AbstractVector{T}; 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], Mhck1::AbstractVector{T}=[0.1, 1.0]) where{T}
ssss
    vhth[1] = x[1]
    vhth2[1] = vhth[1] ^ 2

    vhth2[2] = (Mhck1[2] - (nh[1] .* vhth2[1])) / nh[2]
    vhth[2] = vhth2[2]^0.5

    nj = 2
    # (l,j) = (0,4) = (0,2(nj+1))
    out[nj-1] = sum(nh .* vhth2 .^ nj) - Mhck1[nj+1]
end
function king_fMnhC2!(out::AbstractVector{T}, x::AbstractVector{T}, nh::AbstractVector{T}, Mhck1::AbstractVector{T}; 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0]) where{T}

    vhth[1] = x[1]
    vhth2[1] = vhth[1] ^ 2

    vhth2[2] = (Mhck1[2] - (nh[1] .* vhth2[1])) / nh[2]
    vhth[2] = vhth2[2]^0.5

    nj = 2
    # (l,j) = (0,4) = (0,2(nj+1))
    out[nj-1] = sum(nh .* vhth2 .^ nj) - Mhck1[nj+1]
    return vhth[2]
end

# nMode ≥ 2
function king_fMnhC3!(out::AbstractVector{T}, x::AbstractVector{T}, nh::AbstractVector{T};
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], nModel1::Int64=1, 
    Mhck1::AbstractVector{T}=[0.1, 1.0]) where{T}

    vhth[1:nModel1] = x[1:1:end]
    vhth2[1:end-1] = vhth[1:end-1] .^ 2

    Kh9 = (Mhck1[2] - sum(nh[1:end-1] .* vhth2[1:end-1]))
    vhth2[end] = Kh9 / nh[end]
    vhth[end] = vhth2[end]^0.5

    # nj = 2
    # # (l,j) = (0,4) = (0,2(nj))
    # out[nj-1] = sum(nh .* vhth2 .^ nj) - Mhck1[nj+1]

    # nj = 3
    # # (l,j) = (0,6) = (0,2(nj))
    # out[nj-1] = sum(nh .* vhth2 .^ nj) - Mhck1[nj+1]
    for nj in 2:nModel1+1
        out[nj-1] = sum(nh .* vhth2 .^ nj) - Mhck1[nj+1]
    end
end
function king_fMnhC3!(out::AbstractVector{T}, x::AbstractVector{T}, nh::AbstractVector{T}, nModel1::Int64, 
    Mhck1::AbstractVector{T}; vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0]) where{T}

    vhth[1:nModel1] = x[1:1:end]
    vhth2[1:end-1] = vhth[1:end-1] .^ 2

    Kh9 = (Mhck1[2] - sum(nh[1:end-1] .* vhth2[1:end-1]))
    vhth2[end] = Kh9 / nh[end]
    vhth[end] = vhth2[end]^0.5

    # nj = 2
    # # (l,j) = (0,4) = (0,2(nj+1))
    # out[nj-1] = sum(nh .* vhth2 .^ nj) - Mhck1[nj+1]

    for nj in 2:nModel1+1
        out[nj-1] = sum(nh .* vhth2 .^ nj) - Mhck1[nj+1]
    end
    return vhth[end]
end

# The Jacobian matrix: J = zeros(T,nMod-1,nMod-1)
# nMode = 2
function king_fMnhC2_g!(J::AbstractArray{T,N}, x::AbstractVector{T}, nh::AbstractVector{T}; 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], Mhck1::AbstractVector{T}=[0.1, 1.0]) where{T,N}

    fill!(J, 0.0)
    vhth[1] = x[1]
    vhth2[1] = vhth[1] ^ 2

    vhth2[2] = (Mhck1[2] - nh[1] .* vhth2[1]) / nh[2]
    vhth[2] = vhth2[2]^0.5

    nj = 1
    # (l, j) = (0, 4)
    s = 1
    J[nj, s] = 4nh[s] * vhth[s] * (vhth2[s] - vhth2[2])
end

function king_fMnhC2_g!(J::AbstractArray{T,N}, x::AbstractVector{T}, nh::AbstractVector{T}, Mhck1::AbstractVector{T}; 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0]) where{T,N}

    fill!(J, 0.0)
    vhth[1] = x[1]
    vhth2[1] = vhth[1] ^ 2

    vhth2[2] = (Mhck1[2] - nh[1] .* vhth2[1]) / nh[2]
    vhth[2] = vhth2[2]^0.5

    nj = 1
    # (l, j) = (0, 4)
    s = 1
    J[nj, s] = 4nh[s] * vhth[s] * (vhth2[s] - vhth2[2])
    return vhth[2]
end

# nMod ≥ 2
function king_fMnhC3_g!(J::AbstractArray{T,N}, x::AbstractVector{T}, nh::AbstractVector{T}; 
    vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], nModel1::Int64=1, Mhck1::AbstractVector{T}=[0.1, 1.0]) where{T,N}

    fill!(J, 0.0)
    vhth[1:nModel1] = x[1:1:end]
    vhth2[1:end-1] = vhth[1:end-1] .^ 2

    vhth2[end] = (Mhck1[2] - sum(nh[1:end-1] .* vhth2[1:end-1])) / nh[end]
    vhth[end] = vhth2[end]^0.5
    
    # nj = 1
    # # (l, j) = (0, 4)   # j = 2(nj+1)
    # # j2 = (nj + 1)
    # for s in 1:nModel1
    #     J[nj, s] = 4nh[s] * vhth[s] * (vhth2[s] - vhth2[end])
    # end

    # (l, j) = (0, 2(nj+1))
    for nj in 1:nModel1
        for s in 1:nModel1
            # j2 = (nj + 1)
            J[nj, s] = 2(nj + 1) * nh[s] * vhth[s] * (vhth2[s]^nj - vhth2[end]^nj)
        end
    end
end

function king_fMnhC3_g!(J::AbstractArray{T,N}, x::AbstractVector{T}, nh::AbstractVector{T}, nModel1::Int64, 
    Mhck1::AbstractVector{T}; vhth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0]) where{T,N}

    fill!(J, 0.0)
    vhth[1:nModel1] = x[1:1:end]
    vhth2[1:end-1] = vhth[1:end-1] .^ 2

    vhth2[end] = (Mhck1[2] - sum(nh[1:end-1] .* vhth2[1:end-1])) / nh[end]
    vhth[end] = vhth2[end]^0.5
    
    # nj = 1
    # # (l, j) = (0, 4)   # j = 2(nj+1)
    # for s in 1:nModel1
    #     J[nj, s] = 4nh[s] * vhth[s] * (vhth2[s] - vhth2[end])
    # end

    # nj = 2
    # # (l, j) = (0, 6)   # j = 2(nj+1)
    # J[nj, s+1] = 6nh[s] * vhth[nj] * (vhth2[s]^2 - vhth2[end]^2)
    
    # (l, j) = (0, 2(nj+1))
    for nj in 1:nModel1
        for s in 1:nModel1
            # j2 = (nj + 1)
            J[nj, s] = 2(nj + 1) * nh[s] * vhth[s] * (vhth2[s]^nj - vhth2[end]^nj)
        end
    end
    return vhth[end]
end