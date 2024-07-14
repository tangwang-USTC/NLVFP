

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
    naik,vthik,nModel = fl0king01_fMNKf!(naik1,vthik1,nModelk1,
                naik,vthik,nModelk,DnuTh,Mhck1,njMs,NL_solve,rtol_DnuTi;
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
  
"""

# [ite,nMod]
function fl0king01_fMNKf_to_C!(naik1::AbstractVector{T}, vthik1::AbstractVector{T}, nModelk1::Int64,
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
        # @show NK, nModelk, size(naik1), size(Mhck1)
        nModelk1 = 1NK
        yfitNK = zeros(T,2nModelk1)
        fl0king01_fMNKf!(naik1,vthik1,nModelk1,DnuTh,yfitNK,Mhck1,NL_solve;
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
    
            @error("11KMNK: Underfitting: Optimization process for `naik,vthik` is not converged when",nModelk1)
            @show fmtf2.(vthik1)
        end
    else
        # @show NK, nModelk, size(naik1), size(Mhck1)
        nModelk1 = 1NK
        yfitNK = zeros(T,2nModelk1)
        fl0king01_fMNKf!(naik1,vthik1,nModelk1,DnuTh,yfitNK,Mhck1,NL_solve;
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
            printstyled("21NK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
            if yfitNKM ≤ atol_nuTi_optim
                printstyled("21NK, yfitNKM=", fmtf2.(yfitNKM), color=:green,"\n")
            else
                printstyled("21NK, yfitNK=", fmtf2.(yfitNK), color=:blue,"\n")
            end
        else
            printstyled("22NK, (Dnh,DTh)=", (DnuTh[1], DnuTh[3], yfitNKM), color=:red,"\n")
            if yfitNKM ≤ atol_nuTi_optim
                printstyled("22NK, yfitNKM=", fmtf2.(yfitNKM), color=:green,"\n")
            else
                printstyled("22NK, yfitNK=", fmtf2.(yfitNK), color=:blue,"\n")
            end
    
            @error("21NK: Underfitting: Optimization process for `naik,vthik` is not converged when",nModelk1)
            @show fmtf2.(vthik1)
        end
    end
    return naik1, vthik1, nModelk1
end

"""
  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
    fl0king01_fMNKf!(naik,vthik,nModel,DnuTh,yfit,Mhck1,NL_solve;
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
  
"""

# [nMod]
function fl0king01_fMNKf!(naik::AbstractVector{T}, vthik::AbstractVector{T}, nModel::Int64,
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
    
    res = fl0king01_fMNKf(deepcopy(x0), naik[vecMod], vthik[vecMod], nModel, Mhck1[1:nModel2,1], NL_solve; 
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

    king_fMNKf!(yfit, xfit, naik[vecMod];vhth=vthik[vecMod], nModel=nModel, Mhck1=Mhck1[1:nModel2,1])
end

function fl0king01_fMNKf(x0::AbstractVector{T}, 
    nai::AbstractVector{T}, vthi::AbstractVector{T}, nModel::Int64,
    Mhck1::AbstractVector{T}, NL_solve::Symbol;
    lbs::AbstractVector{T}=[0.5], ubs::AbstractVector{T}=[2.0],
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT,
    NL_solve_method::Symbol=:newton) where {T}

    king01_fMNKf!(out, x) = king_fMNKf!(out, x, nai;vhth=vthi, nModel=nModel, Mhck1=Mhck1)
    if NL_solve == :LeastSquaresOptim
        if is_Jacobian
            J!(J, x) = king_fMNKf_g!(J, x, nai; vhth=vthi, nModel=nModel)
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fMNKf!, (g!)=J!, output_length=length(x0), autodiff=autodiff)
        else
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fMNKf!, output_length=length(x0), autodiff=autodiff)
        end
        res = optimize!(nls, optimizer(factor), iterations=maxIterKing, show_trace=show_trace,
            x_tol=p_tol, f_tol=f_tol, g_tol=g_tol, lower=lbs, upper=ubs)
    elseif NL_solve == :NLsolve
        if is_Jacobian
            Js!(J, x) = king_fMNKf_g!(J, x, nai; vhth=vthi, nModel=nModel)
            nls = OnceDifferentiable(king01_fMNKf!, Js!, x0, similar(x0))
            if NL_solve_method == :trust_region
                res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            elseif NL_solve_method == :newton
                res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            end
        else
            nls = OnceDifferentiable(king01_fMNKf!, x0, similar(x0))
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
    king_fMNKf!(out, x, nh;vhth=vhth, nMod=nMod, Mhck1=Mhck1)

"""

function king_fMNKf!(out::AbstractVector{T}, x::AbstractVector{T}, nh::AbstractVector{T};
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

function king_fMNKf_g!(J, x::AbstractVector{T}, nh::AbstractVector{T};
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
