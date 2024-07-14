

"""

  When `is_nai_const = true`
       `∑(nai) = nh `
       `∑(nai .* uai) = uh `
       `∑(nai .* vthi^2) = Th `

  Optimization of the amplitude function `fhl0` by using the `King` functions 
    to finding the optimized parameters `(n̂ₛ,ûₛ,v̂ₜₕₛ) = (nai,uai,vthi)`. 

  The general moments is renormalized as: 
    
    `M̂ⱼₗᵐ*`= M̂ⱼₗᵐ / CjLL2(j,L)`.
 
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
    naik,vthik,nModel = fl0king01_fMnh!(naik1,vthik1,nModelk1,nModelk,
                DnuTh,Mhck1,NL_solve,rtol_DnuTi;
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
  
"""
# [ite,nMod]
function fl0king01_fMnh!(naik1::AbstractVector{T}, vthik1::AbstractVector{T}, nModelk1::Int64, nModelk::Int64,
    DnuTh::AbstractVector{T}, Mhck1::Matrix{T}, NL_solve::Symbol, rtol_DnuTi::T; 

    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT, 
    Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1],
    is_optim_CnIK::Bool=false) where {T}
    
    nModelk1 = deepcopy(nModelk)
    yfit = zeros(nModelk1)
    fl0king01_fMnh!(naik1,vthik1,nModelk1,DnuTh,yfit,Mhck1,NL_solve;
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)

    yfitM = maximum(abs.(yfit))
    is_converged_nMod = norm([DnuTh[1], DnuTh[3]]) ≤ atol_nuTi_optim
    if is_converged_nMod
        printstyled("(Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
        if yfitM ≤ atol_nuTi_optim
            printstyled("yfitM=", fmtf2.(yfitM), color=:green,"\n")
        else
            printstyled("yfit=", fmtf2.(yfit), color=:blue,"\n")
        end
    else
        printstyled("(Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
        if yfitM ≤ atol_nuTi_optim
            printstyled("yfitM=", fmtf2.(yfitM), color=:green,"\n")
        else
            printstyled("yfit=", fmtf2.(yfit), color=:blue,"\n")
        end

        ## Checking whether overdetermined, `nModelk1!
        if nModelk1 == 1
            @warn("22: Optimization process for `naik1,vthik1` is not converged when `nModelk1 = 1`.")
        else
            if is_nhnMod_adapt
                is_nMod_renew, naik1,vthik1,nModelk1 = nMod_update(naik1,vthik1,nModelk1;rtol_DnuTi=rtol_DnuTi)
            else
                is_nMod_renew = false
            end

            # Updating `naik1,vthik1` according to `Mhck1` with the new `nModelk1`
            if is_nMod_renew
                yfit = zeros(nModelk1)
                fl0king01_fMnh!(naik1,vthik1,nModelk1,DnuTh,yfit,Mhck1,NL_solve;
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                    
                is_converged_nMod = norm([DnuTh[1], DnuTh[3]]) ≤ atol_nuTi_optim
                if is_converged_nMod
                    printstyled("(Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                else
                    printstyled("(Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                    @error("21: Optimization process for `naik1,vthik1` is not converged with iteration for `nMod_update`.")
                end
            else
                @error("20: Optimization process for `naik1,vthik1` is not converged but `Dvthi ≤ rtol_DnuTi`.")
            end
        end
    end

    return naik1, vthik1, nModelk1
end

"""
  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
    fl0king01_fMnh!(naik,vthik,nModel,DnuTh,yfit,Mhck1,NL_solve;
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
  
"""
# [nMod], yfit = zeros(nModel)
function fl0king01_fMnh!(naik::AbstractVector{T}, vthik::AbstractVector{T}, nModel::Int64, 
    DnuTh::AbstractVector{T}, yfit::AbstractVector{T}, Mhck1::Matrix{T}, NL_solve::Symbol; 
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT, 
    Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1]) where {T}

    vecMod = 1:nModel
    # @show nModel, vthik
    # @show Mhck1[1+nModel,1] .- 1

    # The parameter limits for MCF plasma.
    # naifit = naik
    x0 = zeros(nModel)
    lbs = zeros(nModel)
    ubs = zeros(nModel)
    
    # vthi
    # if nModel == 1
    #     lbs[1] = 1.0
    #     ubs[1] = 1.0
    #     x0[1] = 1.0
    # else
        for i in vecMod
            ubs[i] = min(vhthMax, Nspan_optim_nuTi[3] * vthik[i])
            lbs[i] = max(vhthMin, vthik[i] / Nspan_optim_nuTi[3])
            x0[i] = vthik[i]
        end
    # end

    res = fl0king01_fMnh(deepcopy(x0), naik[vecMod], nModel, Mhck1[2:nModel+1,1], NL_solve; 
        lbs=lbs, ubs=ubs, optimizer=optimizer, factor=factor, autodiff=autodiff,
        is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)

    if NL_solve == :LeastSquaresOptim
        vthik[vecMod] = res.minimizer         # the vector of best model1 parameters
        niter = res.iterations
        is_converged = res.converged
        xssr = res.ssr                         # sum(abs2, fcur)
    elseif NL_solve == :NLsolve
        vthik[vecMod] = res.zero         # the vector of best model1 parameters
        niter = res.iterations
        is_converged = res.f_converged
        xssr = res.residual_norm                         # sum(abs2, fcur)
    elseif NL_solve == :JuMP
        fgfgg
    end

    DnuTh[1] = sum(naik[vecMod]) - 1

    # # T̂ = ∑ₖ (n̂ₖv̂ₜₕₖ²) - 1.
    DnuTh[3] = sum(naik[vecMod] .* vthik[vecMod] .^ 2) - 1                                   # Thfit .- 1

    king_fMnh!(yfit, vthik[vecMod], naik[vecMod]; Mhck1=Mhck1[2:nModel+1,1], nModel=nModel)
end

"""
  Inputs:
    Mhck1: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
    res = fl0king01_fMnh(x0, naik, nModel, Mhck1, NL_solve; 
                    vhth=vhth,vhth2=vhth2,lbs=lbs, ubs=ubs,
                    optimizer=optimizer, factor=factor, autodiff=autodiff,
                    is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)
  
"""

# [nMod ≥ 1], Optimization
function fl0king01_fMnh(x0::AbstractVector{T}, nai::AbstractVector{T}, nModel::Int64, 
    Mhck1::AbstractVector{T}, NL_solve::Symbol;
    lbs::AbstractVector{T}=[0.5], ubs::AbstractVector{T}=[2.0],
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT,
    NL_solve_method::Symbol=:newton) where {T}

    king01_fMnh!(out, x) = king_fMnh!(out, x, nai; Mhck1=Mhck1, nModel=nModel)
    if NL_solve == :LeastSquaresOptim
        if is_Jacobian
            J!(J, x) = king_fMnh_g!(J, x, nai; nModel=nModel)
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fMnh!, (g!)=J!, output_length=length(x0), autodiff=autodiff)
        else
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fMnh!, output_length=length(x0), autodiff=autodiff)
        end
        res = optimize!(nls, optimizer(factor), iterations=maxIterKing, show_trace=show_trace,
            x_tol=p_tol, f_tol=f_tol, g_tol=g_tol, lower=lbs, upper=ubs)
    elseif NL_solve == :NLsolve
        if is_Jacobian
            Js!(J, x) = king_fMnh_g!(J, x, nai; nModel=nModel)
            nls = OnceDifferentiable(king01_fMnh!, Js!, x0, similar(x0))
            if NL_solve_method == :trust_region
                res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            elseif NL_solve_method == :newton
                res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            end
        else
            nls = OnceDifferentiable(king01_fMnh!, x0, similar(x0))
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
    king_fMnh!(out, x; Mhck1=Mhck1, nMod=nMod)

"""
# nMode ≥ 1
function king_fMnh!(out::AbstractVector{T}, vhth::AbstractVector{T}, nh::AbstractVector{T}; 
    Mhck1::AbstractVector{T}=[1.0, 1.0], nModel::Int64=1) where{T}

    if nModel == 1
        nj = 1
        out[nj] = sum(nh .* vhth .^ 2) - Mhck1[nj]
    else
        vhth2 = vhth .^ 2
        nj = 1
        # (l,j) = (0,2)
        out[nj] = sum(nh .* vhth2) - Mhck1[nj]

        nj += 1
        # (l,j) = (0,4) = (0,2(nj))
        out[nj] = sum(nh .* vhth2 .^ 2) - Mhck1[nj]
        
        # nj += 1
        # # (l,j) = (0,6) = (0,2(nj))
        # out[nj] = sum(nh .* vhth2 .^ nj) - Mhck1[nj]
        for nj in 3:nModel
            out[nj] = sum(nh .* vhth2 .^ nj) - Mhck1[nj]
        end
    end
end

function king_fMnh_g!(J, vhth::AbstractVector{T}, nh::AbstractVector{T}; nModel::Int64=1) where{T}

    fill!(J, 0.0)
    if nModel == 1
        J[1, 1] = 2nh[1] * vhth[1]
    else
        nj = 1
        # (l, j) = (0, 2)
        for s in 1:nModel
            J[nj, s] = 2nh[s] * vhth[s]
        end

        nj += 1                      # nj = 2        2nMod
        # (l,j) = (0,4) = (0,2nj)
        for s in 1:nModel
            J[nj, s] = 4nh[s] * vhth[s]^3
        end

        # j = 2nj           # 4 
        # for s in 1:nModel
        #     J[nj, s] = 4 * nh[s] * vhth[s] .^ (4 - 1)
        # end
        # j = 2nj           # 6
        # for s in 1:nModel
        #     J[nj, s] = j * nh[s] * vhth[s] .^ (6 - 1)
        # end
        for nj in 3:nModel
            j = 2nj
            for s in 1:nModel
                J[nj, s] = j * nh[s] * vhth[s] .^ (j - 1)
            end
        end
        # ~,R = qr(J)
        # dataJ = DataFrame(J,:auto)
        # dataQ = DataFrame(Q,:auto)
        # dataR = DataFrame(R,:auto)
    end
end
