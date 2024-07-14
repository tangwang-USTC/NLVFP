

"""
  # When `is_nai_const == true`
  #      `∑(nai .* uai) = uha`
  #      `∑(nai .* vthi * (1 + 2/3 * (uai / vthi)^2)) = Mh*(1,1) = Th + 2/3 * uh^2`

  Optimization of the amplitude function `fhl0` by using the `King` functions 
    to finding the optimized parameters `(n̂ₛ,ûₛ,v̂ₜₕₛ) = (nai,uai,vthi)`. 

  The general moments is renormalized as: 
    
    `M̂ⱼₗᵐ*`= M̂ⱼₗᵐ / CjLL2(j,L)`.
 
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

  Inputs:
    Mhcsl0: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
    nModel,naik,uaik,vthik = fl0king01_fDM!(naik,uaik,vthik,DnuTh,Mhcsl0,nModel,NL_solve,rtol_DnuTi;
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
  
"""

# [ite,nMod]
function fl0king01_fDM!(naik::AbstractVector{T}, uaik::AbstractVector{T}, vthik::AbstractVector{T}, DnuTh::AbstractVector{T},
    Mhcsl0::AbstractVector{T}, nModel::Int64, NL_solve::Symbol, rtol_DnuTi::T; 
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT, 
    Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1]) where {T}
    
    yfit = fl0king01_fDM!(naik,uaik,vthik,DnuTh,Mhcsl0,nModel,NL_solve;
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
        @show 2, uaik

        ## Checking whether overdetermined, `nModel!
        if nModel == 1
            @warn("32: Optimization process for `naik,uaik,vthik` is not converged when `nModel = 1`.")
        else
            is_nMod_renew, naik, uaik, vthik, nModel = nMod_update(naik, uaik, nModel, vthik;rtol_DnuTi=rtol_DnuTi)
            
            # Updating `naik,uaik,vthik` according to `Mhcsl0` with the new `nModel`
            if is_nMod_renew
                fl0king01_fDM!(naik,uaik,vthik,DnuTh,Mhcsl0,nModel,NL_solve;
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
                    
                is_converged_nMod = norm([DnuTh[1], DnuTh[3]]) ≤ atol_nuTi_optim
                if is_converged_nMod
                    printstyled("(Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:green,"\n")
                else
                    printstyled("(Dnh,DTh)=", (DnuTh[1], DnuTh[3]), color=:red,"\n")
                    @warn("31: Optimization process for `naik,uaik,vthik` is not converged with iteration for `nMod_update`.")
                end
                degfgbn
            else
                @warn("30: Optimization process for `naik,uaik,vthik` is not converged but `Dvthi ≤ rtol_DnuTi`.")
            end
        end
        edfgv
    end
    
    return nModel,naik,uaik,vthik
end

"""
  Inputs:
    Mhcsl0: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
    is_converged_nMod = fl0king01_fDM!(naik,uaik,vthik,DnuTh,Mhcsl0,nModel,NL_solve;
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi)
  
"""

# [nMod]
function fl0king01_fDM!(naik::AbstractVector{T}, uaik::AbstractVector{T}, vthik::AbstractVector{T}, DnuTh::AbstractVector{T}, 
    Mhcsl01::AbstractVector{T}, nModel::Int64, NL_solve::Symbol;
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT, 
    Nspan_optim_nuTi::AbstractVector{T}=[1.1,1.1,1.1]) where {T}

    @show nModel
    # vthik
    if nModel == 1
        naik[1], uaik[1], vthik[1] = fl0king01_fDM(Mhcsl01)
        DnuTh .= 0.0
        yfit = zeros(nModel)
    else
        # The parameter limits for MCF plasma.
        # naifit = naik
        nModel1 = nModel - 1
        x0 = zeros(2nModel1)      # [uai1, vthi1, uai2, vthi2, ⋯]
        lbs = zeros(2nModel1)
        ubs = zeros(2nModel1)

        # uaik
        vec = 1:nModel1
        for i in vec
            i2 = 2i - 1
            x0[i2] = deepcopy(uaik[i])
            if abs(uaik[i]) < uhMin
                ubs[i2] = 10 * uhMin
                lbs[i2] = -10 * uhMin
            else
                usign = sign(uaik[i])
                if usign == 1
                    ubs[i2] = min(uhMax, Nspan_optim_nuTi[2] * uaik[i])
                    lbs[i2] = uaik[i] / Nspan_optim_nuTi[2]
                else
                    ubs[i2] = uaik[i] / Nspan_optim_nuTi[2]
                    lbs[i2] = max(- uhMax, Nspan_optim_nuTi[2] * uaik[i])
                end
            end
        end

        for i in vec
            i2 = 2i
            ubs[i2] = min(vhthMax, Nspan_optim_nuTi[3] * vthik[i])
            lbs[i2] = max(vhthMin, vthik[i] / Nspan_optim_nuTi[3])
            x0[i2] = vthik[i]
        end

        uh,vhth,uvth,vhth2,uvth2 = zeros(T,nModel),zeros(T,nModel),zeros(T,nModel),zeros(T,nModel),zeros(T,nModel)
        if nModel == 2
            @show nModel, length(x0)
            res = fl0king01_fDM(deepcopy(x0), naik, Mhcsl01, NL_solve; 
                uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2,lbs=lbs, ubs=ubs,
                optimizer=optimizer, factor=factor, autodiff=autodiff,
                is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)
        else
            res = fl0king01_fDM(deepcopy(x0), naik, Mhcsl01, NL_solve, nModel1; 
                uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2,lbs=lbs, ubs=ubs,
                optimizer=optimizer, factor=factor, autodiff=autodiff,
                is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)
        end
    
        if NL_solve == :LeastSquaresOptim
            xfit = res.minimizer         # the vector of best model1 parameters
            niter = res.iterations
            is_converged = res.converged
            xssr = res.ssr                         # sum(abs2, fcur)
        elseif NL_solve == :NLsolve
            xfit = res.zero         # the vector of best model1 parameters
            niter = res.iterations
            is_converged = res.f_converged
            xssr = res.residual_norm                         # sum(abs2, fcur)
        elseif NL_solve == :JuMP
            fgfgg
        end
    
        uaik[vec] = xfit[1:2:end]
        vthik[vec] = xfit[2:2:end]

        yfit = zeros(2nModel1)
        if nModel == 2
            uaik[end], vthik[end] = king_fDM2!(yfit, xfit, naik, Mhcsl01;uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2)
        else
            uaik[end], vthik[end] = king_fDM2!(yfit, xfit, naik, Mhcsl01, nModel1;uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2)
        end
    

        uaik[naik .≤ atol_n] .= 0.0
        uhafit = sum(naik .* uaik)
    
        DnuTh[1] = sum(naik) - 1
    
        # # T̂ = ∑ₖ n̂ₖ (v̂ₜₕₖ² + 2/3 * ûₖ²) - 2/3 * û², where `û = ∑ₖ(n̂ₖûₖ) / ∑ₖ(n̂ₖ)`
        # Thfit = sum(naik .* (vthik .^ 2 + 2 / 3 * uaik .^ 2)) - 2 / 3 * uhafit .^ 2
        DnuTh[3] = sum(naik .* (vthik .^ 2 + 2 / 3 * uaik .^ 2)) - 2 / 3 * uhafit .^ 2 -1      # Thfit .- 1
    end
    return yfit
end

"""
  Inputs:
    Mhcsl0: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.
  
  Outputs:
  res = fl0king01_fDM(x0, naik, Mhcsl01, NL_solve; 
                    uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2,lbs=lbs, ubs=ubs,
                    optimizer=optimizer, factor=factor, autodiff=autodiff,
                    is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)
  res = fl0king01_fDM(x0, naik, Mhcsl01, NL_solve, nModel1; 
                    uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2,lbs=lbs, ubs=ubs,
                    optimizer=optimizer, factor=factor, autodiff=autodiff,
                    is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, NL_solve_method=NL_solve_method)
  
"""

# nMode = 1
function fl0king01_fDM(Mhcsl01::AbstractVector{T}) where{T}

    # nj = 1
    # (l,j) = (1,1)
    uh = Mhcsl01[1]
    
    # nj += 1
    # (l,j) = (0,2)
    # 1 .+ 2 / 3 * uh .^ 2 == Mhcsl01[nj]
    Mhcsl01[2] = 1 .+ 2 / 3 * uh .^ 2
    return 1.0, uh, 1.0
    # return nh, uh, vhth
end

# nMod = 2
function fl0king01_fDM(x0::AbstractVector{T}, nai::AbstractVector{T}, 
    Mhcsl01::AbstractVector{T}, NL_solve::Symbol;
    uh::AbstractVector{T}=[0.1, 1.0], vhth::AbstractVector{T}=[0.1, 1.0], 
    uvth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], uvth2::AbstractVector{T}=[0.1, 1.0],
    lbs::AbstractVector{T}=[-uhMax, 0.8], ubs::AbstractVector{T}=[uhMax, 1.2],
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT,
    NL_solve_method::Symbol=:newton) where {T}

    king01_fDM!(out, x) = king_fDM2!(out, x, nai;uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2,Mhcsl01=Mhcsl01)
    if NL_solve == :LeastSquaresOptim
        @show length(nai), length(uh)
        if is_Jacobian
            J!(J, x) = king_fDM2_g!(J, x, nai;uh=uh,vhth=vhth,vhth2=vhth2,Mhcsl01=Mhcsl01)
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fDM!, (g!)=J!, output_length=length(x0), autodiff=autodiff)
        else
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fDM!, output_length=length(x0), autodiff=autodiff)
        end
        res = optimize!(nls, optimizer(factor), iterations=maxIterKing, show_trace=show_trace,
            x_tol=p_tol, f_tol=f_tol, g_tol=g_tol, lower=lbs, upper=ubs)
    elseif NL_solve == :NLsolve
        if is_Jacobian
            Js!(J, x) = king_fDM2_g!(J, x, nai;uh=uh,vhth=vhth,vhth2=vhth2,Mhcsl01=Mhcsl01)
            nls = OnceDifferentiable(king01_fDM!, Js!, x0, similar(x0))
            if NL_solve_method == :trust_region
                res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            elseif NL_solve_method == :newton
                res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            end
        else
            nls = OnceDifferentiable(king01_fDM!, x0, similar(x0))
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

# [nMod ≥ 3]
function fl0king01_fDM(x0::AbstractVector{T}, nai::AbstractVector{T}, 
    Mhcsl01::AbstractVector{T}, NL_solve::Symbol, nModel1::Int64;
    uh::AbstractVector{T}=[0.1, 1.0], vhth::AbstractVector{T}=[0.1, 1.0], 
    uvth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], 
    lbs::AbstractVector{T}=[-uhMax, 0.8], ubs::AbstractVector{T}=[uhMax, 1.2],
    optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
    is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=epsT, f_tol::Float64=epsT, g_tol::Float64=epsT,
    NL_solve_method::Symbol=:newton) where {T}

    king01_fDM!(out, x) = king_fDM2!(out, x, nai, nModel1;uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2,Mhcsl01=Mhcsl01)
    if NL_solve == :LeastSquaresOptim
        if is_Jacobian
            J!(J, x) = king_fDM2_g!(J, x, nai, nModel1;uh=uh,vhth=vhth,vhth2=vhth2,Mhcsl01=Mhcsl01)
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fDM!, (g!)=J!, output_length=length(x0), autodiff=autodiff)
        else
            nls = LeastSquaresProblem(x=x0, (f!)=king01_fDM!, output_length=length(x0), autodiff=autodiff)
        end
        res = optimize!(nls, optimizer(factor), iterations=maxIterKing, show_trace=show_trace,
            x_tol=p_tol, f_tol=f_tol, g_tol=g_tol, lower=lbs, upper=ubs)
    elseif NL_solve == :NLsolve
        if is_Jacobian
            Js!(J, x) = king_fDM2_g!(J, x, nai, nModel1;uh=uh,vhth=vhth,vhth2=vhth2,Mhcsl01=Mhcsl01)
            nls = OnceDifferentiable(king01_fDM!, Js!, x0, similar(x0))
            if NL_solve_method == :trust_region
                res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            elseif NL_solve_method == :newton
                res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
                    iterations=maxIterKing, show_trace=show_trace)
            end
        else
            nls = OnceDifferentiable(king01_fDM!, x0, similar(x0))
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
    out: = zeros(nModel1,nModel1)
    x: = x(nModel1)
    nh: = nai
    uh: = uai
    vhth: = vthi
    nModel1 = nModel - 1
    x: = x(nModel1)
    Mhcsl0: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.

  Outputs:
    king_fDM2!(out, x, nh;uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2,Mhcsl01=Mhcsl01)
    uai[end], vthi[end] = king_fDM2!(out, x, nh, Mhcsl01;uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2)
    king_fDM2!(out, x, nh, nModel1;uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2,Mhcsl01=Mhcsl01)
    uai[end], vthi[end] = king_fDM2!(out, x, nh, Mhcsl01, nModel1;uh=uh,vhth=vhth,uvth=uvth,vhth2=vhth2,uvth2=uvth2)

"""

# When `is_nai_const == true`
#      `∑(nai .* uai) = Mh(2,0) = uha`
#      `∑(nai .* vthi^2 * (1 + 2/3 * (uai / vthi)^2)) = Mh(1,1) = Th + 2/3 * uh^2`
# nMode = 1

# nMode = 2
function king_fDM2!(out::AbstractVector{T}, x::AbstractVector{T}, nh::AbstractVector{T}; 
    uh::AbstractVector{T}=[0.1, 1.0], vhth::AbstractVector{T}=[0.1, 1.0], 
    uvth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhcsl01::AbstractVector{T}=[0.1, 1.0]) where{T}

    uh[1] = x[1]
    vhth[1] = x[2]
    vhth2[1] = vhth[1] ^ 2
    uvth2[1] = uh[1] ^ 2 / vhth2[1]

    uh[2] = (Mhcsl01[1] - nh[1] .* uh[1]) / nh[2]
    vhth2[2] = (Mhcsl01[2] - (nh[1] .* vhth2[1]) .* (1 .+ 2 / 3 * uvth2[1])) / nh[2] - 2 / 3 * uh[2] ^ 2
    vhth[2] = vhth2[2]^0.5
    uvth[2] = uh[2] ./ vhth[2]
    uvth2[2] = uvth[2] .^ 2   # uh .^ 2 ./ vhth .^ 2

    nj = 1
    # (l,j) = (1,3)
    out[nj] = sum((nh .* uh .* vhth2) .* (1 .+ 2 / 5 * uvth2)) - Mhcsl01[nj+2]

    nj += 1
    # # (l,j) = (0,4)
    out[nj] = sum((nh .* vhth2 .^ 2) .* (1 .+ 4 / 3 * uvth2 .+ 4 / 15 * uvth2 .^ 2)) - Mhcsl01[nj+2]
end

function king_fDM2!(out::AbstractVector{T}, x::AbstractVector{T}, nh::AbstractVector{T}, Mhcsl01; 
    uh::AbstractVector{T}=[0.1, 1.0], vhth::AbstractVector{T}=[0.1, 1.0], 
    uvth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0]) where{T}

    uh[1] = x[1]
    vhth[1] = x[2]
    vhth2[1] = vhth[1] ^ 2
    uvth2[1] = uh[1] ^ 2 / vhth2[1]

    uh[2] = (Mhcsl01[1] - nh[1] .* uh[1]) / nh[2]
    vhth2[2] = (Mhcsl01[2] - (nh[1] .* vhth2[1]) .* (1 .+ 2 / 3 * uvth2[1])) / nh[2] - 2 / 3 * uh[2] ^ 2
    vhth[2] = vhth2[2]^0.5
    uvth[2] = uh[2] ./ vhth[2]
    uvth2[2] = uvth[2] .^ 2   # uh .^ 2 ./ vhth .^ 2

    nj = 1
    # (l,j) = (1,3)
    out[nj] = sum((nh .* uh .* vhth2) .* (1 .+ 2 / 5 * uvth2)) - Mhcsl01[nj+2]

    nj += 1
    # # (l,j) = (0,4)
    out[nj] = sum((nh .* vhth2 .^ 2) .* (1 .+ 4 / 3 * uvth2 .+ 4 / 15 * uvth2 .^ 2)) - Mhcsl01[nj+2]
    return uh[2], vhth[2]
end

# nMode ≥ 3
function king_fDM2!(out::AbstractVector{T}, x::AbstractVector{T}, nh::AbstractVector{T}, nModel1::Int64;
    uh::AbstractVector{T}=[0.1, 1.0], vhth::AbstractVector{T}=[0.1, 1.0], 
    uvth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0], Mhcsl01::AbstractVector{T}=[0.1, 1.0]) where{T}

    uh[1:end-1] = x[1:2:end]
    vhth[1:end-1] = x[2:2:end]
    vhth2[1:end-1] = vhth[1:end-1] .^ 2
    uvth[1:end-1] = uh[1:end-1] ./ vhth[1:end-1]
    uvth2[1:end-1] = uvth[1:end-1] .^ 2   # uh .^ 2 ./ vhth .^ 2

    uh[end] = (Mhcsl01[1] - sum(nh[1:end-1] .* uh[1:end-1])) / nh[end]
    vhth2[end] = (Mhcsl01[2] - sum((nh[1:end-1] .* vhth2[1:end-1]) .* (1 .+ 2 / 3 * uvth2[1:end-1]))) / nh[end] - 2 / 3 * (uh[end]) .^ 2
    vhth[end] = vhth2[end]^0.5
    uvth[end] = uh[end] ./ vhth[end]
    uvth2[end] = uvth[end] .^ 2   # uh .^ 2 ./ vhth .^ 2

    nj = 1
    # (l,j) = (1,3)
    out[nj] = sum((nh .* uh .* vhth2) .* (1 .+ 2 / 5 * uvth2)) - Mhcsl01[nj+2]


    # @show nModel1, size(out), length(nh), length(x)
    # rtgh
    nj += 1
    # (l,j) = (0,4)
    l = 0
    j = 4
    out[nj] = sum((nh .* vhth2 .^ 2) .* (1 .+ 4 / 3 * uvth2 .+ 4 / 15 * uvth2 .^ 2)) - Mhcsl01[nj+2]

    for kM in 2:nModel1
        nj += 1
        (l, j) = (1, nj)
        N = (j - l) / 2 |> Int
        k = 1:N
        ck = [2^k * binomial(N, k) / prod(5:2:2(l+k)+1) for k in 1:N]
        out[nj] = sum((nh .* uh .* vhth .^ (j - l)) .* (1 .+ [sum(ck .* uvth[s] .^ (2k)) for s in 1:nModel1])) - Mhcsl01[nj+2]

        nj += 1
        (l, j) = (0, nj)
        N = j / 2 |> Int
        k = 1:N
        ck = [2^k * binomial(N, k) / prod(3:2:2k+1) for k in 1:N]
        out[nj] = sum((nh .* vhth .^ j) .* (1 .+ [sum(ck .* uvth[s] .^ (2k)) for s in 1:nModel1])) - Mhcsl01[nj+2]
    end
end

function king_fDM2!(out::AbstractVector{T}, x::AbstractVector{T}, nh::AbstractVector{T}, Mhcsl01::AbstractVector{T}, nModel1::Int64;
    uh::AbstractVector{T}=[0.1, 1.0], vhth::AbstractVector{T}=[0.1, 1.0], 
    uvth::AbstractVector{T}=[0.1, 1.0], vhth2::AbstractVector{T}=[0.1, 1.0], 
    uvth2::AbstractVector{T}=[0.1, 1.0]) where{T}

    uh[1:end-1] = x[1:2:end]
    vhth[1:end-1] = x[2:2:end]
    vhth2[1:end-1] = vhth[1:end-1] .^ 2
    uvth[1:end-1] = uh[1:end-1] ./ vhth[1:end-1]
    uvth2[1:end-1] = uvth[1:end-1] .^ 2   # uh .^ 2 ./ vhth .^ 2

    uh[end] = (Mhcsl01[1] - sum(nh[1:end-1] .* uh[1:end-1])) / nh[end]
    vhth2[end] = (Mhcsl01[2] - sum((nh[1:end-1] .* vhth2[1:end-1]) .* (1 .+ 2 / 3 * uvth2[1:end-1]))) / nh[end] - 2 / 3 * (uh[end]) .^ 2
    vhth[end] = vhth2[end]^0.5
    uvth[end] = uh[end] ./ vhth[end]
    uvth2[end] = uvth[end] .^ 2   # uh .^ 2 ./ vhth .^ 2

    nj = 1
    # (l,j) = (1,3)
    out[nj] = sum((nh .* uh .* vhth2) .* (1 .+ 2 / 5 * uvth2)) - Mhcsl01[nj+2]


    @show nModel1, size(out), length(nh), length(x)
    rtgh
    nj += 1
    # (l,j) = (0,4)
    l = 0
    j = 4
    out[nj] = sum((nh .* vhth2 .^ 2) .* (1 .+ 4 / 3 * uvth2 .+ 4 / 15 * uvth2 .^ 2)) - Mhcsl01[nj+2]

    for kM in 2:nModel1
        nj += 1
        (l, j) = (1, nj)
        N = (j - l) / 2 |> Int
        k = 1:N
        ck = [2^k * binomial(N, k) / prod(5:2:2(l+k)+1) for k in 1:N]
        out[nj] = sum((nh .* uh .* vhth .^ (j - l)) .* (1 .+ [sum(ck .* uvth[s] .^ (2k)) for s in 1:nModel1])) - Mhcsl01[nj+2]

        nj += 1
        (l, j) = (0, nj)
        N = j / 2 |> Int
        k = 1:N
        ck = [2^k * binomial(N, k) / prod(3:2:2k+1) for k in 1:N]
        out[nj] = sum((nh .* vhth .^ j) .* (1 .+ [sum(ck .* uvth[s] .^ (2k)) for s in 1:nModel1])) - Mhcsl01[nj+2]
    end
    return uh[end], vhth[end]
end

"""
  Inputs:
    J: = zeros(nModel1,nModel1)
    x: = x(nModel1)
    nh: = nai
    uh: = uai[1:nModel1]
    vhth: = vthi[1:nModel1]
    nModel1 = nModel - 1

  Outputs:
    king_fDM2_g!(J, x, nh;uh=uh,vhth=vhth,vhth2=vhth2, Mhcsl01)
    king_fDM2_g!(J, x, nModel1, nh;uh=uh,vhth=vhth,vhth2=vhth2, Mhcsl01)
"""

# The Jacobian matrix: J = zeros(T,nMod-1,nMod-1)
# nMode = 2
function king_fDM2_g!(J::AbstractArray{T,N}, x::AbstractVector{T}, nh::AbstractVector{T}; 
    uh::AbstractVector{T}=[0.1, 1.0], vhth::AbstractVector{T}=[0.1, 1.0], 
    vhth2::AbstractVector{T}=[0.1, 1.0], Mhcsl01::AbstractVector{T}=[0.1, 1.0]) where{T,N}

    fill!(J, 0.0)
    uh[1] = x[1]
    vhth[1] = x[2]
    vhth2[1] = vhth[1] ^ 2
    uh[2] = (Mhcsl01[1] - nh[1] .* uh[1]) / nh[2]
    vhth2[2] = (Mhcsl01[2] - (nh[1] .* vhth2[1]) .* (1 .+ 2 / 3 * (uh[1] ^ 2 / vhth2[1]))) / nh[2] - 2 / 3 * uh[2] ^ 2
    vhth[2] = vhth2[2]^0.5

    uh9 = uh[end]
    uh92 = uh9 ^2
    vhth92 = vhth2[end]
    
    nj = 1
    # (l,j) = (1,3)
    s = 1
    s2 = 2(s - 1)
    CM1 = -1.0 + 4/3 * uh9 * (uh9 - uh[s]) / vhth92
    J[nj, s2+1] = nh[s] * (vhth2[s] + CM1 * vhth92 + 6 / 5 * (uh[s]^2 - uh92))
    J[nj, s2+2] = 2nh[s] * vhth[s] * (uh[s] - uh9)

    nj += 1
    # (l, j) = (0, 4)
    CM11 = 4 / 3
    CM112 = 2CM11
    CM12 = 4 / 15
    CM32 = - 4CM12
    s = 1
    s2 = 2(s - 1)
    CMu2 = CM112 * (uh9 - uh[s]) * vhth92 * (vhth[s] / vhth[end])
    CM2 = 2 / 3 * (uh92 - uh9 * uh[s]) / vhth92
    CM31 = CM112 * (CM2 - 1)
    J[nj, s2+1] = nh[s] * (CMu2 + CM112 * uh[s] * vhth2[s] + CM31 * uh9 * vhth92 + CM32 * (uh92 * uh9 - uh[s]^3))
    J[nj, s2+2] = 4nh[s] * vhth[s] * ((vhth2[s] - vhth92) + 2 / 3 * (uh[s]^2 - uh92))
end

# nMode ≥ 3
function king_fDM2_g!(J::AbstractArray{T,NN}, x::AbstractVector{T}, nh::AbstractVector{T}, nModel1::Int64; 
    uh::AbstractVector{T}=[0.1, 1.0], vhth::AbstractVector{T}=[0.1, 1.0], 
    vhth2::AbstractVector{T}=[0.1, 1.0], Mhcsl01::AbstractVector{T}=[0.1, 1.0]) where{T,NN}

    fill!(J, 0.0)
    vec = 1:nModel1
    uh[vec] = x[1:2:end]
    vhth[vec] = x[2:2:end]
    vhth2[vec] = vhth[vec] .^ 2

    nh9 = nh[end]

    uh[end] = (Mhcsl01[1] - sum(nh[vec] .* uh[vec])) / nh9
    uh9 = uh[end]
    uh92 = uh9 ^2

    vhth2[end] = (Mhcsl01[2] - ((nh[vec] .* vhth2[vec]) .* (1 .+ 2 / 3 * (uh[vec] .^ 2 ./ vhth2[vec])))) / nh9 - 2 / 3 * uh9 ^ 2
    vhth[end] = vhth2[end]^0.5
    vhth92 = vhth2[end]
    
    nj = 1
    # (l,j) = (1,3)
    for s in vec
        s2 = 2(s - 1)
        CM1 = -1.0 + 4/3 * uh9 * (uh9 - uh[s]) / vhth92
        J[nj, s2+1] = nh[s] * (vhth2[s] + CM1 * vhth92 + 6 / 5 * (uh[s]^2 - uh92))
        J[nj, s2+2] = 2nh[s] * vhth[s] * (uh[s] - uh9)
    end

    nj += 1
    # (l, j) = (0, 4)
    CM11 = 4 / 3
    CM112 = 2CM11
    CM12 = 4 / 15
    CM32 = - 4CM12
    for s in vec
        s2 = 2(s - 1)
        CMu2 = CM112 * (uh9 - uh[s]) * vhth92 * (vhth[s] / vhth[end])
        CM2 = 2 / 3 * (uh92 - uh9 * uh[s]) / vhth92
        CM31 = CM112 * (CM2 - 1)
        J[nj, s2+1] = nh[s] * (CMu2 + CM112 * uh[s] * vhth2[s] + CM31 * uh9 * vhth92 + CM32 * (uh92 * uh9 - uh[s]^3))
        J[nj, s2+2] = 4nh[s] * vhth[s] * ((vhth2[s] - vhth92) + 2 / 3 * (uh[s]^2 - uh92))
    end

    uvth = uh ./ vhth
    
    for kM in 2:nModel1
        ertgh
        nj += 1
        (l, j) = (1, nj)
        N = (j - l) / 2 |> Int
        k = 1:N
        ck = [2^k * binomial(N, k) / prod(5:2:2(l+k)+1) for k in 1:N]
        # ck = [2^k * binomial(N, k) / prod((2l+3):2:2(l+k)+1) for k in 1:N]
        for s in vec
            s2 = 2(s - 1)
            J[nj, s2+1] = nh[s] * vhth[s]^(j - l) * (l + sum((2k .+ l) .* ck .* uvth[s] .^ (2k)))
            J[nj, s2+2] = nh[s] * uh[s] * vhth[s]^(j - 2) * ((j - l) - sum((2k .+ (l - j)) .* ck .* uvth[s] .^ (2k)))
        end

        nj += 1
        (l, j) = (0, nj)
        N = j / 2 |> Int
        k = 1:N
        ck = [2^k * binomial(N, k) / prod(3:2:2k+1) for k in 1:N]
        for s in vec
            s2 = 2(s - 1)
            J[nj, s2+1] = nh[s] * vhth[s]^(j - l) * sum(2k .* ck .* uvth[s] .^ (2k .- 1))
            J[nj, s2+2] = j * nh[s] * vhth[s]^(j - 1) * (1 + sum((1 .- 2 / j * k) .* ck .* uvth[s] .^ (2k)))
        end
    end
end
