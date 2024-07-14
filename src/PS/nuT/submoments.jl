"""
  When `is_nai_const = true`
  
  In orthonormalization method of the moments, the following procedure will give
  the sub-moments `nhai, uhai, vhathi` of `f̂₀(v̂)` and `{f̂₁(v̂)}` according the conservation laws with using `MsnnEven.jl`.

  The parameters `naik,uaik,vthik` are computed according to the functions of re-normalized moments `M̂ⱼₗᵐ*`(= M̂ⱼₗᵐ / CjLL2(j,L))`;
  
  K̂a*`= 2/3 * K̂a
  
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.


  # Updating the timestep according to the local linear model: `yₖ₊₁ = yₖ + dt * ∂ₜyₖ`
"""

"""
  Inputs:
    i

  Outputs:
    submoment!(naik1,uaik1,vthik1,nModk1,
            LMk,naik,uaik,vthik,nModk,
            Mhck1,errMhcop,nMjMs,
            Rdtsabk1,Rdtsabk,ns,
            Nspan_optim_nuTi,kt,tk,dtk;
            Nspan_nuTi_max=Nspan_nuTi_max,
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
            is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
            is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,is_NK_nai_const=is_NK_nai_const,
            is_nuTi_initial=is_nuTi_initial,is_fixed_NK=is_fixed_NK,
            is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
"""

# 2.5D, [nMod,ns], the first `jᵗʰ`-order moments of the `ℓᵗʰ`-order coefficient of distribution function on the entire velocity axis domain.
#  Mhck1
function submoment!(naik1::Vector{AbstractVector{T}},uaik1::Vector{AbstractVector{T}},vthik1::Vector{AbstractVector{T}},nModk1::Vector{Int64},
    LMk::Vector{Int64},naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},nModk::Vector{Int64},
    Mhck1::Vector{Matrix{T}},errMhcop::Vector{Matrix{T}},
    nMjMs::Vector{Int64},edtnIKTs::AbstractArray{T,N2},
    Rdtsabk1::T,Rdtsabk::T,ns::Int64,Nspan_optim_nuTi::AbstractVector{T},
    NK::Int64,NKmax::Int64,kt::Int64,tk::T,dtk::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7, 
    optimizer=Dogleg,factor=QR(),autodiff::Symbol=:central,
    is_Jacobian::Bool=true,show_trace::Bool=false, maxIterKing::Int64=200,
    p_tol::Float64=1e-27,f_tol::Float64=1e-27,g_tol::Float64=1e-27,
    is_optim_CnIK::Bool=false,is_nhnMod_adapt::Bool=false,
    is_NK_adapt_max::Bool=false,is_nai_const::Bool=true,#is_NK_nai_const::Vector{Bool}=[false,false],
    is_nuTi_initial::Bool=true,is_fixed_NK::Bool=false,
    is_nMod_update_back::Bool=false,is_nMod_update_advance::Bool=false) where {T,N2}

    DnuTh = zeros(3,ns)
    @show nModk, is_nai_const
    # @show nModk, vthik1
    dtk0 = 1dtk
    if is_NKCnhC
        Nspan_optim_nuTi0 = ones(T,3) 
        for isp in 1:ns
            @show isp
            @show nModk1[isp], vthik1[isp]
            if norm(uaik[isp]) ≤ atol_IKTh
                naik1[isp],vthik1[isp],nModk1[isp] = fl0king01_fMNKCnhC!(
                            naik1[isp],vthik1[isp],nModk1[isp],
                            LMk[isp],DnuTh[:,isp],Mhck1[isp],errMhcop[isp],
                            nMjMs[isp],edtnIKTs[:,isp],
                            Rdtsabk1,Rdtsabk,NL_solve,rtol_DnuTi,NK,NKmax,kt,tk,dtk;
                            optimizer=optimizer,factor=factor,autodiff=autodiff,
                            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi,
                            is_nhnMod_adapt=is_nhnMod_adapt,is_NK_adapt_max=is_NK_adapt_max,
                            is_NK_nai_const=is_NK_nai_const,is_nuTi_initial=is_nuTi_initial,
                            is_fixed_NK=is_fixed_NK,
                            is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
    
                # @show 5567 nModk1[isp], nModk[isp]
                # @show vthik1[isp]
                # @show naik1[isp]
                if nModk[isp] ≥ 222222
                    for k in 1:min(nModk1[isp],nModk[isp])
                        if vthik1[isp][k] > vthik[isp][k]
                            ratio = vthik1[isp][k] / vthik[isp][k]
                            if abs(ratio - 1.0) > rtol_DnIK 
                                dtk = dtk0 * rtol_DnIK / (vthik1[isp][k] - vthik[isp][k])
                            end
                        else
                            ratio = vthik[isp][k] / vthik1[isp][k]
                            if abs(ratio - 1.0) > rtol_DnIK 
                                dtk = dtk0 * rtol_DnIK / (vthik[isp][k] - vthik1[isp][k])
                            end
                        end
                        if ratio ≤ Nspan_nuTi_max[2]
                            Nspan_optim_nuTi0[3] = max(Nspan_optim_nuTi0[3], 1.0 + 2(ratio - 1.0))
                        else
                            Nspan_optim_nuTi0[3] = max(Nspan_optim_nuTi0[3], Nspan_nuTi_max[2])
                            @warn("31: `ratio_vthi` has been larger that `Nspan_nuTi_max`!",ratio)
                        end
                    end
                end
    
                if nModk1[isp] ≠ nModk[isp]
                    uaik1[isp] = zero.(naik1[isp])
                    printstyled("`nModk` will be updated in the optimization process of `nai, vthi`,isp=",isp,color=:red,"\n")
                else
                    uaik1[isp] .= 0.0
                end
                # println("******************************************************")
                # @show nModk1[isp], nModk[isp], uaik1[isp]
            else
                wedsfvg
            end
        end
    else
        Nspan_optim_nuTi0 = ones(T,3) 
        if is_NKC
            for isp in 1:ns
                @show isp
                @show nModk1[isp],vthik1[isp]
                # if kt ≥ 4
                #     is_nai_const_k = is_nai_const
                #     is_nhnMod_adapt_k = is_nhnMod_adapt
                # else
                #     if kt ≥ 3
                #         nModk1[isp] = NKk
                #     end
                #     is_nai_const_k = false
                #     is_nhnMod_adapt_k = true
                # end
                if norm(uaik[isp]) ≤ atol_IKTh
                    naik1[isp],vthik1[isp],nModk1[isp] = fl0king01_fMNKC!(
                                naik1[isp],vthik1[isp],nModk1[isp],
                                LMk[isp],DnuTh[:,isp],Mhck1[isp],errMhcop[isp],
                                nMjMs[isp],edtnIKTs[:,isp],
                                Rdtsabk1,Rdtsabk,NL_solve,rtol_DnuTi,kt,tk,dtk;
                                optimizer=optimizer,factor=factor,autodiff=autodiff,
                                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi,
                                is_nhnMod_adapt=is_nhnMod_adapt,is_NK_adapt_max=is_NK_adapt_max,
                                is_nai_const=is_nai_const,is_nuTi_initial=is_nuTi_initial,
                                is_fixed_NK=is_fixed_NK)
        
                    @show nModk1[isp], nModk[isp]
                    @show naik1[isp]
                    @show vthik1[isp]
                    if nModk[isp] ≥ 222222
                        for k in 1:min(nModk1[isp],nModk[isp])
                            if vthik1[isp][k] > vthik[isp][k]
                                ratio = vthik1[isp][k] / vthik[isp][k]
                                if abs(ratio - 1.0) > rtol_DnIK 
                                    dtk = dtk0 * rtol_DnIK / (vthik1[isp][k] - vthik[isp][k])
                                end
                            else
                                ratio = vthik[isp][k] / vthik1[isp][k]
                                if abs(ratio - 1.0) > rtol_DnIK 
                                    dtk = dtk0 * rtol_DnIK / (vthik[isp][k] - vthik1[isp][k])
                                end
                            end
                            if ratio ≤ Nspan_nuTi_max[2]
                                Nspan_optim_nuTi0[3] = max(Nspan_optim_nuTi0[3], 1.0 + 2(ratio - 1.0))
                            else
                                Nspan_optim_nuTi0[3] = max(Nspan_optim_nuTi0[3], Nspan_nuTi_max[2])
                                @warn("31: `ratio_vthi` has been larger that `Nspan_nuTi_max`!",ratio)
                            end
                        end
                    end
        
                    if nModk1[isp] ≠ nModk[isp]
                        uaik1[isp] = zero.(naik1[isp])
                        printstyled("`nModk` will be updated in the optimization process of `nai, vthi`,isp=",isp,color=:red,"\n")
                    else
                        uaik1[isp] .= 0.0
                    end
                    # println("******************************************************")
                    # @show nModk1[isp], nModk[isp], uaik1[isp]
                else
                    wedsfvg
                end
            end
        else
            for isp in 1:ns
                # @show isp
                if norm(uaik1[isp]) ≤ atol_IKTh
                    naik1[isp],vthik1[isp],nModk1[isp] = fl0king01_fMnhC!(
                                naik1[isp],vthik1[isp],nModk1[isp],
                                DnuTh[:,isp],Mhck1[isp],errMhcop[isp],NL_solve,rtol_DnuTi;
                                optimizer=optimizer,factor=factor,autodiff=autodiff,
                                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,Nspan_optim_nuTi=Nspan_optim_nuTi,
                                is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
                                is_nuTi_initial=false)
                    if nModk1[isp] ≥ 2
                        for k in 1:nModk1[isp]
                            if vthik1[isp][k] > vthik[isp][k]
                                ratio = vthik1[isp][k] / vthik[isp][k]
                                if abs(ratio - 1.0) > rtol_DnIK 
                                    dtk = dtk0 * rtol_DnIK / (vthik1[isp][k] - vthik[isp][k])
                                end
                            else
                                ratio = vthik[isp][k] / vthik1[isp][k]
                                if abs(ratio - 1.0) > rtol_DnIK 
                                    dtk = dtk0 * rtol_DnIK / (vthik[isp][k] - vthik1[isp][k])
                                end
                            end
                            if ratio ≤ Nspan_nuTi_max[2]
                                Nspan_optim_nuTi0[3] = max(Nspan_optim_nuTi0[3], 1.0 + 2(ratio - 1.0))
                            else
                                Nspan_optim_nuTi0[3] = max(Nspan_optim_nuTi0[3], Nspan_nuTi_max[2])
                                @warn("31: `ratio_vthi` has been larger that `Nspan_nuTi_max`!",ratio)
                            end
                        end
                    end
                    @show errMhcop[isp]
        
                    if nModk1[isp] ≠ nModk[isp]
                        uaik1[isp] = zero.(naik1[isp])
                        printstyled("`nModk` will be updated in the optimization process of `nai, vthi`,isp=",isp,color=:red,"\n")
                    else
                        uaik1[isp] .= 0.0
                    end
                else
                    if nModk[isp] ≥ 2
                        for k in 1:nModel
                            if vthik1[isp][k] > vthik[isp][k]
                                ratio = vthik1[isp][k] / vthik[isp][k]
                                if abs(ratio - 1.0) > rtol_DnIK 
                                    dtk = dtk0 * rtol_DnIK / (vthik1[isp][k] - vthik[isp][k])
                                end
                            else
                                ratio = vthik[isp][k] / vthik1[isp][k]
                                if abs(ratio - 1.0) > rtol_DnIK 
                                    dtk = dtk0 * rtol_DnIK / (vthik[isp][k] - vthik1[isp][k])
                                end
                            end
                            if ratio ≤ Nspan_nuTi_max[2] 
                                Nspan_optim_nuTi0[3] = max(Nspan_optim_nuTi0[3], 1.0 + 2(ratio - 1.0))
                            else
                                Nspan_optim_nuTi0[3] = max(Nspan_optim_nuTi0[3], Nspan_nuTi_max[2])
                                @warn("31: `ratio_vthi` has been larger that `Nspan_nuTi_max`!",ratio)
                            end
        
                            ## 
                            uka0, uka = abs(vthik1[isp][k]), abs(uaik[isp][k])
                            if uka0 ≥ atol_IKTh
                                if uka ≥ atol_IKTh
                                    if uka > uka0
                                        ratio = uka / uka0
                                        if abs(ratio - 1.0) > rtol_DnIK 
                                            dtk = dtk0 * rtol_DnIK / (uka - uka0)
                                        end
                                    else
                                        ratio = uka0 / uka
                                        if abs(ratio - 1.0) > rtol_DnIK 
                                            dtk = dtk0 * rtol_DnIK / (uka0 - uka)
                                        end
                                    end
                                    if ratio ≤ Nspan_nuTi_max[2] 
                                        Nspan_optim_nuTi0[2] = max(Nspan_optim_nuTi0[2], 1.0 + 2(ratio - 1.0))
                                    else
                                        Nspan_optim_nuTi0[2] = max(Nspan_optim_nuTi0[2], Nspan_nuTi_max[2])
                                        @warn("21: `ratio_uai` has been larger that `Nspan_nuTi_max`!",ratio)
                                    end
                                else
                                    ratio = uka0 / (uka + atol_IKTh)
                                    if abs(ratio - 1.0) > rtol_DnIK 
                                        dtk = dtk0 * rtol_DnIK / (uka0 - uka)
                                    end
        
                                    if ratio ≤ Nspan_nuTi_max[2] 
                                        Nspan_optim_nuTi0[2] = max(Nspan_optim_nuTi0[2], 1.0 + 2(ratio - 1.0))
                                    else
                                        Nspan_optim_nuTi0[2] = max(Nspan_optim_nuTi0[2], Nspan_nuTi_max[2])
                                        @warn("22: `ratio_uai` has been larger that `Nspan_nuTi_max`!",ratio)
                                    end
                                end
                            else
                                if uka ≥ atol_IKTh
                                    ratio = uka / (uka0 + atol_IKTh)
                                    if abs(ratio - 1.0) > rtol_DnIK 
                                        dtk = dtk0 * rtol_DnIK / (uka - uka0)
                                    end
                                    if ratio ≤ Nspan_nuTi_max[2] 
                                        Nspan_optim_nuTi0[2] = max(Nspan_optim_nuTi0[2], 1.0 + 2(ratio - 1.0))
                                    else
                                        Nspan_optim_nuTi0[2] = max(Nspan_optim_nuTi0[2], Nspan_nuTi_max[2])
                                        @warn("23: `ratio_uai` has been larger that `Nspan_nuTi_max`!",ratio)
                                    end
                                else
                                end
                            end
                        end
                    end
                    
                    if nModel - nModk[isp] ≠ 0
                        printstyled("`nModk` will be updated in the optimization process of `nai, vthi`,isp=",isp,color=:red,"\n")
                    end
                end
            end
        end
        for k in 2:3
            Nspan_optim_nuTi[k] = max(Nspan_optim_nuTi0[k], Nspan_nuTi_max[1])
        end
    end
    # @show 5577, nModk1, vthik1
    # @show naik1
    # dfvgbnm
    return dtk
end

"""
  Outputs:
    submomentN!(na1,Ia1,Ka1,vtha1,naik,uaik,vthik,ma,nk,uk,vthk,nModk)
    na1,Ia1,Ka1,vtha1 = submomentN!(naik,uaik,vthik,ma,nk,uk,vthk,nModk)
"""

# [ns], nk = nk[1:nMod], In theory
function submomentN!(na1::AbstractVector{T},Ia1::AbstractVector{T},Ka1::AbstractVector{T},
    vtha1::AbstractVector{T},naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    ma::AbstractVector{T},nk::Vector{AbstractVector{T}},uk::Vector{AbstractVector{T}},vthk::Vector{AbstractVector{T}},
    nMod::Vector{Int64},ns::Int64;is_nMod_renew::Vector{Bool}=ones(Bool,ns)) where{T}
    for isp in 1:ns
        if is_nMod_renew[isp]
            na1[isp],Ia1[isp],Ka1[isp],vtha1[isp] = submomentN!(naik[isp],uaik[isp],vthik[isp],
                                                   ma[isp],nk[isp],uk[isp],vthk[isp],nMod[isp])
            naik[isp][nMod[isp]+1:end] .= 0.0
        end
    end
end

# []
function submomentN!(naik::AbstractVector{T},uaik::AbstractVector{T},vthik::AbstractVector{T},
    ma::T,nk::AbstractVector{T},uk::AbstractVector{T},vthk::AbstractVector{T},nMod::Int64) where{T}
    
    if nMod == 1
        naik[:] = [1.0]
        vthik[:] = [1.0]
        uaik[:] = [uk[1] / vthk[1]]
        Ia1p = ma * nk[1] .* uk[1]
        Ka1p = 0.5ma * (nk[1] .* (1.5 * vthk[1].^2 + uk[1].^2))
        return nk[1], Ia1p, Ka1p, vthk[1]
    else
        vec = 1:nMod
        na1 = sum_kbn(nk[vec])
        Ia1 = ma * sum_kbn(nk[vec] .* uk[vec])
        Ka1 = 0.5ma * sum_kbn(nk[vec] .* (1.5 * vthk[vec].^2 + uk[vec].^2))
        rho = ma * na1
        vtha1 = (2/3 * (2Ka1 / rho - (Ia1 / rho)^2)) ^0.5
        
        naik[vec] = nk[vec] / na1
        uaik[vec] = uk[vec] / vtha1
        vthik[vec] = vthk[vec] / vtha1
        return na1, Ia1, Ka1, vtha1
    end
end

"""
  Outputs:
    IKk0 = IK_initial(ma, na, vth, naik, uaik, vthik, nMod, ns)
"""

function IK_initial(ma::AbstractVector{T}, na::AbstractVector{T}, vth::AbstractVector{T}, 
    naik::Vector{AbstractVector{T}}, uaik::Vector{AbstractVector{T}}, vthik::Vector{AbstractVector{T}}, 
    nMod::Vector{Int64},ns::Int64) where{T}

    IK0a = zeros(maximum(nMod),2,ns)
    for isp in 1:ns
        rho = ma[isp] * na[isp]
        Ic = rho * vth[isp]
        Kc = rho * vth[isp]^2 / 2
        for k in 1:nMod[isp]
            IK0a[k,1,isp] = Kc * naik[isp][k] * (1.5 * (vthik[isp][k])^2 + (uaik[isp][k])^2)
            IK0a[k,2,isp] = Ic * naik[isp][k] * uaik[isp][k]
        end
    end
    return IK0a
end

"""
  Outputs:
    nuTi_vec!(x0_nuTi,nMod, naik, uaik, vthik)
"""
# [nMod, ns]


# [nMod]
function nuTi_vec!(x0_nuTi::AbstractVector{T},nMod::Int64, naik::AbstractVector{T}, 
    uaik::AbstractVector{T}, vthik::AbstractVector{T}) where{T}
    
    i = 1
    x0_nuTi[1:3] = [naik[i], uaik[i], vthik[i]]
    for i in 2:nMod
        x0_nuTi[3(i - 1)+1:3+3(i - 1)] = [naik[i], uaik[i], vthik[i]]
    end
end
