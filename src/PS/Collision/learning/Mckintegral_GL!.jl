
"""
  A `s`-stage integral at the `kᵗʰ` step with implicit Euler method or trapezoidal method with `Niter_stage`: 

  Level of the algorithm
    kᵗʰ: the time step level
    s: the stage level during `kᵗʰ` time step
    i: the inner iteration level during `sᵗʰ` stage
    
  Inputs:
    nk1 = deepcopy(nk)
    Ik1 = deepcopy(Ik)
    Kk1 = deepcopy(Kk)
    vthk1 = deepcopy(vthk)              # vth_(k+1)_(i) which will be changed in the following codes
    Mck1 = deepcopy(Mck)
    Rck1i::Vector{Any} = [Rck; Rck11; Rck12; ⋯; Rck1i]
    Rck1: = Rck
    Rck1[njMs+1,1,:]                # `w3k = Rdtvth = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`
    Nstage::Int64 ≥ 3, where `Nstage ∈ N⁺`
    Rck1N = Vector{Any}(undef,Nstage)
            When `Nstage=1` go back to the Euler method;
            When `Nstage=2` which will give the trapezoidal method;

  Outputs:
    err_dtnIK1 = Mck1integrals_GLegendre!(Rck1N, Mck1, Rck1, err_Rck1, Mhck1, 
        vhk, vhe, nvG, nc0, nck, ocp, vGdom, nvlevele0, nvlevel0, LMk, LM1k,  
        muk, Mμk, Munk, Mun1k, Mun2k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nk1, vthk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
        Ik1, Kk1, vthk1i, Mck, Rck, RMcsk, Mhck, nk, Ik, Kk, vthk, dtk, A, b, c, s;
        NL_solve=NL_solve, is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
        is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
        L_shape=L_shape, eps_fup=eps_fup, eps_flow=eps_flow, jMax=jMax,
        maxiter_vGm=maxiter_vGm, is_out_shape=is_out_shape,
        abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels,
         orderVconst=orderVconst, vGm_limit=vGm_limit,
        is_vth_ode=is_vth_ode,is_corrections=is_corrections,i_iter_rs2=i_iter_rs2,
        alg_embedded=alg_embeddedi_iter_rs3=i_iter_rs3)
"""

# [kᵗʰ,s,i], `alg_embedded = :Trapz` as the initial guess values at the general inner nodes.

function Mck1integrals_GLegendre!(Rck1N::Vector{Any},
    Mck1::AbstractArray{T,N},Rck1::AbstractArray{T,N},
    err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::Vector{AbstractVector{T}},vhe::AbstractVector{StepRangeLen},
    nvG::Vector{Int64},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},vGdom::AbstractArray{T,N2},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},LMk::Vector{Int64},LM1k::Int64,
    muk::AbstractArray{T,N2},Mμk::AbstractArray{T,N2},Munk::AbstractArray{T,N2},
    Mun1k::AbstractArray{T,NM1},Mun2k::AbstractArray{T,NM2},
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nk1::AbstractVector{T},
    vthk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},
    Ik1::AbstractVector{T},Kk1::AbstractVector{T},vthk1i::AbstractVector{T},
    Mck::AbstractArray{T,N},Rck::AbstractArray{T,N},
    RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},nk::AbstractVector{T},
    Ik::AbstractVector{T},Kk::AbstractVector{T},vthk::AbstractVector{T},
    dtk::T,A::AbstractArray{T,N2},b::AbstractVector{T},c::AbstractVector{T},s::Int64;
    NL_solve::Symbol=:NLsolve, is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,
    L_shape::Int64=0,eps_fup::T=1e-17,eps_flow::T=1e-18,jMax::Int64=1,
    maxiter_vGm::Int64=100,is_out_shape::Bool=false,
    abstol::Float64=epsT5,reltol::Float64=1e-5,vadaptlevels::Int=4,
    orderVconst::Vector{Int64}=[1,1],vGm_limit::Vector{T}=[5.0,20],
    is_vth_ode::Bool=true,is_corrections::Vector{Bool}=[true,false,false],
    i_iter_rs2::Int64=0,alg_embedded::Symbol=:Trapz,i_iter_rs3::Int64=0) where{T,N,N2,NM1,NM2}

    # vthk1i = zeros(T,ns)
    Mck1integrals!(Rck1N,Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nk1, vthk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
        Ik1, Kk1, vthk1i, Mck, Rck, RMcsk, Mhck, nk, Ik, Kk, vthk, dtk, c, s;
        NL_solve=NL_solve, is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
        is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
        L_shape=L_shape, eps_fup=eps_fup, eps_flow=eps_flow, jMax=jMax,
        maxiter_vGm=maxiter_vGm, is_out_shape=is_out_shape,
        abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels,
         orderVconst=orderVconst, vGm_limit=vGm_limit,
        is_vth_ode=is_vth_ode,is_corrections=is_corrections,
        i_iter_rs2=i_iter_rs2,alg_embedded=alg_embedded)
    
    i_iter3 = 0
    Mck1_copy = deepcopy(Mck1[1:njMs,:,:])           # tk + dtk * c[end]

    ρk1 = mak1 .* nk1

    # Computing the relative errors
    RerrDMck1 = 1.0
    RerrDMck1_up = deepcopy(RerrDMck1)
    ratio_DMc = 1.0          # = RerrDMck1 / RerrDMck1_up - 1
    if s == 1
        while i_iter3 < i_iter_rs3
            i_iter3 += 1
            @show i_iter3, RerrDMck1
            if RerrDMck1 ≤ RerrDMc
                break
            else
                Mck1 = Mck + dtk * A[1,1] * Rck1
    
                # Updating the values of `Rck1`, parameter such as `vth` will be updated according to `Mck1` at `k+1`
                err_dtnIK1 = Rck_update!(Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                    nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                    mak1, Zqk1, nk1, vthk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, Ik1, Kk1, Mck1, dtk;
                    NL_solve=NL_solve, is_normal=is_normal,
                    restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                    autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                    rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                    optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                    is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                    is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
                    L_shape=L_shape, eps_fup=eps_fup, eps_flow=eps_flow, jMax=jMax,
                    maxiter_vGm=maxiter_vGm, is_out_shape=is_out_shape,
                    abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels,
                     orderVconst=orderVconst, vGm_limit=vGm_limit,
                    is_corrections=is_corrections,is_vth_ode=true)
                RerrDMck1 = evaluate_RerrDMc(Mck1[1:njMs,:,:],Mck1_copy)
                ratio_DMc = abs(RerrDMck1 / (RerrDMck1_up + epsT) - 1)
                if ratio_DMc ≤ Ratio_DMc
                    break
                end
                Mck1_copy = deepcopy(Mck1[1:njMs,:,:])
                RerrDMck1_up = deepcopy(RerrDMck1)
            end
        end
    else
        Mck1N = deepcopy(Rck1N)
        while i_iter3 < i_iter_rs3
            i_iter3 += 1
            @show i_iter3, RerrDMck1
            if RerrDMck1 ≤ RerrDMc
                break
            else
                ii = 1
                jj = 1
                Rck1 = A[ii,jj] * Rck1N[jj]          # `k_eff`, the effective derivatives
                for jj in 2:s
                    Rck1 += A[ii,jj] * Rck1N[jj]
                end
                Mck1N[ii] = Mck + dtk * Rck1
                for ii in 2:s
                    jj = 1
                    Rck1 = A[ii,jj] * Rck1N[jj]
                    for jj in 2:s
                        Rck1 += A[ii,jj] * Rck1N[jj]
                    end
                    Mck1N[ii] = Mck + dtk * Rck1
                end
                # Mck1N_fM_check(Mck,Mck1N,s,nsk1,i_iter3)
                Mck1N_fM_check(Mck,Mck1N,ρk1,s,nsk1,i_iter3)
    
                # Updating the values of `Rck1N`, parameter such as `vth` will be updated according to `Mck1N`
                ii = 1
                Mck1 = Mck1N[ii]
                println("--------------------------------")
                @show ii
                # Rck1 = zero.(Rck)
                err_dtnIK1 = Rck_update!(Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                    nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                    mak1, Zqk1, nk1, vthk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, Ik1, Kk1, Mck1, dtk;
                    NL_solve=NL_solve, is_normal=is_normal,
                    restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                    autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                    rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                    optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                    is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                    is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
                    L_shape=L_shape, eps_fup=eps_fup, eps_flow=eps_flow, jMax=jMax,
                    maxiter_vGm=maxiter_vGm, is_out_shape=is_out_shape,
                    abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels,
                     orderVconst=orderVconst, vGm_limit=vGm_limit,
                    is_corrections=is_corrections,is_vth_ode=true)
                Rck1N[ii] = deepcopy(Rck1)
                for ii in 2:s
                    println("--------------------------------")
                    @show ii
                    Mck1 = Mck1N[ii]
                    err_dtnIK1 = Rck_update!(Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                        mak1, Zqk1, nk1, vthk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, Ik1, Kk1, Mck1, dtk;
                        NL_solve=NL_solve, is_normal=is_normal,
                        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                        is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
                        L_shape=L_shape, eps_fup=eps_fup, eps_flow=eps_flow, jMax=jMax,
                        maxiter_vGm=maxiter_vGm, is_out_shape=is_out_shape,
                        abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels,
                         orderVconst=orderVconst, vGm_limit=vGm_limit,
                        is_corrections=is_corrections,is_vth_ode=true)
                    Rck1N[ii] = deepcopy(Rck1)
                end

                RerrDMck1 = evaluate_RerrDMc(Mck1[1:njMs,:,:],Mck1_copy)
                ratio_DMc = abs(RerrDMck1 / (RerrDMck1_up + epsT) - 1)
                if ratio_DMc ≤ Ratio_DMc
                    break
                end
                Mck1_copy = deepcopy(Mck1[1:njMs,:,:])
                RerrDMck1_up = deepcopy(RerrDMck1)
            end
        end

        # Computing the effective derivatives in `Gauss-Legendre` algorithm: Rck1 = sum(b * Rck1N) at `k+1` grid
        if s == 2
            Rck1 = (Rck1N[1] + Rck1N[2]) / 2
        else
            s1 = 1
            Rck1 = b[s1] * Rck1N[s1]
            for s1 in 2:s
                Rck1 += b[s1] * Rck1N[s1]
            end
        end
    end
    if i_iter3 ≥ i_iter_rs3
        @warn(`rs3: The maximum number of iteration reached before the "LobattoIIIA4" method to be convergence!!!`)
    end
    @show i_iter3

    # Computing the values of `Mck1` at `k+1` step and its effective derivatives `Rck1 = keff` 
    # respective to time by using the explicit Euler method (ExEuler)
    Mck1 = deepcopy(Mck)
    δvthk1 = zero.(vthk1)
    err_dtnIK1 = Mck1integral!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ,  
        mak1, Zqk1, nk1, vthk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Ik1, Kk1, δvthk1, RMcsk, Mhck, Ik, Kk, vthk, dtk;
        NL_solve=NL_solve, is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
        is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
        L_shape=L_shape, eps_fup=eps_fup, eps_flow=eps_flow, jMax=jMax,
        maxiter_vGm=maxiter_vGm, is_out_shape=is_out_shape,
        abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels,
         orderVconst=orderVconst, vGm_limit=vGm_limit,
        is_vth_ode=is_vth_ode,is_corrections=is_corrections)
end


function Mck1N_fM_check(Mck::AbstractArray{T,N},Mck1N::Vector{Any},s::Int64,ns::Int64,i_iter3::Int64) where{T,N}

    s1 = s + 1
    Mvec = zeros(s1,ns)
    for nj in 1:njMs
        for isp in 1:ns
            ss = 1
            Mvec[ss,isp] = Mck[nj,1,isp]
            for ss in 2:s1
                Mvec[ss,isp] = Mck1N[ss-1][nj,1,isp]
            end
        end
        isp = 1
        ylabel = string("Mcka")
        label = string("(nj,i_iter)=",(nj,i_iter3))
        pppa = plot(Mvec[:,isp],label=label,ylabel=ylabel)
        isp = 2
        ylabel = string("Mckb")
        label = string("nj=",nj)
        pppb = plot(Mvec[:,isp],label=label,ylabel=ylabel)
        display(plot(pppa,pppb,layout=(2,1)))
    end
end


function Mck1N_fM_check(Mck::AbstractArray{T,N},Mck1N::Vector{Any},
    ρk::AbstractVector{T},s::Int64,ns::Int64,i_iter3::Int64) where{T,N}

    s1 = s + 1
    Mvec = zeros(s1,ns)
    vthvec = zeros(s1,ns)
    for nj in 1:njMs
        for isp in 1:ns
            ss = 1
            Mvec[ss,isp] = Mck[nj,1,isp]
            for ss in 2:s1
                Mvec[ss,isp] = Mck1N[ss-1][nj,1,isp]
            end
        end
        if nj == 1
            # isp = 1
            # ylabel = string("Mcka")
            # label = string("(nj,i_iter)=",(nj,i_iter3))
            # pppa = plot(Mvec[:,isp],label=label,ylabel=ylabel)
            # isp = 2
            # ylabel = string("Mckb")
            # label = string("nj=",nj)
            # pppb = plot(Mvec[:,isp],label=label,ylabel=ylabel)
            # display(plot(pppa,pppb,layout=(2,1)))
        elseif nj == 2
            j = 2(nj - 1)
            title = string("(nj,i_iter)=",(nj,i_iter3))

            isp = 1
            vthvec[:,isp] = (Mvec[:,isp] / ρk[isp]).^0.5
            Mhvec = Mtheorems_Mc(Mvec[:,isp], ρk[isp], vthvec[:,isp],j) .- 1.0
            label = string("a")
            ylabel = string("Mck")
            pppa = plot(Mvec[:,isp],label=label,ylabel=ylabel,title=title)
            ylabel = string("DMhck")
            xlabel = string("DMhc1=",fmtf2(Mhvec[1]))
            pppha = plot(Mhvec,label=label,ylabel=ylabel,xlabel=xlabel)

            isp = 2
            vthvec[:,isp] = (Mvec[:,isp] / ρk[isp]).^0.5
            Mhvec = Mtheorems_Mc(Mvec[:,isp], ρk[isp], vthvec[:,isp],j) .- 1.0
            label = string("b")
            pppb = plot(Mvec[:,isp],label=label)
            xlabel = string("DMhc1=",fmtf2(Mhvec[1]))
            ppphb = plot(Mhvec,label=label,xlabel=xlabel)
            display(plot(pppa,pppha,pppb,ppphb,layout=(2,2)))
        else
            j = 2(nj - 1)
            title = string("(nj,i_iter)=",(nj,i_iter3))

            isp = 1
            println("8888888888888888888888888888888888")
            @show fmtf4.(diff(Mvec[:,isp]))
            Mhvec = Mtheorems_Mc(Mvec[:,isp], ρk[isp], vthvec[:,isp],j) .- 1.0
            label = string("a")
            ylabel = string("Mck")
            pppa = plot(Mvec[:,isp],label=label,ylabel=ylabel,title=title)
            ylabel = string("DMhck")
            xlabel = string("DMhc1=",fmtf2(Mhvec[1]))
            pppha = plot(Mhvec,label=label,ylabel=ylabel,xlabel=xlabel)

            isp = 2
            Mhvec = Mtheorems_Mc(Mvec[:,isp], ρk[isp], vthvec[:,isp],j) .- 1.0
            label = string("b")
            pppb = plot(Mvec[:,isp],label=label)
            xlabel = string("DMhc1=",fmtf2(Mhvec[1]))
            ppphb = plot(Mhvec,label=label,xlabel=xlabel)
            display(plot(pppa,pppha,pppb,ppphb,layout=(2,2)))
        end
    end
end