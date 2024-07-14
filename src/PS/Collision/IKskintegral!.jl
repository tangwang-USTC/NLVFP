
"""
  Inputs:
    IKk:  [[[K], [I]], nMod, ns]

  Outputs:
    IKk1integralk!(dtIKk1,IKk1,pstk, Nstep;is_normal=is_normal,
               restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM,is_boundaryv0=is_boundaryv0,
               is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
               
               eps_fup=eps_fup,eps_flow=eps_flow,
               maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
               abstol=abstol,reltol=reltol,
               vadaptlevels=vadaptlevels,gridv_type=gridv_type,
               is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,

               i_iter_rs2=i_iter_rs2,is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
               is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
               alg_embedded=alg_embedded,is_Cerror_dtnIKTs=is_Cerror_dtnIKTs,
               rtol_dtsa=rtol_dtsa)
"""

# [k,s,i], alg_embedded ∈ [:Trapz, :ImMidpoint, :Range2, :Heun2, Raslton2, :Alshina2], o = 2
# :ExMidpoint = :Range2 
# :CN = :CrankNicolson = LobattoIIIA2 = :Trapz
function IKk1integralk!(dtIKk1::AbstractArray{T,N}, IKk1::AbstractArray{T,N},
    ps::Dict{String,Any}, Nstep::Int64; is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_boundaryv0::Bool=false,

    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    
    i_iter_rs2::Int64=10,is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    alg_embedded::Symbol=:Trapz,is_Cerror_dtnIKTs::Bool=true,
    is_dtk_GKMM_dtIK::Bool=true,rtol_dtsa::T=1e-8,ratio_dtk1::T=1.2) where{T,N}
    
                      # ratio_dtk1 = dtk1 / dtk
    # ps_copy = deepcopy(ps)
    tk = deepcopy(ps["tk"])
    dtk = deepcopy(ps["dt"])
    tauk = ps["tauk"]

    nsk1 = ps["ns"]
    mak1 = ps["ma"]
    Zqk1 = ps["Zq"]
    spices = ps["spices"]

    nak1 = ps["nk"]
    Iak1 = ps["Ik"]
    Kak1 = ps["Kk"]
    vathk1 = ps["vthk"]               # vth_(k)
    sak1 = deepcopy(ps["sak"])
    dtsabk1 = deepcopy(ps["dtsabk"])

    nnv = ps["nnv"]
    ocp = ps["ocp"]

    nModk1 = ps["nModk"]
    naik = ps["naik"]
    uaik = ps["uaik"]
    vthik = ps["vthik"]
    LMk = ps["LMk"]
    ρk1 = mak1 .* nak1
    edtnIKTs = deepcopy(ps["edtnIKTsk"])

    nvG = 2 .^ nnv .+ 1

    sk1 = zero.(IKk1[:,1,:])
    Rdtsaak = dtsaa_initial(nModk1,nsk1)
    dtnIKs = zeros(4,2)
    DThk1 = zeros(T, ns)             # δT̂
    is_nMod_renew = zeros(Bool,nsk1)

    k = 0       # initial step to calculate the values `ℭ̂ₗ⁰` and `w3k = Rdtvth = 𝒲 / 3`
    # where `is_update_nuTi = false` and `naik, uaik, vthik` are convergent according to `fvL`

    @show dtk, i_iter_rs2, alg_embedded
    δvthi = zeros(T,nsk1)

    vGdom = [0.0, 10.0]            # The initial guess vaules of boundaries of velocity axis
    nk1 = deepcopy(naik)
    uk1 = deepcopy(uaik)
    vthk1 = deepcopy(vthik)
    nuTk1_sub_initial!(nk1,uk1,vthk1,nak1,vathk1,naik,uaik,vthik,nModk1,nsk1)

    # # Updating the entropy
    entropy_fDM!(sk1,mak1,nk1,vthk1,IKk1[:,2,:],IKk1[:,2,:],nModk1,nsk1)
    # entropy_fDM!(sak1,mak1,nak1,vathk1,Iak1,Kak1,nsk1)

    # Updating the entropy change rate
    dtKa = zeros(T,nsk1)
    dtIa = zeros(T,nsk1)
    dtk1 = 1dtk
    for isp in 1:nsk1
        k = 1
        dtKa[isp] = dtIKk1[k,1,isp]
        dtIa[isp] = dtIKk1[k,2,isp]
        for k in 2:nModk1[isp]
            dtKa[isp] += dtIKk1[k,1,isp]
            dtIa[isp] += dtIKk1[k,2,isp]
        end
    end

    # uhak = Iak1 ./ (mak1 .* nak1 .* vathk1)
    # # Rdtsabk = entropyN_rate_fDM(mak1,nak1,vathk1,uhak,dtIa,dtKa,nsk1)
    # dtsabk = entropy_rate_fDM(mak1,vathk1,uhak,dtIa,dtKa,nsk1)
    # @show sak1, dtsabk / sum(sak1)
    Rdtsabk1 = entropyN_rate_fDM(mak1,nak1,vathk1,Iak1,Kak1,dtIa,dtKa,nsk1)
    
    # dtk = dt_RdtnIK(dtk,dtIKk1,IKk1,nModk1,nsk1;rtol_DnIK=rtol_DnIK)
    # dtk == dt_ratio * tauk[1] || printstyled("0: The time step is decided by `dtIK/Ik` instead of `tauk`!",color=:purple,"\n")
    if alg_embedded == :ExEuler
    else
        IKk = deepcopy(IKk1)
        dtIKk = deepcopy(dtIKk1)
        vthk1i = deepcopy(vthk1)          # zeros(T,2)
        if alg_embedded == :ImEuler
        else
            if alg_embedded == :Trapz 
                count = 0
                for k in 1:Nstep
                    # parameters
                    tk = deepcopy(ps["tk"])
                    dtk = deepcopy(ps["dt"])
                    Nt_save = ps["Nt_save"]
                    count_save = ps["count_save"]
                    
                    if is_dtk_GKMM_dtIK
                        dtk = min(ratio_dtk1 * dtk, dt_ratio * tauk[1])
                        # @show 1,dtk
                        dtk = dt_RdtnIK(dtk,dtIKk,IKk,nModk1,nsk1;rtol_DnIK=rtol_DnIK)
                        dtk == dt_ratio * tauk[1] || printstyled("The time step is decided by `dtIK/Ik` instead of `tauk`!",color=:purple,"\n")
                        # @show 2,dtk
                    else
                        # dtk = min(ratio_dtk1 * dtk, dt_ratio * tauk[1])
                        dtk *= ratio_dtk1
                        # @show 1,dtk
                    end

                    println()
                    println("**************------------******************------------*********")
                    printstyled("k=",k,",tk,dt,Rdt=",fmtf2.([ps["tk"],dtk,dtk/ps["tk"]]),"\n";color=:blue)
                    
                    dtk1 = IKk1integrali_rs2!(dtIKk1,IKk1,Rdtsaak,sk1,
                                dtnIKs,edtnIKTs,nvG,ocp,vGdom,LMk, 
                                CΓ,εᵣ,mak1,Zqk1,spices,nk1,uk1,vthk1, 
                                nModk1, DThk1, vthk1i,δvthi,IKk,dtIKk,dtk;
                                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                                is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               
                                eps_fup=eps_fup,eps_flow=eps_flow,
                                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                                abstol=abstol,reltol=reltol,
                                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                 
                                is_check_conservation_dtM=is_check_conservation_dtM,
                                i_iter_rs2=i_iter_rs2,alg_embedded=alg_embedded,
                                is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                                is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)

                    # Updating the moments and submoments from `nk1,uk1,vthk1`
                    submomentN!(nak1,Iak1,Kak1,vathk1,naik,uaik,vthik,mak1,nk1,uk1,vthk1,nModk1,nsk1)
                    
                    # # # Updating the parameters `nModk1`
                    if prod(nModk1) ≥ 2
                        # reducing the number of `nModk1` according to `Rdtsaak` and updating `naik, uaik, vthik`
                        nMod_update!(is_nMod_renew, nModk1, Rdtsaak, naik, uaik, vthik, nsk1;rtol_dtsa=rtol_dtsa)

                        if is_fixed_timestep == false
                            if sum(is_nMod_renew) > 0

                                # Updating `nk1,uk1,vthk1` owing to the reduced parameters `nModk1`
                                nuTk1_sub_update!(nk1,uk1,vthk1,nak1,vathk1,naik,uaik,vthik,nModk1,nsk1,is_nMod_renew)
                                
                                # Updating `IKk1` from `nk1,uk1,vthk1`
                                nIKk_update!(IKk1,mak1,nk1,uk1,vthk1,nModk1;is_nMod_renew=is_nMod_renew)

                               # Updating `dtIKk1`
                                dtk1 = dtIKk_update!(dtIKk1,IKk1,Rdtsaak,sk1,dtnIKs,edtnIKTs,nvG,ocp,vGdom,LMk,
                                        CΓ,εᵣ,mak1,Zqk1,spices,nk1,uk1,vthk1,nModk1,DThk1,dtk1;
                                        is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                                        autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                                        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                                        is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               
                                        eps_fup=eps_fup,eps_flow=eps_flow,
                                        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                                        abstol=abstol,reltol=reltol,
                                        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                                        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                         
                                        is_check_conservation_dtM=is_check_conservation_dtM,
                                        is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                                        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
                            
                                tau_fM!(tauk, mak1, Zqk1, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                                printstyled("0: Updating the time scale, tau=", tauk,color=:green,"\n")
                                count = 0
                            else
                                count += 1
                                if count == count_tau_update
                                    tau_fM!(tauk, mak1, Zqk1, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                                    printstyled("1: Updating the time scale, tau=", tauk,color=:green,"\n")
                                    count = 0
                                end
                            end
                        else
                            # fujjkkkk
                        end
                    elseif is_fixed_timestep == false
                        count += 1
                        if count == count_tau_update
                            tau_fM!(tauk, mak1, Zqk1, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                            printstyled("3: Updating the time scale, tau=", tauk,color=:green,"\n")
                            count = 0
                        end
                    end

                    # [ns], Updating the entropy and its change rate with assumpation of dauble-Maxwellian distribution
                    entropy_fDM!(sak1,mak1,nak1,vathk1,Iak1,Kak1,nsk1)                        
                    
                    # Updating the total effects of Coulomb collisions for every sub-components
                    for isp in 1:nsk1
                        k = 1
                        dtKa[isp] = dtIKk1[k,1,isp]
                        dtIa[isp] = dtIKk1[k,2,isp]
                        for k in 2:nModk1[isp]
                            dtKa[isp] += dtIKk1[k,1,isp]
                            dtIa[isp] += dtIKk1[k,2,isp]
                        end
                    end

                    # dtsabk1 = dtsak1 + dtsbk1
                    # [nsk1 = 2] Iahk = uak1 ./ vathk1
                    dtsabk1 = entropy_rate_fDM(mak1,vathk1,Iak1 ./ (ρk1 .* vathk1),dtIa,dtKa,nsk1)
                    
                    Rdtsabk1 = dtsabk1 / sum(sak1)
                    # Rdtsabk1 = entropyN_rate_fDM(mak1,nak1,vathk1,Iak1,Kak1,dtIa,dtKa,nsk1)
                    # @show Rdtsabk1
                    
                    # Updating the entropy of sub-components of distribution functions for `(k+1)ᵗʰ` step
                    entropy_fDM!(sk1,mak1,nk1,vthk1,IKk1[:,2,:],IKk1[:,2,:],nModk1,nsk1)
                    
                    # # updating the distribution function and parameters at `(k+1)ᵗʰ` step
                    IKk[:,:,:] = deepcopy(IKk1)
                    dtIKk[:,:,:] = deepcopy(dtIKk1)

                    ps["tk"] = tk + dtk
                    ps["dt"] = dtk1
                    # ps["nk"] = deepcopy(nk1)
                    ps["Ik"] = deepcopy(Iak1)
                    ps["Kk"] = deepcopy(Kak1)
                    ps["naik"] = deepcopy(naik)
                    ps["uaik"] = deepcopy(uaik)
                    ps["vthik"] = deepcopy(vthik)
                    ps["nModk"] = deepcopy(nModk1)
                    ps["sak"] = deepcopy(sak1)
                    ps["dtsabk"] = deepcopy(dtsabk1)
                    ps["edtnIKTsk"] = deepcopy(edtnIKTs)
    
                    # Saving the dataset at `(k+1)ᵗʰ` step
                    if count_save == Nt_save
                        ps["count_save"] = 1
                        data_IKs_saving(ps;is_Cerror_dtnIKTs=is_Cerror_dtnIKTs)
                    else
                        ps["count_save"] = count_save + 1
                    end

                    # Terminating the progrom when I. reaches the maximum time moment; II. number of time step; III. equilibrium state.
                    if abs(Rdtsabk1) ≤ rtol_dtsa_terminate
                        @warn("The system has reached the equilibrium state when", Rdtsabk1)
                        break
                    elseif tk > ps["tspan"][2]
                        @warn("The system has reached the maximum time moment at", tk)
                        break
                    end
                end
            else
            end
        end
    end
end

"""

  Inputs:
    Rdtsaa: To decide whether reaches a new bifurcation point according to the 第二类独立特征可分辨判据。

  Outputs:
    dtk1 = IKk1integrali_rs2!(dtIKk1,IKk1,Rdtsaak,sk1,dtnIKs,edtnIKTs,
               vhk,nvG,ocp,vGdom,nvlevele0,nvlevel0,LMk,
               CΓ,εᵣ,mak1,Zqk1,spices,nk1,uk1,vthk1,
               nModk1,DThk1,vthk1i,δvthi,IKk,dtIKk,dtk1,dtk;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
               is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               
               eps_fup=eps_fup,eps_flow=eps_flow,
               maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
               abstol=abstol,reltol=reltol,
               vadaptlevels=vadaptlevels,gridv_type=gridv_type,
               is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,

               is_check_conservation_dtM=is_check_conservation_dtM,
               i_iter_rs2=i_iter_rs2,alg_embedded=alg_embedded,
               is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
               is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
"""

# [sᵗʰ,i], alg_embedded ∈ [:Trapz, :Heun2, :ImMidpoint],    rs = 2, o = 2
function IKk1integrali_rs2!(dtIKk1i::AbstractArray{T,N},IKk1::AbstractArray{T,N},
    Rdtsaak::Vector{TA2},sk1::AbstractArray{T,N2},
    dtnIKs::AbstractArray{T,N2},edtnIKTs::AbstractArray{T,N2},
    nvG::Int64,ocp::Int64,vGdom::AbstractVector{T},LMk::Vector{Int64},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::Vector{Int64},spices::Vector{Symbol},
    nk1::Vector{TA},uk1::Vector{TA},vthk1::Vector{TA},
    nModk1::Vector{Int64},DThk1::AbstractVector{T},vthk1i::Vector{TA},δvthi::AbstractVector{T},
    IKk::AbstractArray{T,N},dtIKk::AbstractArray{T,N},dtk::T;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    
    i_iter_rs2::Int64=0,alg_embedded::Symbol=:Trapz,
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,TA,TA2,N,N2}

    i_iter = 0                  # and Checking the `i = 1` iteration
    # Applying the explicit Euler step
    # δvthi = ones(T,2)        # = vthk1_i / vthk1_i1 .- 1
    RDIK = dtIKk1i ./ IKk1 * dtk
    if maximum(abs.(RDIK)) > 1.1 * rtol_DnIK
        printstyled("RDIK > rtol_DnIK,", RDIK,color=:red,"\n")
    else
        printstyled("RDIK > rtol_DnIK,", RDIK,color=:green,"\n")
    end
    
    dtk1 = 1dtk
    dtk1 = IKk1integral!(dtIKk1i,IKk1,Rdtsaak,sk1,dtnIKs,edtnIKTs,nvG,ocp,vGdom,LMk,
               CΓ,εᵣ,mak1,Zqk1,spices,nk1,uk1,vthk1,nModk1,DThk1,dtk1;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
               is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               
               eps_fup=eps_fup,eps_flow=eps_flow,
               maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
               abstol=abstol,reltol=reltol,
               vadaptlevels=vadaptlevels,gridv_type=gridv_type,
               is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,

               is_check_conservation_dtM=is_check_conservation_dtM,
               is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
               is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
    
    # If `i_iter_rs2 ≤ 0`, then degenerate into the explicit Euler method (ExEuler)
    if alg_embedded == :Trapz
        vthk1i[:] = deepcopy(vthk1[:])
        # dtIKk1i: = (dtIKk + dtIKk1) / 2       inputs
        #          = dtIKk1                     outputs
        while i_iter < i_iter_rs2
            i_iter += 1
            IKk1[:,:,:] = deepcopy(IKk)
            dtIKk1i[:,:,:] = (dtIKk + dtIKk1i) / 2        # dtIKk1k = dtIKk_(k+1/2)
            dtk1 = 1dtk
            dtk1 = IKk1integral!(dtIKk1i,IKk1,Rdtsaak,sk1,dtnIKs,edtnIKTs,nvG,ocp,vGdom,LMk,
                       CΓ,εᵣ,mak1,Zqk1,spices,nk1,uk1,vthk1,nModk1,DThk1,dtk1;
                       is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                       autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                       p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                       rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
                       is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               
                       eps_fup=eps_fup,eps_flow=eps_flow,
                       maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                       abstol=abstol,reltol=reltol,
                       vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                       is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
        
                       is_check_conservation_dtM=is_check_conservation_dtM,
                       is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                       is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
            # δvthi = vthk1i ./ vthk1 .- 1                # = vthk1_i / vthk1_i1 .- 1
            for isp in 1:2
                nModel = nModk1[isp] 
                k = 1
                δvthi[isp] = abs(vthk1i[isp][k] ./ vthk1[isp][k] .- 1)
                if nModel ≥ 2
                    for k in 2:nModel
                        δvthi[isp] = max(δvthi[isp], abs(vthk1i[isp][k] ./ vthk1[isp][k] .- 1))
                    end
                end
            end
            if maximum(δvthi) ≤ rtol_vthi
                break
            end
            vthk1i[:] = deepcopy(vthk1)
            # @show i_iter, δvthi
        end
        # dtIKk1k = dtIK_(k+1/2)
        if i_iter > i_iter_rs2
            @warn(`The maximum number of iteration reached before the Heun method to be convergence!!!`)
        end
    # elseif alg_embedded == :Heun         # i_iter_rs2 = 1
    else
        dfgbn
    end
    return dtk1
end

"""

  Inputs:
    Rdtsaa: To decide whether reaches a new bifurcation point according to the 第二类独立特征可分辨判据。

  Outputs:
    dtk1 = dtIKk_update!(dtIKk,IKk1,Rdtsaak,sk1,dtnIKs,edtnIKTs,
               vhk,nvG,ocp,vGdom,nvlevele0,nvlevel0,LMk,
               CΓ,εᵣ,mak1,Zqk1,spices,nk1,uk1,vthk1,nModk1,DThk,dtk1;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
               is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               
               eps_fup=eps_fup,eps_flow=eps_flow,
               maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
               abstol=abstol,reltol=reltol,
               vadaptlevels=vadaptlevels,gridv_type=gridv_type,
               is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,

               is_check_conservation_dtM=is_check_conservation_dtM,
               is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
               is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
    dtk1 = IKk1integral!(dtIKk,IKk1,Rdtsaak,sk1,dtnIKs,edtnIKTs,
               vhk,nvG,ocp,vGdom,nvlevele0,nvlevel0,LMk,
               CΓ,εᵣ,mak1,Zqk1,spices,nk1,uk1,vthk1,nModk1,DThk,dtk1;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
               is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               
               eps_fup=eps_fup,eps_flow=eps_flow,
               maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
               abstol=abstol,reltol=reltol,
               vadaptlevels=vadaptlevels,gridv_type=gridv_type,
               is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,

               is_check_conservation_dtM=is_check_conservation_dtM,
               is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
               is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
"""

# [iᵗʰ,ns], `tk = 0` or after the bifurcation, `dtIKk .= 0.0`, `IKk` may be updated according to the M-theorems
function dtIKk_update!(dtIKk::AbstractArray{T,N3},IKk1::AbstractArray{T,N3},
    Rdtsaak::Vector{TA2},sk1::AbstractArray{T,N2},
    dtnIKs::AbstractArray{T,N2},edtnIKTs::AbstractArray{T,N2},
    nvG::Int64,ocp::Int64,vGdom::AbstractVector{T},LMk::Vector{Int64},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::Vector{Int64},spices::Vector{Symbol},
    nk1::Vector{TA},uk1::Vector{TA},vthk1::Vector{TA},
    nModk1::Vector{Int64},DThk::AbstractVector{T},dtk1::T;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,

    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,TA,TA2,N2,N3}
    
    # Calculate the parameters `nk1,uk1,vthk1` from IKk1
    nuT_sub_update!(uk1,vthk1,mak1,nk1,IKk1,nModk1)

    dtk1 = dtIKab!(dtIKk,Rdtsaak,sk1,dtnIKs,edtnIKTs,nvG,ocp,vGdom,LMk,
           CΓ,εᵣ,mak1,Zqk1,spices,nk1,uk1,vthk1,nModk1,DThk,dtk1;
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
           is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               
           eps_fup=eps_fup,eps_flow=eps_flow,
           maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
           abstol=abstol,reltol=reltol,
           vadaptlevels=vadaptlevels,gridv_type=gridv_type,
           is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,

           is_check_conservation_dtM=is_check_conservation_dtM,
           is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
           is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
           
    return dtk1
end

# [iᵗʰ], alg_embedded == ∈ [:ExEuler, :ImEuler, :Trapz]
function IKk1integral!(dtIKk::AbstractArray{T,N3},IKk1::AbstractArray{T,N3},
    Rdtsaak::Vector{TA2},sk1::AbstractArray{T,N2},
    dtnIKs::AbstractArray{T,N2},edtnIKTs::AbstractArray{T,N2},
    nvG::Int64,ocp::Int64,vGdom::AbstractVector{T},LMk::Vector{Int64},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::Vector{Int64},spices::Vector{Symbol},
    nk1::Vector{TA},uk1::Vector{TA},vthk1::Vector{TA},
    nModk1::Vector{Int64},DThk::AbstractVector{T},dtk1::T;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0,20],
    abstol::Float64=epsT5,reltol::Float64=1e-5,
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,
    
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true) where{T,TA,TA2,N2,N3}
    
    # Update `IK1 = IKk + dt * dtIKs` and the relative parameters
    IKk1integral0!(IKk1,dtIKk,nModk1,dtk1)

    # Calculate the parameters `nk1,uk1,vthk1` from `IKk1`
    nuT_sub_update!(uk1,vthk1,mak1,nk1,IKk1,nModk1)

    # # Computing the M-functions

    # # Computing the re-normalized moments

    # # Renew the values of `IKk1` according to the quantities of `Mhck1` and `RMcsk1`

    dtk1 = dtIKab!(dtIKk,Rdtsaak,sk1,dtnIKs,edtnIKTs,nvG,ocp,vGdom,LMk,
           CΓ,εᵣ,mak1,Zqk1,spices,nk1,uk1,vthk1,nModk1,DThk,dtk1;
           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
           rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, 
           is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               
           eps_fup=eps_fup,eps_flow=eps_flow,
           maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
           abstol=abstol,reltol=reltol,
           vadaptlevels=vadaptlevels,gridv_type=gridv_type,
           is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,

           is_check_conservation_dtM=is_check_conservation_dtM,
           is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
           is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)

    # Checking the laws of the dissipation
    return dtk1
end

# [], alg_embedded::Symbol ∈ [:ExEuler, :ImEuler, :Trapz]
function IKk1integral0!(IKk1::AbstractArray{T,N},dtIKk::AbstractArray{T,N},nMod::Vector{Int64},dtk1::T) where{T,N}
    
    for isp in 1:2
        for k in 1:nMod[isp] 
            IKk1[k,:,isp] += dtk1 * dtIKk[k,:,isp]
        end
    end
end

"""
  The re-normalized moments of the first-two orders amplitude functions `fvL` 

  A. When `fM`, the number of the unkown moments is `nj ≥ nModk1` and

    i. `j ∈ 2:2:2nj` for `ℓ = 0`

  B. When `fDM`, the number of the unkown moments is `nj ≥ 2nMod` and`

    i. `j ∈ 2:2:2([nj/2])` for `ℓ = 0```
    ii. `j ∈ 1:2:2([nj/2])` for `ℓ = 1`

  Inputs:
  Outputs:
"""

  