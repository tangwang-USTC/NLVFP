
"""
  A single-step multistage RK algorithm with inner iteration for
  the Fokker-Planck collision equations. The inner iteration is performed by
  the embedded implicit methods (implicit Euler method, the trapezoidal method or LabottaIIIA4 method)
  or the explicit methods such as explicit Euler, the Heun's method and so on.
  
  For `Mck` or `nak, vthk, Mhck`

  The criterions which are used to decide whether the algorithm is convergence or not are determined by the following characteristics:

    `criterions = [ps["DThk"]; ps["err_dtnIK"]; δvathi]`
  
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nModk1 = 1` gives `Î = û`.
  
  Level of the algorithm
    k: the time step level
    s: the stage level during `kᵗʰ` time step
    i: the inner iteration level during `sᵗʰ` stage
    
  Inputs:
    Rck1[njMs+1,1,:]                    # `w3k = Rdtvath = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`
    orders=order_dvδtf
    is_δtfvLaa = 0          # [0,     1   ]
                            # [dtfab, dtfa]
    uCk: The relative velocity during the Coulomb collision process.

  Outputs:
    Mck1integralk!(Rck1, Mck1, pst0, Nstep; orderRK=orderRK,
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
        is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
        eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
        abstol=abstol,reltol=reltol,
        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
        is_vth_ode=is_vth_ode, is_corrections=is_corrections, 
        i_iter_rs2=i_iter_rs2, is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
        dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
        alg_embedded=alg_embedded, is_MjMs_max=is_MjMs_max,
        is_moments_out=is_moments_out, is_Cerror_dtnIKTs=is_Cerror_dtnIKTs)

"""

# [k,s,i], alg_embedded ∈ [:Trapz, :ImMidpoint, :Range2, :Heun2, Raslton2, :Alshina2], o = 2
# :ExMidpoint = :Range2 
# :CN = :CrankNicolson = LobattoIIIA2 = :Trapz
function Mck1integralk!(Rck1::AbstractArray{T,N}, Mck1::AbstractArray{T,N}, 
    ps::Dict{String,Any}, Nstep::Int64; orderRK::Int64=2,
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,

    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,

    is_vth_ode::Bool=true,is_corrections::Vector{Bool}=[true,false,false],
    i_iter_rs2::Int64=10,is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false,
    alg_embedded::Symbol=:Trapz,is_MjMs_max::Bool=false, 
    is_moments_out::Bool=false,is_Cerror_dtnIKTs::Bool=true,
    is_dtk_GKMM_dtIK::Bool=true,rtol_dtsa::T=1e-8,ratio_dtk1::T=1.2) where{T,N}


    # ratio_dtk1 = dtk / dtk
    # ps_copy = deepcopy(ps)
    tk = deepcopy(ps["tk"])
    tauk = ps["tauk"]

    nsk1 = ps["ns"]
    mak1 = ps["ma"]
    Zqk1 = ps["Zq"]
    spices = ps["spices"]

    nak = ps["nk"]
    Iak = ps["Ik"]
    Kak = ps["Kk"]
    vathk = ps["vthk"]               # vath_(k)
    sak1 = deepcopy(ps["sak"])
    dtsabk1 = deepcopy(ps["dtsabk"])

    nnv = ps["nnv"]
    # nc0, nck = ps["nc0"], ps["nck"]
    ocp = ps["ocp"]
    vGdom = ps["vGm"]

    # vhe = ps["vhe"]
    # vhk = ps["vhk"]
    # nvlevel0, nvlevele0 = ps["nvlevel0"], ps["nvlevele0"]          # nvlevele = nvlevel0[nvlevele0]

    nModk = ps["nModk"]
    nMjMs = ps["nMjMs"]
    naik = ps["naik"]
    uaik = ps["uaik"]
    vthik = ps["vthik"]

    LMk = ps["LMk"]
    # muk, Mμk, Munk, Mun1k, Mun2k = ps["muk"], ps["Mμk"], ps["Munk"], ps["Mun1k"], ps["Mun2k"]
    # w3k, err_dtnIK, DThk = ps["w3k"], ps["err_dtnIK"], ps["DThk"]         # w3k = vₜₕ⁻¹∂ₜvₜₕ
    Mhck = ps["Mhck"]
    edtnIKTs = deepcopy(ps["edtnIKTsk"])
    CRDn = ps["CRDnk"]                          # The discrete error of number density conservation
    Nspan_optim_nuTi = Nspan_nuTi_max[2] * ones(T,3)              # [nai, uai, vthi]

    Rdtsabk1 = 0.0

    nvG = 2 .^ nnv .+ 1
    LM1k = maximum(LMk) + 1

    DThk1 = zeros(T, ns)             # δT̂
    is_nMod_renew = zeros(Bool,nsk1)

    k = 0       # initial step to calculate the values `ℭ̂ₗ⁰` and `w3k = Rdtvath = 𝒲 / 3`
    # where `is_update_nuTi = false` and `nai, uai, vthik` are convergent according to `fvL`

    # # # # Updating the conservative momentums `n, I, K`
    nak1 = deepcopy(nak)
    Iak1 = deepcopy(Iak)
    Kak1 = deepcopy(Kak)
    vathk1 = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes

    # Structure preserving
    nModk1 = deepcopy(nModk)
    # if nModk1 ≠ nModk
    #     Mhinitial_fM!(nMjMs,nsk1,nModk1;njMs_fix=njMs_fix,
    #               is_MjMs_max=is_MjMs_max,is_LM1_full=is_LM1_full)
    # end

    Mhck1 = deepcopy(Mhck)
    @show Ia, Mck1[1,2,:]    
    # edsdrgrrgrg
    err_Rck12 = zero.(Rck1[1:njMs, :, 1:2])      # Storages for two-spices collision `Cab`
    if nsk1 ≥ 3
        Rck12 = zero.(Rck1[:,:,1:2])
        edtnIKTs2 = zero.(edtnIKTs[:,1:2])
        DThk12 = zeros(T,2)
        naik2 = Vector{AbstractVector{T}}(undef,2)
        uaik2 = Vector{AbstractVector{T}}(undef,2)
        vthik2 = Vector{AbstractVector{T}}(undef,2)
    end

    dtk = ps["dt"]
    @show dtk, i_iter_rs2, alg_embedded
    # RdMsk = zeros(T, 3, ns)                      # [RdIak, RdKak, Rdvathk]
    # δvathi = zeros(T,ns)
    # criterions = [ps["DThk"]; ps["err_dtnIK"]; δvathi] 
    
    Rck = deepcopy(Rck1)
    tauk = ps["tauk"]
    ρk1 = mak1 .* nak1

    dtKIak = zeros(T,2,ns)                #  [[K, I], ns]
    dtKIak[1,:] = Rck1[2,1,:] * CMcKa     # K
    dtKIak[2,:] = Rck1[1,2,:]             # I

    @show is_fvL_CP,dtk
    @show 0, dtKIak[1,:]


    Mck = deepcopy(Mck1)
    vathk1i = deepcopy(vathk)          # zeros(T,nsk1)
    if orderRK ≤ 2
        count = 0
        k = 1
        done = true
        while done
            # parameters
            tk = deepcopy(ps["tk"])
            dtk = deepcopy(ps["dt"])
            Nt_save = ps["Nt_save"]
            count_save = ps["count_save"]
            
            # println()
            println("**************------------******************------------*********")
            printstyled("k=",k,",tk,dt,Rdt=",fmtf2.([ps["tk"],dtk,dtk/ps["tk"]]),"\n";color=:blue)

            # @show 0, Mhck1[1][1,2], Mhck1[2][1,2]


            # a = sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
            # @show 70, a
            # b = sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
            # @show 70, b
            # if abs(a) + abs(b) > 1e-10
            #     @show a,b
            #     # sdnghghghg
            # end

            if nsk1 == 2
                dtk1 = Mck1integrali_rs2!(Mck1,Rck1,edtnIKTs,err_Rck12,Mhck1,
                    nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
                    CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, 
                    nModk1, nMjMs, DThk1, Iak1, Kak1, vathk1i, 
                    Mck, Rck, nak, vathk, Nspan_optim_nuTi, dtk;
                    Nspan_nuTi_max=Nspan_nuTi_max,
                    NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
                    restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                    autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                    rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                    optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                    is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                    is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
                    
                    eps_fup=eps_fup,eps_flow=eps_flow,
                    maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                    abstol=abstol,reltol=reltol,
                    vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                    is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                    
                    is_vth_ode=is_vth_ode, is_corrections=is_corrections,
                    i_iter_rs2=i_iter_rs2,is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                    is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                    dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)

                # @show 73, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
                # @show 73, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
                # @show vathk1,uaik[2][1],Mck1[1,2,2]
    
                # # # Updating the parameters `nModk1`
                if prod(nModk1) ≥  2
                        
                    # reducing the number of `nModk1` according to and updating `naik, uaik, vthik`
                    nMod_update!(is_nMod_renew, nModk1, naik, uaik, vthik, nsk1)

                    if is_fixed_timestep == false
                        if sum(is_nMod_renew) > 0

                            # Updating `Mhck1` and `Mck1` owing to the reduced parameters `naik, uaik, vthik`
                            Mhck1 = MsnntL2fL0(Mhck1,nMjMs,LMk,nsk1,naik,uaik,vthik,nModk1;is_renorm=is_renorm)
                            MckMhck!(Mck1,Mhck1,ρa,vathk1,LMk,nsk1,nMjMs)

                            # Updating `Rck1` owing to the reduced parameters `naik, uaik, vthik`
                            uak1 = Mck1[1,2,:] ./ ρk1
                            # @show 78, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
                            # @show 78, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
                            dtk1 = dtMcab2!(Rck1,edtnIKTs,err_Rck12,
                                    nvG, ocp, vGdom,  LMk, LM1k, naik, uaik, vthik, 
                                    CΓ, εᵣ, mak1, Zqk1, spices, nak1, uak1, vathk1,
                                    nModk1,nMjMs,DThk1, dtk1;
                                    is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                                    rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                                    is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,
                                    is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,

                                    eps_fup=eps_fup,eps_flow=eps_flow,
                                    maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                                    abstol=abstol,reltol=reltol,
                                    vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                                    is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                                    
                                    is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                                    is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)

                                    # @show 79, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
                                    # @show 79, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
                            tau_fM!(tauk, mak1, Zqk1, spices, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                            printstyled("0: Updating the time scale, tau=", tauk,color=:green,"\n")
                            count = 0
                        else
                            count += 1
                            if count == count_tau_update
                                tau_fM!(tauk, mak1, Zqk1, spices, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                                printstyled("1: Updating the time scale, tau=", tauk,color=:green,"\n")
                                count = 0
                            end
                        end
                    else
                        fujjkkkk
                    end
                elseif is_fixed_timestep == false
                    count += 1
                    if count == count_tau_update
                        tau_fM!(tauk, mak1, Zqk1, spices, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                        printstyled("2: Updating the time scale, tau=", tauk,color=:green,"\n")
                        count = 0
                    end
                end
                CRDn[1] = min(abs(edtnIKTs[1,1]),abs(edtnIKTs[1,2]))
            else
                dtk1 = Mck1integrali_rs2!(Mck1,Rck1,edtnIKTs,
                    Rck12,edtnIKTs2,CRDn,err_Rck12,DThk12,naik2,uaik2,vthik2,Mhck1,
                    nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
                    CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, 
                    nsk1, nModk1, nMjMs, DThk1, Iak1, Kak1, vathk1i, 
                    Mck, Rck, nak, vathk, Nspan_optim_nuTi, dtk;
                    Nspan_nuTi_max=Nspan_nuTi_max,
                    NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
                    restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                    autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                    rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                    optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                    is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                    is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
                    eps_fup=eps_fup,eps_flow=eps_flow,
                    maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                    abstol=abstol,reltol=reltol,
                    vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                    is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                    is_vth_ode=is_vth_ode, is_corrections=is_corrections,
                    i_iter_rs2=i_iter_rs2,is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                    is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                    dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)

                # # # Updating the parameters `nModk1`
                if prod(nModk1) ≥  2
                    # reducing the number of `nModk1` according to and updating `naik, uaik, vthik`
                    nMod_update!(is_nMod_renew, nModk1, naik, uaik, vthik, nsk1)

                    if is_fixed_timestep == false
                        if sum(is_nMod_renew) > 0

                            # Updating `Mhck1` and `Mck1` owing to the reduced parameters `naik, uaik, vthik`
                            Mhck1 = MsnntL2fL0(Mhck1,nMjMs,LMk,nsk1,naik,uaik,vthik,nModk1;is_renorm=is_renorm)
                            MckMhck!(Mck1,Mhck1,ρa,vathk1,LMk,nsk1,nMjMs)

                            # Updating `Rck1` owing to the reduced parameters `naik, uaik, vthik`
                            uak1 = Mck1[1,2,:] ./ ρk1
                            if nsk1 == 2
                            else
                            end
                            dtk1 = dtMcabn!(Rck1,edtnIKTs,
                                    Rck12,edtnIKTs2,CRDn,err_Rck12,DThk12,naik2,uaik2,vthik2,
                                    nvG, ocp, vGdom,  LMk, LM1k, naik, uaik, vthik, 
                                    CΓ, εᵣ, mak1, Zqk1, spices, nak1, uak1, vathk1,
                                    nsk1,nModk1,nMjMs,DThk1, dtk1;
                                    is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                                    rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                                    is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,
                                    is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
                                    eps_fup=eps_fup,eps_flow=eps_flow,
                                    maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                                    abstol=abstol,reltol=reltol,
                                    vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                                    is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                                    is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                                    is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)

                            tau_fM!(tauk, mak1, Zqk1, spices, nak1, vathk1, Coeff_tau, 
                                    naik2, vthik2, naik, vthik, nModk1, nsk1)
                            printstyled("0: Updating the time scale, tau=", tauk,color=:green,"\n")
                            count = 0
                        else
                            count += 1
                            if count == count_tau_update
                                tau_fM!(tauk, mak1, Zqk1, spices, nak1, vathk1, Coeff_tau, 
                                        naik2, vthik2, naik, vthik, nModk1, nsk1)
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
                        tau_fM!(tauk, mak1, Zqk1, spices, nak1, vathk1, Coeff_tau, 
                                naik2, vthik2, naik, vthik, nModk1, nsk1)
                        printstyled("2: Updating the time scale, tau=", tauk,color=:green,"\n")
                        count = 0
                    end
                end
            end
            @show Ia[2], Iak[2], Iak1[2]

            # Updating `Iak1` and `Kak1` from `Mck1`
            Kak1 = Mck1[2,1,:] * CMcKa 
            Iak1 = Mck1[1,2,:]
            dtKIak[1,:] = Rck1[2,1,:] * CMcKa     # dtKa
            dtKIak[2,:] = Rck1[1,2,:]             # dtIa

            # [ns], Updating the entropy and its change rate with assumpation of dauble-Maxwellian distribution
            entropy_fDM!(sak1,mak1,nak1,vathk1,Iak1,Kak1,nsk1)                        

            # dtsabk1 = dtsak1 + dtsbk1
            # [nsk1 = 2] Iahk = uak1 ./ vathk1
            dtsabk1 = entropy_rate_fDM(mak1,vathk1,Iak1 ./ (ρk1 .* vathk1),dtKIak[2,:],dtKIak[1,:],nsk1)
            
            Rdtsabk1 = dtsabk1 / sum(sak1)
            # Rdtsabk1 = entropyN_rate_fDM(mak1,nak1,vathk1,Iak1,Kak1,dtKIak[2,:],dtKIak[1,:],nsk1)
            # @show Rdtsabk1

            # When `nₖ₊₁ˢ`, `Iₖ₊₁ˢ` and `Kₖ₊₁ˢ` reach convergence, conservations and
            # the constraint equation is satisfied: `K̂ = 3/2 + Î`,

            is_corrections[1] ? nothing : nak = deepcopy(nak1)
            Iak = deepcopy(Iak1)
            Kak = deepcopy(Kak1)
            vathk = deepcopy(vathk1)
            Mck = deepcopy(Mck1)
            Rck = deepcopy(Rck1)
            Mhck = deepcopy(Mhck1)
            
            # Updating the time-step at `(k+1)ᵗʰ` step according to `τTab` and `τRdtIKk1`
            if is_fixed_timestep
                dtk1 = dtk
            else
                if is_dtk_GKMM_dtIK
                    if is_dt_tau
                        dtk1 = dt_tau(dtk1,tauk[1];ratio_dtk1=ratio_dtk1,dt_ratio=dt_ratio)
                    else
                        dtk1 = dt_tau_warn(dtk1,tauk[1];ratio_dtk1=ratio_dtk1,dt_ratio=dt_ratio)
                    end
                    dtk1 = dt_RdtnIK(dtk1,dtKIak,Iak,Kak,nsk1;rtol_DnIK=rtol_DnIK)
                    # dtk1 = dt_RdtnIK(dtk1,dtKIak,min(abs.(Iak),abs.(Iak1)),Kak1,nsk1;rtol_DnIK=rtol_DnIK)
                else
                    dtk1 = min(dtk * ratio_dtk1, dtk1)
                end
            end

            ps["tk"] = tk + dtk
            # ps["tauk"] = tauk
            ps["dt"] = dtk1
            # ps["nak"] = deepcopy(nak1)
            ps["Ik"] = deepcopy(Iak1)
            ps["Kk"] = deepcopy(Kak1)
            # ps["uCk"] = deepcopy(uCk1)
            ps["DThk"] = deepcopy(DThk1)
            ps["naik"] = deepcopy(naik)
            ps["uaik"] = deepcopy(uaik)
            ps["vthik"] = deepcopy(vthik)
            ps["nModk"] = deepcopy(nModk1)
            ps["vGm"] = vGdom

            ps["LMk"] = deepcopy(LMk)
            ps["nMjMs"] = deepcopy(nMjMs)
            ps["Mhck"] = deepcopy(Mhck1)
            ps["sak"] = deepcopy(sak1)
            ps["dtsabk"] = deepcopy(dtsabk1)
            ps["edtnIKTsk"] = deepcopy(edtnIKTs)
            ps["CRDnk"][1] = deepcopy(CRDn[1])

            # Saving the dataset at `(k+1)ᵗʰ` step
            if count_save == Nt_save
                ps["count_save"] = 1
                data_Ms_saving(ps;is_moments_out=is_moments_out,is_Cerror_dtnIKTs=is_Cerror_dtnIKTs)
            else
                ps["count_save"] = count_save + 1
            end

            # Terminating the progrom when I. reaches the maximum time moment; II. number of time step; III. equilibrium state.
            if abs(Rdtsabk1) ≤ rtol_dtsa_terminate
                RDTab33vec[iCase,3] = ps["tk"]
                @warn("The system has reached the equilibrium state when", Rdtsabk1)
                break
            else
                if ps["tk"] > ps["tspan"][2]
                    if abs(tk - ps["tspan"][2]) / tk ≤ rtol_tk
                        RDTab33vec[iCase,3] = tk
                        @warn("The system has reached the maximum time moment at", tk)
                        break
                    else
                        ps["tk"] = ps["tspan"][2]
                        dtk1 = ps["tk"] - tk
                        ps["dt"] = dtk1 
                    end
                else
                    if k > Nstep
                        RDTab33vec[iCase,3] = ps["tk"]
                        @warn("The system has reached the maximum iterative step", Nstep)
                        break
                    end
                end
                k += 1
            end
        end
    else
        errorsddd
    end
end

"""
  A `s`-stage integral at the `kᵗʰ` step with implicit Euler method or Trapezoidal method with `Niter_stage`: 

  Level of the algorithm
    sᵗʰ: the stage level during `kᵗʰ` time step
    i: the inner iteration level during `sᵗʰ` stage
    
  Inputs:
    nak1 = deepcopy(nak)
    Iak1 = deepcopy(Iak)
    Kak1 = deepcopy(Kak)
    vathk1 = deepcopy(vathk)       # vath_(k+1)_(i) which will be changed in the following codes
    Mck1 = deepcopy(Mck)
    Rck1i: = Rck1 .= 0             # which will be changed in the following codes

    During the iterative process `while ⋯ end`

    Inputs:
      :TrapzMck scheme, `2ᵗʰ`-order scheme for `na`, `Ta`, but `0ᵗʰ`-order scheme for other normalzied kinetic moments `Mhck`
        
        Rck1i: = (Rck + Rck1) / 2      
    
      :TrapzMhck scheme, `2ᵗʰ`-order scheme for all kinetic moments `Mck`
      
        Rck1i: = (Rck + Rck1) / 2  ,      for `n, I, K`
        Mhck1: = TrapzMhck(Mhck,Mhck1),   for `Mhck`

      outputs:
           = Rck1       

  Outputs:
    dtk1 = Mck1integrali_rs2!(Mck1,Rck1,edtnIKTs,CRDn,err_Rck12,
        Mhck1, nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
        CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1,
        nsk1, nModk1, nMjMs, DThk1,
        Iak1, Kak1, vathk1i, Mck, Rck, nak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
        is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
        eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
        abstol=abstol,reltol=reltol,
        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
        
        is_vth_ode=is_vth_ode, is_corrections=is_corrections,
        i_iter_rs2=i_iter_rs2, is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
        dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
"""
# [sᵗʰ,i], alg_embedded ∈ [:TrapzMck],    rs = 2, o = 2
# ns ≥ 3
function Mck1integrali_rs2!(Mck1::AbstractArray{T,N},Rck1i::AbstractArray{T,N},edtnIKTs::AbstractArray{T,N2},
    Rck1i2::AbstractArray{T,N},edtnIKTs2::AbstractArray{T,N2},CRDn::AbstractVector{T},
    err_Rck12::AbstractArray{T,N},DThk12::AbstractVector{T},
    naik2::Vector{AbstractVector{T}},uaik2::Vector{AbstractVector{T}},vthik2::Vector{AbstractVector{T}},Mhck1::Vector{Matrix{T}},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2},LMk::Vector{Int64},LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},spices::Vector{Symbol},
    nak1::AbstractVector{T},vathk1::AbstractVector{T},
    nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},DThk1::AbstractVector{T},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},vathk1i::AbstractVector{T},
    Mck::AbstractArray{T,N},Rck::AbstractArray{T,N},nak::AbstractVector{T},vathk::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},dtk::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,

    is_vth_ode::Bool=true,is_corrections::Vector{Bool}=[true,false,false],
    i_iter_rs2::Int64=0,is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false) where{T,N,N2}

    i_iter = 0                  # and Checking the `i = 1` iteration
    # Applying the explicit Euler step
    δvathi = ones(T,nsk1)        # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
    δvathk1 = zeros(T,nsk1)      # = vathk1 ./ vathk1
    # Rck1i .= 0.0

    dtk1 = 1dtk
    dtk1 = Mck1integral!(Mck1,Rck1i,Mck,edtnIKTs,
        Rck1i2,edtnIKTs2,CRDn,err_Rck12,DThk12,naik2,uaik2,vthik2,Mhck1,
        nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
        CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, 
        nsk1, nModk1, nMjMs, DThk1, Iak1, Kak1, δvathk1, Nspan_optim_nuTi, dtk1;
        Nspan_nuTi_max=Nspan_nuTi_max,
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
        is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
        eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
        abstol=abstol,reltol=reltol,
        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
        is_vth_ode=is_vth_ode,
        is_corrections=is_corrections,is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
        dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
    vathk1i[:] = deepcopy(vathk1)
    # Rck1Ex = deepcopy(Rck1i)
    
    # @show 71, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
    # @show 71, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]

    # # If `i_iter_rs2 ≤ 0`, then degenerate into the explicit Euler method (ExEuler)
    δvathi_up = zeros(T,nsk1)
    while i_iter < i_iter_rs2
        i_iter += 1
        nak1[:] = deepcopy(nak)
        vathk1[:] = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
        Rck1i[:,:,:] = (Rck + Rck1i) / 2         # Rck1k = Rc_(k+1/2)
        dtk1 = 1dtk
        
        dtk1 = Mck1integral!(Mck1,Rck1i,Mck,edtnIKTs,
            Rck1i2,edtnIKTs2,CRDn,err_Rck12,DThk12,naik2,uaik2,vthik2,Mhck1,
            nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
            CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, 
            nsk1, nModk1, nMjMs, DThk1, Iak1, Kak1, δvathk1, Nspan_optim_nuTi, dtk1;
            Nspan_nuTi_max=Nspan_nuTi_max,
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
            restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
            rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
            optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
            is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
            is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
            is_vth_ode=is_vth_ode,
            is_corrections=is_corrections,is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
            is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
            dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
        δvathi = vathk1i ./ vathk1 .- 1                # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
        ratio_vathi = δvathi - δvathi_up

        # # Rck1 = Rck1i
        if norm(ratio_vathi) ≤ rtol_vthi || norm(δvathi) ≤ atol_vthi
            break
        end
        vathk1i[:] = deepcopy(vathk1)
    end

    # Rck1k = Rc_(k+1/2)
    if i_iter ≥ i_iter_rs2
        @warn(`The maximum number of iteration reached before the Trapz method to be convergence!!!`)
    end
    # @show 722, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
    # @show 722, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
    # @show vathk1,uaik[2][1],Mck1[1,2,2]
    return dtk1
end

# ns = 2
function Mck1integrali_rs2!(Mck1::AbstractArray{T,N},Rck1i::AbstractArray{T,N},
    edtnIKTs::AbstractArray{T,N2},err_Rck12::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2},LMk::Vector{Int64},LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},spices::Vector{Symbol},
    nak1::AbstractVector{T},vathk1::AbstractVector{T},
    nModk1::Vector{Int64},nMjMs::Vector{Int64},DThk1::AbstractVector{T},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},vathk1i::AbstractVector{T},
    Mck::AbstractArray{T,N},Rck::AbstractArray{T,N},
    nak::AbstractVector{T},vathk::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},dtk::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,

    is_vth_ode::Bool=true,is_corrections::Vector{Bool}=[true,false,false],
    i_iter_rs2::Int64=0,is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false) where{T,N,N2}

    i_iter = 0                  # and Checking the `i = 1` iteration
    # Applying the explicit Euler step
    δvathi = ones(T,2)        # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
    δvathk1 = zeros(T,2)      # = vathk1 ./ vathk1
    # Rck1i .= 0.0
    dtk1 = 1dtk
    
    dtk1 = Mck1integral!(Mck1,Rck1i,Mck,edtnIKTs,err_Rck12,Mhck1,
        nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
        CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, 
        nModk1, nMjMs, DThk1, Iak1, Kak1, δvathk1, Nspan_optim_nuTi, dtk1;
        Nspan_nuTi_max=Nspan_nuTi_max,
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
        is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
        eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
        abstol=abstol,reltol=reltol,
        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
        is_vth_ode=is_vth_ode,
        is_corrections=is_corrections,is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
        dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
    vathk1i[:] = deepcopy(vathk1)
    # @show 71, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
    # @show 71, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
    # Rck1Ex = deepcopy(Rck1i)
    
    # # If `i_iter_rs2 ≤ 0`, then degenerate into the explicit Euler method (ExEuler)
    δvathi_up = zeros(T,2)
    if orderRK == 1
        while i_iter < i_iter_rs2
            i_iter += 1
            nak1[:] = deepcopy(nak)
            vathk1[:] = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
            # Rck1i[:,:,:] = (Rck + Rck1i) / 2        # Rck1k = Rc_(k+1/2)
            dtk1 = 1dtk
            
            dtk1 = Mck1integral!(Mck1,Rck1i,Mck,edtnIKTs,err_Rck12,Mhck1, 
                nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
                CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, 
                nModk1, nMjMs, DThk1, Iak1, Kak1, δvathk1, Nspan_optim_nuTi, dtk1;
                Nspan_nuTi_max=Nspan_nuTi_max,
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
                restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
                eps_fup=eps_fup,eps_flow=eps_flow,
                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                abstol=abstol,reltol=reltol,
                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                is_vth_ode=is_vth_ode,
                is_corrections=is_corrections,is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
            δvathi = vathk1i ./ vathk1 .- 1                # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
            ratio_vathi = δvathi - δvathi_up
    
            # # Rck1 = Rck1i
            if norm(ratio_vathi) ≤ rtol_vthi || norm(δvathi) ≤ atol_vthi
                break
            end
            # @show 72, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
            # @show 72, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
            vathk1i[:] = deepcopy(vathk1)
            @show i_iter, δvathk1, δvathi
        end
    elseif orderRK == 2
        while i_iter < i_iter_rs2
            i_iter += 1
            nak1[:] = deepcopy(nak)
            vathk1[:] = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
            Rck1i[:,:,:] = (Rck + Rck1i) / 2        # Rck1k = Rc_(k+1/2)
            dtk1 = 1dtk
            
            dtk1 = Mck1integral!(Mck1,Rck1i,Mck,edtnIKTs,err_Rck12,Mhck1, 
                nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
                CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, 
                nModk1, nMjMs, DThk1, Iak1, Kak1, δvathk1, Nspan_optim_nuTi, dtk1;
                Nspan_nuTi_max=Nspan_nuTi_max,
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
                restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
                eps_fup=eps_fup,eps_flow=eps_flow,
                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                abstol=abstol,reltol=reltol,
                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                is_vth_ode=is_vth_ode,
                is_corrections=is_corrections,is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
            δvathi = vathk1i ./ vathk1 .- 1                # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
            ratio_vathi = δvathi - δvathi_up
    
            # # Rck1 = Rck1i
            if norm(ratio_vathi) ≤ rtol_vthi || norm(δvathi) ≤ atol_vthi
                break
            end
            # @show 72, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
            # @show 72, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
            vathk1i[:] = deepcopy(vathk1)
            @show i_iter, δvathk1, δvathi
        end
    else
        dfcvghggh
    end
    # Rck1k = Rc_(k+1/2)
    if i_iter ≥ i_iter_rs2
        @warn(`The maximum number of iteration reached before the Trapz method to be convergence!!!`)
    end
    # @show 722, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
    # @show 722, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
    # @show vathk1,uaik[2][1],Mck1[1,2,2]
    return dtk1
end

"""
  Integral at the `sᵗʰ` stage with implicit Euler method with `Niter_stage`: 

  Level of the algorithm
    i=0ᵗʰ: the inner iteration level during `sᵗʰ` stage
    
  Inputs:
    nak1 = deepcopy(nak)
    Iak1 = deepcopy(Iak)
    Kak1 = deepcopy(Kak)
    vathk:
    vathk1 = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
    Mck1 = deepcopy(Mck)
    Rck1: = Rck,   i = 0, the explicit Euler method          when input
          = Rck1i, i ≥ 1, the implicit Euler method
          = (Rck + Rck1i)/2, i ≥ 1, the Trapezoidal method
          = Rck1                                             when outputs
    Rck1[njMs+1,1,:]                # `w3k = Rdtvath = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`


  Outputs:
    dtk = Mck1integral!(Mck1,Rck1,Mck,edtnIKTs,err_Rck12,Mhck1, 
        nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
        CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1,
        nsk1, nModk1, nMjMs, DThk1, Iak1, Kak1, δvathk1, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
        is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
        eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
        abstol=abstol,reltol=reltol,
        vadaptlevels=vadaptlevels,gridv_type=gridv_type,
        is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
        is_vth_ode=is_vth_ode,
        is_corrections=is_corrections,is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
        dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
"""

# [iᵗʰ], alg_embedded == ∈ [:ExEuler], `Mck1= Mck`
# # ns ≥ 3
function Mck1integral!(Mck1::AbstractArray{T,N},Rck1::AbstractArray{T,N},Mck::AbstractArray{T,N},
    edtnIKTs::AbstractArray{T,N2},Rck12::AbstractArray{T,N},
    edtnIKTs2::AbstractArray{T,N2},CRDn::AbstractVector{T},
    err_Rck12::AbstractArray{T,N},DThk12::AbstractVector{T},
    naik2::Vector{AbstractVector{T}},uaik2::Vector{AbstractVector{T}},vthik2::Vector{AbstractVector{T}},Mhck1::Vector{Matrix{T}},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2},LMk::Vector{Int64},LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},spices::Vector{Symbol},
    nak1::AbstractVector{T},vathk1::AbstractVector{T},
    nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},DThk1::AbstractVector{T},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},δvathk1::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},dtk1::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,

    is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false) where{T,N,N2}
    
    ρk1 = mak1 .* nak1
    # `vathk1 = zeros(nsk1)`

    Mck1integral0!(Mck1,Mck,Rck1,nsk1,dtk1)

    # Calculate the parameters `nak1,vathk1,Iak1,Kak1` from `Mck1`
    nIKT_update!(nak1,vathk1,Iak1,Kak1,δvathk1,mak1,nsk1,Mck1;
                is_vth_ode=is_vth_ode,is_corrections_na=is_corrections[1])
    RerrDKab = sum(Kak1) / Kab0 - 1
    if abs(RerrDKab) > rtol_nIK_warn
        @warn("Energy conservation is not achieved:",RerrDKab)
        if abs(RerrDKab) > rtol_nIK_error
            @error("Energy conservation is not achieved:",RerrDKab)
        end
    end

    # # Computing the re-normalized moments
    if is_Ms_nuT
        # Renew the values of `Mhck1` according to the quantities of `Mck1`
        MhcknuT!(Mhck1,Mck1,ρk1,vathk1,LMk,nsk1,nMjMs,nModk1;is_renorm=is_renorm)
    else
        MhckMck!(Mhck1,Mck1[1:njMs,:,:],ρk1,LMk,nsk1,nMjMs,vathk1)
    end

    # # # Computing the re-normalized kinetic dissipative forces `Rhck1`
    # Rhck1 = deepcopy(Mhck1)
    # MhckMck!(Rhck1,Rck1[1:njMs,:,:],ρk1,LMk,nsk1,nMjMs,vathk1)

    # uk = Iak1 ./ ρk1
    if prod(nModk1) == 1
        # # Calculate the parameters `nai, uai, vthi`
        submoment!(naik, uaik, vthik, Mhck1, nsk1)

        # Msnnt = deepcopy(Mhck1)
        # Msnnt = MsnntL2fL0(Msnnt,nMjMs,LMk,nsk1,uaik;is_renorm=is_renorm)
        # @show Msnnt - Mhck1
        # edsfghbnm1

        dtk1 = dtMcabn!(Rck1,edtnIKTs,
            Rck12,edtnIKTs2,CRDn,err_Rck12,DThk12, nvG, ocp, vGdom, LMk, LM1k,
            CΓ, εᵣ, mak1, Zqk1, spices, nak1, Iak1 ./ ρk1, vathk1,
            nsk1, nMjMs, DThk1, dtk1;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
            rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM,is_boundaryv0=is_boundaryv0,
            is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
            is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
            is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
    else
        if dtk_order_Rc == :min && is_dtk_order_Rcaa == false
            dtk1 = dtMcab!(Rck1,edtnIKTs,
                Rck12,edtnIKTs2,CRDn,err_Rck12,DThk12,naik2,uaik2,vthik2,Mhck1, 
                nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
                CΓ, εᵣ, mak1, Zqk1, spices, nak1, Iak1 ./ ρk1, vathk1,
                nsk1, nModk1, nMjMs, DThk1, Nspan_optim_nuTi, dtk1;
                Nspan_nuTi_max=Nspan_nuTi_max,
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
                restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
                eps_fup=eps_fup,eps_flow=eps_flow,
                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                abstol=abstol,reltol=reltol,
                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
        else
            Mck12 = deepcopy(Mck1[:,:,1:2])
            dtk1 = dtMcab!(Rck1,edtnIKTs,
                Rck12,edtnIKTs2,CRDn,err_Rck12,DThk12,naik2,uaik2,vthik2,Mhck1, 
                nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
                CΓ, εᵣ, mak1, Zqk1, spices, Mck1, Mck12, nak1, Iak1 ./ ρk1, vathk1,
                nsk1, nModk1, nMjMs, DThk1, Nspan_optim_nuTi, dtk1;
                Nspan_nuTi_max=Nspan_nuTi_max,
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
                restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
                eps_fup=eps_fup,eps_flow=eps_flow,
                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                abstol=abstol,reltol=reltol,
                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
        end
        Rck1[njMs+1,1,:] .*= vathk1      # ∂ₜvₜₕ
    end
    # @show 74, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
    # @show 74, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
    return dtk1
end

# ns = 2
function Mck1integral!(Mck1::AbstractArray{T,N},Rck1::AbstractArray{T,N},Mck::AbstractArray{T,N},
    edtnIKTs::AbstractArray{T,N2},err_Rck12::AbstractArray{T,N},Mhck1::Vector{Matrix{T}},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2},LMk::Vector{Int64},LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},spices::Vector{Symbol},
    nak1::AbstractVector{T},vathk1::AbstractVector{T},
    nModk1::Vector{Int64},nMjMs::Vector{Int64},DThk1::AbstractVector{T},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},δvathk1::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},dtk1::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],nsk1::Int64=2,
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,

    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9,

    is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false) where{T,N,N2}
    
    ρk1 = mak1 .* nak1
    # `vathk1 = zeros(nsk1)`
    
    Mck1integral0!(Mck1,Mck,Rck1,nsk1,dtk1)
    
    # Calculate the parameters `nak1,vathk1,Iak1,Kak1` from `Mck1`
    nIKT_update!(nak1,vathk1,Iak1,Kak1,δvathk1,mak1,nsk1,Mck1;
                is_vth_ode=is_vth_ode,is_corrections_na=is_corrections[1])

    RerrDKab = sum(Kak1) / Kab0 - 1
    if abs(RerrDKab) > rtol_nIK_warn
        @warn("Energy conservation is not achieved:",RerrDKab)
        if abs(RerrDKab) > rtol_nIK_error
            @error("Energy conservation is not achieved:",RerrDKab)
        end
    end

    # # Computing the re-normalized moments 
    if is_Ms_nuT                                # for `nModk1 = 1`
        # Renew the values of `Mhck1` according to the quantities of `Mck1`
        MhcknuT!(Mhck1,Mck1,ρk1,vathk1,LMk,nsk1,nMjMs,nModk1;is_renorm=is_renorm)
    else
        MhckMck!(Mhck1,Mck1[1:njMs,:,:],ρk1,LMk,nsk1,nMjMs,vathk1)
    end
    # @show 3, Mhck1[1][1,2], Mhck1[2][1,2] 

    # # # Computing the re-normalized kinetic dissipative forces `Rhck1`
    # Rhck1 = deepcopy(Mhck1)
    # MhckMck!(Rhck1,Rck1[1:njMs,:,:],ρk1,LMk,nsk1,nMjMs,vathk1)

    # uk = Iak1 ./ ρk1
    if prod(nModk1) == 1
        # # Calculate the parameters `nai, uai, vthi`
        submoment!(naik, uaik, vthik, Mhck1, nsk1)

        # Msnnt = deepcopy(Mhck1)
        # Msnnt = MsnntL2fL0(Msnnt,nMjMs,LMk,nsk1,uaik;is_renorm=is_renorm)
        # @show Msnnt - Mhck1
        # # edsfghbnm2

        dtk1 = dtMcab2!(Rck1,edtnIKTs,err_Rck12, nvG, ocp, vGdom, LMk, LM1k,
            CΓ, εᵣ, mak1, Zqk1, spices, nak1, Iak1 ./ ρk1, vathk1, nMjMs, DThk1, dtk1;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
            rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM,is_boundaryv0=is_boundaryv0,
            is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f,
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
            is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
            is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
    else
        if dtk_order_Rc == :min && is_dtk_order_Rcaa == false
            dtk1 = dtMcab!(Rck1,edtnIKTs,err_Rck12,Mhck1,  
                nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
                CΓ, εᵣ, mak1, Zqk1, spices, nak1, Iak1 ./ ρk1, vathk1,
                nModk1, nMjMs, DThk1, Nspan_optim_nuTi, dtk1;
                Nspan_nuTi_max=Nspan_nuTi_max,
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
                restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
                eps_fup=eps_fup,eps_flow=eps_flow,
                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                abstol=abstol,reltol=reltol,
                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
        else
            dtk1 = dtMcab!(Rck1,edtnIKTs,err_Rck12,Mhck1,  
                nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
                CΓ, εᵣ, mak1, Zqk1, spices, Mck1, nak1, Iak1 ./ ρk1, vathk1,
                nModk1, nMjMs, DThk1, Nspan_optim_nuTi, dtk1;
                Nspan_nuTi_max=Nspan_nuTi_max,
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
                restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
                is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
                eps_fup=eps_fup,eps_flow=eps_flow,
                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                abstol=abstol,reltol=reltol,
                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit,
                is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa)
        end
    end
    Rck1[njMs+1,1,:] .*= vathk1      # ∂ₜvₜₕ
    # @show 74, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
    # @show 74, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
    
    return dtk1
end


"""
  The `iᵗʰ` iteration of with Euler method or Trapezoidal method: 

  Inputs:
    Mck1::AbstractArray{T,3} = Mck, which will be changed
    Rck1: = Rvthk1k^3 * Rck,   i = 0, the explicit Euler method
          = Rvthk1i^3 * Rck1i, i ≥ 1, the implicit Euler method
          = (Rvthk1k^3 * Rck + Rvthk1i^3 * Rck1i)/2, i ≥ 1, the Trapezoidal method
    Rck1[njMs+1,1,:]                    # `w3k1 = Rdtvath = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`
    Rvthk1k = vathk1 / vathk
    Rvthk1i = vathk1i1 / vathk1i
    Rck1[njMs+1,1,:]                    # `w3k1 = Rdtvath = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`

  Outputs:
    Mck1integral0!(Mck1,Mck,Rck1,nsk1,dtk)
"""

# [], alg_embedded::Symbol ∈ [:ExEuler, :ImEuler, :Trapz]
function Mck1integral0!(Mck1::AbstractArray{T,N},Mck::AbstractArray{T,N},Rck1::AbstractArray{T,N},nsk1::Int64,dtk::T) where{T,N}
    
    for isp in 1:nsk1
        Mck1[:,:,isp] = Mck[:,:,isp] + dtk * Rck1[:,:,isp]
    end
end



