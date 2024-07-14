
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
    Mck1integralk!(Mck, pst0, Nstep;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        is_vth_ode=is_vth_ode, is_corrections=is_corrections, i_iter_rs2=i_iter_rs2,
        Nstage=Nstage, Nstage_improved=Nstage_improved, 
        s_iter_max=s_iter_max, alg_embedded=alg_embedded,
        is_optimδtfvL=is_optimδtfvL,residualMethod_FP0D=residualMethod_FP0D, 
        orders=order_dvδtf, is_boundv0=is_boundv0, Nsmooth=Nsmooth,
        order_smooth=order_smooth, order_smooth_itp=order_smooth_itp, order_nvc_itp=order_nvc_itp,
        abstol_Rdy=abstol_Rdy, k_δtf=k_δtf, Nitp=Nitp,
        nvc0_limit=nvc0_limit, L1nvc_limit=L1nvc_limit,is_LM_const=is_LM_const,
        is_moments_out=is_moments_out,is_MjMs_max=is_MjMs_max)

"""

# [k,s,i], alg_embedded ∈ [:Trapz, :ImMidpoint, :Range2, :Heun2, Raslton2, :Alshina2], o = 2
# :ExMidpoint = :Range2 
# :CN = :CrankNicolson = LobattoIIIA2 = :Trapz
function Mck1integralk!(Rck1::AbstractArray{T,N}, Mck1::AbstractArray{T,N}, ps::Dict{String,Any}, Nstep::Int64; 
    NL_solve::Symbol=:NLsolve, 
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,

    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,

    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false,jMax::Int64=1,

    L_shape::Int64=0,eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,is_out_shape::Bool=false,
    abstol::Float64=epsT5,reltol::Float64=1e-5,vadaptlevels::Int=4,
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],
    is_warn::Bool=false,is_nMod_adapt::Bool=false,

    i_iter_rs2::Int64=10,alg_embedded::Symbol=:Trapz, 
    rtol_DnIK::T=0.1,is_dtk_GKMM_dtIK::Bool=true,
    i_iter_rs3::Int64=10,s_iter_max::Int64=0,Nstage::Int64=2,Nstage_improved::Symbol=:last,
    is_extrapolate_FLn::Bool=true,is_optimδtfvL::Bool=false,residualMethod_FP0D::Int64=1, 
    orders::Int64=2, is_boundv0::Vector{Bool}=[true, false, false], Nsmooth::Int=3,
    order_smooth::Int64=3, order_smooth_itp::Int64=1, order_nvc_itp::Int64=4,
    abstol_Rdy::AbstractVector{T}=[0.45, 0.45, 0.45], k_δtf::Int64=2, Nitp::Int64=10,
    nvc0_limit::Int64=4, L1nvc_limit::Int64=3, is_LM_const::Bool=false,
    is_moments_out::Bool=false, is_MjMs_max::Bool=false) where{T,N}

    ratio_dtk = 1.1                  # ratio_dtk1 = dtk / dtk
    # ps_copy = deepcopy(ps)
    tk = deepcopy(ps["tk"])
    nsk1 = ps["ns"]
    mak1 = ps["ma"]
    Zqk1 = ps["Zq"]
    nak = ps["nk"]
    Iak = ps["Ik"]
    Kak = ps["Kk"]
    vathk = ps["vthk"]               # vath_(k)

    nnv = ps["nnv"]
    nc0, nck, ocp = ps["nc0"], ps["nck"], ps["ocp"]
    vGdom = ps["vGm"]

    vhk = ps["vhk"]
    vhe = ps["vhe"]
    nvlevel0, nvlevele0 = ps["nvlevel0"], ps["nvlevele0"]          # nvlevele = nvlevel0[nvlevele0]

    nModk = ps["nModk"]
    naik = ps["naik"]
    uaik = ps["uaik"]
    vthik = ps["vthik"]

    LMk = ps["LMk"]
    muk, Mμk, Munk, Mun1k, Mun2k = ps["muk"], ps["Mμk"], ps["Munk"], ps["Mun1k"], ps["Mun2k"]
    # w3k, err_dtnIK, DThk = ps["w3k"], ps["err_dtnIK"], ps["DThk"]         # w3k = vₜₕ⁻¹∂ₜvₜₕ
    nMjMs = ps["nMjMs"]
    RMcsk = ps["RMcsk"]
    Mhck = ps["Mhck"]

    nvG = 2 .^ nnv .+ 1
    LM1k = maximum(LMk) + 1

    DThk1 = zeros(T, ns)             # δT̂
    is_nMod_renew = zeros(Bool,nsk1)
    tauk = ps["tauk"]

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
    RMcsk1 = deepcopy(Mck1[1:njMs, :, 1])
    err_Rck1 = deepcopy(Rck1[1:njMs, :, :])


    dtk = ps["dt"]
    @show dtk, (Nstage, Nstage_improved), (i_iter_rs2, s_iter_max), alg_embedded
    # RdMsk = zeros(T, 3, ns)                      # [RdIak, RdKak, Rdvathk]
    # δvathi = zeros(T,ns)
    # criterions = [ps["DThk"]; ps["err_dtnIK"]; δvathi]
    
    if alg_embedded == :ExEuler
        δvathk1 = zeros(T,nsk1)      # = vathk1 ./ vathk1
        for k in 1:Nstep
            # parameters
            ps["tk"] += dtk
            Nt_save = ps["Nt_save"]
            count_save = ps["count_save"]

            println()
            println("**************------------******************------------*********")
            printstyled("k=",k,",tk=",ps["tk"],"\n";color=:blue)

            dtk = Mck1integral!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
                RMcsk1, Iak1, Kak1, δvathk1, RMcsk, Mhck, Iak, Kak, vathk, dtk; 
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                orderVconst=orderVconst, 
                vGm_limit=vGm_limit, is_vth_ode=is_vth_ode,
                is_corrections=is_corrections)
            if is_Lagrange
                uCk = ps["uCk"]         # the relative velocity at the `kᵗʰ` step 
                                        # which gives `Mck = Mck'` and `Rck = Rck' in the Lagrange coordinate system.
                uk1 = Iak1 ./ (mak1 .* nak1)
                uCk1 = (vathk1[1] * uk1[2] + vathk1[2] * uk1[1]) / sum(vathk1)
                if abs(uCk1) ≤ errnIKc
                    uCk1 = 0.0
                else
                    # DuCk1 = uCk - uCk1
                    # RDuCk1 = DuCk1 / uCk1               #  (uCk / uCk1) - 1.0
                    is_uCk_renew = uCk_renew(uCk,uCk1)  # Whether renew the relative velocity during the Coulomb collision process
                    if is_uCk_renew
                        ps["uCk"] = deepcopy(uCk1)
    
                        # Update the parameters from coordinate `uCk` to coordinate `uCk1`
                        Mck_renew!(Mck1,uCk,uCk1)
                        dtk = Rck_update!(Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                            nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
                            RMcsk1, Iak1, Kak1, Mck1,dtk; 
                            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                            is_corrections=is_corrections,is_vth_ode=false,is_warn=is_warn)
                    else
                        uCk1 = deepcopy(uCk)
                    end
                end
            end

            # When `nₖ₊₁ˢ`, `Iₖ₊₁ˢ` and `Kₖ₊₁ˢ` reach convergence, conservations and
            # the constraint equation is satisfied: `K̂ = 3/2 + Î`,

            # @show 3, vathk1
            # # updating the distribution function and parameters at `(k+1)ᵗʰ` step
            is_corrections[1] ? nothing : nak = deepcopy(nak1)
            Iak = deepcopy(Iak1)
            Kak = deepcopy(Kak1)
            vathk = deepcopy(vathk1)
            RMcsk = deepcopy(RMcsk1)
            Mhck = deepcopy(Mhck1)
            # if norm(Iak) ≥ epsT1000
            #     djfjkggk
            # end

            # ps["nak"] = deepcopy(nak1)
            ps["Ik"] = deepcopy(Iak1)
            ps["Kk"] = deepcopy(Kak1)
            ps["DThk"] = deepcopy(DThk1)
            ps["naik"] = deepcopy(naik)
            ps["uaik"] = deepcopy(uaik)
            ps["vthik"] = deepcopy(vthik)
            ps["nModk"] = deepcopy(nModk1)
            ps["vGm"] = vGdom

            ps["LMk"] = deepcopy(LMk)
            ps["nMjMs"] = deepcopy(nMjMs)
            ps["Mhck"] = deepcopy(Mhck1)
            # ps["Mhck"]

            # Saving the dataset at `(k+1)ᵗʰ` step
            if count_save == Nt_save
                ps["count_save"] = 1
                data_Ms_saving(ps;is_moments_out=is_moments_out)
            else
                ps["count_save"] = count_save + 1
            end
        end
    else
        Mck = deepcopy(Mck1)
        vathk1i = deepcopy(vathk)          # zeros(T,nsk1)
        if alg_embedded == :ImEuler
            for k in 1:Nstep
                # parameters
                ps["tk"] += dtk
                Nt_save = ps["Nt_save"]
                count_save = ps["count_save"]

                println()
                println("**************------------******************------------*********")
                printstyled("k=",k,",tk=",ps["tk"],"\n";color=:blue)

                # uCk = ps["uCk"]         # the relative velocity at the `kᵗʰ` step 
                #                         # which gives `Mck = Mck'` and `Rck = Rck' in the Lagrange coordinate system.

                Mck1integrali!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                    nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                    mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
                    Iak1, Kak1, vathk1i, Mck, RMcsk, Mhck, nak, Iak, Kak, vathk, dtk;
                    NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                    orderVconst=orderVconst, 
                    vGm_limit=vGm_limit, is_vth_ode=is_vth_ode, 
                    is_corrections=is_corrections, i_iter_rs2=i_iter_rs2)
                # uk1 = Iak1 ./ (mak1 .* nak1)
                # uCk1 = sum(vathk1 .* (Iak1 ./ (mak1 .* nak1))) / sum(vathk1)

                # When `nₖ₊₁ˢ`, `Iₖ₊₁ˢ` and `Kₖ₊₁ˢ` reach convergence, conservations and
                # the constraint equation is satisfied: `K̂ = 3/2 + Î`,

                # # updating the distribution function and parameters at `(k+1)ᵗʰ` step
                is_corrections[1] ? nothing : nak = deepcopy(nak1)
                Iak = deepcopy(Iak1)
                Kak = deepcopy(Kak1)
                vathk = deepcopy(vathk1)
                Mck = deepcopy(Mck1)
                Mhck = deepcopy(Mhck1)
                RMcsk = deepcopy(RMcsk1)
                @show Mhck[1][1:3]
                @show Mhck[2][1:3]

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
                # ps["Mhck"]

                # Saving the dataset at `(k+1)ᵗʰ` step
                if count_save == Nt_save
                    ps["count_save"] = 1
                    data_Ms_saving(ps;is_moments_out=is_moments_out)
                else
                    ps["count_save"] = count_save + 1
                end
            end
        else
            if alg_embedded == :Trapz 
                Rck = deepcopy(Rck1)
                tauk = ps["tauk"]
                count = 0

                dtIKak = zeros(T,2,ns)                #  [[K, I], ns]
                dtIKak[1,:] = Rck1[2,1,:] * CMcKa     # K
                dtIKak[2,:] = Rck1[1,2,:]             # I
                # @show Kak = Mck1[2,1,:] * CMcKa  ./ Ka
                # @show Iak = Mck1[1,2,:] ./ Ia
                for k in 1:Nstep
                    # parameters
                    tk = deepcopy(ps["tk"])
                    dtk = deepcopy(ps["dt"])
                    Nt_save = ps["Nt_save"]
                    count_save = ps["count_save"]
                    
                    if is_dtk_GKMM_dtIK
                        dtk = min(ratio_dtk * dtk, dt_ratio * tauk[1])
                        # @show 1,dtk
                        dtk = dt_DnIK(dtk,dtIKak,Iak,Kak,nsk1;rtol_DnIK=rtol_DnIK)
                        # dtk == dt_ratio * tauk[1] || printstyled("The time step is decided by `dtIKa/IKa` instead of `tauk`!",color=:purple,"\n")
                        # @show 2,dtk
                    else
                        # dtk = min(ratio_dtk1 * dtk, dt_ratio * tauk[1])
                        dtk *= ratio_dtk1
                        @show 1,dtk
                    end
    
                    println()
                    println("**************------------******************------------*********")
                    printstyled("k=",k,",tk,dt,Rdt=",fmtf2.([ps["tk"],dtk,dtk/ps["tk"]]),"\n";color=:blue)

                    # uCk = ps["uCk"]         # the relative velocity at the `kᵗʰ` step 
                    #                         # which gives `Mck = Mck'` and `Rck = Rck' in the Lagrange coordinate system.
    
                    dtk1 = Mck1integrali_rs2!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
                        Iak1, Kak1, vathk1i, Mck, Rck, RMcsk, Mhck, nak, Iak, Kak, vathk, dtk;
                        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                        is_vth_ode=is_vth_ode, is_corrections=is_corrections, 
                        is_warn=true,is_nMod_adapt=false, i_iter_rs2=i_iter_rs2)
                    
                    # Updating `Iak1` and `Kak1` from `Mck1`
                    Kak1 = Mck1[2,1,:] * CMcKa 
                    Iak1 = Mck1[1,2,:]
                    dtIKak[1,:] = Rck1[2,1,:] * CMcKa     # K
                    dtIKak[2,:] = Rck1[1,2,:]             # I

                    # # # Updating the parameters `nModk1`
                    if prod(nModk1) ≥  2
                        if is_nMod_adapt
                            # reducing the number of `nModk1` according to and updating `naik, uaik, vthik`
                            nMod_update!(is_nMod_renew, nModk1, naik, uaik, vthik, nsk1)

                            if is_fixed_timestep == false
                                if sum(is_nMod_renew) > 0

                                    # Updating `Mhck1` and `Mck1` owing to the reduced parameters `naik, uaik, vthik`
                                    Mhck1 = MsnntL2fL0(Mhck1,nMjMs,LMk,nsk1,naik,uaik,vthik,nModk1;is_renorm=is_renorm)
                                    MckMhck!(Mck1,Mhck1,ρa,vathk1,LMk,nsk1,nMjMs)

                                    # Updating `Rck1` owing to the reduced parameters `naik, uaik, vthik`
                                    dtk = dtMcab2!(Rck1,err_Rck1,vhk, nvG, ocp, vGdom, 
                                            nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                                            mak1,Zqk1,nak1,uak1,vathk1,nsk1,nModk1,nMjMs,DThk1, dtk;
                                            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                                            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                                            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                                            rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
                                            is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,
                                            is_check_conservation_dtM=is_check_conservation_dtM,is_fit_f=is_fit_f)

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
                                fujjkkkk
                            end
                        else
                            if is_fixed_timestep == false
                                count += 1
                                if count == count_tau_update
                                    tau_fM!(tauk, mak1, Zqk1, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                                    printstyled("2: Updating the time scale, tau=", tauk,color=:green,"\n")
                                    count = 0
                                end
                            end
                        end
                    elseif is_fixed_timestep == false
                        count += 1
                        if count == count_tau_update
                            tau_fM!(tauk, mak1, Zqk1, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                            printstyled("2: Updating the time scale, tau=", tauk,color=:green,"\n")
                            count = 0
                        end
                    end

                    # Updating the entropy and its change rate
                    # entropy_fDM!(sak1,mak1,nak1,vathk,Iak1,Kak1,nsk1)

                    # dtsabk = entropy_rate_fDM(mak1,vathk,Ihk,dtIa,dtKa,nsk1)
                    # @show sak1, dtsabk / sum(sak1)

                    Rdtsabk1 = entropyN_rate_fDM(mak1,nak1,vathk,Iak1,Kak1,dtIa,dtKa,nsk1)
                    @show Rdtsabk1
    
                    # When `nₖ₊₁ˢ`, `Iₖ₊₁ˢ` and `Kₖ₊₁ˢ` reach convergence, conservations and
                    # the constraint equation is satisfied: `K̂ = 3/2 + Î`,
    
                    # # updating the distribution function and parameters at `(k+1)ᵗʰ` step
                    # if is_Ms_nuT
                    #     MhcknuT!(Mhck1,Mck1,nMjMs,LMk,mak1 .* nak1,vathk1,nsk1,nModk1;is_renorm=is_renorm)
                    # end

                    is_corrections[1] ? nothing : nak = deepcopy(nak1)
                    Iak = deepcopy(Iak1)
                    Kak = deepcopy(Kak1)
                    vathk = deepcopy(vathk1)
                    Mck = deepcopy(Mck1)
                    Rck = deepcopy(Rck1)
                    Mhck = deepcopy(Mhck1)
                    RMcsk = deepcopy(RMcsk1)
    
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
                    # ps["Mhck"]
    
                    # Saving the dataset at `(k+1)ᵗʰ` step
                    if count_save == Nt_save
                        ps["count_save"] = 1
                        data_Ms_saving(ps;is_moments_out=is_moments_out)
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
            elseif alg_embedded == :RK4 || alg_embedded == :LobattoIIIA4
                Rck = deepcopy(Rck1)
                for k in 1:Nstep
                    # parameters
                    ps["tk"] += dtk
                    Nt_save = ps["Nt_save"]
                    count_save = ps["count_save"]
    
                    println()
                    println("**************------------******************------------*********")
                    printstyled("k=",k,",tk=",ps["tk"],"\n";color=:blue)
                
                    Mck1integrali_rs3!(Mck1, Rck1, err_Rck1, Mhck1, vhk, vhe, 
                        nvG, nc0, nck, ocp, vGdom, nvlevele0, nvlevel0, LMk, LM1k,  
                        muk, Mμk, Munk, Mun1k, Mun2k, naik, uaik, vthik, CΓ, εᵣ, 
                        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
                        Iak1, Kak1, vathk1i, Mck, Rck, RMcsk, Mhck, nak, Iak, Kak, vathk, dtk;
                        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                        i_iter_rs2=i_iter_rs2,i_iter_rs3=i_iter_rs3,alg_embedded=alg_embedded)
    
                    # When `nₖ₊₁ˢ`, `Iₖ₊₁ˢ` and `Kₖ₊₁ˢ` reach convergence, conservations and
                    # the constraint equation is satisfied: `K̂ = 3/2 + Î`,
    
                    # # updating the distribution function and parameters at `(k+1)ᵗʰ` step
                    is_corrections[1] ? nothing : nak = deepcopy(nak1)
                    Iak = deepcopy(Iak1)
                    Kak = deepcopy(Kak1)
                    vathk = deepcopy(vathk1)
                    Mck = deepcopy(Mck1)
                    Rck = deepcopy(Rck1)
                    Mhck = deepcopy(Mhck1)
                    RMcsk = deepcopy(RMcsk1)
                    vathk1i = deepcopy(vathk)
    
                    # ps["nak"] = deepcopy(nak1)
                    ps["Ik"] = deepcopy(Iak1)
                    ps["Kk"] = deepcopy(Kak1)
                    ps["DThk"] = deepcopy(DThk1)
                    ps["naik"] = deepcopy(naik)
                    ps["uaik"] = deepcopy(uaik)
                    ps["vthik"] = deepcopy(vthik)
                    ps["nModk"] = deepcopy(nModk1)
                    ps["vGm"] = vGdom
    
                    ps["LMk"] = deepcopy(LMk)
                    ps["nMjMs"] = deepcopy(nMjMs)
                    ps["Mhck"] = deepcopy(Mhck1)
                    # ps["Mhck"]
    
                    # Saving the dataset at `(k+1)ᵗʰ` step
                    if count_save == Nt_save
                        ps["count_save"] = 1
                        data_Ms_saving(ps;is_moments_out=is_moments_out)
                    else
                        ps["count_save"] = count_save + 1
                    end
                end
            elseif alg_embedded == :RK438 || alg_embedded == :CRK44
                Rck = deepcopy(Rck1)
                for k in 1:Nstep
                    # parameters
                    ps["tk"] += dtk
                    Nt_save = ps["Nt_save"]
                    count_save = ps["count_save"]
    
                    println()
                    println("**************------------******************------------*********")
                    printstyled("k=",k,",tk=",ps["tk"],"\n";color=:blue)
                
                    Mck1integrali_rs4!(Mck1, Rck1, err_Rck1, Mhck1, vhk, vhe, 
                        nvG, nc0, nck, ocp, vGdom, nvlevele0, nvlevel0, LMk, LM1k,  
                        muk, Mμk, Munk, Mun1k, Mun2k, naik, uaik, vthik, CΓ, εᵣ, 
                        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
                        Iak1, Kak1, vathk1i, Mck, Rck, RMcsk, Mhck, nak, Iak, Kak, vathk, dtk;
                        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                        i_iter_rs2=i_iter_rs2,i_iter_rs3=i_iter_rs3,alg_embedded=alg_embedded)
    
                    # When `nₖ₊₁ˢ`, `Iₖ₊₁ˢ` and `Kₖ₊₁ˢ` reach convergence, conservations and
                    # the constraint equation is satisfied: `K̂ = 3/2 + Î`,
    
                    # # updating the distribution function and parameters at `(k+1)ᵗʰ` step
                    is_corrections[1] ? nothing : nak = deepcopy(nak1)
                    Iak = deepcopy(Iak1)
                    Kak = deepcopy(Kak1)
                    vathk = deepcopy(vathk1)
                    Mck = deepcopy(Mck1)
                    Rck = deepcopy(Rck1)
                    Mhck = deepcopy(Mhck1)
                    RMcsk = deepcopy(RMcsk1)
                    vathk1i = deepcopy(vathk)
    
                    # ps["nak"] = deepcopy(nak1)
                    ps["Ik"] = deepcopy(Iak1)
                    ps["Kk"] = deepcopy(Kak1)
                    ps["DThk"] = deepcopy(DThk1)
                    ps["naik"] = deepcopy(naik)
                    ps["uaik"] = deepcopy(uaik)
                    ps["vthik"] = deepcopy(vthik)
                    ps["nModk"] = deepcopy(nModk1)
                    ps["vGm"] = vGdom
    
                    ps["LMk"] = deepcopy(LMk)
                    ps["nMjMs"] = deepcopy(nMjMs)
                    ps["Mhck"] = deepcopy(Mhck1)
                    # ps["Mhck"]
    
                    # Saving the dataset at `(k+1)ᵗʰ` step
                    if count_save == Nt_save
                        ps["count_save"] = 1
                        data_Ms_saving(ps;is_moments_out=is_moments_out)
                    else
                        ps["count_save"] = count_save + 1
                    end
                end
            elseif alg_embedded == :GLegendre
                Rck = deepcopy(Rck1)
                s = Nstage
                A, b, c = construct_GL(s,T)
                Rck1N = Vector{Any}(undef,s)
                for k in 1:Nstep
                    # parameters
                    ps["tk"] += dtk
                    Nt_save = ps["Nt_save"]
                    count_save = ps["count_save"]
    
                    println()
                    println("**************------------******************------------*********")
                    printstyled("k=",k,",tk=",ps["tk"],"\n";color=:blue)

                    Mck1integrals_GLegendre!(Rck1N, Mck1, Rck1, err_Rck1, Mhck1, 
                        vhk, vhe, nvG, nc0, nck, ocp, vGdom, nvlevele0, nvlevel0, LMk, LM1k,  
                        muk, Mμk, Munk, Mun1k, Mun2k, naik, uaik, vthik, CΓ, εᵣ, 
                        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
                        Iak1, Kak1, vathk1i, Mck, Rck, RMcsk, Mhck, nak, Iak, Kak, vathk, dtk, A, b, c, s;
                        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                        i_iter_rs2=i_iter_rs2,alg_embedded=:Trapz,i_iter_rs3=i_iter_rs3)
                
                    # When `nₖ₊₁ˢ`, `Iₖ₊₁ˢ` and `Kₖ₊₁ˢ` reach convergence, conservations and
                    # the constraint equation is satisfied: `K̂ = 3/2 + Î`,
    
                    # # updating the distribution function and parameters at `(k+1)ᵗʰ` step
                    is_corrections[1] ? nothing : nak = deepcopy(nak1)
                    Iak = deepcopy(Iak1)
                    Kak = deepcopy(Kak1)
                    vathk = deepcopy(vathk1)
                    Mck = deepcopy(Mck1)
                    Rck = deepcopy(Rck1)
                    Mhck = deepcopy(Mhck1)
                    RMcsk = deepcopy(RMcsk1)
                    vathk1i = deepcopy(vathk)
    
                    # ps["nak"] = deepcopy(nak1)
                    ps["Ik"] = deepcopy(Iak1)
                    ps["Kk"] = deepcopy(Kak1)
                    ps["DThk"] = deepcopy(DThk1)
                    ps["naik"] = deepcopy(naik)
                    ps["uaik"] = deepcopy(uaik)
                    ps["vthik"] = deepcopy(vthik)
                    ps["nModk"] = deepcopy(nModk1)
                    ps["vGm"] = vGdom
    
                    ps["LMk"] = deepcopy(LMk)
                    ps["nMjMs"] = deepcopy(nMjMs)
                    ps["Mhck"] = deepcopy(Mhck1)
                    # ps["Mhck"]
    
                    # Saving the dataset at `(k+1)ᵗʰ` step
                    if count_save == Nt_save
                        ps["count_save"] = 1
                        data_Ms_saving(ps;is_moments_out=is_moments_out)
                    else
                        ps["count_save"] = count_save + 1
                    end
                end
            else
                errorsddd
            end
        end
    end
    # @show dtk, (Nstage, Nstage_improved), (i_iter_rs2,i_iter_rs3, s_iter_max), alg_embedded
end

"""
  A `s`-stage integral at the `kᵗʰ` step with implicit Euler method or trapezoidal method with `Niter_stage`: 

  Level of the algorithm
    kᵗʰ: the time step level
    s: the stage level during `kᵗʰ` time step
    i: the inner iteration level during `sᵗʰ` stage
    
  Inputs:
    nak1 = deepcopy(nak)
    Iak1 = deepcopy(Iak)
    Kak1 = deepcopy(Kak)
    vathk1 = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
    Mck1 = deepcopy(Mck)
    Rck1i::Vector{Any} = [Rck; Rck11; Rck12; ⋯; Rck1i]
    Rck1: = Rck
    Rck1[njMs+1,1,:]                # `w3k = Rdtvath = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`
    Nstage::Int64 ≥ 3, where `Nstage ∈ N⁺`
    Rck1N = Vector{Any}(undef,Nstage)
            When `Nstage=1` go back to the Euler method;
            When `Nstage=2` which will give the trapezoidal method;

  Outputs:
    dtk = Mck1integrals!(Rck1N, Nstage, Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, Mck, RMcsk, Mhck, nak, Iak, Kak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        Nstage_improved=Nstage_improved,s_iter_max=s_iter_max)
    dtk = Mck1integrals!(Rck1N, Nstage, Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, Mck, Rck, RMcsk, Mhck, nak, Iak, Kak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        Nstage_improved=Nstage_improved,s_iter_max=s_iter_max,alg_embedded=alg_embedded)
"""

# [kᵗʰ,s,i], alg_embedded =:ImEuler, Nstage ≥ 3, 

# N1im_n, `Nstage_improved = :every` for every stage
# N1im_N, `Nstage_improved = :last` for the last stage
function Mck1integrals!(Rck1N::Vector{Any},Nstage::Int64,Mck1::AbstractArray{T,N},
    Rck1::AbstractArray{T,N},err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::AbstractVector{T},nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nvlevele0::Vector{Int64}, nvlevel0::Vector{Int64}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nak1::AbstractVector{T},
    vathk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},Mck::AbstractArray{T,N},
    RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},nak::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],
    i_iter_rs2::Int64=0,Nstage_improved::Symbol=:last,s_iter_max::Int64=0) where{T,N,N2}

    s1 = 1
    Rck1N[s1] = deepcopy(Rck1)

    s1 += 1
    Mcks = deepcopy(Mck)
    vathks = deepcopy(vathk)
    vathk1i = deepcopy(vathk)
    dtk = Mck1integrali!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
        Iak1, Kak1, vathk1i, Mcks, RMcsk, Mhck, nak, Iak, Kak, vathks, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        is_vth_ode=is_vth_ode,is_corrections=is_corrections,i_iter_rs2=i_iter_rs2)
    Rck1N[s1] = deepcopy(Rck1)

    if Nstage_improved == :last
        for s1 in 3:Nstage
            Mcks = deepcopy(Mck1)
            vathks = deepcopy(vathk1)
            RMcsk, Mhck = deepcopy(RMcsk1), deepcopy(Mhck1)
            # nak = deepcopy(nak1)
            Iak, Kak = deepcopy(Iak1), deepcopy(Iak1)
            dtk = Mck1integrali!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
                Iak1, Kak1, vathk1i, Mcks, RMcsk, Mhck, nak, Iak, Kak, vathks, dtk;
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                is_vth_ode=is_vth_ode,is_corrections=is_corrections,i_iter_rs2=i_iter_rs2)
            Rck1N[s1] = deepcopy(Rck1)
        end
    
        # Reupdating the values of `Mck1` according to the high-order RK algorithm for the last stage
        if Nstage == 3
        elseif Nstage == 4
        elseif Nstage == 5
        elseif Nstage == 6
        elseif Nstage == 7
        elseif Nstage == 8
        elseif Nstage == 9
        elseif Nstage == 10
        elseif Nstage == 11
        elseif Nstage == 12
        elseif Nstage == 13
        elseif Nstage == 14
        elseif Nstage == 15
        elseif Nstage == 16
        elseif Nstage == 17
        elseif Nstage == 18
        else
            kwskjjb
        end
    elseif Nstage_improved == :every
        for s1 in 3:Nstage
            Mcks = deepcopy(Mck1)
            vathks = deepcopy(vathk1)
            RMcsk, Mhck = deepcopy(RMcsk1), deepcopy(Mhck1)
            # nak = deepcopy(nak1)
            Iak, Kak = deepcopy(Iak1), deepcopy(Iak1)
            dtk = Mck1integrali!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
                Iak1, Kak1, vathk1, Mcks, RMcsk, Mhck, nak, Iak, Kak, vathks, dtk;
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                is_vth_ode=is_vth_ode,is_corrections=is_corrections,i_iter_rs2=i_iter_rs2)
            Rck1N[s1] = deepcopy(Rck1)
    
            # Reupdating the values of `Mck1` according to the high-order RK algorithm for every stage
            if s1 == 3
            elseif s1 == 4
            elseif s1 == 5
            elseif s1 == 6
            elseif s1 == 7
            elseif s1 == 8
            elseif s1 == 9
            elseif s1 == 10
            elseif s1 == 11
            elseif s1 == 12
            elseif s1 == 13
            elseif s1 == 14
            elseif s1 == 15
            elseif s1 == 16
            elseif s1 == 17
            elseif s1 == 18
            else
                kwskjjb
            end
        end
    else
        eedfgnhmn
    end
end

# N1im, `Nstage_improved = :no` 
function Mck1integrals!(Nstage::Int64,Mck1::AbstractArray{T,N},Rck1::AbstractArray{T,N},
    err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::AbstractVector{T},nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nvlevele0::Vector{Int64}, nvlevel0::Vector{Int64}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nak1::AbstractVector{T},
    vathk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},Mck::AbstractArray{T,N},
    RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},nak::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],
    i_iter_rs2::Int64=0) where{T,N,N2}

    # s1 = 2
    Mcks = deepcopy(Mck)
    vathks = deepcopy(vathk)
    vathk1i = deepcopy(vathk)
    dtk = Mck1integrali!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
        Iak1, Kak1, vathk1i, Mcks, RMcsk, Mhck, nak, Iak, Kak, vathks, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        is_vth_ode=is_vth_ode,is_corrections=is_corrections,i_iter_rs2=i_iter_rs2)

    for s1 in 3:Nstage
        Mcks = deepcopy(Mck1)
        vathks = deepcopy(vathk1)
        RMcsk, Mhck = deepcopy(RMcsk1), deepcopy(Mhck1)
        # nak = deepcopy(nak1)
        Iak, Kak = deepcopy(Iak1), deepcopy(Iak1)
        dtk = Mck1integrali!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
            nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
            Iak1, Kak1, vathk1i, Mcks, RMcsk, Mhck, nak, Iak, Kak, vathks, dtk;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
            is_vth_ode=is_vth_ode,is_corrections=is_corrections,i_iter_rs2=i_iter_rs2)
    end
end

# [kᵗʰ,s,i], alg_embedded ∈ [:Trapz, :ImMidpoint, :Range2, :Heun2, Raslton2, :Alshina2], o = 2

# N1im_n, `Nstage_improved = :every` for every stage
# N1im_N, `Nstage_improved = :last` for the last stage
function Mck1integrals!(Rck1N::Vector{Any},Nstage::Int64,Mck1::AbstractArray{T,N},
    Rck1::AbstractArray{T,N},err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::AbstractVector{T},nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nvlevele0::Vector{Int64}, nvlevel0::Vector{Int64}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nak1::AbstractVector{T},
    vathk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},Mck::AbstractArray{T,N},
    Rck::AbstractArray{T,N},RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},nak::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],
    i_iter_rs2::Int64=0,Nstage_improved::Symbol=:last,
    s_iter_max::Int64=0,alg_embedded::Symbol=:Trapz) where{T,N,N2}

    s1 = 1
    Rck1N[s1] = deepcopy(Rck1)

    s1 += 1
    Mcks = deepcopy(Mck)
    vathks = deepcopy(vathk)
    vathk1i = deepcopy(vathk)
    Rcks = deepcopy(Rck)
    dtk = Mck1integrali_rs2!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
        Iak1, Kak1, vathk1i, Mcks, Rcks, RMcsk, Mhck, nak, Iak, Kak, vathks, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        i_iter_rs2=i_iter_rs2,alg_embedded=alg_embedded,is_warn=is_warn)
    Rck1N[s1] = deepcopy(Rck1)
    
    if Nstage_improved == :last
        for s1 in 3:Nstage
            Mcks = deepcopy(Mck1)
            vathks = deepcopy(vathk1)
            Rcks = deepcopy(Rck1)
            RMcsk, Mhck = deepcopy(RMcsk1), deepcopy(Mhck1)
            # nak = deepcopy(nak1)
            Iak, Kak = deepcopy(Iak1), deepcopy(Iak1)
            dtk = Mck1integrali_rs2!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
                Iak1, Kak1, vathk1i, Mcks, Rcks, RMcsk, Mhck, nak, Iak, Kak, vathks, dtk;
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                i_iter_rs2=i_iter_rs2,alg_embedded=alg_embedded,is_warn=is_warn)
            Rck1N[s1] = deepcopy(Rck1)
        end
    
        # Reupdating the values of `Mck1` according to the high-order RK algorithm for the last stage
        if Nstage == 3
        elseif Nstage == 4
        elseif Nstage == 5
        elseif Nstage == 6
        elseif Nstage == 7
        elseif Nstage == 8
        elseif Nstage == 9
        elseif Nstage == 10
        elseif Nstage == 11
        elseif Nstage == 12
        elseif Nstage == 13
        elseif Nstage == 14
        elseif Nstage == 15
        elseif Nstage == 16
        elseif Nstage == 17
        elseif Nstage == 18
        else
            kwskjjb
        end
    elseif Nstage_improved == :every
        for s1 in 3:Nstage
            Mcks = deepcopy(Mck1)
            vathks = deepcopy(vathk1)
            Rcks = deepcopy(Rck1)
            RMcsk, Mhck = deepcopy(RMcsk1), deepcopy(Mhck1)
            # nak = deepcopy(nak1)
            Iak, Kak = deepcopy(Iak1), deepcopy(Iak1)
            dtk = Mck1integrali_rs2!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
                Iak1, Kak1, vathk1i, Mcks, Rcks, RMcsk, Mhck, nak, Iak, Kak, vathks, dtk;
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                i_iter_rs2=i_iter_rs2,alg_embedded=alg_embedded,is_warn=is_warn)
            Rck1N[s1] = deepcopy(Rck1)
    
            # Reupdating the values of `Mck1` according to the high-order RK algorithm for every stage
            if s1 == 3
            elseif s1 == 4
            elseif s1 == 5
            elseif s1 == 6
            elseif s1 == 7
            elseif s1 == 8
            elseif s1 == 9
            elseif s1 == 10
            elseif s1 == 11
            elseif s1 == 12
            elseif s1 == 13
            elseif s1 == 14
            elseif s1 == 15
            elseif s1 == 16
            elseif s1 == 17
            elseif s1 == 18
            else
                kwskjjb
            end
        end
    else
        egnndss
    end
end

# N1im, `Nstage_improved = :no`
function Mck1integrals!(Nstage::Int64,Mck1::AbstractArray{T,N},
    Rck1::AbstractArray{T,N},err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::AbstractVector{T},nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nvlevele0::Vector{Int64}, nvlevel0::Vector{Int64}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nak1::AbstractVector{T},
    vathk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},Mck::AbstractArray{T,N},
    Rck::AbstractArray{T,N},RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},nak::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],
    i_iter_rs2::Int64=0,alg_embedded::Symbol=:Trapz) where{T,N,N2}

    # s1 = 2
    Mcks = deepcopy(Mck)
    vathks = deepcopy(vathk)
    Rcks = deepcopy(Rck)
    dtk = Mck1integrali_rs2!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
            nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
            RMcsk1, Iak1, Kak1, vathk1i, Mcks, Rcks, RMcsk, Mhck, nak, Iak, Kak, vathks, dtk;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
            i_iter_rs2=i_iter_rs2,alg_embedded=alg_embedded,is_warn=is_warn)

    for s1 in 3:Nstage
        Mcks = deepcopy(Mck1)
        vathks = deepcopy(vathk1)
        Rcks = deepcopy(Rck1)
        RMcsk, Mhck = deepcopy(RMcsk1), deepcopy(Mhck1)
        # nak = deepcopy(nak1)
        Iak, Kak = deepcopy(Iak1), deepcopy(Iak1)
        dtk = Mck1integrali_rs2!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
            nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
            Iak1, Kak1, vathk1i, Mcks, Rcks, RMcsk, Mhck, nak, Iak, Kak, vathks, dtk;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
            i_iter_rs2=i_iter_rs2,alg_embedded=alg_embedded,is_warn=is_warn)
    end
end

# [kᵗʰ,s,i], `alg_embedded = :Trapz` as the initial guess values at the general inner nodes.
function Mck1integrals!(Rck1N::Vector{Any},Mck1::AbstractArray{T,N},
    Rck1::AbstractArray{T,N},err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::AbstractVector{T},nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nvlevele0::Vector{Int64}, nvlevel0::Vector{Int64}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nak1::AbstractVector{T},
    vathk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},vathk1i::AbstractVector{T},
    Mck::AbstractArray{T,N},Rck::AbstractArray{T,N},
    RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},nak::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},
    dtk::T,c::AbstractVector{T},s::Int64;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],
    i_iter_rs2::Int64=0,alg_embedded::Symbol=:Trapz,is_warn::Bool=false) where{T,N,N2}

    s1 = 1
    # vathk1i = zeros(T,ns)
    dtk = Mck1integrali_rs2!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
        Iak1, Kak1, vathk1i, Mck, Rck, RMcsk, Mhck, nak, Iak, Kak, vathk, dtk * c[s1];
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        i_iter_rs2=i_iter_rs2,alg_embedded=alg_embedded,is_warn=is_warn)
    Rck1N[s1] = deepcopy(Rck1)
    
    for s1 in 2:s
        dtk = Mck1integrali_rs2!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
            nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
            Iak1, Kak1, vathk1i, Mck, Rck, RMcsk, Mhck, nak, Iak, Kak, vathk, dtk * c[s1];
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
            i_iter_rs2=i_iter_rs2,alg_embedded=alg_embedded,is_warn=is_warn)
        Rck1N[s1] = deepcopy(Rck1)
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
    Rck1i: = Rck1 .= 0              # which will be changed in the following codes

  Outputs:
"""

# [sᵗʰ,i], alg_embedded = :ExEuler                          rs = 1, o = 1

# [sᵗʰ,i], alg_embedded = :ImEuler,                         rs = 2, o = 1
function Mck1integrali!(Mck1::AbstractArray{T,N},Rck1i::AbstractArray{T,N},
    err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::AbstractVector{T},nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nvlevele0::Vector{Int64}, nvlevel0::Vector{Int64}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nak1::AbstractVector{T},
    vathk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},vathk1i::AbstractVector{T},
    Mck::AbstractArray{T,N},RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},nak::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],i_iter_rs2::Int64=0) where{T,N,N2}
    
    # Rck1Ex, Rck1Im, Rck1k = Rc_(k+1/2)
    i_iter = 0                  # and Checking the `i = 1` iteration

    # Applying the explicit Euler step
    δvathi = ones(T,nsk1)        # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
    δvathk1 = zeros(T,nsk1)      # = vathk1 ./ vathk1
    # Rck1i .= 0.0
    dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, δvathk1, RMcsk, Mhck, Iak, Kak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
    vathk1i[:] = deepcopy(vathk1)
    # Rck1Ex = deepcopy(Rck1i)
    
    # If `i_iter_rs2 ≤ 0`, then degenerate into be the explicit Euler method (ExEuler)
    while i_iter < i_iter_rs2
        i_iter += 1
        nak1 = deepcopy(nak)
        vathk1 = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
        Mck1 = deepcopy(Mck)
        dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
            nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
            Iak1, Kak1, δvathk1, vathk1i, RMcsk, Mhck, Iak, Kak, vathk, dtk;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        δvathi = vathk1i ./ vathk1 .- 1                # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
        if norm(δvathi) ≤ rtol_vthi
            break
        end
        vathk1i[:] = deepcopy(vathk1)
        # @show i_iter, δvathk1, δvathi
    end
    # # Rck1Im = deepcopy(Rck1i)
    if i_iter ≥ i_iter_rs2
        @warn(`The maximum number of iteration reached before the implicit Euler method to be convergence!!!`)
    end
end

# [sᵗʰ,i], alg_embedded ∈ [:Trapz, :Heun2, :ImMidpoint],    rs = 2, o = 2
function Mck1integrali_rs2!(Mck1::AbstractArray{T,N},Rck1i::AbstractArray{T,N},
    err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::AbstractVector{T},nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nvlevele0::Vector{Int64}, nvlevel0::Vector{Int64}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nak1::AbstractVector{T},
    vathk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},vathk1i::AbstractVector{T},
    Mck::AbstractArray{T,N},Rck::AbstractArray{T,N},
    RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},nak::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],
    is_warn::Bool=false,is_nMod_adapt::Bool=false,
    i_iter_rs2::Int64=0,alg_embedded::Symbol=:Trapz) where{T,N,N2}

    i_iter = 0                  # and Checking the `i = 1` iteration
    # Applying the explicit Euler step
    δvathi = ones(T,nsk1)        # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
    δvathk1 = zeros(T,nsk1)      # = vathk1 ./ vathk1
    # Rck1i .= 0.0
    dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, δvathk1, RMcsk, Mhck, Iak, Kak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        is_warn=is_warn,is_nMod_adapt=is_nMod_adapt)
    vathk1i[:] = deepcopy(vathk1)
    # Rck1Ex = deepcopy(Rck1i)
    
    # If `i_iter_rs2 ≤ 0`, then degenerate into the explicit Euler method (ExEuler)
    if alg_embedded == :Trapz

        # Rck1i: = (Rck + Rck1) / 2         inputs
        #        = Rck1                     outputs
        while i_iter < i_iter_rs2
            i_iter += 1
            nak1 = deepcopy(nak)
            vathk1 = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
            Mck1 = deepcopy(Mck)
            Rck1i = (Rck + Rck1i) / 2        # Rck1k = Rc_(k+1/2)
            dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
                nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
                Iak1, Kak1, δvathk1, vathk1i, RMcsk, Mhck, Iak, Kak, vathk, dtk;
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                is_warn=is_warn,is_nMod_adapt=is_nMod_adapt)
            δvathi = vathk1i ./ vathk1 .- 1                # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
            # Rck1 = Rck1i
            if norm(δvathi) ≤ rtol_vthi
                break
            end
            vathk1i[:] = deepcopy(vathk1)
            # @show i_iter, δvathk1, δvathi
        end
        # Rck1k = Rc_(k+1/2)
        if i_iter ≥ i_iter_rs2
            @warn(`The maximum number of iteration reached before the Heun method to be convergence!!!`)
        end
    # elseif alg_embedded == :Heun         # i_iter_rs2 = 1
    else
        dfgbn
    end
    return dtk
end

"""
  Inputs:
  Outputs:
    dtk = Mck1integrali_rs3!(Mck1, Rck1i, err_Rck1, Mhck1, 
            vhk, vhe, nvG, nc0, nck, ocp, vGdom, nvlevele0, nvlevel0, LMk, LM1k,  
            muk, Mμk, Munk, Mun1k, Mun2k, naik, uaik, vthik, CΓ, εᵣ, 
            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
            RMcsk1, Iak1, Kak1, vathk1i, Mck, Rck, RMcsk, Mhck, nak, Iak, Kak, vathk, dtk;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
            i_iter_rs2=i_iter_rs2,i_iter_rs3=i_iter_rs3,alg_embedded=alg_embedded)
"""
# [sᵗʰ,i], alg_embedded ∈ :[RK4, :LobattoIIIA4],            rs = 3, o = 4
function Mck1integrali_rs3!(Mck1::AbstractArray{T,N},Rck1i::AbstractArray{T,N},
    err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::Vector{AbstractVector{T}},vhe::AbstractVector{StepRangeLen},
    nvG::Vector{Int64},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},vGdom::AbstractArray{T,N2},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},LMk::Vector{Int64},LM1k::Int64,
    muk::AbstractArray{T,N2},Mμk::AbstractArray{T,N2},Munk::AbstractArray{T,N2},
    Mun1k::AbstractArray{T,NM1},Mun2k::AbstractArray{T,NM2},
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nak1::AbstractVector{T},
    vathk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},vathk1i::AbstractVector{T},
    Mck::AbstractArray{T,N},Rck::AbstractArray{T,N},
    RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},nak::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],i_iter_rs2::Int64=0,
    i_iter_rs3::Int64=0,alg_embedded::Symbol=:LobattoIIIA4) where{T,N,N2,NM1,NM2}
    
    is_k23 = false
    dtk2 = dtk / 2
    # k1 = deepcopy(Rck)

    ################################################################## rs = 2
    i_iter = 0                  # and Checking the `i = 1` iteration
    # Applying the explicit Euler step for the midpoint
    δvathi = ones(T,nsk1)        # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
    δvathk1 = zeros(T,nsk1)      # = vathk1 ./ vathk1
    # Rck1i .= 0.0
    dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, δvathk1, RMcsk, Mhck, Iak, Kak, vathk, dtk2;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
    vathk1i[:] = deepcopy(vathk1)
    k2 = deepcopy(Rck1i)                 # Rck1Ex
    
    # If `i_iter_rs2 ≤ 0`, `explicit Euler` method for the midpoint
    #    `i_iter_rs2 = 1`, `:Heun `
    #    `i_iter_rs2 ≥ 2`, `:Trapz` for the midpoint

    # Rck1i: = (Rck + Rck1) / 2         inputs
    #        = Rck1                     outputs
    while i_iter < i_iter_rs2
        i_iter += 1
        nak1 = deepcopy(nak)
        vathk1 = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
        Mck1 = deepcopy(Mck)
        Rck1i = (Rck + Rck1i) / 2        # Rck1k = Rc_(k+1/2)
        dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
            nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
            Iak1, Kak1, δvathk1, vathk1i, RMcsk, Mhck, Iak, Kak, vathk, dtk2;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        δvathi = vathk1i ./ vathk1 .- 1                # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
        if norm(δvathi) ≤ rtol_vthi
            break
        end
        vathk1i[:] = deepcopy(vathk1)
        # @show i_iter, δvathk1, δvathi
    end
    if alg_embedded == :LobattoIIIA4
        Mck01 = deepcopy(Mck1)
    end

    if i_iter ≥ i_iter_rs2
        @warn(`The maximum number of iteration reached before the Heun method to be convergence!!!`)
    end

    ################################################################## rs = 3, :RK4 → :LobattoIIIA4
    # Where applying the `k23` instead of the standard `k3` when calculating the value of `k4`
    if is_k23
        dfgvbn 
        k3 = deepcopy(Rck1i)
        k23 = (Rck + Rck1i) / 2
        Rck1i = deepcopy(k23)        # Rck1k = Rc_(k+1/2)
    else
        k3 = deepcopy(Rck1i)                 # The standard derivatives `k3` which calculated by the implicit Euler method during in `RK4` algorithm
    end
    # This step is performed by using the explicit Euler method (ExEuler)
    dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, δvathk1, RMcsk, Mhck, Iak, Kak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
    # k4 = Rck1i
    
    # Computing the effective derivatives in `RK4` algorithm: keff = (k1 + 2(k2 + k3) + k4) / 6
    if is_k23
        Rck1i = (Rck + 4k23 + Rck1i) / 6
    else
        Rck1i = (Rck + 2(k2 + k3) + Rck1i) / 6
    end

    # Computing the values of `Mck1` at `k+1` step and its derivatives `Rck1` respective to time by using the explicit Euler method
    dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, δvathk1, RMcsk, Mhck, Iak, Kak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
    
    # Applying the implicit algorithm such as `LobattoIIIA4` to be the `Corrector` step based on the previous `Predictor` step
    if alg_embedded == :LobattoIIIA4
        # Rck01 = k2
        corrector_LobattoIIIA4!(Mck1, Rck1i, Mck01, k2, err_Rck1, Mhck, vhk, vhe, 
            nvG, nc0, nck, ocp, vGdom, nvlevele0, nvlevel0, LMk, LM1k,  
            muk, Mμk, Munk, Mun1k, Mun2k, naik, uaik, vthik, CΓ, εᵣ, 
            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
            Mck, Rck, RMcsk, Iak, Kak, dtk;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
            is_corrections=is_corrections,i_iter_rs3=i_iter_rs3)
    else
        # ddfhnb
        # for i in i_iter_rs3
        # end
    end
end

# vathk = zeros(nsk); parameters such as `vath` will be updated according to `Mck1`
# Rck01, Mck01 = Rck_half, Mck_half
function corrector_LobattoIIIA4!(Mck1::AbstractArray{T,N},Rck1::AbstractArray{T,N},
    Mck01::AbstractArray{T,N},Rck01::AbstractArray{T,N},
    err_Rck::AbstractArray{T,N},Mhck::Vector{Any},
    vhk::Vector{AbstractVector{T}},vhe::AbstractVector{StepRangeLen},nvG::Vector{Int64},nc0::Vector{Int64},
    nck::Vector{Int64},ocp::Vector{Int64},vGdom::AbstractArray{T,N2},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},LMk::Vector{Int64},LM1k::Int64,
    muk::AbstractArray{T,N2},Mμk::AbstractArray{T,N2},Munk::AbstractArray{T,N2},
    Mun1k::AbstractArray{T,NM1},Mun2k::AbstractArray{T,NM2},
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak::AbstractVector{T},Zqk::AbstractVector{Int64},nak::AbstractVector{T},
    vathk::AbstractVector{T},nsk::Int64,nModk::Vector{Int64},nMjMs::Vector{Int64},
    DThk::AbstractVector{T},Mck::AbstractArray{T,N},Rck::AbstractArray{T,N},
    RMcsk::AbstractArray{T,N2},Iak::AbstractVector{T},Kak::AbstractVector{T},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_corrections::Vector{Bool}=[true,false,false],
    i_iter_rs3::Int64=10,is_warn::Bool=false) where{T,N,N2,NM1,NM2}

    Mck1_copy = deepcopy(Mck1[1:njMs,:,:])

    i_iter = 0
    dtk3 = dtk / 3
    Rck58 = Rck * (5/8)
    Mck01[:,:,:] = Mck + dtk3 * (Rck58 + Rck01 - Rck1 / 8) 
    Mck1[:,:,:] = Mck + dtk3 / 2 * (Rck + Rck01 * 4 + Rck1) 

    # Updating the values of `Rck01`, parameters such as `vath` will be updated according to `Mck01`
    dtk = Rck_update!(Rck01, err_Rck, Mhck, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak, Zqk, nak, vathk, nsk, nModk, nMjMs, DThk, RMcsk, Iak, Kak, Mck01,dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        is_corrections=is_corrections,is_vth_ode=true,is_warn=is_warn)

    # Updating the values of `Rck1`, parameter such as `vath` will be updated according to `Mck1`
    dtk = Rck_update!(Rck1, err_Rck, Mhck, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak, Zqk, nak, vathk, nsk, nModk, nMjMs, DThk, RMcsk, Iak, Kak, Mck1,dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        is_corrections=is_corrections,is_vth_ode=true,is_warn=is_warn)
    
    # Computing the relative errors
    RerrDMck1 = evaluate_RerrDMc(Mck1[1:njMs,:,:],Mck1_copy)
    RerrDMck1_up = deepcopy(RerrDMck1)
    ratio_DMc = 1.0          # = RerrDMck1 / RerrDMck1_up - 1
    while i_iter < i_iter_rs3
        i_iter += 1
        # @show i_iter, RerrDMck1
        if RerrDMck1 ≤ RerrDMc
            break
        else
            Mck01[:,:,:] = Mck + dtk3 * (Rck58 + Rck01 - Rck1 / 8) 
            Mck1[:,:,:] = Mck + dtk3 / 2 * (Rck + Rck01 * 4 + Rck1) 

            # Updating the values of `Rck01`, parameters such as `vath` will be updated according to `Mck01`
            dtk = Rck_update!(Rck01, err_Rck, Mhck, vhk, nvG, ocp, vGdom, 
                nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                mak, Zqk, nak, vathk, nsk, nModk, nMjMs, DThk, RMcsk, Iak, Kak, Mck01,dtk;
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                is_corrections=is_corrections,is_vth_ode=true,is_warn=is_warn)

            # Updating the values of `Rck1`, parameter such as `vath` will be updated according to `Mck1`
            dtk = Rck_update!(Rck1, err_Rck, Mhck, vhk, nvG, ocp, vGdom, 
                nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
                mak, Zqk, nak, vathk, nsk, nModk, nMjMs, DThk, RMcsk, Iak, Kak, Mck1,dtk;
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
                is_corrections=is_corrections,is_vth_ode=true,is_warn=is_warn)
            RerrDMck1 = evaluate_RerrDMc(Mck1[1:njMs,:,:],Mck1_copy)
            ratio_DMc = abs(RerrDMck1 / (RerrDMck1_up + epsT) - 1)
            if ratio_DMc ≤ Ratio_DMc
                break
            end
            Mck1_copy = deepcopy(Mck1[1:njMs,:,:])
            RerrDMck1_up = deepcopy(RerrDMck1)
        end
    end
    if i_iter ≥ i_iter_rs3
        @warn(`rs3: The maximum number of iteration reached before the "LobattoIIIA4" method to be convergence!!!`)
    end
    # @show i_iter
end

function evaluate_RerrDMc(Mck1,Mck1_copy)
    
    # Rerr = 
    return maximum(abs.(((Mck1[:,1,:] .+ epsT)  ./ (Mck1_copy[:,1,:] .+ epsT)) .- 1.0))
end

# [sᵗʰ,i], alg_embedded ∈ [:RK438, :CRK44],            rs = 4, o = 4
function Mck1integrali_rs4!(Mck1::AbstractArray{T,N},Rck1i::AbstractArray{T,N},
    err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::Vector{AbstractVector{T}},vhe::AbstractVector{StepRangeLen},
    nvG::Vector{Int64},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},vGdom::AbstractArray{T,N2},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},LMk::Vector{Int64},LM1k::Int64,
    muk::AbstractArray{T,N2},Mμk::AbstractArray{T,N2},Munk::AbstractArray{T,N2},
    Mun1k::AbstractArray{T,NM1},Mun2k::AbstractArray{T,NM2},
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nak1::AbstractVector{T},
    vathk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},vathk1i::AbstractVector{T},
    Mck::AbstractArray{T,N},Rck::AbstractArray{T,N},
    RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},nak::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],i_iter_rs2::Int64=0,
    i_iter_rs3::Int64=0,alg_embedded::Symbol=:CRK44) where{T,N,N2,NM1,NM2}
    
    is_k23 = false
    dtk2 = dtk / 2
    # k1 = deepcopy(Rck)

    i_iter = 0                  # and Checking the `i = 1` iteration

    ################################################################## rs = 2
    # Applying the explicit Euler step for the second point
    δvathi = ones(T,nsk1)        # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
    δvathk1 = zeros(T,nsk1)      # = vathk1 ./ vathk1
    # Rck1i .= 0.0
    dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, δvathk1, RMcsk, Mhck, Iak, Kak, vathk, dtk2;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
    vathk1i[:] = deepcopy(vathk1)
    k2 = deepcopy(Rck1i)                 # Rck1Ex
    
    # If `i_iter_rs2 ≤ 0`, `explicit Euler` method for the midpoint
    #    `i_iter_rs2 = 1`, `:Heun `
    #    `i_iter_rs2 ≥ 2`, `:Trapz` for the midpoint

    # Rck1i: = (Rck + Rck1i) / 2         inputs
    #        = Rck1i                     outputs
    while i_iter < i_iter_rs2
        i_iter += 1
        nak1 = deepcopy(nak)
        vathk1 = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
        Mck1 = deepcopy(Mck)
        Rck1i = (Rck + Rck1i) / 2        # Rck1k = Rc_(k+1/2)
        dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
            nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, RMcsk1, 
            Iak1, Kak1, δvathk1, vathk1i, RMcsk, Mhck, Iak, Kak, vathk, dtk2;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        δvathi = vathk1i ./ vathk1 .- 1                # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
        if norm(δvathi) ≤ rtol_vthi
            break
        end
        vathk1i[:] = deepcopy(vathk1)
        # @show i_iter, δvathk1, δvathi
    end
    if alg_embedded == :CRK44            # :LobattoIIIA4
        Mck01 = deepcopy(Mck1)
    end

    if i_iter ≥ i_iter_rs2
        @warn(`The maximum number of iteration reached before the Heun method to be convergence!!!`)
    end

    ################################################################## rs = 3, :RK4 → :LobattoIIIA4
    # Where applying the `k23` instead of the standard `k3` when calculating the value of `k4`
    if is_k23
        dfgvbn 
        k3 = deepcopy(Rck1i)
        k23 = (Rck + Rck1i) / 2
        Rck1i = deepcopy(k23)        # Rck1k = Rc_(k+1/2)
    else
        k3 = deepcopy(Rck1i)                 # The standard derivatives `k3` which calculated by the implicit Euler method during in `RK4` algorithm
    end
    # This step is performed by using the explicit Euler method (ExEuler)
    dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, δvathk1, RMcsk, Mhck, Iak, Kak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
    # k4 = Rck1i
    
    # Computing the effective derivatives in `RK4` algorithm: keff = (k1 + 2(k2 + k3) + k4) / 6
    if is_k23
        Rck1i = (Rck + 4k23 + Rck1i) / 6
    else
        Rck1i = (Rck + 2(k2 + k3) + Rck1i) / 6
    end

    # Computing the values of `Mck1` at `k+2/3` step and its derivatives `Rck1` respective to time by using the explicit Euler method
    dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, δvathk1, RMcsk, Mhck, Iak, Kak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
    
    # Applying the implicit algorithm such as `LobattoIIIA4` to be the `Corrector` step based on the previous `Predictor` step
    if alg_embedded == :CRK44             # :LobattoIIIA4
        # Rck01 = k2
        corrector_LobattoIIIA4!(Mck1, Rck1i, Mck01, k2, err_Rck1, Mhck, vhk, vhe, 
            nvG, nc0, nck, ocp, vGdom, nvlevele0, nvlevel0, LMk, LM1k,  
            muk, Mμk, Munk, Mun1k, Mun2k, naik, uaik, vthik, CΓ, εᵣ, 
            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
            Mck, Rck, RMcsk, Iak, Kak, dtk;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
            is_corrections=is_corrections,i_iter_rs3=i_iter_rs3)
    else
        # ddfhnb
        # for i in i_iter_rs3
        # end
    end

    ################################################################## rs = 4, :RK4 → :LobattoIIIA4
    Mck_3 = deepcopy(Mck1)         # The third point
    k3 = deepcopy(Rck1i)

    # Computing the value of `k4` according to the rule of `RK438`
    # k1, k2 = Rck, k2
    Rck1i[:,:,:] = k3 - k2 + Rck    # k4_predict          
    dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, δvathk1, RMcsk, Mhck, Iak, Kak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
    # Computing the effective derivatives in `RK4` algorithm: keff = (k1 + 3(k2 + k3) + k4) / 8
    Rck1i[:,:,:] = (Rck + 3(k2 + k3) + Rck1i) / 8

    # Computing the values of `Mck1` at `k+1` step and its derivatives `Rck1` 
    # respective to time by using the explicit Euler method (ExEuler)
    dtk = Mck1integral!(Mck1, Rck1i, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, δvathk1, RMcsk, Mhck, Iak, Kak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
    
    # Applying the implicit algorithm such as `LobattoIIIA4` to be the `Corrector` step based on the previous `Predictor` step
    if alg_embedded == :CRK44               # :LobattoIIIA4
        # Rck01 = k3
        # Mck, Rck = Mck01, k2    # at the second point
        # Mck01, k2 = Mck_3, k3
        corrector_LobattoIIIA4!(Mck1, Rck1i, Mck_3, k3, err_Rck1, Mhck, vhk, vhe, 
            nvG, nc0, nck, ocp, vGdom, nvlevele0, nvlevel0, LMk, LM1k,  
            muk, Mμk, Munk, Mun1k, Mun2k, naik, uaik, vthik, CΓ, εᵣ, 
            mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
            Mck01, k2, RMcsk, Iak, Kak, dtk;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
            is_corrections=is_corrections,i_iter_rs3=i_iter_rs3)
    else
        # ddfhnb
        # for i in i_iter_rs3
        # end
    end
end

# [sᵗʰ,i], alg_embedded = :RK5,                             rs = 5, o = 5
function Mck1integrali_rs5!()
end

# # [sᵗʰ,i], alg_embedded ∈ [:Range2, Raslton2, :Alshina2, :QinZhang,], o = 2
# [sᵗʰ,i], alg_embedded ∈ [Kutta3, :Heun3, Ralston3, RK3, SSPRK3, :BS3, :OwrenZen3, :Alshina3, RadauIA, RadauIIA], o = 3
# [sᵗʰ,i], alg_embedded ∈ [:RK42, :OwrenZen4, LabottaIIIA4/C4Star],  
# [sᵗʰ,i], alg_embedded ∈ [:Ralston5, :Runge51 :BS5, :OwrenZen5, :DP5, :Tsit5, :SIR54],      o = 5

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
    dtk = Rck_update!(Rck, err_Rck, Mhck, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak, Zqk, nak, vathk, nsk, nModk, nMjMs, DThk, RMcsk, Iak, Kak, Mck,dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        is_corrections=is_corrections,is_vth_ode=is_vth_ode,
        is_warn=is_warn,is_nMod_adapt=is_nMod_adapt)
    dtk = Mck1integral!(Mck1, Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak1, Zqk1, nak1, vathk1, nsk1, nModk1, nMjMs, DThk1, 
        RMcsk1, Iak1, Kak1, δvathk1, vathk1i, RMcsk, Mhck, Iak, Kak, vathk, dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        is_warn=is_warn,is_nMod_adapt=is_nMod_adapt)
"""

# [iᵗʰ],  `tk = 0`, Rck .= 0.0, according to `Mck` which may be updated according to the M-theorems
function Rck_update!(Rck::AbstractArray{T,N},err_Rck::AbstractArray{T,N},Mhck::Vector{Any},
    vhk::AbstractVector{T},nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nvlevele0::Vector{Int64}, nvlevel0::Vector{Int64}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak::AbstractVector{T},Zqk::AbstractVector{Int64},nak::AbstractVector{T},
    vathk::AbstractVector{T},nsk::Int64,nModk::Vector{Int64},nMjMs::Vector{Int64},
    DThk::AbstractVector{T},RMcsk::AbstractArray{T,N2},
    Iak::AbstractVector{T},Kak::AbstractVector{T},Mck::AbstractArray{T,N},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_corrections::Vector{Bool}=[true,false,false],
    is_vth_ode::Bool=false,is_warn::Bool=false,is_nMod_adapt::Bool=false) where{T,N,N2}

    ρk = mak .* nak
    nIKT_update!(mak,nak,vathk,nsk,Iak,Kak,Mck;is_corrections=is_corrections,is_vth_ode=is_vth_ode)
    # uk = Iak1 ./ ρk1

    # Computing the M-functions
    if is_check_Mcs
        RMcskn1 = deepcopy(RMcsk)
        Mtheorems_RMcs!(RMcsk,Mck[1:njMs,:,:],ρk,nsk)
        # is_low = DataFrame(RMcsk .< RMcsk,:auto)
        is_low_RMcs = DataFrame((RMcsk - RMcskn1) ./ abs.(RMcsk),:auto)
        @show is_low_RMcs
    end

    # # Computing the re-normalized moments
    if is_check_Mhc
        Mhckn1 = deepcopy(Mhck)
    end
    # MhckMck!(Mhck,Mck[1:njMs,:,:],ρk,LMk,nsk,nMjMs,vathk)
    MhckMck!(Mhck,Mck[1:njMs,:,:],ρk,LMk,nsk,nMjMs)
    if is_check_Mhc
        isp = 1
        is_low_RMca = DataFrame(Mhck[isp] - Mhckn1[isp],:auto)
        isp = 2
        is_low_RMcb = DataFrame(Mhck[isp] - Mhckn1[isp],:auto)
        @show is_low_RMca
        @show is_low_RMcb
    end

    # # # Renew the values of `Mck` according to the quantities of `Mhck` and `RMcsk`
    # if is_Mtheo 
    #     for isp in 1:nsk
    #         Mhck[isp][:,1,:] .= 1.0
    #     end
    #     a = Mck[1:njMs,:,:]
    #     MckMhck!(a,Mhck,ρk,vathk,LMk,nsk,nMjMs)
    #     Mck[1:njMs,:,:] = a

    #     # Computing the M-functions
    #     if is_check_Mcs
    #         Mtheorems_RMcs!(RMcsk,Mck[1:njMs,:,:],ρk,nsk)
    #         is_low2_RMcs = DataFrame((RMcsk - RMcskn1) ./ abs.(RMcsk),:auto)
    #         @show is_low2_RMcs
    #     end
    # end
    
    # Rck .= 0.0
    dtk = dtMcab!(Rck, err_Rck, Mhck, vhk[1], nvG[1], ocp[1], 
        vGdom[:,1], nvlevele0[1], nvlevel0[1], LMk, LM1k,
        naik, uaik, vthik, CΓ, εᵣ, mak, Zqk, nak, 
        Iak1 ./ ρk1, vathk, nsk, nModk, nMjMs, DThk,dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        orderVconst=orderVconst, 
        vGm_limit=vGm_limit,is_warn=is_warn,is_nMod_adapt=is_nMod_adapt,
        is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK)

    # # Computing the values of `∂ₜvₜₕ` acoording to `w3k = Rdtvath = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3
    Rck[njMs+1,1,:] .*= vathk

    # Checking the laws of the dissipation
    if is_check_dtMcs
        RdtMcsk = deepcopy(Rck[:,:,1])
        RdtMcsd2l!(RdtMcsk,Rck,njMs,ρk,Iak,Kak)
        RdtMcsk_csv = DataFrame(RdtMcsk,:auto)
        @show RdtMcsk_csv
        # ggjjkkk
        
        # `RdtMcsk ≤ 0` in theory
        # RdtMcsk .≤ epsT1000
    end
    return dtk
end

# [iᵗʰ], alg_embedded == :ExEuler
function Mck1integral!(Mck1::AbstractArray{T,N},Rck1::AbstractArray{T,N},
    err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::AbstractVector{T},nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nvlevele0::Vector{Int64}, nvlevel0::Vector{Int64}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nak1::AbstractVector{T},
    vathk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},Iak1::AbstractVector{T},
    Kak1::AbstractVector{T},δvathk1::AbstractVector{T},RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],
    is_warn::Bool=false,is_nMod_adapt::Bool=false) where{T,N,N2}
    
    ρk1 = mak1 .* nak1
    # `vathk1 = zeros(nsk1)`
    Mck1integral0!(Mck1,Rck1,mak1,nak1,vathk1,nsk1,Iak1,Kak1,δvathk1,Iak,Kak,vathk,dtk;
           is_vth_ode=is_vth_ode,is_corrections=is_corrections)
    # uk = Iak1 ./ ρk1

    # Renew the values of `Mhck1` when `nMod = 1` and 
    # Renew the values of `Mck1` according to the quantities of `Mhck1` and `RMcsk1`
    if is_Ms_nuT
        MhcknuT!(Mhck1,Mck1,nMjMs,LMk,ρk1,vathk1,nsk1,nModk1;is_renorm=is_renorm)
    end

    # # Computing the M-functions
    if is_warn && is_check_Mcs
        Mtheorems_RMcs!(RMcsk1,Mck1[1:njMs,:,:],ρk1,nsk1)
        low_RMcs = DataFrame((RMcsk1 - RMcsk) ./ abs.(RMcsk),:auto)
        @show low_RMcs
    end

    # # Computing the re-normalized moments
    # MhckMck!(Mhck1,Mck1[1:njMs,:,:],ρk1,LMk,nsk1,nMjMs,vathk1)
    MhckMck!(Mhck1,Mck1[1:njMs,:,:],ρk1,LMk,nsk1,nMjMs)

    # if is_warn && is_check_Mhc
    #     isp = 1
    #     N_low_RMca = DataFrame(Mhck1[isp] - Mhck[isp],:auto)
    #     isp = 2
    #     N_low_RMcb = DataFrame(Mhck1[isp] - Mhck[isp],:auto)
    #     @show N_low_RMca
    #     @show N_low_RMcb
    # end

    # # # Renew the values of `Mck1` according to the quantities of `Mhck1` and `RMcsk1`
    # if is_Mtheo 
    #     for isp in 1:nsk1
    #         Mhck1[isp][:,1,:] .= 1.0
    #     end
    #     a = Mck1[1:njMs,:,:]
    #     MckMhck!(a,Mhck1,ρk1,vathk1,LMk,nsk1,nMjMs)
    #     Mck1[1:njMs,:,:] = a
    # end

    dtk = dtMcab!(Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, nvlevele0, nvlevel0, LMk, LM1k,
        naik, uaik, vthik, CΓ, εᵣ, mak1, Zqk1, 
        nak1, Iak1 ./ ρk1, vathk1, nsk1, nModk1, nMjMs, DThk1,dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        orderVconst=orderVconst, 
        vGm_limit=vGm_limit,is_warn=is_warn,is_nMod_adapt=is_nMod_adapt,
        is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK)
    Rck1[njMs+1,1,:] .*= vathk      # ∂ₜvₜₕ,  

    # Checking the laws of the dissipation
    if is_warn && is_check_dtMcs
        RdtMcsk1 = deepcopy(Rck1[:,:,1])
        RdtMcsd2l!(RdtMcsk1,Rck1,njMs,ρk1,Iak1,Kak1)
        N_RdtMcsk1_csv = DataFrame(RdtMcsk1,:auto)
        @show N_RdtMcsk1_csv

        # `RdtMcsk1 ≤ 0` in theory
        # RdtMcsk1 .≤ epsT1000
    end
    return dtk
end

# [iᵗʰ], alg_embedded == ∈ [:ImEuler, :Trapz]
function Mck1integral!(Mck1::AbstractArray{T,N},Rck1::AbstractArray{T,N},
    err_Rck1::AbstractArray{T,N},Mhck1::Vector{Any},
    vhk::AbstractVector{T},nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nvlevele0::Vector{Int64}, nvlevel0::Vector{Int64}, LMk::Vector{Int64}, LM1k::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},nak1::AbstractVector{T},
    vathk1::AbstractVector{T},nsk1::Int64,nModk1::Vector{Int64},nMjMs::Vector{Int64},
    DThk1::AbstractVector{T},RMcsk1::AbstractArray{T,N2},Iak1::AbstractVector{T},
    Kak1::AbstractVector{T},δvathk1::AbstractVector{T},vathk1i::AbstractVector{T},
    RMcsk::AbstractArray{T,N2},Mhck::Vector{Any},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},dtk::T;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true,  
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
    Msj_adapt::Vector{Int64}=[0,2],orderVconst::Vector{Int64}=[1,1],
    vGm_limit::Vector{T}=[5.0,20],is_vth_ode::Bool=true,
    is_corrections::Vector{Bool}=[true,false,false],
    is_warn::Bool=false,is_nMod_adapt::Bool=false) where{T,N,N2}
    
    ρk1 = mak1 .* nak1
    # `vathk1 = zeros(nsk1)`
    Mck1integral0!(Mck1,Rck1,mak1,nak1,vathk1,nsk1,Iak1,Kak1,δvathk1,Iak,Kak,vathk,dtk;
           is_vth_ode=is_vth_ode,is_corrections=is_corrections)
    # uk = Iak1 ./ ρk1

    # # Computing the M-functions
    if is_warn && is_check_Mcs
        Mtheorems_RMcs!(RMcsk1,Mck1[1:njMs,:,:],ρk1,nsk1)
        low_RMcs = DataFrame((RMcsk1 - RMcsk) ./ abs.(RMcsk),:auto)
        @show low_RMcs
    end

    # # Computing the re-normalized moments
    # MhckMck!(Mhck1,Mck1[1:njMs,:,:],ρk1,LMk,nsk1,nMjMs,vathk1)
    MhckMck!(Mhck1,Mck1[1:njMs,:,:],ρk1,LMk,nsk1,nMjMs)
    
    # Renew the values of `Mhck1` when `nMod = 1`
    # Renew the values of `Mck1` according to the quantities of `Mhck1` and `RMcsk1`
    if is_Ms_nuT
        MhcknuT!(Mhck1,Mck1,nMjMs,LMk,ρk1,vathk1,nsk1,nModk1;is_renorm=is_renorm)
    end

    if is_warn && is_check_Mhc
        isp = 1
        low_RMca = DataFrame(Mhck1[isp] - Mhck[isp],:auto)
        isp = 2
        low_RMcb = DataFrame(Mhck1[isp] - Mhck[isp],:auto)
        @show low_RMca
        @show low_RMcb
    end

    # # # Renew the values of `Mck1` according to the quantities of `Mhck1` and `RMcsk1`
    # if is_Mtheo 
    #     for isp in 1:nsk1
    #         Mhck1[isp][:,1,:] .= 1.0
    #     end
    #     a = Mck1[1:njMs,:,:]
    #     MckMhck!(a,Mhck1,ρk1,vathk1,LMk,nsk1,nMjMs)
    #     Mck1[1:njMs,:,:] = a
    # end

    dtk = dtMcab!(Rck1, err_Rck1, Mhck1, vhk, nvG, ocp, vGdom, nvlevele0, nvlevel0, LMk, LM1k,
        naik, uaik, vthik, CΓ, εᵣ, mak1, Zqk1, 
        nak1, Iak1 ./ ρk1, vathk1, nsk1, nModk1, nMjMs, DThk1,dtk;
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
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
        is_nMod_adapt=is_nMod_adapt,
        is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK)
    Rck1[njMs+1,1,:] .*= vathk1      # ∂ₜvₜₕ
    # @show vathk1i

    # Checking the laws of the dissipation
    if is_warn && is_check_dtMcs
        RdtMcsk1 = deepcopy(Rck1[:,:,1])
        RdtMcsd2l!(RdtMcsk1,Rck1,njMs,ρk1,Iak1,Kak1)
        RdtMcsk1_csv = DataFrame(RdtMcsk1,:auto)
        @show RdtMcsk1_csv

        # `RdtMcsk1 ≤ 0` in theory
        # RdtMcsk1 .≤ epsT1000
    end
    return dtk
end

"""
  The `iᵗʰ` iteration of with implicit Euler method or Trapezoidal method: 

  Inputs:
    nak1::Vector{Int64} = nak, which will be changed
    Iak1: = zeros(Int64,nsk1)
    Kak1: = zeros(Int64,nsk1)
    vathk:
    vathk1::Vector{Int64} = vathk, which will be changed
    Mck1::AbstractArray{T,3} = Mck, which will be changed
    Rck1: = Rck,   i = 0, the explicit Euler method
          = Rck1i, i ≥ 1, the implicit Euler method
          = (Rck + Rck1i)/2, i ≥ 1, the Trapezoidal method
    Rck1[njMs+1,1,:]                    # `w3k1 = Rdtvath = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`
    Rck[njMs+1,1,:]                     # `w3k

  Outputs:
    nIKT_update!(mak,nak,vathk,nsk,Iak,Kak,Mck;
            is_corrections=is_corrections,is_vth_ode=is_vth_ode)
    Mck1integral0!(Mck1,Rck1,mak1,nak1,vathk1,nsk1,Iak1,Kak1,δvathk1,Iak,Kak,vathk,dtk;
            is_vth_ode=is_vth_ode,is_corrections=is_corrections)
"""

# [], `tk = 0` ,  vathk = deepcopy(Mck[njMs+1,1,:])
function nIKT_update!(mak::AbstractVector{T},nak::AbstractVector{T},vathk::AbstractVector{T},
    nsk::Int64,Iak::AbstractVector{T},Kak::AbstractVector{T},Mck::AbstractArray{T,N},;
    is_corrections::Vector{Bool}=[true,false,false],is_vth_ode::Bool=false) where{T,N}
    
    nsp_vec = 1:nsk

    # # # # Updating the conservative momentums `n, I, K`
    if is_corrections[1] == false
        for isp in nsp_vec
            nak[isp] = Mck[1, 1, isp] / mak[isp]
        end
    else
        for isp in nsp_vec
            Mck[1, 1, isp] = nak[isp] * mak[isp]
        end
    end
    ρk = mak .* nak

    for isp in nsp_vec
        Iak[isp] = Mck[1, 2, isp]
        Kak[isp] = Mck[2, 1, isp] * CMcKa       # Mck[2, 1, isp] * 3 / 4
    end
    if is_vth_ode
        vathk[:] = Mck[njMs+1,1,:]
        if is_check_vth
            δvathk = zeros(nsk)
            for isp in nsp_vec
                if Iak[isp] == 0.0
                    δvathk[isp] = (Mck[2, 1, isp] / ρk[isp])^0.5 / vathk[isp] - 1
                else
                    δvathk[isp] = ((Mck[2, 1, isp] / ρk[isp] - 2/3 * (Iak[isp] / ρk[isp])^2))^0.5 / vathk[isp] - 1
                end
            end
            norm(δvathk) ≤ epsT1000 || @warn("ode: The initial values of `Iak, Kak` and `vathk` are not consistent!",δvathk)
        end
    else
        for isp in nsp_vec
            if Iak[isp] == 0.0
                vathk[isp] = (Mck[2, 1, isp] / ρk[isp])^0.5
            else
                vathk[isp] = ((Mck[2, 1, isp] / ρk[isp] - 2/3 * (Iak[isp] / ρk[isp])^2))^0.5
            end
        end
        if is_check_vth
            δvathk = zeros(nsk)
            for isp in nsp_vec
                if Iak[isp] == 0.0
                    δvathk[isp] = Mck[njMs+1,1,isp] / vathk[isp] - 1
                else
                    δvathk[isp] = Mck[njMs+1,1,isp] / vathk[isp] - 1
                end
            end
            norm(δvathk) ≤ epsT1000 || @warn("The initial values of `Iak, Kak` and `vathk` are not consistent!",δvathk)
        end
    end
end

# [], alg_embedded::Symbol ∈ [:ExEuler, :ImEuler, :Trapz]
function Mck1integral0!(Mck1::AbstractArray{T,N},Rck1::AbstractArray{T,N},
    mak1::AbstractVector{T},nak1::AbstractVector{T},vathk1::AbstractVector{T},
    nsk1::Int64,Iak1::AbstractVector{T},Kak1::AbstractVector{T},δvathk1::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},vathk::AbstractVector{T},dtk::T;
    is_vth_ode::Bool=true,is_corrections::Vector{Bool}=[true,false,false]) where{T,N}
    
    nsp_vec = 1:nsk1
    # @show KK = Rck1[2,1,:] * CMcKa 
    # @show II = Rck1[1,2,:] / 3
    for isp in nsp_vec
        Mck1[:,:,isp] += dtk * Rck1[:,:,isp]
    end

    # # # # Updating the conservative momentums `n, I, K`
    if is_corrections[1] == false
        for isp in nsp_vec
            nak1[isp] = Mck1[1, 1, isp] / mak1[isp]
        end
    else
        for isp in nsp_vec
            Mck1[1, 1, isp] = nak1[isp] * mak1[isp]
        end
    end
    ρk1 = mak1 .* nak1

    for isp in nsp_vec
        Iak1[isp] = Mck1[1, 2, isp]
        Kak1[isp] = Mck1[2, 1, isp] * CMcKa       # Mck1[2, 1, isp] * 3 / 4
    end
        
    # Conservations
    # Mconservations!(Iak1, Kak1, Iak, Kak;errnIKTh=errnIKTh,
    #     rtol_IKTh=rtol_IKTh,rtol_IKTh_err=rtol_IKTh_err,
    #     is_corrections=is_corrections,residualMethod_FP0D=residualMethod_FP0D)
    
    # Computing the values of `vathk1` in two different ways
    if is_vth_ode
        for isp in nsp_vec
            vathk1[isp] = vathk[isp] + dtk * Rck1[njMs+1,1,isp]
            if Iak1[isp] == 0.0
                δvathk1[isp] = (Mck1[2, 1, isp] / ρk1[isp])^0.5 / vathk1[isp] - 1
            else
                δvathk1[isp] = ((Mck1[2, 1, isp] / ρk1[isp] - 2/3 * (Iak1[isp] / ρk1[isp])^2))^0.5 / vathk1[isp] - 1
            end
        end
    else
        for isp in nsp_vec
            # Computing the value of `vathk1`
            δvathk1[isp] = vathk[isp] + dtk * Rck1[njMs+1,1,isp]
            if Iak1[isp] == 0.0
                vathk1[isp] = (Mck1[2, 1, isp] / ρk1[isp])^0.5
            else
                vathk1[isp] = ((Mck1[2, 1, isp] / ρk1[isp] - 2/3 * (Iak1[isp] / ρk1[isp])^2))^0.5
            end
            # δvathk1[isp] = vathk1[isp] / δvathk1[isp]
            δvathk1[isp] \= vathk1[isp]
            δvathk1[isp] -= 1
        end
    end
end

# [], alg_embedded ∈ [Kutta3],         o = 3
# [], alg_embedded ∈ [:RK4],           o = 4
# [], alg_embedded ∈ [:RK5],           o = 5

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

