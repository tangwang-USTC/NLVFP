
"""
  A single-step multistage RK algorithm with inner iteration for
  the Fokker-Planck collision equations. The inner iteration is performed by
  the embedded implicit methods (implicit Euler method, the trapezoidal method)
  or the explicit methods such as explicit Euler, Heun3 or Kutta3 method.
  
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
    orderEmbeded: (=2, default), when `orderRK ≥ 3` will be actived
    rsEmbeded: (=2, default), when `orderEmbeded ≥ 3` will be actived

  Outputs:
    Mck1integralk!(Rck1, Mck1, pst0, Nstep; 
        orderRK=orderRK, rsRK=rsRK, iterRK=iterRK, 
        orderEmbeded=orderEmbeded, rsEmbeded=rsEmbeded, iterEmbeded=iterEmbeded, 
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
        is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
        dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
        is_MjMs_max=is_MjMs_max,ratio_dtk1=ratio_dtk1,
        is_moments_out=is_moments_out,is_Cerror_dtnIKTs=is_Cerror_dtnIKTs,
        is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
        is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
        is_fixed_NK=is_fixed_NK,is_nMod_update_back=is_nMod_update_back)

"""

# [k,s,i],
function Mck1integralk!(Rck1::AbstractArray{T,N}, Mck1::AbstractArray{T,N}, 
    ps::Dict{String,Any}, Nstep::Int64; 
    orderRK::Int64=2, rsRK::Int64=2, iterRK::Int64=3,
    orderEmbeded::Int64=2, rsEmbeded::Int64=2, iterEmbeded::Int64=3,
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
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false,
    is_MjMs_max::Bool=false,ratio_dtk1::T=1.2,
    is_moments_out::Bool=false,is_Cerror_dtnIKTs::Bool=true,
    is_dtk_GKMM_dtIK::Bool=true,rtol_dtsa::T=1e-8,
    is_optim_CnIK::Bool=false,is_nhnMod_adapt::Bool=false,
    is_NK_adapt_max::Bool=false,is_nai_const::Bool=true,is_fixed_NK::Bool=false,
    is_nMod_update_back::Bool=false,is_nMod_update_advance::Bool=false) where{T,N}

    # ratio_dtk1 = dtk / dtk
    # ps_copy = deepcopy(ps)
    tk = deepcopy(ps["tk"])
    if is_fixed_timestep == false
        tauk = ps["tauk"]
    end

    nsk1 = ps["ns"]
    mak1 = ps["ma"]
    Zqk1 = ps["Zq"]
    spices = ps["spices"]

    nak = ps["nk"]
    Iak = ps["Ik"]
    Kak = ps["Kk"]
    vathk = ps["vthk"]               # vath_(k)
    sak = deepcopy(ps["sak"])
    dtsabk = deepcopy(ps["dtsabk"])
    Rdtsabk = dtsabk / sum(sak)
    # @show 0, Rdtsabk
    # @show dtsabk, sak
    # wqedgfb


    nnv = ps["nnv"]
    # nc0, nck = ps["nc0"], ps["nck"]
    ocp = ps["ocp"]
    vGdom = ps["vGm"]

    # vhe = ps["vhe"]
    # vhk = ps["vhk"]
    # nvlevel0, nvlevele0 = ps["nvlevel0"], ps["nvlevele0"]          # nvlevele = nvlevel0[nvlevele0]

    naik = ps["naik"]
    uaik = ps["uaik"]
    vthik = ps["vthik"]
    nModk = ps["nModk"]

    # muk, Mμk, Munk, Mun1k, Mun2k = ps["muk"], ps["Mμk"], ps["Munk"], ps["Mun1k"], ps["Mun2k"]
    LMk = ps["LMk"]
    # w3k = ps["w3k"]
    # err_dtnIK = ps["err_dtnIK"]         # w3k = vₜₕ⁻¹∂ₜvₜₕ
    DThk = ps["DThk"]                     # δT̂
    Mhck = ps["Mhck"]
    errMhcop = ps["errMhcop"]           # The errors of renormalized kinetic moments in moment optimization step (solving the characteristic equations)
    nMjMs = ps["nMjMs"]
    RDMck1 = ps["RDMck1"]

    edtnIKTs = deepcopy(ps["edtnIKTsk"])
    CRDn = ps["CRDnk"]                          # The discrete error of number density conservation
    Nspan_optim_nuTi = Nspan_nuTi_max[2] * ones(T,3)              # [nai, uai, vthi]

    nvG = 2 .^ nnv .+ 1
    LM1k = maximum(LMk) + 1

    is_nMod_renew = zeros(Bool,nsk1)

    k = 0       # initial step to calculate the values `ℭ̂ₗ⁰` and `w3k = Rdtvath = 𝒲 / 3`
    # where `is_update_nuTi = false` and `nai, uai, vthik` are convergent according to `fvL`

    # # # # Updating the conservative momentums `n, I, K`
    nak1 = deepcopy(nak)
    Iak1 = deepcopy(Iak)
    Kak1 = deepcopy(Kak)
    vathk1 = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
    sak1 = deepcopy(sak)
    dtsabk1 = deepcopy(dtsabk)
    Rdtsabk1 = deepcopy(Rdtsabk)

    naik1 = deepcopy(naik)
    uaik1 = deepcopy(uaik)
    vthik1 = deepcopy(vthik)
    # Structure preserving
    nModk1 = deepcopy(nModk)
    DThk1 = deepcopy(DThk)             # δT̂
    Mhck1 = deepcopy(Mhck)
    ρk1 = mak1 .* nak1

    errMhc = deepcopy(Mhck)
    @show Ia, Mck1[1,2,:]    
    # edsdrgrrgrg
    errRhck12 = zero.(Rck1[1:njMs, :, 1:2])      # Storages for two-spices collision `Cab`
    if nsk1 ≥ 3
        Rck12 = zero.(Rck1[:,:,1:2])
        edtnIKTs2 = zero.(edtnIKTs[:,1:2])
        DThk12 = zeros(T,2)
        naik2 = Vector{AbstractVector{T}}(undef,2)
        uaik2 = Vector{AbstractVector{T}}(undef,2)
        vthik2 = Vector{AbstractVector{T}}(undef,2)
    end

    dtk = ps["dt"]
    # RdMsk = zeros(T, 3, ns)                      # [RdIak, RdKak, Rdvathk]
    # δvathi = zeros(T,ns)
    # criterions = [ps["DThk"]; ps["err_dtnIK"]; δvathi] 
    
    dtMhck1 = deepcopy(Mhck1)
    Rhck1 = deepcopy(Mhck1)
    Rhck = deepcopy(Mhck1)
    MhckMck!(Rhck1,Rck1,nMjMs,LMk,ρk1,vathk1,nsk1)
    Rdtvthk1 = Rck1[end,1,:] ./ vathk1
    # @show Rdtvthk1

    println(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
    dtMhck!(dtMhck1,Rhck1,Mhck1,nMjMs,LMk,Rdtvthk1,nsk1;L_Mh_limit=L_Mh_limit)
    dtMhck = deepcopy(dtMhck1)
    # isp = 1
    # if norm(dtMhck1[isp][1:2,1]) ≥ epsT1000
    #     printstyled("Rhck1 =", fmtf2.(Rhck1[isp][:,1]), color=:green,"\n")
    #     printstyled("dtMhck1a =", fmtf2.(dtMhck1[isp][:,1]), color=:green,"\n")
    # else
    #     printstyled("Rhck1 =", fmtf2.(Rhck1[isp][:,1]), color=:green,"\n")
    #     printstyled("dtMhck1a =", fmtf2.(dtMhck1[isp][:,1]), color=:red,"\n")
    # end
    # isp = 2
    # if norm(dtMhck1[isp][1:2,1]) ≥ epsT1000
    #     printstyled("Rhck1 =", fmtf2.(Rhck1[isp][:,1]), color=:green,"\n")
    #     printstyled("dtMhck1b =", fmtf2.(dtMhck1[isp][:,1]), color=:green,"\n")
    # else
    #     printstyled("Rhck1 =", fmtf2.(Rhck1[isp][:,1]), color=:green,"\n")
    #     printstyled("dtMhck1b =", fmtf2.(dtMhck1[isp][:,1]), color=:red,"\n")
    # end

    Rck = deepcopy(Rck1)

    dtKIak = zeros(T,2,ns)                #  [[K, I], ns]
    dtKIak[1,:] = Rck1[2,1,:] * CMcKa     # K
    dtKIak[2,:] = Rck1[1,2,:]             # I

    @show is_fvL_CP,dtk
    @show 0, dtKIak[1,:], NK

    Mck = deepcopy(Mck1)
    vathk1i = deepcopy(vathk)          # zeros(T,nsk1)
    count = 0
    k = 1
    RDMhck1max = ones(T,ns)
    done = true
    if orderRK ≤ 2
        while done
            # parameters
            tk = deepcopy(ps["tk"])
            dtk = deepcopy(ps["dt"])
            Nt_save = ps["Nt_save"]
            count_save = ps["count_save"]

            if NCase ≥ 2
                koutput = 2^(iCase-1)
                if k == koutput
                    is_plot_DflKing = true
                    is_plot_dfln_thoery= true
                else
                    is_plot_DflKing = false
                    is_plot_dfln_thoery = false
                end
            end
            if tk ≤ 0.05
                is_nuTi_initial = true
            else
                is_nuTi_initial = false
            end
            if is_NKCnhC
                if tk ≤ 0.05
                    rtol_DnuTi = 8e-1
                else
                    rtol_DnuTi = epsT
                end
            end
            if is_nMod_update_advance_tk
                if k ≤ 2
                    is_nMod_update_advance = false
                else
                    is_nMod_update_advance = true
                end
            else
                is_nMod_update_advance = false
            end
            if k ≤ 3
                NKk1, NKmaxk1 = 1NK0, 1NKmax0
            else
                NKk1, NKmaxk1 = 1NKk, 1NKmax
            end
            # NKk1, NKmaxk1 = 1NK, 1NKmax
            
            # println()
            println("**************------------******************------------*********")
            printstyled("k=",k,",tk,dt,Rdt=",fmtf2.([ps["tk"],dtk,dtk/ps["tk"]]),"\n";color=:blue)

            dtk1 = 1dtk
            if nsk1 == 2
                # @show 8880, Mhck1[1][:,1] .- 1
                dtk1 = Mck1integrali_rs2!(Mck1, Rck1, RDMck1, edtnIKTs, errRhck12, 
                    Mhck1, Mhck, Rhck1, Rhck, dtMhck1, dtMhck, 
                    errMhc, errMhcop, RDMhck1max, nMjMs,
                    nvG, ocp, vGdom, LMk, LM1k, 
                    naik1, uaik1, vthik1, nModk1, naik, uaik, vthik, nModk, 
                    CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, Rdtsabk1, DThk1, Iak1, Kak1, vathk1i, 
                    Mck, Rck, nak, vathk, Rdtsabk, Nspan_optim_nuTi, NKk1, NKmaxk1, k, tk, dtk;
                    orderEmbeded=orderRK,iterEmbeded=iterRK,
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
                    is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
                    is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                    dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
                    is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
                    is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
                    is_nuTi_initial=is_nuTi_initial,is_fixed_NK=is_fixed_NK,
                    is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)

                dtMhck1 = deepcopy(Mhck1)
                Rhck1 = deepcopy(Mhck1)
                MhckMck!(Rhck1,Rck1,nMjMs,LMk,ρk1,vathk1,nsk1)
                Rdtvthk1 = Rck1[end,1,:] ./ vathk1
                # @show 8881, Mhck1[1][:,1] .- 1
                # @show Rdtvthk1
    
                # println(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;")
                dtMhck!(dtMhck1,Rhck1,Mhck1,nMjMs,LMk,Rdtvthk1,nsk1;L_Mh_limit=L_Mh_limit)
                # isp = 1
                # @show  1 / Nτ_fix
                # if norm(dtMhck1[isp][1:2,1]) ≥ epsT1000
                #     printstyled("Rhck1 =", fmtf2.(Rhck1[isp][:,1]), color=:green,"\n")
                #     printstyled("dtMhck1a =", fmtf2.(dtMhck1[isp][:,1]), color=:green,"\n")
                #     printstyled("DdtMhck1a =", fmtf2.(dtMhck1[isp][:,1]-Rhck1[isp][:,1]), color=:green,"\n")
                #     @show fmtf2.(dtMhck1[isp][:,1] / Nτ_fix)
                # else
                #     printstyled("Rhck1 =", fmtf2.(Rhck1[isp][:,1]), color=:green,"\n")
                #     printstyled("dtMhck1a =", fmtf2.(dtMhck1[isp][:,1]), color=:red,"\n")
                #     printstyled("DdtMhck1a =", fmtf2.(dtMhck1[isp][:,1]-Rhck1[isp][:,1]), color=:green,"\n")
                #     @show fmtf2.(dtMhck1[isp][:,1] / Nτ_fix)
                # end
                # isp = 2
                # if norm(dtMhck1[isp][1:2,1]) ≥ epsT1000
                #     printstyled("Rhck1 =", fmtf2.(Rhck1[isp][:,1]), color=:green,"\n")
                #     printstyled("dtMhck1b =", fmtf2.(dtMhck1[isp][:,1]), color=:green,"\n")
                #     printstyled("DdtMhck1a =", fmtf2.(dtMhck1[isp][:,1]-Rhck1[isp][:,1]), color=:green,"\n")
                #     @show fmtf2.(dtMhck1[isp][:,1] / Nτ_fix)
                # else
                #     printstyled("Rhck1 =", fmtf2.(Rhck1[isp][:,1]), color=:green,"\n")
                #     printstyled("dtMhck1b =", fmtf2.(dtMhck1[isp][:,1]), color=:red,"\n")
                #     printstyled("DdtMhck1a =", fmtf2.(dtMhck1[isp][:,1]-Rhck1[isp][:,1]), color=:green,"\n")
                #     @show fmtf2.(dtMhck1[isp][:,1] / Nτ_fix)
                # end

                # # # Updating the parameters `nModk1`
                if is_nhnMod_adapt && prod(nModk1) ≥  255555
                    # reducing the number of `nModk1` according to and updating `naik, uaik, vthik`
                    nMod_update!(is_nMod_renew, naik1, uaik1, vthik1, nModk1, nsk1)
                    # @show is_nMod_renew, nModk1
                    # @show naik1

                    if is_fixed_timestep == false
                        frgnm
                        if sum(is_nMod_renew) > 0

                            # Updating `Mhck1` and `Mck1` owing to the reduced parameters `naik, uaik, vthik`
                            Mhck1 = MsnntL2fL0(Mhck1,nMjMs,LMk,naik1,uaik1,vthik1,nModk1,nsk1;is_renorm=is_renorm)
                            MckMhck!(Mck1,Mhck1,nMjMs,LMk,ρk1,vathk1,nsk1)

                            # Updating `Rck1` owing to the reduced parameters `naik, uaik, vthik`
                            uak1 = Mck1[1,2,:] ./ ρk1
                            # @show 78, sum(naik[2] .* uaik[2]) .* vathk1[2] - Mck1[1,2,2] / ρa[2]
                            # @show 78, sum(naik[1] .* uaik[1]) .* vathk1[1] - Mck1[1,2,1] / ρa[1]
                            dtk1 = dtMcab2!(Rck1, edtnIKTs, errRhck12, nMjMs,
                                    nvG, ocp, vGdom,  LMk, LM1k, 
                                    naik1, uaik1, vthik1, nModk1, naik, uaik, vthik, nModk, 
                                    CΓ, εᵣ, mak1, Zqk1, spices, nak1, uak1, vathk1, 
                                    DThk1, dtk1;
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
                            if is_nai_const
                                tau_fM!(tauk, mak1, Zqk1, spices, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                                printstyled("0: Updating the time scale, tau=", tauk,color=:green,"\n")
                                count = 0
                            end
                        else
                            if is_nai_const
                                count += 1
                                if count == count_tau_update
                                    tau_fM!(tauk, mak1, Zqk1, spices, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                                    printstyled("1: Updating the time scale, tau=", tauk,color=:green,"\n")
                                    count = 0
                                end
                            end
                        end
                    else
                        if sum(is_nMod_renew) > 0
                            # Updating `Mhck1` and `Mck1` owing to the reduced parameters `naik, uaik, vthik`
                            Mhck1 = MsnntL2fL0(Mhck1,nMjMs,LMk,naik1,uaik1,vthik1,nModk1,nsk1;is_renorm=is_renorm)
                            MckMhck!(Mck1,Mhck1,nMjMs,LMk,ρk1,vathk1,nsk1)

                            # Updating `Rck1` owing to the reduced parameters `naik, uaik, vthik`
                            uak1 = Mck1[1,2,:] ./ ρk1
                            
                            dtk1 = dtMcab2!(Rck1, edtnIKTs, errRhck12, nMjMs,
                                    nvG, ocp, vGdom, LMk, LM1k, naik1, uaik1, vthik1, nModk1, 
                                    CΓ, εᵣ, mak1, Zqk1, spices, nak1, uak1, vathk1, DThk1, dtk1;
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
                        end
                    end
                end
                CRDn[1] = min(abs(edtnIKTs[1,1]),abs(edtnIKTs[1,2]))
            else
                dtk1 = Mck1integrali_rs2!()

                # # # Updating the parameters `nModk1`
                if prod(nModk1) ≥  2
                    # reducing the number of `nModk1` according to and updating `naik, uaik, vthik`
                    nMod_update!(is_nMod_renew, naik, uaik, vthik, nModk, nsk1)

                    if is_fixed_timestep == false
                        if sum(is_nMod_renew) > 0

                            # Updating `Mhck1` and `Mck1` owing to the reduced parameters `naik, uaik, vthik`
                            Mhck1 = MsnntL2fL0(Mhck1,nMjMs,LMk,naik1,uaik1,vthik1,nModk1,nsk1;is_renorm=is_renorm)
                            MckMhck!(Mck1,Mhck1,nMjMs,LMk,ρk1,vathk1,nsk1)

                            # Updating `Rck1` owing to the reduced parameters `naik, uaik, vthik`
                            uak1 = Mck1[1,2,:] ./ ρk1
                            if nsk1 == 2
                            else
                            end
                            dtk1 = dtMcabn!()

                            if is_nai_const
                                tau_fM!(tauk, mak1, Zqk1, spices, nak1, vathk1, Coeff_tau, 
                                        naik2, vthik2, naik, vthik, nModk1, nsk1)
                                printstyled("0: Updating the time scale, tau=", tauk,color=:green,"\n")
                                count = 0
                            end
                        else
                            if is_nai_const
                                count += 1
                                if count == count_tau_update
                                    tau_fM!(tauk, mak1, Zqk1, spices, nak1, vathk1, Coeff_tau, 
                                            naik2, vthik2, naik, vthik, nModk1, nsk1)
                                    printstyled("1: Updating the time scale, tau=", tauk,color=:green,"\n")
                                    count = 0
                                end
                            end
                        end
                    else
                        # fujjkkkk
                    end
                elseif is_fixed_timestep == false
                    if is_nai_const
                        count += 1
                        if count == count_tau_update
                            tau_fM!(tauk, mak1, Zqk1, spices, nak1, vathk1, Coeff_tau, 
                                    naik2, vthik2, naik, vthik, nModk1, nsk1)
                            printstyled("2: Updating the time scale, tau=", tauk,color=:green,"\n")
                            count = 0
                        end
                    end
                end
            end
            # @show 8882, Mhck1[1][:,1] .- 1

            # Updating `Iak1` and `Kak1` from `Mck1`
            Kak1 = Mck1[2,1,:] * CMcKa 
            Iak1 = Mck1[1,2,:]

            # RDKab = sum(Kak1) / Kab0 - 1
            # @show 7773,nModk1, RDKab
            # dtKIak[1,:] = Rck1[2,1,:] * CMcKa     # dtKa
            # dtKIak[2,:] = Rck1[1,2,:]             # dtIa
            # if abs(RDKab) > rtol_nIK_warn
            #     @warn("0, Energy conservation is not achieved:",RDKab)
            #     if abs(RDKab) > rtol_nIK_error
            #         @error("0, Energy conservation is not achieved:",RDKab)
            #     end
            #     egfdbf
            # end

            Rdtsabk = deepcopy(Rdtsabk1)
            # [ns], Updating the entropy and its change rate with assumpation of dauble-Maxwellian distribution
            entropy_fDM!(sak1,mak1,nak1,vathk1,Iak1,Kak1,nsk1)                        

            # dtsabk1 = dtsak1 + dtsbk1
            # [nsk1 = 2] Iahk = uak1 ./ vathk1
            dtsabk1 = entropy_rate_fDM(mak1,vathk1,Iak1 ./ (ρk1 .* vathk1),dtKIak[2,:],dtKIak[1,:],nsk1)
            
            Rdtsabk1 = dtsabk1 / sum(sak1)
            # @show dtsabk1 , sak1
            # Rdtsabk1 = entropyN_rate_fDM(mak1,nak1,vathk1,Iak1,Kak1,dtKIak[2,:],dtKIak[1,:],nsk1)
            @show Rdtsabk1
            # sdfvb

            # When `nₖ₊₁ˢ`, `Iₖ₊₁ˢ` and `Kₖ₊₁ˢ` reach convergence, conservations and
            # the constraint equation is satisfied: `K̂ = 3/2 + Î`,

            is_corrections[1] ? nothing : nak = deepcopy(nak1)
            Iak = deepcopy(Iak1)
            Kak = deepcopy(Kak1)
            vathk = deepcopy(vathk1)
            Mck = deepcopy(Mck1)
            Rck = deepcopy(Rck1)
            Mhck = deepcopy(Mhck1)
            @show 8885, Mhck1[1][:,1] .- 1
            @show 8885, Mhck1[2][:,1] .- 1
            # erfgh
            naik = deepcopy(naik1)
            uaik = deepcopy(uaik1)
            vthik = deepcopy(vthik1)
            nModk = deepcopy(nModk1)
            # @show nModk1
            # @show vthik1
            # @show uaik1
            # @show naik1
            
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
            if is_fixed_timestep
            else
                ps["dt"] = dtk1
            end
            # isp = 1
            # @show nModk1[isp], sum(naik1[isp][1:nModk1[isp]] .* vthik1[isp][1:nModk1[isp]].^2) .- 1
            # println("..............................................")
            # isp = 2
            # @show nModk1[isp], sum(naik1[isp][1:nModk1[isp]] .* vthik1[isp][1:nModk1[isp]].^2) .- 1
            # println("..............................................")

            # ps["tauk"] = tauk
            # ps["nak"] = deepcopy(nak1)
            ps["Ik"] = deepcopy(Iak1)
            ps["Kk"] = deepcopy(Kak1)
            # ps["uCk"] = deepcopy(uCk1)
            ps["DThk"] = deepcopy(DThk1)
            ps["naik"] = deepcopy(naik1)
            ps["uaik"] = deepcopy(uaik1)
            ps["vthik"] = deepcopy(vthik1)
            ps["nModk"] = deepcopy(nModk1)
            ps["vGm"] = vGdom

            ps["LMk"] = deepcopy(LMk)
            ps["nMjMs"] = deepcopy(nMjMs)
            ps["Mhck"] = deepcopy(Mhck1)
            ps["errMhcop"] = deepcopy(errMhcop)
            ps["RDMck1"] = deepcopy(RDMck1)
            ps["sak"] = deepcopy(sak1)
            ps["dtsabk"] = deepcopy(dtsabk1)
            ps["edtnIKTsk"] = deepcopy(edtnIKTs)
            ps["CRDnk"][1] = deepcopy(CRDn[1])
            # @show ps["errMhcop"]

            # Saving the dataset at `(k+1)ᵗʰ` step
            if count_save == Nt_save
                ps["count_save"] = 1
                # if k ≥ 3
                    data_Ms_saving(ps;is_moments_out=is_moments_out,is_Cerror_dtnIKTs=is_Cerror_dtnIKTs)
                # end
            else
                ps["count_save"] = count_save + 1
            end
            if NCase ≥ 2
                if k == koutput
                    RDTab33vec[iCase,3] = ps["tk"]
                    # RDuab33vec[iCase,3] = ps["tk"]
                end
            end

            # Terminating the progrom when I. reaches the maximum time moment; II. number of time step; III. equilibrium state.
            if abs(Rdtsabk1) ≤ rtol_dtsa_terminate
                if k ≥ 5
                    @warn("The system has reached the equilibrium state when", Rdtsabk1)
                    break
                end
            else
                if ps["tk"] > ps["tspan"][2]
                    # if abs(tk - ps["tspan"][2]) / tk ≤ rtol_tk
                        @warn("The system has reached the maximum time moment at", tk)
                        break
                    # else
                    #     ps["tk"] = ps["tspan"][2]
                    #     dtk1 = ps["tk"] - tk
                    #     ps["dt"] = dtk1
                    # end
                else
                    if k > Nstep
                        @warn("The system has reached the maximum iterative step", Nstep)
                        break
                    end
                end
                k += 1
            end
        end
    else
        if orderRK == 4
            eddddd
        else
            errorsddd
        end
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
    dtk1 = Mck1integrali_rs2!(Mck1, Rck1, RDMck1, edtnIKTs, CRDn, errRhck12, 
        Mhck1, Mhck, Rhck1, Rhck, dtMhck1, dtMhck,errMhc, errMhcop, nMjMs,
        nvG, ocp, vGdom, LMk, LM1k, 
        naik1, uaik1, vthik1, nModk1, naik, uaik, vthik, nModk, 
        CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, nsk1, DThk1,
        Iak1, Kak1, vathk1i, Mck, Rck, nak, vathk, NK, NKmax, kt, tk, dtk;
        orderEmbeded=orderEmbeded,iterEmbeded=iterEmbeded, 
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
        is_extrapolate_FLn=is_extrapolate_FLn,rtol_DnIK=rtol_DnIK,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
        dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
        is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
        is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
        is_nuTi_initial=is_nuTi_initial,is_fixed_NK=is_fixed_NK,
        is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
"""
# [sᵗʰ,i], [:Trapz, :LobattoIIIA2],    rs = 2, o = 2
# ns ≥ 3

# ns = 2
function Mck1integrali_rs2!(Mck1::AbstractArray{T,N},Rck1i::AbstractArray{T,N},RDMck1::AbstractArray{T,N}, 
    edtnIKTs::AbstractArray{T,N2},errRhck12::AbstractArray{T,N},
    Mhck1::Vector{Matrix{T}},Mhck::Vector{Matrix{T}},Rhck1::Vector{Matrix{T}},Rhck::Vector{Matrix{T}},
    dtMhck1::Vector{Matrix{T}},dtMhck::Vector{Matrix{T}},errMhc::Vector{Matrix{T}},
    errMhcop::Vector{Matrix{T}},RDMhck1max::AbstractVector{T},nMjMs::Vector{Int64},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2},LMk::Vector{Int64},LM1k::Int64,
    naik1::Vector{AbstractVector{T}},uaik1::Vector{AbstractVector{T}},vthik1::Vector{AbstractVector{T}},nModk1::Vector{Int64},
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},nModk::Vector{Int64},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},spices::Vector{Symbol},
    nak1::AbstractVector{T},vathk1::AbstractVector{T},Rdtsabk1::T,DThk1::AbstractVector{T},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},vathk1i::AbstractVector{T},
    Mck::AbstractArray{T,N},Rck::AbstractArray{T,N},
    nak::AbstractVector{T},vathk::AbstractVector{T},Rdtsabk::T,
    Nspan_optim_nuTi::AbstractVector{T},NK::Int64,NKmax::Int64,kt::Int64,tk::T,dtk::T;
    orderEmbeded::Int64=2,iterEmbeded::Int64=0,
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
    is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false,
    is_optim_CnIK::Bool=false,is_nhnMod_adapt::Bool=false,
    is_NK_adapt_max::Bool=false,is_nai_const::Bool=true,
    is_nuTi_initial::Bool=true,is_fixed_NK::Bool=false,
    is_nMod_update_back::Bool=false,is_nMod_update_advance::Bool=false) where{T,N,N2}

    i_iter = 0                  # and Checking the `i = 1` iteration
    # Applying the explicit Euler step
    δvathi = ones(T,2)        # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
    δvathk1 = zeros(T,2)      # = vathk1 ./ vathk1
    # Rck1i .= 0.0
    dtk1 = 1dtk
    @show 11, nModk1, nModk, vthik1
    # @show naik1
    if iterEmbeded == 0
        dtk1 = Mck1ExEuler!(Mck1, Rck1i, RDMck1, Mck, edtnIKTs, errRhck12, 
            Mhck1, Mhck, Rhck1, dtMhck1,
            errMhc, errMhcop, RDMhck1max, nMjMs,
            nvG, ocp, vGdom, LMk, LM1k, 
            naik1, uaik1, vthik1, nModk1,  naik, uaik, vthik, nModk, 
            CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, Rdtsabk1, Rdtsabk, 
            DThk1, Iak1, Kak1, δvathk1, Nspan_optim_nuTi, NK, NKmax, kt, tk, dtk1;
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
            dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
            is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
            is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
            is_nuTi_initial=is_nuTi_initial,is_fixed_NK=is_fixed_NK,
            is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
        vathk1i[:] = deepcopy(vathk1)
        return dtk1
    else
        dtk1 = Mck1ExEuler!(Mck1, Rck1i, Mck, edtnIKTs, errRhck12, 
            Mhck1, Mhck, Rhck1, dtMhck1,
            errMhc, errMhcop, RDMhck1max, nMjMs,
            nvG, ocp, vGdom, LMk, LM1k, 
            naik1, uaik1, vthik1, nModk1,  naik, uaik, vthik, nModk, 
            CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, Rdtsabk1, Rdtsabk, 
            DThk1, Iak1, Kak1, δvathk1, Nspan_optim_nuTi, NK, NKmax, kt, tk, dtk1;
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
            dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
            is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
            is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
            is_nuTi_initial=is_nuTi_initial,is_fixed_NK=is_fixed_NK,
            is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
        vathk1i[:] = deepcopy(vathk1)
    end
    
    # # If `iterEmbeded ≤ 0`, then degenerate into the explicit Euler method (ExEuler)
    δvathi_up = zeros(T,2)
    if orderEmbeded == 1
        while i_iter < iterEmbeded
            i_iter += 1
            println("....................")
            @show i_iter
            nak1[:] = deepcopy(nak)
            vathk1[:] = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
            # Rck1i[:,:,:] = (Rck + Rck1i) / 2        # Rck1k = Rc_(k+1/2)
            dtk1 = 1dtk
            
            dtk1 = Mck1integral!(Mck1, Rck1i, RDMck1, Mck, edtnIKTs, errRhck12, 
                Mhck1, Mhck, Rhck1, dtMhck1,
                errMhc, errMhcop, RDMhck1max, nMjMs,
                nvG, ocp, vGdom, LMk, LM1k, 
                naik1, uaik1, vthik1, nModk1, naik, uaik, vthik, nModk, 
                CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, Rdtsabk1, Rdtsabk, 
                DThk1, Iak1, Kak1, δvathk1, Nspan_optim_nuTi, NK, NKmax, kt, tk, dtk1;
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
                dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
                is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=false,
                is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
                is_nuTi_initial=is_nuTi_initial,is_fixed_NK=is_fixed_NK,
                is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
            δvathi = vathk1i ./ vathk1 .- 1                # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
            ratio_vathi = δvathi - δvathi_up
    
            # # Rck1 = Rck1i
            if norm(ratio_vathi) ≤ rtol_vthi || norm(δvathi) ≤ atol_vthi
                break
            end

            vathk1i[:] = deepcopy(vathk1)
            @show 1,i_iter, δvathk1, δvathi
        end
    elseif orderEmbeded == 2
        while i_iter < iterEmbeded
            i_iter += 1
            println("....-----------..........")
            @show i_iter
            nak1[:] = deepcopy(nak)
            vathk1[:] = deepcopy(vathk)              # vath_(k+1)_(i) which will be changed in the following codes
            Rck1i[:,:,:] = (Rck + Rck1i) / 2        # Rck1k = Rc_(k+1/2)
            dtMhck1[:,:,:] = (dtMhck + dtMhck1) / 2        # Rck1k = Rc_(k+1/2)
            # @show Rck[nMjMs[1]+1,1,1], Rck1i[nMjMs[1]+1,1,1]
            dtk1 = 1dtk
            # @show 550, nModk1, vthik1
            
            dtk1 = Mck1integral!(Mck1, Rck1i, RDMck1, Mck, edtnIKTs, errRhck12, 
                Mhck1, Mhck, Rhck1, dtMhck1,
                errMhc, errMhcop, RDMhck1max, nMjMs,
                nvG, ocp, vGdom, LMk, LM1k, 
                naik1, uaik1, vthik1, nModk1, naik, uaik, vthik, nModk, 
                CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, Rdtsabk1, Rdtsabk, 
                DThk1, Iak1, Kak1, δvathk1, Nspan_optim_nuTi, NK, NKmax, kt, tk, dtk1;
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
                dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
                is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=false,
                is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
                is_nuTi_initial=is_nuTi_initial,is_fixed_NK=is_fixed_NK,
                is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
            δvathi = vathk1i ./ vathk1 .- 1                # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
            ratio_vathi = δvathi - δvathi_up
    
            # # Rck1 = Rck1i
            if norm(ratio_vathi) ≤ rtol_vthi || norm(δvathi) ≤ atol_vthi
                break
            end
            vathk1i[:] = deepcopy(vathk1)
            @show 2,i_iter, δvathk1, δvathi
        end
    else
        dfcvghggh
    end
    if i_iter ≥ iterEmbeded
        @warn(`The maximum number of iteration reached before the Trapz method to be convergence!!!`)
    end
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
    dtk = Mck1integral!(Mck1, Rck1, Mck, edtnIKTs, errRhck12, 
        Mhck1, Mhck, Rhck1, dtMhck1,
        errMhc, errMhcop, RDMhck1max, nMjMs, 
        nvG, ocp, vGdom, LMk, LM1k, 
        naik1, uaik1, vthik1, nModk1, naik, uaik, vthik, nModk, 
        CΓ, εᵣ, mak1, Zqk1, spices, nak1, vathk1, 
        nsk1, DThk1, Iak1, Kak1, δvathk1, NK, NKmax, kt, tk, dtk1;
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
        dtk_order_Rc=dtk_order_Rc,is_dtk_order_Rcaa=is_dtk_order_Rcaa,
        is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
        is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
        is_nuTi_initial=is_nuTi_initial,is_fixed_NK=is_fixed_NK,
        is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
"""

# [iᵗʰ], 
# # ns ≥ 3

# ns = 2
function Mck1ExEuler!(Mck1::AbstractArray{T,N},Rck1::AbstractArray{T,N},Mck::AbstractArray{T,N},
    edtnIKTs::AbstractArray{T,N2},errRhck12::AbstractArray{T,N},
    Mhck1::Vector{Matrix{T}},Mhck::Vector{Matrix{T}},Rhck1::Vector{Matrix{T}},
    dtMhck1::Vector{Matrix{T}},errMhc::Vector{Matrix{T}},
    errMhcop::Vector{Matrix{T}},RDMhck1max::AbstractVector{T},nMjMs::Vector{Int64},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2},LMk::Vector{Int64},LM1k::Int64,
    naik1::Vector{AbstractVector{T}},uaik1::Vector{AbstractVector{T}},vthik1::Vector{AbstractVector{T}},nModk1::Vector{Int64},
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},nModk::Vector{Int64},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},spices::Vector{Symbol},
    nak1::AbstractVector{T},vathk1::AbstractVector{T},Rdtsabk1::T,Rdtsabk::T,DThk1::AbstractVector{T},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},δvathk1::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},NK::Int64,NKmax::Int64,kt::Int64,tk::T,dtk::T;
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
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false,
    is_optim_CnIK::Bool=false,is_nhnMod_adapt::Bool=false,
    is_NK_adapt_max::Bool=false,is_nai_const::Bool=true,
    is_nuTi_initial::Bool=true,is_fixed_NK::Bool=false,
    is_nMod_update_back::Bool=false,is_nMod_update_advance::Bool=false) where{T,N,N2}
    
    dtk1 = 1dtk
    ρk1 = mak1 .* nak1
    # `vathk1 = zeros(nsk1)`
    
    Mck1integral0!(Mck1,Mck,Rck1,nsk1,dtk1)
    
    # Calculate the parameters `nak1,vathk1,Iak1,Kak1` from `Mck1`
    nIKT_update!(nak1,vathk1,Iak1,Kak1,δvathk1,mak1,nsk1,Mck1;
                is_vth_ode=is_vth_ode,is_corrections_na=is_corrections[1])
    
    RerrDKab = sum(Kak1) / Kab0 - 1
    if abs(RerrDKab) > rtol_nIK_warn
        @warn("2, Energy conservation is not achieved:",RerrDKab)
        if abs(RerrDKab) > rtol_nIK_error
            @error("2, Energy conservation is not achieved:",RerrDKab)
        end
    end
    # # Computing the re-normalized moments 
    MhckMck!(Mhck1,Mck1[1:njMs,:,:],nMjMs,LMk,ρk1,vathk1,nsk1)
    # @show 221, fmtf2.(Mhck1[1][:,1] .- 1)
    # @show 221, fmtf2.(Mhck1[2][:,1] .- 1)

    # ODE for `∂ₜℳ̂ⱼₗ⁰(tk1) = dtMhck1`
    if is_dMhck1_Mhck
        Mhck1integral0!(Mhck1,Mhck,dtMhck1,nsk1,dtk1)
        # @show 223, fmtf2.(Mhck1[1][:,1] .- 1)
        # @show 223, fmtf2.(Mhck1[2][:,1] .- 1)
    end

    # @show 552, nModk1, vthik1

    dtk1 = submoment!(naik1, uaik1, vthik1, nModk1, 
            LMk, naik, uaik, vthik, nModk, 
            Mhck1, errMhcop, nMjMs, 
            edtnIKTs, Rdtsabk1, Rdtsabk, ns,
            Nspan_optim_nuTi,NK,NKmax,kt,tk,dtk1;
            Nspan_nuTi_max=Nspan_nuTi_max,
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
            is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
            is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
            is_nuTi_initial=is_nuTi_initial,is_fixed_NK=is_fixed_NK,
            is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
   
    # Updating `Mhck1` and `Mck1` owing to the reduced parameters `naik, uaik, vthik`
    if is_Mhc_reformula
        Mhck1 = MsnntL2fL0(Mhck1,nMjMs,LMk,naik1,uaik1,vthik1,nModk1,nsk1;is_renorm=is_renorm)
        MckMhck!(Mck1,Mhck1,nMjMs,LMk,ρk1,vathk1,nsk1)
    end
    # @show 222, fmtf2.(Mhck1[1][:,1] .- 1)
    # @show 222, fmtf2.(Mhck1[2][:,1] .- 1)
    for isp in 1:ns
        if norm(uaik[isp]) ≤ atol_IKTh
            L1 = 1
            RDMhck1max[isp] = maximum(abs.((Mhck1[isp][:,L1] ./ Mhck[isp][:,L1] .- 1) / dtk))
        else
            sddffff
        end
    end
    # @show 131, nModk1, vthik1

    # uk = Iak1 ./ ρk1
    if dtk_order_Rc == :min && is_dtk_order_Rcaa == false
        dtk1 = dtMcab2!(Rck1,edtnIKTs,errRhck12,nMjMs,
               nvG,ocp,vGdom,LMk,LM1k,
               naik1,uaik1,vthik1,nModk1,
               CΓ,εᵣ,mak1,Zqk1,spices,nak1,Iak1 ./ ρk1,vathk1,DThk1,dtk1;
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
    else
        rfgth
        dtk1 = dtMcab2!(Rck1,edtnIKTs,errRhck12,Mck1,nMjMs,
               nvG,ocp,vGdom,LMk,LM1k,
               naik1,uaik1,vthik1,nModk1,
               CΓ,εᵣ,mak1,Zqk1,spices,nak1,Iak1 ./ ρk1,vathk1,DThk1,dtk1;
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
    end
    Rdtvthk1 = Rck1[njMs+1,1,:]
    MhckMck!(Rhck1,Rck1,nMjMs,LMk,ρk1,vathk1,nsk1)
    dtMhck!(dtMhck1,Rhck1,Mhck1,nMjMs,LMk,Rdtvthk1,nsk1;L_Mh_limit=L_Mh_limit)
    Rck1[njMs+1,1,:] .*= vathk1      # ∂ₜvₜₕ
    return dtk1
end
# RDMck1
function Mck1ExEuler!(Mck1::AbstractArray{T,N},Rck1::AbstractArray{T,N},
    RDMck1::AbstractArray{T,N},Mck::AbstractArray{T,N},
    edtnIKTs::AbstractArray{T,N2},errRhck12::AbstractArray{T,N},
    Mhck1::Vector{Matrix{T}},Mhck::Vector{Matrix{T}},Rhck1::Vector{Matrix{T}},
    dtMhck1::Vector{Matrix{T}},errMhc::Vector{Matrix{T}},
    errMhcop::Vector{Matrix{T}},RDMhck1max::AbstractVector{T},nMjMs::Vector{Int64},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2},LMk::Vector{Int64},LM1k::Int64,
    naik1::Vector{AbstractVector{T}},uaik1::Vector{AbstractVector{T}},vthik1::Vector{AbstractVector{T}},nModk1::Vector{Int64},
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},nModk::Vector{Int64},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},spices::Vector{Symbol},
    nak1::AbstractVector{T},vathk1::AbstractVector{T},Rdtsabk1::T,Rdtsabk::T,DThk1::AbstractVector{T},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},δvathk1::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},NK::Int64,NKmax::Int64,kt::Int64,tk::T,dtk::T;
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
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false,
    is_optim_CnIK::Bool=false,is_nhnMod_adapt::Bool=false,
    is_NK_adapt_max::Bool=false,is_nai_const::Bool=true,
    is_nuTi_initial::Bool=true,is_fixed_NK::Bool=false,
    is_nMod_update_back::Bool=false,is_nMod_update_advance::Bool=false) where{T,N,N2}
    
    dtk1 = 1dtk
    ρk1 = mak1 .* nak1
    # `vathk1 = zeros(nsk1)`
    
    Mck1integral0!(Mck1,Mck,Rck1,nsk1,dtk1)
    RDMck1[:,:,:] = deepcopy(Mck1)

    # Calculate the parameters `nak1,vathk1,Iak1,Kak1` from `Mck1`
    nIKT_update!(nak1,vathk1,Iak1,Kak1,δvathk1,mak1,nsk1,Mck1;
                is_vth_ode=is_vth_ode,is_corrections_na=is_corrections[1])
    
    RerrDKab = sum(Kak1) / Kab0 - 1
    if abs(RerrDKab) > rtol_nIK_warn
        @warn("2, Energy conservation is not achieved:",RerrDKab)
        if abs(RerrDKab) > rtol_nIK_error
            @error("2, Energy conservation is not achieved:",RerrDKab)
        end
    end
    # # Computing the re-normalized moments 
    MhckMck!(Mhck1,Mck1[1:njMs,:,:],nMjMs,LMk,ρk1,vathk1,nsk1)
    @show 331, fmtf2.(Mhck1[1][:,1] .- 1)
    @show 331, fmtf2.(Mhck1[2][:,1] .- 1)

    # ODE for `∂ₜℳ̂ⱼₗ⁰(tk1) = dtMhck1`
    if is_dMhck1_Mhck
        Mhck1integral0!(Mhck1,Mhck,dtMhck1,nsk1,dtk1)
        @show 333, fmtf2.(Mhck1[1][:,1] .- 1)
        @show 333, fmtf2.(Mhck1[2][:,1] .- 1)
    end

    # @show 553, nModk1, vthik1

    dtk1 = submoment!(naik1, uaik1, vthik1, nModk1, 
            LMk, naik, uaik, vthik, nModk, 
            Mhck1, errMhcop, nMjMs, 
            edtnIKTs, Rdtsabk1, Rdtsabk, ns,
            Nspan_optim_nuTi,NK,NKmax,kt,tk,dtk1;
            Nspan_nuTi_max=Nspan_nuTi_max,
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
            is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
            is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
            is_nuTi_initial=is_nuTi_initial,is_fixed_NK=is_fixed_NK,
            is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)
    
    # Updating `Mhck1` and `Mck1` owing to the reduced parameters `naik, uaik, vthik`
    # @show 3320, fmtf2.(Mhck1[1][:,1] .- 1)
    if is_Mhc_reformula
        # @show 555, nModk1, vthik1
        # @show naik1
        # sdfgbh
        Mhck1 = MsnntL2fL0(Mhck1,nMjMs,LMk,naik1,uaik1,vthik1,nModk1,nsk1;is_renorm=is_renorm)
        MckMhck!(Mck1,Mhck1,nMjMs,LMk,ρk1,vathk1,nsk1)
    end
    @show 332, fmtf2.(Mhck1[1][:,1] .- 1)
    @show 332, fmtf2.(Mhck1[2][:,1] .- 1)
    # edfgbn
    for isp in 1:ns
        if norm(uaik[isp]) ≤ atol_IKTh
            L1 = 1
            # @show 552, fmtf2.(RDMck1[:,L1,isp])
            # @show 552, fmtf2.(Mck1[:,L1,isp])
            # @show 552, fmtf2.(Mck[:,L1,isp])
            RDMhck1max[isp] = maximum(abs.((Mhck1[isp][:,L1] ./ Mhck[isp][:,L1] .- 1) / dtk))
            RDMck1[:,L1,isp] = (RDMck1[:,L1,isp] - Mck1[:,L1,isp]) ./ (abs.(RDMck1[:,L1,isp] - Mck[:,L1,isp]) .+ epsT)
            # @show 552, fmtf2.(RDMck1[:,L1,isp])
        else
            sddffff
        end
    end
    # @show 556, nModk1, vthik1
    
    # uk = Iak1 ./ ρk1
    if dtk_order_Rc == :min && is_dtk_order_Rcaa == false
        dtk1 = dtMcab2!(Rck1,edtnIKTs,errRhck12,nMjMs,
               nvG,ocp,vGdom,LMk,LM1k,
               naik1,uaik1,vthik1,nModk1,
               CΓ,εᵣ,mak1,Zqk1,spices,nak1,Iak1 ./ ρk1,vathk1,DThk1,dtk1;
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
    else
        dtk1 = dtMcab2!(Rck1,edtnIKTs,errRhck12,Mck1,nMjMs,
               nvG,ocp,vGdom,LMk,LM1k,
               naik1,uaik1,vthik1,nModk1,
               CΓ,εᵣ,mak1,Zqk1,spices,nak1,Iak1 ./ ρk1,vathk1,DThk1,dtk1;
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
    end
    # @show 557, nModk1, vthik1
    Rdtvthk1 = Rck1[njMs+1,1,:]
    MhckMck!(Rhck1,Rck1,nMjMs,LMk,ρk1,vathk1,nsk1)
    dtMhck!(dtMhck1,Rhck1,Mhck1,nMjMs,LMk,Rdtvthk1,nsk1;L_Mh_limit=L_Mh_limit)
    Rck1[njMs+1,1,:] .*= vathk1      # ∂ₜvₜₕ
    # @show 558, nModk1, vthik1
    return dtk1
end

function Mck1integral!(Mck1::AbstractArray{T,N},Rck1::AbstractArray{T,N},
    RDMck1::AbstractArray{T,N},Mck::AbstractArray{T,N},
    edtnIKTs::AbstractArray{T,N2},errRhck12::AbstractArray{T,N},
    Mhck1::Vector{Matrix{T}},Mhck::Vector{Matrix{T}},Rhck1::Vector{Matrix{T}},
    dtMhck1::Vector{Matrix{T}},errMhc::Vector{Matrix{T}},
    errMhcop::Vector{Matrix{T}},RDMhck1max::AbstractVector{T},nMjMs::Vector{Int64},
    nvG::Vector{Int64}, ocp::Vector{Int64}, vGdom::AbstractArray{T,N2},LMk::Vector{Int64},LM1k::Int64,
    naik1::Vector{AbstractVector{T}},uaik1::Vector{AbstractVector{T}},vthik1::Vector{AbstractVector{T}},nModk1::Vector{Int64},
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},nModk::Vector{Int64},
    CΓ::T,εᵣ::T,mak1::AbstractVector{T},Zqk1::AbstractVector{Int64},spices::Vector{Symbol},
    nak1::AbstractVector{T},vathk1::AbstractVector{T},Rdtsabk1::T,Rdtsabk::T,DThk1::AbstractVector{T},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},δvathk1::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},NK::Int64,NKmax::Int64,kt::Int64,tk::T,dtk::T;
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
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false,
    is_optim_CnIK::Bool=false,is_nhnMod_adapt::Bool=false,
    is_NK_adapt_max::Bool=false,is_nai_const::Bool=true,
    is_nuTi_initial::Bool=true,is_fixed_NK::Bool=false,
    is_nMod_update_back::Bool=false,is_nMod_update_advance::Bool=false) where{T,N,N2}
    
    dtk1 = 1dtk
    ρk1 = mak1 .* nak1
    # `vathk1 = zeros(nsk1)`
    
    Mck1integral0!(Mck1,Mck,Rck1,nsk1,dtk1)
    RDMck1[:,:,:] = deepcopy(Mck1)
    
    # Calculate the parameters `nak1,vathk1,Iak1,Kak1` from `Mck1`
    nIKT_update!(nak1,vathk1,Iak1,Kak1,δvathk1,mak1,nsk1,Mck1;
                is_vth_ode=is_vth_ode,is_corrections_na=is_corrections[1])

    RerrDKab = sum(Kak1) / Kab0 - 1
    if abs(RerrDKab) > rtol_nIK_warn
        @warn("3, Energy conservation is not achieved:",RerrDKab)
        if abs(RerrDKab) > rtol_nIK_error
            @error("3, Energy conservation is not achieved:",RerrDKab)
        end
    end

    # # Computing the re-normalized moments 
    MhckMck!(Mhck1,Mck1[1:njMs,:,:],nMjMs,LMk,ρk1,vathk1,nsk1)
    # @show 441, fmtf2.(Mhck1[1][:,1] .- 1)
    # @show 441, fmtf2.(Mhck1[2][:,1] .- 1)

    # ODE for `∂ₜℳ̂ⱼₗ⁰(tk1) = dtMhck1`
    if is_dMhck1_Mhck
        Mhck1integral0!(Mhck1,Mhck,dtMhck1,nsk1,dtk1)
        # @show 443, fmtf2.(Mhck1[1][:,1] .- 1)
        # @show 443, fmtf2.(Mhck1[2][:,1] .- 1)
        # @show 554, nModk1, vthik1
    end

    dtk1 = submoment!(naik1, uaik1, vthik1, nModk1, 
            LMk, naik, uaik, vthik, nModk, 
            Mhck1, errMhcop, nMjMs, 
            edtnIKTs, Rdtsabk1, Rdtsabk, ns,
            Nspan_optim_nuTi,NK,NKmax,kt,tk,dtk1;
            Nspan_nuTi_max=Nspan_nuTi_max,
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
            is_optim_CnIK=is_optim_CnIK,is_nhnMod_adapt=is_nhnMod_adapt,
            is_NK_adapt_max=is_NK_adapt_max,is_nai_const=is_nai_const,
            is_nuTi_initial=is_nuTi_initial,is_fixed_NK=is_fixed_NK,
            is_nMod_update_back=is_nMod_update_back,is_nMod_update_advance=is_nMod_update_advance)

    # Updating `Mhck1` and `Mck1` owing to the reduced parameters `naik, uaik, vthik`
    if is_Mhc_reformula
        Mhck1 = MsnntL2fL0(Mhck1,nMjMs,LMk,naik1,uaik1,vthik1,nModk1,nsk1;is_renorm=is_renorm)
        MckMhck!(Mck1,Mhck1,nMjMs,LMk,ρk1,vathk1,nsk1)
    end
    # @show 442, fmtf2.(Mhck1[1][:,1] .- 1)
    # @show 442, fmtf2.(Mhck1[2][:,1] .- 1)
    
    for isp in 1:ns
        if norm(uaik[isp]) ≤ atol_IKTh
            L1 = 1
            # @show 552, fmtf2.(RDMck1[:,L1,isp])
            # @show 552, fmtf2.(Mck1[:,L1,isp])
            # @show 552, fmtf2.(Mck[:,L1,isp])
            RDMhck1max[isp] = maximum(abs.((Mhck1[isp][:,L1] ./ Mhck[isp][:,L1] .- 1) / dtk))
            RDMck1[:,L1,isp] = (RDMck1[:,L1,isp] - Mck1[:,L1,isp]) ./ (abs.(RDMck1[:,L1,isp] - Mck[:,L1,isp]) .+ epsT)
            # @show 552, fmtf2.(RDMck1[:,L1,isp])
        else
            sddffff
        end
    end

    # uk = Iak1 ./ ρk1
    if dtk_order_Rc == :min && is_dtk_order_Rcaa == false
        dtk1 = dtMcab2!(Rck1,edtnIKTs,errRhck12,nMjMs,
               nvG,ocp,vGdom,LMk,LM1k,
               naik1,uaik1,vthik1,nModk1,
               CΓ,εᵣ,mak1,Zqk1,spices,nak1,Iak1 ./ ρk1,vathk1,DThk1,dtk1;
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
    else
        dtk1 = dtMcab2!(Rck1,edtnIKTs,errRhck12,Mck1,nMjMs,
               nvG,ocp,vGdom,LMk,LM1k,
               naik1,uaik1,vthik1,nModk1,
               CΓ,εᵣ,mak1,Zqk1,spices,nak1,Iak1 ./ ρk1,vathk1,DThk1,dtk1;
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
    end
    Rdtvthk1 = Rck1[njMs+1,1,:]
    MhckMck!(Rhck1,Rck1,nMjMs,LMk,ρk1,vathk1,nsk1)
    dtMhck!(dtMhck1,Rhck1,Mhck1,nMjMs,LMk,Rdtvthk1,nsk1;L_Mh_limit=L_Mh_limit)
    Rck1[njMs+1,1,:] .*= vathk1      # ∂ₜvₜₕ
    
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

# [], algEmbedded::Symbol ∈ [:ExEuler, :ImEuler, :Trapz]
function Mck1integral0!(Mck1::AbstractArray{T,N},Mck::AbstractArray{T,N},Rck1::AbstractArray{T,N},nsk1::Int64,dtk::T) where{T,N}
    
    for isp in 1:nsk1
        Mck1[:,:,isp] = Mck[:,:,isp] + dtk * Rck1[:,:,isp]
    end
end

function Mhck1integral0!(Mhck1::Vector{Matrix{T}},Mhck::Vector{Matrix{T}},dtMhck1::Vector{Matrix{T}},nsk1::Int64,dtk::T) where{T}
    
    for isp in 1:nsk1
        @show isp
        # sumMh = sum(Mhck1[isp][:,1] .- 1)
        # @show sumMh
        # if sumMh ≤ epsT1000
        #     Mhck1[isp] = Mhck[isp] + dtk * dtMhck1[isp]
        # end
        sumMh = sum(dtMhck1[isp][:,1])
        # @show sumMh
        if sumMh ≥ epsT1000
            @show 88810, Mhck1[isp][:,1] .- 1
            Mhck1[isp] = Mhck[isp] + dtk * dtMhck1[isp]
            Mhck1[isp][2,1] = 1.
            # @show 8881, Mhck[isp][:,1] .- 1
            @show 8881, Mhck1[isp][:,1] .- 1
            printstyled("Mhck1: scheme 1, Mck",color=:red,"\n")
        else
            @show 8882, Mhck1[isp][:,1] .- 1
            @show 8820 Mhck[isp][:,1] + dtk * dtMhck1[isp][:,1] .- 1
            # @show 8882, Mhck[isp][:,1] .- 1
            # @show 8882, Mhck1[isp][:,1] .- 1
            printstyled("Mhck1: scheme 2, Mhck",color=:green,"\n")
        end
        # rtghj
    end
end


