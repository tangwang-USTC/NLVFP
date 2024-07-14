
"""
  A single-step multistage RK algorithm with inner iteration for
  the Fokker-Planck collision equations. The inner iteration is performed by
  the embedded implicit methods (implicit Euler method, the trapezoidal method or LabottaIIIA4 method)
  or the explicit methods such as explicit Euler, the Heun's method and so on.
  
  For `fvL`

  The criterions which are used to decide whether the algorithm is convergence or not are determined by the following characteristics:

    `criterions = [ps["DThk"]; ps["err_dtnIK"]; δvathi]`
  
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nModk1 = 1` gives `Î = û`.
  
  Level of the algorithm
    k: the time step level
    s: the stage level during `kᵗʰ` time step
    i: the inner iteration level during `sᵗʰ` stage
    
  Inputs:
    orders=order_dvδtf
    is_δtfvLaa = 0          # [0,     1   ]
                            # [dtfab, dtfa]
    uCk: The relative velocity during the Coulomb collision process.

  Outputs:
    fvLck1integralk!(dtfvLc0k1, pst0, Nstep; orderRK=orderRK,
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
function fvLck1integralk!(fvL0k1::Vector{Matrix{T}}, 
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

    is_corrections::Vector{Bool}=[true,false,false],
    i_iter_rs2::Int64=10,is_extrapolate_FLn::Bool=true,rtol_DnIK::T=0.1,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    dtk_order_Rc::Symbol=:mid,is_dtk_order_Rcaa::Bool=false,
    alg_embedded::Symbol=:Trapz,is_MjMs_max::Bool=false, 
    is_moments_out::Bool=false,is_Cerror_dtnIKTs::Bool=true,
    is_dtk_GKMM_dtIK::Bool=true,rtol_dtsa::T=1e-8,ratio_dtk1::T=1.2) where{T}


    # ratio_dtk1 = dtk / dtk
    # ps_copy = deepcopy(ps)
    tk = deepcopy(ps["tk"])
    tauk = ps["tauk"]

    nsk1 = ps["ns"]
    ma = ps["ma"]
    Zq = ps["Zq"]
    spices = ps["spices"]

    nak = ps["nk"]
    Iak = ps["Ik"]
    Kak = ps["Kk"]
    vathk = ps["vthk"]               # vath_(k)
    sak1 = deepcopy(ps["sak"])
    dtsabk1 = deepcopy(ps["dtsabk"])

    nnv = ps["nnv"]
    nc0, nck = ps["nc0"], ps["nck"]
    ocp = ps["ocp"]
    vGdom = ps["vGm"]

    vhek = ps["vhe"]
    vhkk = ps["vhk"]
    nvlevel0, nvlevele0 = ps["nvlevel0"], ps["nvlevele0"]          # nvlevele = nvlevel0[nvlevele0]

    nModk = ps["nModk"]
    nMjMs = ps["nMjMs"]
    naik = ps["naik"]
    uaik = ps["uaik"]
    vthik = ps["vthik"]

    LMk = ps["LMk"]
    # muk, Mμk, Munk, Mun1k, Mun2k = ps["muk"], ps["Mμk"], ps["Munk"], ps["Mun1k"], ps["Mun2k"]
    # w3k, err_dtnIK, DThk = ps["w3k"], ps["err_dtnIK"], ps["DThk"]         # w3k = vₜₕ⁻¹∂ₜvₜₕ
    Mhck = ps["Mhck"]
    dtnIKs = zeros(T,3,2)
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
    if nsk1 ≥ 3
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
    
    tauk = ps["tauk"]
    ρk1 = ma .* nak1

    dtKIak = zeros(T,2,ns)                #  [[K, I], ns]
    dtKIak[1,:] = deepcopy(Kak)           # K
    dtKIak[2,:] = deepcopy(Iak)           # I

    @show is_fvL_CP,dtk
    @show 0, dtKIak[1,:]

    ve = deepcopy(vhek)
    vk = deepcopy(vhkk)
    fvLc0k = deepcopy(fvL0k1)

    for isp in nsp_vec
        ve[isp] = vhek[isp] * vathk[isp]
        vk[isp] = vhkk[isp] * vathk[isp]
        fvLc0k[isp] *= (na[isp] / vathk[isp]^3)
    end
    fvLc0k1 = deepcopy(fvLc0k)

    fvL0k = deepcopy(fvL0k1)
    dtfvL0k = deepcopy(fvL0k1)
    if prod(nModk) == 1
        FP0D2Vab2!(dtfvL0k,fvL0k1,vhkk,nvG,nc0,nck,ocp,
                nvlevele0,nvlevel0,LMk,LM1k,
                CΓ,εᵣ,ma,Zq,spices,nak,[uaik[1][1],uaik[2][1]],vathk;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs, 
                is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn)
    else
        FP0D2Vab2!(dtfvL0k,fvL0k1,vhkk,nvG,nc0,nck,ocp,
               nvlevele0,nvlevel0,LMk,LM1k,naik, uaik, vthik,
               CΓ,εᵣ,ma,Zq,spices,nak,vathk,nModk;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               is_extrapolate_FLn=is_extrapolate_FLn)
    end

    # # Calculating the rate of change of the kinetic moments in the Lagrange coordinate system.
    # Rc = deepcopy(Rck1)
    # aa = Rc[1:njMs,:,:]
    # err_Rc = deepcopy(Rc)
    # if gridv_type == :uniform
    #     dtMcsd2l!(aa,err_Rc,dtfvL0k,vhek,nMjMs,ρk1,vathk,LMk,nsk1;is_renorm=is_renorm)
    # elseif gridv_type == :chebyshev
    #     if is_dtf_MS_IK
    #         dtMcsd2l!(aa,err_Rc,dtfvL0k,vhek,nvG[1],nMjMs,ρk1,vathk,LMk,nsk1;
    #                   is_renorm=is_renorm,is_norm_error=is_norm_error)
    #     else
    #         dtMcsd2l!(aa,err_Rc,dtfvL0k,vhek,nvG,nMjMs,ρk1,vathk,LMk,nsk1;
    #                   is_renorm=is_renorm,is_norm_error=is_norm_error)
    #     end
    # end

    dtfvLc0k = deepcopy(dtfvL0k)
    for isp in nsp_vec
        dtfvLc0k[isp] *= (na[isp] / vathk[isp]^3)
    end
    # Computing the `dtn, dtI, dtK`              # wrong now
    dtnIKsc!(dtnIKs,edtnIKTs,dtfvLc0k,ve,ma,nak,vathk,nsk1;
            atol_nIK=atol_nIK,is_norm_error=is_norm_error)
    # @show dtI ./ dtnIKs[2,:] .- 1
    # @show dtK ./ dtnIKs[3,:] .- 1

    if orderRK ≤ 2
        count = 0
        vathk1i = deepcopy(vathk)          # zeros(T,nsk1)
        δvathi = zeros(T,nsk1)
        dtfvLc0k1 = deepcopy(dtfvLc0k)
        errMhc = deepcopy(Mhck1)

        LMk1 = 1LMk
        LM1k1 = 1LM1k
        vGdomk = deepcopy(vGdom)
        vGdomk1 = deepcopy(vGdom)
        vhek1 = deepcopy(vhek)
        vhkk1 = deepcopy(vhkk)
        nc0k1, nckk1 = 1nc0, 1nck
        nvlevele0k1, nvlevel0k1 = deepcopy(nvlevele0), deepcopy(nvlevel0)
        vathk_up = deepcopy(vathk)
        k = 1
        done = true
        while done
            # parameters
            tk = deepcopy(ps["tk"])
            if is_fixed_timestep
            else
                dtk = deepcopy(ps["dt"])
            end
            Nt_save = ps["Nt_save"]
            count_save = ps["count_save"]
            koutput = 2^(n21 + iCase - 1)
            if k == koutput
                is_plot_DflKing = true
                is_plot_dfln_thoery= true
            else
                is_plot_DflKing = false
                is_plot_dfln_thoery = false
            end
            
            # println()
            println("**************------------******************------------*********")
            printstyled("k=",k,",tk,dt,Rdt=",fmtf2.([ps["tk"],dtk,dtk/ps["tk"]]),"\n";color=:blue)

            dtk1 = 1dtk
            title = string(L"k,t=",fmtf2(tk))
            isp = 1
            iFv = 2
            if 2 == 1
                vec22 = vhek[isp] .< 2
                # Courant number
                CnCFLa = abs.(dtk * dtfvLc0k[isp][vec22,1] ./ abs.(fvLc0k1[isp][vec22,1] - fvLc0k[isp][vec22,1]) .+ epsT)
                if is_plot_DflKing
                    if spices[isp] == spices[iFv]
                        label = L"a"
                    else
                        label = string(spices[isp])
                    end
                    ylabel = L"Courant number"
                    pdtkCFLa = plot(vhek[isp][vec22], CnCFLa,label=label,ylabel=ylabel,title=title)
                end
    
                vec22 = vhek[iFv] .< 2
                # Courant number
                CnCFLb = abs.(dtk * dtfvLc0k[iFv][vec22,1] ./ abs.(fvLc0k1[iFv][vec22,1] - fvLc0k[iFv][vec22,1]) .+ epsT)
                if is_plot_DflKing
                    if spices[isp] == spices[iFv]
                        label = L"b"
                    else
                        label = string(spices[iFv])
                    end
                    pdtkCFLb = plot(vhek[iFv][vec22], CnCFLb,label=label,ylabel=ylabel)
                    if is_display_dtCFL
                        display(plot(pdtkCFLa,pdtkCFLb,layout=(2,1)))
                    end
                end
                if k == koutput
                    RDKing00vec[iCase,5] = min(minimum(CnCFLa), minimum(CnCFLb))
                end
            end

            vathk1i = deepcopy(vathk)          # zeros(T,nsk1)
            # Updating the meshgrids
            vHadapt1D!(vhek1,vhkk1, vGdomk1, nvG, nc0k1, nckk1, ocp, 
                  nvlevele0k1, nvlevel0k1, naik,uaik,vthik, nModk1, nsk1;
                  eps_fup=eps_fup,eps_flow=eps_flow,
                  maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                  abstol=abstol,reltol=reltol,
                  vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                  is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)

            DvGdomk = maximum(abs.(vGdomk1[2,:] ./ vGdomk[2,:] .- 1))
            Dvathk = maximum(abs.(vathk_up ./ vathk .- 1))
            if max(DvGdomk, Dvathk) ≥ rtol_vGdom
                @show DvGdomk, Dvathk
                fvgnggggg
                for isp in nsp_vec
                    ve[isp] = vhek1[isp] * vathk[isp]
                    vk[isp] = vhkk1[isp] * vathk[isp]
                    vhek[isp] = ve[isp] / vathk[isp]
                    vhkk[isp] = vk[isp] / vathk[isp]
                end

                fvL0k1 = Vector{Matrix{T}}(undef,nsk1)
                for isp in 1:nsk1
                    fvL0k1[isp] = zeros(T,nvG[isp],LMk1[isp] + 3)
                end
                LM1k1, fvL0k1 = fvLDMz!(fvL0k1, vhek1, LMk1, 2, [uaik[1][1],uaik[2][1]]; 
                                L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)
                if prod(nModk) == 1
                    FP0D2Vab2!(dtfvL0k,fvL0k1,vhkk1,nvG,nc0k1,nckk1,ocp,
                            nvlevele0k1, nvlevel0k1,LMk1,LM1k1,
                            CΓ,εᵣ,ma,Zq,spices,nak,[uaik[1][1],uaik[2][1]],vathk;
                            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs, 
                            is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                            is_extrapolate_FLn=is_extrapolate_FLn)
                else
                    FP0D2Vab2!(dtfvL0k,fvL0k1,vhkk1,nvG,nc0k1,nckk1,ocp,
                           nvlevele0k1, nvlevel0k1,LMk1,LM1k1,naik, uaik, vthik,
                           CΓ,εᵣ,ma,Zq,spices,nak,vathk,nModk;
                           is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                           autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                           p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                           is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                           is_extrapolate_FLn=is_extrapolate_FLn)
                end

                RhnnEvens!(Mhck1, errMhc, fvL0k1, vhek1, nMjMs, LMk1, nsk1; 
                          is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)
                dtfvLc0k = deepcopy(dtfvL0k)
                fvLc0k = deepcopy(fvL0k1)
                for isp in nsp_vec
                    fvLc0k[isp] *= (na[isp] / vathk[isp]^3)
                    dtfvLc0k[isp] *= (na[isp] / vathk[isp]^3)
                end
                fvLc0k1 = deepcopy(fvLc0k)
                dtfvLc0k1 = deepcopy(dtfvLc0k)

                # Computing the `dtn, dtI, dtK`              # wrong now
                dtnIKsc!(dtnIKs,edtnIKTs,dtfvLc0k,ve,ma,nak,vathk,nsk1;
                        atol_nIK=atol_nIK,is_norm_error=is_norm_error)

                vathk_up = deepcopy(vathk)
                vGdomk = deepcopy(vGdomk1)
                nc0, nck = 1nc0k1, 1nckk1
                nvlevele0, nvlevel0 = deepcopy(nvlevele0k1), deepcopy(nvlevel0k1)
            else
                for isp in nsp_vec
                    vhek1[isp] = ve[isp] / vathk[isp]
                    vhkk1[isp] = vk[isp] / vathk[isp]
                    vhek[isp] = ve[isp] / vathk[isp]
                    vhkk[isp] = vk[isp] / vathk[isp]
                end
                fvL0k = deepcopy(fvL0k1)
                fvLc0k = deepcopy(fvL0k)
                for isp in nsp_vec
                    fvLc0k[isp] *= (na[isp] / vathk[isp]^3)
                end
                fvLc0k1 = deepcopy(fvLc0k)
                dtfvLc0k = deepcopy(dtfvLc0k1)
            end
            if nsk1 == 2
                # println("..-----------------------------------------..")
                # @show tk, 0, fvLc0k[2][1:3,1]
                # @show tk, 0, fvLc0k1[2][1:3,1]

                dtkCFL = fvLck1integrali_rs2!(dtfvLc0k1,fvLc0k1,fvLc0k,
                    dtfvLc0k,fvL0k1,fvL0k,dtnIKs,edtnIKTs,Mhck1,errMhc, 
                    nvG, nc0, nck, ocp, ve, vk, vhek1, vhkk1, 
                    nvlevele0,nvlevel0,LMk1, LM1k1, naik, uaik, vthik, vhek, vhkk, LMk,
                    CΓ, εᵣ, ma, Zq, spices, nak1, vathk, Iak, Kak, 
                    nModk1, nMjMs, vathk1i, δvathi, Iak1, Kak1, vathk1, 
                    Nspan_optim_nuTi, dtk1, tk+dtk;orderRK=orderRK,
                    Nspan_nuTi_max=Nspan_nuTi_max,
                    NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
                    restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
                    autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                    optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
                    is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0, is_fit_f=is_fit_f,
                    is_extrapolate_FLn=is_extrapolate_FLn,i_iter_rs2=i_iter_rs2,
                    is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
                    is_plot_DflKing=is_plot_DflKing,is_plot_dfln_thoery=is_plot_dfln_thoery)
                # @show 0, tk, dtkCFL
                # println("..-----------------------------------------..")
                
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
                            dtk1 = dtMcab2!(Rck1,edtnIKTs,err_Rck12,
                                    nvG, ocp, vGdom,  LMk, LM1k, naik, uaik, vthik, 
                                    CΓ, εᵣ, ma, Zq, spices, nak1, uak1, vathk1,
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
                            tau_fM!(tauk, ma, Zq, spices, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                            printstyled("0: Updating the time scale, tau=", tauk,color=:green,"\n")
                            count = 0
                        else
                            count += 1
                            if count == count_tau_update
                                tau_fM!(tauk, ma, Zq, spices, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                                printstyled("1: Updating the time scale, tau=", tauk,color=:green,"\n")
                                count = 0
                            end
                        end
                    else
                        if sum(is_nMod_renew) > 0

                            # Updating `Mhck1` and `Mck1` owing to the reduced parameters `naik, uaik, vthik`
                            Mhck1 = MsnntL2fL0(Mhck1,nMjMs,LMk,nsk1,naik,uaik,vthik,nModk1;is_renorm=is_renorm)
                            MckMhck!(Mck1,Mhck1,ρa,vathk1,LMk,nsk1,nMjMs)

                            # Updating `Rck1` owing to the reduced parameters `naik, uaik, vthik`
                            uak1 = Mck1[1,2,:] ./ ρk1
                            dtk1 = dtMcab2!(Rck1,edtnIKTs,err_Rck12,
                                    nvG, ocp, vGdom,  LMk, LM1k, naik, uaik, vthik, 
                                    CΓ, εᵣ, ma, Zq, spices, nak1, uak1, vathk1,
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
                            tau_fM!(tauk, ma, Zq, spices, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                            printstyled("0: Updating the time scale, tau=", tauk,color=:green,"\n")
                            count = 0
                        else
                            count += 1
                            if count == count_tau_update
                                tau_fM!(tauk, ma, Zq, spices, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                                printstyled("1: Updating the time scale, tau=", tauk,color=:green,"\n")
                                count = 0
                            end
                        end
                    end
                elseif is_fixed_timestep == false
                    count += 1
                    if count == count_tau_update
                        tau_fM!(tauk, ma, Zq, spices, nak1, vathk1, Coeff_tau, naik, vthik, nModk1)
                        printstyled("2: Updating the time scale, tau=", tauk,color=:green,"\n")
                        count = 0
                    end
                end
                CRDn[1] = min(abs(edtnIKTs[1,1]),abs(edtnIKTs[1,2]))
            else
                dfhgvbnhjjj

                # # # Updating the parameters `nModk1`
                if prod(nModk1) ≥  2
                    # reducing the number of `nModk1` according to and updating `naik, uaik, vthik`
                    nMod_update!(is_nMod_renew, nModk1, naik, uaik, vthik, nsk1)

                    if is_fixed_timestep == false
                        wsadfg
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
                                    nvG, ocp, vGdom, LMk, LM1k, naik, uaik, vthik, 
                                    CΓ, εᵣ, ma,Zq, spices, nak1, uak1, vathk1,
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

                            tau_fM!(tauk, ma, Zq, spices, nak1, vathk1, Coeff_tau, 
                                    naik2, vthik2, naik, vthik, nModk1, nsk1)
                            printstyled("0: Updating the time scale, tau=", tauk,color=:green,"\n")
                            count = 0
                        else
                            count += 1
                            if count == count_tau_update
                                tau_fM!(tauk, ma, Zq, spices, nak1, vathk1, Coeff_tau, 
                                        naik2, vthik2, naik, vthik, nModk1, nsk1)
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
                        tau_fM!(tauk, ma, Zq, spices, nak1, vathk1, Coeff_tau, 
                                naik2, vthik2, naik, vthik, nModk1, nsk1)
                        printstyled("2: Updating the time scale, tau=", tauk,color=:green,"\n")
                        count = 0
                    end
                end
            end

            # [ns], Updating the entropy and its change rate with assumpation of dauble-Maxwellian distribution
            entropy_fDM!(sak1,ma,nak1,vathk1,Iak1,Kak1,nsk1)                        

            # dtsabk1 = dtsak1 + dtsbk1
            # [nsk1 = 2] Iahk = uak1 ./ vathk1
            dtsabk1 = entropy_rate_fDM(ma,vathk1,Iak1 ./ (ρk1 .* vathk1),dtnIKs[2,:],dtnIKs[3,:],nsk1)
            
            Rdtsabk1 = dtsabk1 / sum(sak1)
            # Rdtsabk1 = entropyN_rate_fDM(ma,nak1,vathk1,Iak1,Kak1,dtIak1,dtKak1,nsk1)
            # @show Rdtsabk1

            # When `nₖ₊₁ˢ`, `Iₖ₊₁ˢ` and `Kₖ₊₁ˢ` reach convergence, conservations and
            # the constraint equation is satisfied: `K̂ = 3/2 + Î`,

            is_corrections[1] ? nothing : nak = deepcopy(nak1)
            Iak = deepcopy(Iak1)
            Kak = deepcopy(Kak1)
            vathk = deepcopy(vathk1)
            Mhck = deepcopy(Mhck1)
            LMk = deepcopy(LMk1)
            LM1k = deepcopy(LM1k1)
            
            # Updating the time-step at `(k+1)ᵗʰ` step according to `τTab` and `τRdtIKk1`

            # Updating `dtIak1` and `dtKak1`
            dtKIak[1,:] = dtnIKs[3,:]             # dtKa
            dtKIak[2,:] = dtnIKs[2,:]             # dtIa
            if is_fixed_timestep
                dtk1 = dtk
            else
                dtk1 = dtk
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
            @show tk, dtk,dtk1,is_fixed_timestep

            ps["tk"] = tk + dtk
            # ps["tauk"] = tauk
            ps["dt"] = dtk1
            @show ps["tk"],ps["dt"]
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
            if k == koutput
                RDKing33vec[iCase,3] = deepcopy(ps["tk"])
                RDKing33vec[iCase,4] = k
                RDKing00vec[iCase,3] = deepcopy(ps["tk"])
                RDKing00vec[iCase,4] = k
                RDKing00vec[iCase,5] = dtkCFL
            end


            # Terminating the progrom when I. reaches the maximum time moment; II. number of time step; III. equilibrium state.
            if abs(Rdtsabk1) ≤ rtol_dtsa_terminate
                @warn("The system has reached the equilibrium state when", Rdtsabk1)
                break
            else
                if ps["tk"] > ps["tspan"][2]
                    # if abs(tk - ps["tspan"][2]) / ps["tspan"][2] ≤ rtol_tk
                    if tk ≥ ps["tspan"][2]
                        @warn("The system has reached the maximum time moment at", tk)
                        break
                    # else
                    #     ps["tk"] = ps["tspan"][2]
                    #     dtk1 = ps["tk"] - tk
                    #     ps["dt"] = dtk1
                    end
                else
                    if k > Nstep
                        @warn("The system has reached the maximum iterative step", Nstep)
                        break
                    end
                end
            end
            k += 1
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

    During the iterative process `while ⋯ end`

    Inputs:
      vathk1i: = vathk, for the initial stage by Explicit Euler.
               = vath_k_(i-1), for the iterative stage

      outputs:

  Outputs:
    dtkCFL = fvLck1integrali_rs2!(dtfvLc0k1,fvLc0k1,fvLc0k,
        dtfvLc0k,fvL0k1,fvL0k,dtnIKs,edtnIKTs,Mhck1,errMhc, 
        nvG, nc0, nck, ocp, ve, vk, vhek1, vhkk1, 
        nvlevele0,nvlevel0,LMk1, LM1k1, naik, uaik, vthik, vhek, vhkk, LMk,  
        CΓ, εᵣ, ma, Zq, spices, nak1, vathk, Iak, Kak, 
        nModk1, nMjMs, vathk1i, δvathi, Iak1, Kak1, vathk1, 
        Nspan_optim_nuTi, dtk, tk;orderRK=orderRK,
        Nspan_nuTi_max=Nspan_nuTi_max,
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0, is_fit_f=is_fit_f,
        is_extrapolate_FLn=is_extrapolate_FLn,i_iter_rs2=i_iter_rs2,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
        is_plot_DflKing=is_plot_DflKing)
"""
# [sᵗʰ,i], alg_embedded ∈ [:TrapzMck],    rs = 2, o = 2
# ns ≥ 3

# ns = 2
function fvLck1integrali_rs2!(dtfvLc0k1::Vector{Matrix{T}},fvLc0k1::Vector{Matrix{T}},
    fvLc0k::Vector{Matrix{T}},dtfvLc0k::Vector{Matrix{T}},
    fvL0k1::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},
    dtnIKs::AbstractArray{T,N2},edtnIKTs::AbstractArray{T,N2},
    Mhck1::Vector{Matrix{T}},errMhc::Vector{Matrix{T}},
    nvG::Vector{Int64},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},
    ve::Vector{StepRangeLen},vk::Vector{AbstractVector{T}},
    vhek1::Vector{StepRangeLen},vhkk1::Vector{AbstractVector{T}},
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}},LMk1::Vector{Int64},LM1k1::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    vhek::Vector{StepRangeLen},vhkk::Vector{AbstractVector{T}}, LMk::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak1::AbstractVector{T},vathk::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},
    nModk1::Vector{Int64},nMjMs::Vector{Int64},vathk1i::AbstractVector{T},δvathi::AbstractVector{T},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},vathk1::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},dtk::T,tk::T;orderRK::Int64=2,
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),is_Jacobian::Bool=true,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,is_fit_f::Bool=false,

    is_extrapolate_FLn::Bool=true,i_iter_rs2::Int64=0,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    is_plot_DflKing=false,is_plot_dfln_thoery::Bool=false) where{T,N2}

    i_iter = 0                  # and Checking the `i = 1` iteration
    # Applying the explicit Euler step
    # δvathi = ones(T,2)        # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
    
    dtkCFL = dtfvLc0k1ExEuler!(dtfvLc0k1,fvLc0k1,fvLc0k,dtfvLc0k,
        fvL0k1,fvL0k,dtnIKs,edtnIKTs,Mhck1,errMhc, 
        nvG, nc0, nck, ocp, ve, vk, vhek1, vhkk1, 
        nvlevele0,nvlevel0,LMk1, LM1k1, naik, uaik, vthik, vhek, vhkk, LMk,
        CΓ, εᵣ, ma, Zq, spices, nak1, vathk, Iak, Kak, 
        nModk1, nMjMs, Iak1, Kak1, vathk1, Nspan_optim_nuTi, dtk, tk;
        Nspan_nuTi_max=Nspan_nuTi_max,
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0, is_fit_f=is_fit_f,
        is_extrapolate_FLn=is_extrapolate_FLn,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
        is_plot_DflKing=is_plot_DflKing,is_plot_dfln_thoery=is_plot_dfln_thoery)
    vathk1i[:] = deepcopy(vathk1)
    δvathi_up = zeros(T,2)
    
    # println(".....................................................")
    # @show  dtkCFL
    # @show fmtf6(tk), 0, fvLc0k[2][1:3,1]
    # @show fmtf6(tk), 0, fvLc0k1[2][1:3,1]
    # # If `i_iter_rs2 ≤ 0`, then degenerate into the explicit Euler method (ExEuler)
    while i_iter < i_iter_rs2
        i_iter += 1
        @show i_iter
        # dtfvLc0k1[:,:,:] = (dtfvLc0k + dtfvLc0k1) / 2        # Rck1k = Rc_(k+1/2)
        
        fvLck1integral!(dtfvLc0k1,fvLc0k1,fvLc0k,dtfvLc0k,
            fvL0k1,fvL0k,dtnIKs,edtnIKTs,Mhck1,errMhc, 
            nvG, nc0, nck, ocp, ve, vk, vhek1, vhkk1, 
            nvlevele0,nvlevel0,LMk1, LM1k1, naik, uaik, vthik, vhek, vhkk, LMk,
            CΓ, εᵣ, ma, Zq, spices, nak1, vathk, Iak, Kak, 
            nModk1, nMjMs, vathk1i, Iak1, Kak1, vathk1, Nspan_optim_nuTi, dtk, tk;orderRK=orderRK,
            Nspan_nuTi_max=Nspan_nuTi_max,
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
            restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
            optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
            is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0, is_fit_f=is_fit_f,
            is_extrapolate_FLn=is_extrapolate_FLn,
            is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error,
            is_plot_DflKing=is_plot_DflKing,is_plot_dfln_thoery=is_plot_dfln_thoery,iter=i_iter)
        δvathi[:] = vathk1i ./ vathk1 .- 1                # = vathk1_w3k_i / vathk1_w3k_i1 .- 1
        ratio_vathi = δvathi - δvathi_up
        @show i_iter, δvathi
        # println(".....................................................")
        # @show fmtf6(tk), i_iter, fvLc0k[2][1:3,1]
        # @show fmtf6(tk), i_iter, fvLc0k1[2][1:3,1]

        if norm(ratio_vathi) ≤ rtol_vthi || norm(δvathi) ≤ atol_vthi
            break
        end
        vathk1i[:] = deepcopy(vathk1)
    end
    if i_iter ≥ i_iter_rs2
        @error(`The maximum number of iteration reached before the Trapz method to be convergence!!!`)
    end
    return dtkCFL
end

"""
  Integral at the `sᵗʰ` stage with implicit Euler method with `Niter_stage`: 

  Level of the algorithm
    i=0ᵗʰ: the inner iteration level during `sᵗʰ` stage
    
  Inputs:
    nak1 = deepcopy(nak)
    Iak1 = deepcopy(Iak)
    Kak1 = deepcopy(Kak)
    vathk1:
    vathk1i = deepcopy(vathk)              # vath_(k+1)_(i-1) which will be changed in the following codes


  Outputs:
    fvLck1integral!(dtfvLc0k1,fvLc0k1,fvLc0k,dtfvLc0k,
        fvL0k1,fvL0k,dtnIKs,edtnIKTs,Mhck1,errMhc, 
        nvG, nc0, nck, ocp, ve, vk, vhek1, vhkk1, 
        nvlevele0,nvlevel0,LMk1, LM1k1, naik, uaik, vthik, vhek,vhkk,LMk,
        CΓ, εᵣ, ma, Zq, spices, nak1, Iak, Kak, 
        nModk1, nMjMs, vathk1i, Iak1, Kak1, vathk1, Nspan_optim_nuTi, dtk, tk;
        Nspan_nuTi_max=Nspan_nuTi_max,
        NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0, is_fit_f=is_fit_f,
        is_extrapolate_FLn=is_extrapolate_FLn,
        is_enforce_errdtnIKab=is_enforce_errdtnIKab,is_norm_error=is_norm_error)
"""

# [iᵗʰ], alg_embedded == ∈ [:ExEuler], `dtfvLc0k1= dtfvLc0k`
# # # ns ≥ 3

# ns = 2
function dtfvLc0k1ExEuler!(dtfvLc0k1::Vector{Matrix{T}},fvLc0k1::Vector{Matrix{T}},
    fvLc0k::Vector{Matrix{T}},dtfvLc0k::Vector{Matrix{T}},
    fvL0k1::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},
    dtnIKs::AbstractArray{T,N2},edtnIKTs::AbstractArray{T,N2},
    Mhck1::Vector{Matrix{T}},errMhc::Vector{Matrix{T}},
    nvG::Vector{Int64},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},
    ve::Vector{StepRangeLen},vk::Vector{AbstractVector{T}},
    vhek1::Vector{StepRangeLen},vhkk1::Vector{AbstractVector{T}},
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}},LMk1::Vector{Int64},LM1k1::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    vhek::Vector{StepRangeLen},vhkk::Vector{AbstractVector{T}},LMk::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak1::AbstractVector{T},vathk::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},
    nModk1::Vector{Int64},nMjMs::Vector{Int64},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},vathk1::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},dtk::T,tk::T;
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],nsk1::Int64=2,
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),is_Jacobian::Bool=true,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,is_fit_f::Bool=false,

    is_extrapolate_FLn::Bool=true,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    is_plot_DflKing=false,is_plot_dfln_thoery::Bool=false) where{T,N2}
    
    # ρk1 = ma .* nak1
    
    # dtfvLc0k1 = dtfvLc0k
    fvLck1integral0!(fvLc0k1,fvLc0k,dtfvLc0k1,nsk1,dtk)

    # # Computing the `dtn, dtI, dtK`
    # if gridv_type == :uniform
    #     dtMcsd2l!(aa,err_Rc,dtfvLc0k1,vhe,nMjMs,ma.*nak1,vathk1i,LMk1,nsk1;is_renorm=is_renorm)
    # elseif gridv_type == :chebyshev
    #     # if is_dtf_MS_IK
    #     #     dtMcsd2l!(aa,err_Rc,dtfvLc0k1,vhe,nvG[1],nMjMs,ma.*nak1,vathk,LMk,ns;
    #     #               is_renorm=is_renorm,is_norm_error=is_norm_error)
    #     # else
    #     #     dtMcsd2l!(aa,err_Rc,dtfvLc0k1,vhe,nvG,nMjMs,ma.*nak1,vathk,LMk,ns;
    #     #               is_renorm=is_renorm,is_norm_error=is_norm_error)
    #     # end
    # end

    dtnIKsc!(dtnIKs,edtnIKTs,dtfvLc0k1,ve,ma,nak1,vathk,nsk1;
            atol_nIK=atol_nIK,is_norm_error=is_norm_error)
            
    if is_enforce_errdtnIKab
        dtnIKposteriorC!(dtnIKs,edtnIKTs)
    end

    # # Computing the `n, I, K`
    # nIKsc!(nIKs,fvLc0k1[:,1:2,:],ve,ma,nsk1;errnIKc=errnIKc)
    Iak1[:] = Iak + dtk * dtnIKs[2,:]
    Kak1[:] = Kak + dtk * dtnIKs[3,:]

    RerrDKab = sum(Kak1) / Kab0 - 1
    if abs(RerrDKab) > rtol_nIK_warn
        @warn("Energy conservation is not achieved:",RerrDKab)
        if abs(RerrDKab) > rtol_nIK_error
            @error("Energy conservation is not achieved:",RerrDKab)
        end
    end
    # @show dtk * dtnIKs[3,:] ./ Kak

    vthnIK!(vathk1,ma,nak1,Iak1,Kak1,nsk1)

    cf3 = zeros(T,nsk1)
    for isp in 1:nsk1
        cf3[isp] = (nak1[isp] / vathk1[isp]^3)
        fvL0k1[isp] = fvLc0k1[isp] / cf3[isp]
        vhek1[isp] = ve[isp] / vathk1[isp]
        vhkk1[isp] = vk[isp] / vathk1[isp]
    end

    # Computing the normalized kinetic moments
    RhnnEvens!(Mhck1, errMhc, fvL0k1, vhek1, nMjMs, LMk1, nsk1; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)


    Ccol = zeros(T,2)
    # # Updating the FP collision terms according to the `FPS` operators.
    # dtfvLc0k1 = dtfvL0k1
    if prod(nModk1) == 1
        # # Calculate the parameters `nai, uai, vthi`
        submoment!(naik, uaik, vthik, Mhck1, nsk1)

        # # Updating the normalized distribution function `KvL0k1` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        # KvL0k1 = deepcopy(fvL0k1)
        KvL0k1 = Vector{Matrix{T}}(undef,nsk1)
        for isp in 1:nsk1
            KvL0k1[isp] = zeros(T,nvG[isp],LMk1[isp] + 3)
        end
        LM1k1, KvL0k1 = fvLDMz!(KvL0k1, vhek1, LMk1, 2, [uaik[1][1],uaik[2][1]]; 
                        L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)
        dtCFL = FP0D2Vab2!(dtfvLc0k1,KvL0k1,vhkk1,nvG,nc0,nck,ocp,
                nvlevele0,nvlevel0,LMk1,LM1k1,
                CΓ,εᵣ,ma,Zq,spices,Ccol,nak1,[uaik[1][1],uaik[2][1]],vathk1;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs, 
                is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                is_extrapolate_FLn=is_extrapolate_FLn,
                is_plot_dfln_thoery=is_plot_dfln_thoery)
    else
        # # Calculate the parameters `nai, uai, vthi` from the re-normalized moments `ℳ̂ⱼ,ₗ⁰`
        submoment!(naik, uaik, vthik, nModk1, ns, Mhck1,
                Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)

        # # Updating the normalized distribution function `fvL0k1` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        KvL0k1 = deepcopy(fvL0k1)
        LM1k1, KvL0k1 = fvLDMz!(KvL0k1, vhek1, nvG, LMk1, ns, naik, uaik, vthik, nModk1; 
                         L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)

        dtCFL = FP0D2Vab2!(dtfvLc0k1,KvL0k1,vhkk1,nvG,nc0,nck,ocp,
               nvlevele0,nvlevel0,LMk1,LM1k1,naik, uaik, vthik,
               CΓ,εᵣ,ma,Zq,spices,Ccol,nak1,vathk1,nModk1;
               is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
               autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
               p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
               is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
               is_extrapolate_FLn=is_extrapolate_FLn)
    end

    for isp in 1:nsk1
        dtfvLc0k1[isp] *= cf3[isp]
    end
    
    @show is_plot_DflKing, is_plot_dfln_thoery
    if is_plot_DflKing
        iter = 0
        if 1 == 2
            DKing = fvL0k1 - KvL0k1
            Dfing = (fvLc0k1 - fvLc0k)
        else
            KvLc0k1 = deepcopy(KvL0k1)
            for isp in 1:2
                KvLc0k1[isp] *= (nak1[isp] / vathk1[isp]^3)
            end
            DKing = fvLc0k1 - KvLc0k1
            Dfing = (fvLc0k1 - fvLc0k)
        end
        if MultiType == :dt
            xlabel = L"\Delta t"

            if is_L2dtfl0
                isppp = 1
                fl0L2dtv[:,iCase] = deepcopy(fvLc0k1[isppp][:,1])         # For the exact species
    
                if iCase == NCase
                    fl0L2dtv1 = ((sum((fl0L2dtv[:,1:end-1] ./ fl0L2dtv[:,end] .- 1) .^2;dims=1)) .^0.5)[:]
                    fl0L2order = fl0L2dtv1[1] ./ (2^2) .^(0:NCase-2)
        
                    xlabel = L"\Delta t"
                    ylabel = L"L_2^{\Delta t}"
                    label = L"Numerical"
                    pL2dt = plot(1 ./ Nτ_fixvec[1:end-1], fl0L2dtv1,line=(2,:auto),label=label,
                                ylabel=ylabel,yscale=:log10,
                                xlabel=xlabel,xscale=:log10)
                    label = L"2^{nd} order"
                    pL2dt = plot!(1 ./ Nτ_fixvec[1:end-1], fl0L2order,line=(2,:auto),label=label)
                    savefig(string(file_fig_file,"_L2dt",NCase,".png"))
                    display(pL2dt)
                end
            end
    
            if is_L2dtKing
                isppp = 1
                KingL2dtv[:,iCase] = deepcopy(KvLc0k1[isppp][:,1])         # For the exact species
    
                if iCase == NCase
                    KingL2dtv1 = ((sum((KingL2dtv[:,1:end-1] ./ KingL2dtv[:,end] .- 1) .^2;dims=1)) .^0.5)[:]
                    KingL2order = KingL2dtv1[1] ./ (2^2) .^(0:NCase-2)
        
                    xlabel = L"\Delta t"
                    ylabel = L"L_2^{\Delta t}"
                    label = L"Numerical"
                    pL2dt = plot(1 ./ Nτ_fixvec[1:end-1], KingL2dtv1,line=(2,:auto),label=label,
                                ylabel=ylabel,yscale=:log10,legend=legendbR,
                                xlabel=xlabel,xscale=:log10)
                    label = L"2^{nd} order"
                    pL2dt = plot!(1 ./ Nτ_fixvec[1:end-1], KingL2order,line=(2,:auto),label=label)
                    savefig(string(file_fig_file,"_L2dt",NCase,".png"))
                    display(pL2dt)
                    if is_CPUtime== true
                        pCPUtL2dt = plot(pL2dt,pCPUt,layout=(2,1))
                        savefig(string(file_fig_file,"_CPUL2dt",NCase,".png"))
                        display(plot(pCPUtL2dt))
                    end
                end
            end
            if is_L2dtDKing
                isppp = 1
                # DKingL2dtv[:,iCase] = deepcopy(DKing[isppp][:,1])
                DKingL2dtv[:,iCase] = DKing[isppp][:,1] ./ fvLc0k1[isppp][:,1]
    
                if iCase == NCase
                    DKingL2dtv[:,NCase+1] = deepcopy(vhek1[isppp])
    
                    vvvv4 = DKingL2dtv[:,NCase+1] .< 0.3
                    DKingL2dtv1 = ((sum((DKingL2dtv[vvvv4,1:NCase]) .^2;dims=1)) .^0.5)[:]
                    DKingL1order = DKingL2dtv1[1] ./ (2^1) .^(0:NCase-1)
                    DKingL2order = DKingL2dtv1[1] ./ (2^2) .^(0:NCase-1)
        
                    xlabel = L"\Delta t"
                    ylabel = L"L_2^{\Delta t}"
                    label = L"Numerical"
                    pL2dtDKing = plot(1 ./ Nτ_fixvec[1:end], DKingL2dtv1,line=(2,:auto),label=label,
                                ylabel=ylabel,yscale=:log10,
                                xlabel=xlabel,xscale=:log10,legend=legendbR)
                    label = L"1^{nd} order"
                    pL2dtDKing = plot!(1 ./ Nτ_fixvec, DKingL1order,line=(2,:auto),label=label)
                    label = L"2^{nd} order"
                    pL2dtDKing = plot!(1 ./ Nτ_fixvec, DKingL2order,line=(2,:auto),label=label)
                    savefig(string(file_fig_file,"_L2dtDKing",NCase,".png"))
                    display(pL2dtDKing)
                    if is_L2dtKing
                        pL2dtL2dtDKing = plot(pL2dt,pL2dtDKing,layout=(2,1))
                        display(pL2dtL2dtDKing)
                    end
                end
            end
        end

        
        # @show isp, δtf[isp][1:3,1]
        # @show 41, a[2:6]
        if  MultiType == :dt && is_plot_King_convergence
            if is_plot_dfln_thoery && iter == 2
                DKingk = (KvLc0k1 - fvLc0k)
                isp = 1
                iFv = 2
                cfF3 = (4 / pi^2 * nak1[isp] / vathk1[isp]^3 * nak1[iFv] / vathk1[iFv]^3)
                
                ivhn = 57
    
                is_dtf_theory_A = false
                vha5 = vhek1[isp] .< vh5
                if is_dtf_theory_A
                    mM = ma[isp] / ma[iFv]
                    vabth = vathk1[isp] / vathk1[iFv]
                    dtfLn = dtfMab(mM,vabth,vhkk1[isp][nvlevel0[isp][nvlevele0[isp]]])
                    dfth = cfF3 * dtfLn[vha5]
                    af = (Dfing[isp][vha5,1] ./ dfth)
                    aK = (DKingk[isp][vha5,1] ./ dfth)
                    # @show af[1:3]
                    # @show aK[1:3]
    
                    xlabel = string("v̂")
                    label = string("a0,fk1-fk,nnv=",nnv[isp])
                    pafa = plot(vhek1[isp][vha5][2:end],af[2:end] / af[3] .- 1,label=label,xlabel=xlabel)
                    label = string("a0,Kk1-fk,Δt=",1 / Nτ_fix)
                    paKa = plot!(vhek1[isp][vha5][2:end],aK[2:end] / aK[3] .- 1,label=label)
                    ygfdrgf
                else
                    dfth = dtfvLc0k[isp][vha5,1]
                    af = (Dfing[isp][vha5,1] / dtk ./ dfth .- 1)
                    aK = (DKingk[isp][vha5,1] / dtk ./ dfth .- 1)
                    # @show dtk, dfth[1:3]
                    # @show Dfing[isp][1:3,1] / dtk
                    # @show DKingk[isp][1:3,1] / dtk
                    # @show af[1:3]
                    # @show aK[1:3]
    
                    xlabel = string("v̂")
                    ylabel = string("(fk1-fk)/dt / dtfk -1")
                    if spices[1] == spices[2]
                        label = string("a",iter,",nnv=",nnv[isp])
                    else
                        label = string(spices[isp],iter,",nnv=",nnv[isp])
                    end
                    pafa = plot(vhek1[isp][vha5][2:end],af[2:end],label=label,ylabel=ylabel)
    
                    ylabel = string("(Kk1-fk)/dt / dtfk -1")
                    if spices[1] == spices[2]
                        label = string("a",iter,",nnv=",nnv[isp])
                    else
                        label = string(spices[isp],iter,",nnv=",nnv[isp])
                    end
                    paKa = plot(vhek1[isp][vha5][2:end],aK[2:end],label=label,ylabel=ylabel,xlabel=xlabel)
                end
                
                vha5 = vhek1[iFv] .< vh5
                if is_dtf_theory_A
                    mM = ma[iFv] / ma[isp]
                    vabth = vathk1[iFv] / vathk1[isp]
                    dtfLn = dtfMab(mM,vabth,vhkk1[iFv][nvlevel0[iFv][nvlevele0[iFv]]])
                    dfth = cfF3 * dtfLn[vha5]
                    af = (Dfing[iFv][vha5,1] ./ dfth)
                    aK = (DKingk[iFv][vha5,1] ./ dfth)
                    label = string("b",iter,",fk1-fk,nnv=",nnv[iFv])
                    pafb = plot(vhek1[iFv][vha5][2:end],af[2:end] / af[3] .- 1,label=label,xlabel=xlabel)
                    label = string("b0,Kk1-fk,Δt=",1 / Nτ_fix)
                    paKb = plot!(vhek1[iFv][vha5][2:end],aK[2:end] / aK[3] .- 1,label=label)
                else
                    dfth = dtfvLc0k[iFv][vha5,1]
                    af = (Dfing[iFv][vha5,1] / dtk ./ dfth .- 1)
                    aK = (DKingk[iFv][vha5,1] / dtk ./ dfth .- 1)
                    # @show dtk, dfth[1:3]
                    # @show Dfing[iFv][1:3,1] / dtk
                    # @show DKingk[iFv][1:3,1] / dtk
                    # @show af[1:3]
                    # @show aK[1:3]
    
                    if spices[1] == spices[2]
                        label = string("b",iter,",dt=",1/Nτ_fix)
                    else
                        label = string(spices[iFv],iter,",dt=",1/Nτ_fix)
                    end
                    pafb = plot(vhek1[iFv][vha5][2:end],af[2:end],label=label)
                    if spices[1] == spices[2]
                        label = string("b",iter,",dt=",1/Nτ_fix)
                    else
                        label = string(spices[iFv],iter,",dt=",1/Nτ_fix)
                    end
                    paKb = plot(vhek1[iFv][vha5][2:end],aK[2:end],label=label,xlabel=xlabel)
                end
                display(plot(pafa,pafb,paKa,paKb,layout=(2,2)))
            end
            
            if iter < 0
                DKing = fvLc0k1 - KvLc0k1
                Dfing = (fvLc0k1 - fvLc0k)
    
                isp = 1
                label = string("fl0ak,Δt=",1/Nτ_fix)
                pKinga = plot(vhek1[isp],fvLc0k[isp][:,1],line=(2,:auto),label=label)
                label = string("fl0ak1,t=",fmtf6(tk))
                pKinga = plot!(vhek1[isp],fvLc0k1[isp][:,1],line=(2,:auto),label=label)
                label = string("Kl0ak1,iter=",iter)
                pKinga = plot!(vhek1[isp],KvLc0k1[isp][:,1],line=(2,:auto),label=label)
    
                isp = 2
                label = string("fl0bk,vh=",fmtf4(vhek1[isp][ivh]))
                pKingb = plot(vhek1[isp],fvLc0k[isp][:,1],line=(2,:auto),label=label)
                label = string("fl0bk,vh=",fmtf4(vhek1[isp][ivh]*vathk1[isp]))
                pKingb = plot!(vhek1[isp],fvLc0k1[isp][:,1],line=(2,:auto),label=label)
                label = "Kl0bk1"
                pKingb = plot!(vhek1[isp],KvLc0k1[isp][:,1],line=(2,:auto),label=label)
                pKingab = plot(pKinga,pKingb,layout=(1,2))
                if is_display_King
                    display(plot(pKinga,pKingb,layout=(1,2)))
                end
            end
            if iter > 0
                DKing = fvLc0k1 - KvLc0k1
                Dfing = (fvLc0k1 - fvLc0k)
    
                isp = 1
                xlabel = L"\hat{v}"
                ylabel = L"f_l^0"
                label = L"f_l^0(t_k)"
                pKinga = plot(vhek1[isp],fvLc0k[isp][:,1],line=(2,:auto),label=label,xlabel=xlabel,ylabel=ylabel)
                label = L"f_l^{0*}(t_{k+1})"
                pKinga = plot!(vhek1[isp],fvLc0k1[isp][:,1],line=(2,:auto),label=label)
                label = L"f_l^0(t_{k+1})"
                pKinga = plot!(vhek1[isp],KvLc0k1[isp][:,1],line=(2,:auto),label=label)
    
                isp = 2
                label = L"F_l^0(t_k)"
                pKingb = plot(vhek1[isp],fvLc0k[isp][:,1],line=(2,:auto),label=label,xlabel=xlabel)
                label = L"F_l^{0*}(t_{k+1})"
                pKingb = plot!(vhek1[isp],fvLc0k1[isp][:,1],line=(2,:auto),label=label)
                label = L"F_l^0(t_{k+1})"
                pKingb = plot!(vhek1[isp],KvLc0k1[isp][:,1],line=(2,:auto),label=label)
                pKingab = plot(pKinga,pKingb,layout=(1,2))
                if is_display_King
                    display(plot(pKinga,pKingb,layout=(1,2)))
                end
            end
    
            if Cnm_name == :C0
                title = string("C0,Δtₖ=",1/Nτ_fix,",iter=",iter)
            elseif Cnm_name == :C2
                title = string("C2,Δtₖ=",1/Nτ_fix,",iter=",iter)
            end
    
            if 1 == 1
        
                isp = 1
                ylabel = L"\Delta K_l^0"
                label = L"l=0"
                label = string("a, Δt=",1 / Nτ_fix)
                xlabel = L"\hat{v}"
                pDfinga = plot(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel,title=title)
                isp = 2
                # ylabel = "ΔF̂ₗ⁰"
                # label = string(spices[isp])
                label = string("b, Δt=",1 / Nτ_fix)
                pDfingb = plot!(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel,yscale=:log10,xlabel=xlabel)
                # display(plot(pDfinga,pDfingb,layout=(2,1)))
            
                isp = 1
                ylabel = L"\Delta f_l^0"
                label = L"l=0"
                # label = string("a, Δt=",1 / Nτ_fix)
                label = string("a")
                xlabel = L"\hat{v}"
                pDKinga = plot(vhek1[isp], abs.(Dfing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel)
                isp = 2
                # ylabel = L"\Delta F_l^0"
                # label = string(spices[isp])
                # label = string("b, Δt=",1 / Nτ_fix)
                label = string("b")
                pDKingb = plot!(vhek1[isp], abs.(Dfing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel,yscale=:log10,xlabel=xlabel)
                # display(plot(pDKinga,pDKingb,layout=(2,1)))
            
                # RDKing = (KvL0k1 - fvL0k1) ./ Dfing
                isp = 1
                ylabel = L"\delta f_l^0"
                if Cnm_name == :C0
                    DKing[isp] ./= Dfing[isp]
                elseif Cnm_name == :C2
                    DKing[isp] ./= fvLc0k1[isp] * (2^(iCase-1))
                end
                # label = string(spices[isp])
                # label = string("a")
                label = string("a, Δt=",1 / Nτ_fix)
                pRDKinga = plot(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel,yscale=:log10)
                isp = 2
                # ylabel = L"\delta F_l^0"
                if Cnm_name == :C0
                    DKing[isp] ./= Dfing[isp]
                elseif Cnm_name == :C2
                    DKing[isp] ./= fvLc0k1[isp] * (2^(iCase-1))
                end
                # label = string(spices[isp])
                label = string("b, Δt=",1 / Nτ_fix)
                pRDKingb = plot!(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel,yscale=:log10,xlabel=xlabel)
        
        
                isp = 1
                iFv = 2
    
                if is_save_nvG_NCase
                    RDKingvec[1:nvG[isp],iCase,isp] = deepcopy(abs.(DKing[isp][:,1]).+epsT)
                    RDKingvec[1:nvG[iFv],iCase,iFv] = deepcopy(abs.(DKing[iFv][:,1]).+epsT)
                else
                    RDKingvec[1,iCase,isp] = deepcopy(abs.(DKing[isp][ivh,1]).+epsT)
                    RDKingvec[1,iCase,iFv] = deepcopy(abs.(DKing[iFv][ivh,1]).+epsT)
                end
                if is_display_King
                    display(plot(pDKinga,pDfinga,pRDKingb,layout=(3,1)))
                end
                plot(pDKinga,pDfinga,pRDKingb,layout=(3,1))
                title = string("Δtₖ=",1/Nτ_fix,"iter=",iter)
                if Cnm_name == :C0
                    savefig(string(file_fig_file,"_RDKingC0",title,".png"))
                elseif Cnm_name == :C2
                    savefig(string(file_fig_file,"_RDKingC2",title,".png"))
                end
            end
    
            if is_save_nvG_NCase
                if Lmode == :L0
                    RDKing00vec[iCase,isp] = abs.(RDKingvec[ivh,iCase,isp])
                    RDKing00vec[iCase,iFv] = abs.(RDKingvec[ivh,iCase,iFv])
                elseif Lmode == :L1
                    if is_weight_DKing
                        vec05 = vhek1[isp] .< vh5
                        RDKing00vec[iCase,isp] = sum(abs.(RDKingvec[1:nvG[isp],iCase,isp][vec05]) .* KvL0k1[isp][vec05,1])[1] / sum(vec05)
                        vec05 = vhek1[iFv] .< vh5
                        RDKing00vec[iCase,iFv] = sum(abs.(RDKingvec[1:nvG[iFv],iCase,iFv][vec05]) .* KvL0k1[iFv][vec05,1])[1] / sum(vec05)   
                    else
                        vec05 = vhek1[isp] .< vh5
                        RDKing00vec[iCase,isp] = sum(abs.(RDKingvec[1:nvG[isp],iCase,isp][vec05]))[1] / sum(vec05)
                        vec05 = vhek1[iFv] .< vh5
                        RDKing00vec[iCase,iFv] = sum(abs.(RDKingvec[1:nvG[iFv],iCase,iFv][vec05]))[1] / sum(vec05)
                    end
                else
                    dfbvbfdff
                end
            else
                if Lmode == :L0
                    RDKing00vec[iCase,isp] = abs.(RDKingvec[1,iCase,isp])
                    RDKing00vec[iCase,iFv] = abs.(RDKingvec[1,iCase,iFv])
                elseif Lmode == :L1
                    if is_weight_DKing
                        vec05 = vhek1[isp] .< vh5
                        RDKing00vec[iCase,isp] = sum(abs.(RDKingvec[1,iCase,isp][vec05]) .* KvL0k1[isp][vec05,1])[1] / sum(vec05)
                        vec05 = vhek1[iFv] .< vh5
                        RDKing00vec[iCase,iFv] = sum(abs.(RDKingvec[1,iCase,iFv][vec05]) .* KvL0k1[iFv][vec05,1])[1] / sum(vec05)   
                    else
                        vec05 = vhek1[isp] .< vh5
                        RDKing00vec[iCase,isp] = sum(abs.(RDKingvec[1,iCase,isp][vec05]))[1] / sum(vec05)
                        vec05 = vhek1[iFv] .< vh5
                        RDKing00vec[iCase,iFv] = sum(abs.(RDKingvec[1,iCase,iFv][vec05]))[1] / sum(vec05)
                    end
                else
                    dfbvbfdff
                end
            end
    
            isp = 1
            iFv= 2
            wlinecl = 2
            if is_output_convergence
                
                wlinecl = 2
                if 1 == 1
                    if 2 == 1
                        DKing = fvL0k1 - KvL0k1
                        Dfing = (fvLc0k1 - fvLc0k)
                    else
                        KvLc0k1 = deepcopy(KvL0k1)
                        for isp in 1:2
                            KvLc0k1[isp] *= (nak1[isp] / vathk1[isp]^3)
                        end
                        DKing = fvLc0k1 - KvLc0k1
                        Dfing = (fvLc0k1 - fvLc0k)
                    end
            
                    isp = 1
                    ylabel = L"\Delta f_l^0"
                    label = L"l=0"
                    label = string(L"a, \Delta t=",1 / Nτ_fix)
                    label = L"a"
                    xlabel = L"\hat{v}"
                    pDfinga = plot(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(wlinecl,:auto),
                                    ylabel=ylabel)
                    isp = 2
                    # ylabel = "ΔF̂ₗ⁰"
                    # label = string(spices[isp])
                    if spices[1] == spices[2]
                        label = string(L"b, \Delta t=",fmtf2(1 / Nτ_fix))
                    else
                        label = string(spices[2],L"\Delta t=",fmtf2(1 / Nτ_fix))
                    end
                    pDfingb = plot!(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(wlinecl,:auto),
                                    ylabel=ylabel,yscale=:log10)
                    # display(plot(pDfinga,pDfingb,layout=(2,1)))
            
                    isp = 1
                    ylabel = L"\delta f_l^0"
                    if Cnm_name == :C0
                        DKing[isp] ./= Dfing[isp]
                    elseif Cnm_name == :C2
                        DKing[isp] ./= fvLc0k1[isp] * (2^(iCase-1))
                    end
                    # label = string(spices[isp])
                    # label = string(L"a, \Delta t=",1 / Nτ_fix)
                    if spices[1] == spices[2]
                        if Cnm_name == :C0
                            label = L"a,C0"
                        elseif Cnm_name == :C2
                            label = L"a,C2"
                        end
                    else
                        if Cnm_name == :C0
                            label = string(spices[1],L",C0")
                        elseif Cnm_name == :C2
                            label = string(spices[1],L",C2")
                        end
                    end
                    pRDKinga = plot(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(wlinecl,:auto),
                                    ylabel=ylabel,yscale=:log10,legend=legendbR)
                    isp = 2
                    if Cnm_name == :C0
                        DKing[isp] ./= Dfing[isp]
                    elseif Cnm_name == :C2
                        DKing[isp] ./= fvLc0k1[isp] * (2^(iCase-1))
                    end
                    if spices[1] == spices[2]
                        label = string(L"b, \Delta t=",fmtf2(1 / Nτ_fix))
                    else
                        label = string(spices[2],L",\Delta t=",fmtf2(1 / Nτ_fix))
                    end
                    pRDKingb = plot!(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(wlinecl,:auto),
                                    ylabel=ylabel,yscale=:log10,
                                    xlabel=xlabel)
                           
                end
    
                isp = 1
                iFv = 2
                if Cnm_name == :C0
                    title = string(L"C0,\hat{v}=", fmtf2(vhe[1][ivh]))
                elseif Cnm_name == :C2
                    title = string(L"C2,\hat{v}=", fmtf2(vhe[1][ivh]))
                end
                if length(Nτ_fixvec) ≥ 2
                    # DKing *= Nτ_fix
            
                    wlinecl = 2
            
                    ######################################################## Convergence
    
                    if is_plot_Convergence
                        methodvv = 1
                        if is_save_nvG_NCase
                            if methodvv == 2
                                # RDKing33vec[:,isp] = reverse(abs.(RDKingvec[ivh,:,isp]))
                                RDKing33vec[:,isp] = (abs.(RDKingvec[ivh,:,isp]))
                                orderavec = order_converg(RDKing33vec[:,isp])
                            
                                # RDKing33vec[:,iFv] = reverse(abs.(RDKingvec[ivh,:,iFv]))
                                RDKing33vec[:,iFv] = (abs.(RDKingvec[ivh,:,iFv]))
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            elseif methodvv == 1
                                RDKing33vec[:,isp] = RDKing00vec[:,isp]
                                RDKing33vec[:,iFv] = RDKing00vec[:,iFv]
        
                                orderavec = order_converg(RDKing33vec[:,isp])
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = L"\Delta f_l^0"
                            elseif methodvv == 3
                                RDKing33vec[:,isp] = abs.(RDKingvec[ivh,1:end,isp] .- RDKingvec[ivh,end,isp])
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[ivh,1:end,iFv] .- RDKingvec[ivh,end,iFv])
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            else
                                RDKing33vec[:,isp] = abs.(RDKingvec[ivh,1:end,isp] .- RDKingvec[ivh,end,isp]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[ivh,1:end,iFv] .- RDKingvec[ivh,end,iFv]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            end
                        else
                            if methodvv == 2
                                # RDKing33vec[:,isp] = reverse(abs.(RDKingvec[1,:,isp]))
                                RDKing33vec[:,isp] = (abs.(RDKingvec[1,:,isp]))
                                orderavec = order_converg(RDKing33vec[:,isp])
                            
                                # RDKing33vec[:,iFv] = reverse(abs.(RDKingvec[1,:,iFv]))
                                RDKing33vec[:,iFv] = (abs.(RDKingvec[1,:,iFv]))
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            elseif methodvv == 1
                                RDKing33vec[:,isp] = RDKing00vec[:,isp]
                                RDKing33vec[:,iFv] = RDKing00vec[:,iFv]
        
                                orderavec = order_converg(RDKing33vec[:,isp])
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = L"\Delta f_l^0"
                            elseif methodvv == 3
                                RDKing33vec[:,isp] = abs.(RDKingvec[1,1:end,isp] .- RDKingvec[1,end,isp])
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[1,1:end,iFv] .- RDKingvec[1,end,iFv])
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            else
                                RDKing33vec[:,isp] = abs.(RDKingvec[1,1:end,isp] .- RDKingvec[1,end,isp]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[1,1:end,iFv] .- RDKingvec[1,end,iFv]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            end
                        end
                        if is_display_King
                            display(plot(pDKinga,pDfinga,pRDKingb,layout=(3,1)))
                        end
                    
                        ordera = sum(orderavec) / length(orderavec)
                        orderb = sum(orderbvec) / length(orderbvec)
                    
                        RDKing33vec1 = deepcopy(RDKing33vec)
                        RDKing33vec2 = deepcopy(RDKing33vec)
                        # RDKing33vec3 = deepcopy(RDKing33vec)
                        # RDKing33vec4 = deepcopy(RDKing33vec)
                        for iii in 1:NCase-1
                            RDKing33vec1[iii,:] = RDKing33vec1[1,:] / (2^1) .^ (iii-1)
                            RDKing33vec2[iii,:] = RDKing33vec1[1,:] / (2^2) .^ (iii-1)
                            # RDKing33vec3[iii,:] = RDKing33vec1[1,:] / (2^3) .^ (iii-1)
                            # RDKing33vec4[iii,:] = RDKing33vec1[1,:] / (2^4) .^ (iii-1)
                        end
                        xlabel = L"\Delta t"
                        il1 = 1
                        il2 = 2
                        ic1 = 1
                        ic2type = 2
                        if ic2type == 1
                            ic2 = ic1 + 1
                        else
                            ic2 = ic1 + 6
                        end
                        vecdt = 1:length(Nτ_fixvec)-1
                        dtvec = 1 ./ Nτ_fixvec[vecdt]
                    
                        if methodvv == 1
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"a,\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = L"a,L1"
                                elseif  Lmode == :L2
                                    label = L"a,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[1],L",\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[1],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[1],L",L2")
                                end
                            end
                            # ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,linetypes[il1]),
                            #             ylabel=ylabel)
                            ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel,xscale=:log10)
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"b,\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = L"b,L1"
                                elseif  Lmode == :L2
                                    label = L"b,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[2],L",\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[2],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[2],L",L2")
                                end
                            end
                            # ppb = plot(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,linetypes[il2]),
                            ppb = plot(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel,xscale=:log10,legend=legendbR)
                        else
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"a,\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = L"a,L1"
                                elseif  Lmode == :L2
                                    label = L"a,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[1],L",\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[1],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[1],L",L2")
                                end
                            end
                            ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,:auto))
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"b,\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = L"b,L1"
                                elseif  Lmode == :L2
                                    label = L"b,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[2],L",\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[2],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[2],L",L2")
                                end
                            end
                            ppb = plot!(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel,xscale=:log10,legend=legendbR)
                        end
                        
                        # order
                        if methodvv ≠ 1
                            if 2 == 1
                                if ordera ≤ 2
                                    ic1 += 1
                                    if ic2type == 1
                                        ic2 = ic1 + 1
                                    else
                                        ic2 = ic1 + 6
                                    end
                                    il1 += 1
                                    il2 += 1
                                    label = L"a, 1^{th} order"
                                    plot!(dtvec, RDKing33vec1[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                    if 1 ≤ ordera
                                        label = L"a, 2^{nd} order"
                                        plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                    end
                                else
                                    if 2 ≤ ordera ≤ 3
                                        ic1 += 1
                                        if ic2type == 1
                                            ic2 = ic1 + 1
                                        else
                                            ic2 = ic1 + 6
                                        end
                                        il1 += 1
                                        il2 += 1
                                        label = L"a, 2^{nd} order"
                                        plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                        label = L"a, 3^{th} order"
                                        plot!(dtvec, RDKing33vec3[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                    else
                                        if 3 ≤ ordera ≤ 4
                                            ic1 += 1
                                            if ic2type == 1
                                                ic2 = ic1 + 1
                                            else
                                                ic2 = ic1 + 6
                                            end
                                            il1 += 1
                                            il2 += 1
                                            label = L"a, 3^{th} order"
                                            plot!(dtvec, RDKing33vec3[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                            label = L"a, 4^{th} order"
                                            plot!(dtvec, RDKing33vec4[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                        else
                                            @show ordera
                                            sdfgbn
                                        end
                                    end
                                end
                                
                                if orderb ≤ 2
                                    ic1 += 1
                                    if ic2type == 1
                                        ic2 = ic1 + 1
                                    else
                                        ic2 = ic1 + 6
                                    end
                                    il1 += 1
                                    il2 += 1
                                    label = L"b, 1^{th} order"
                                    ppa = plot!(dtvec, RDKing33vec1[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                                ylabel=ylabel,yscale=:log10,
                                                xlabel=xlabel)
                                    if 1 ≤ ordera
                                        label = L"b, 2^{nd} order"
                                        ppa = plot!(dtvec, RDKing33vec2[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                    ylabel=ylabel,yscale=:log10,
                                                    xlabel=xlabel)
                                    end
                                else
                                    if 2 ≤ ordera ≤ 3
                                        ic1 += 1
                                        if ic2type == 1
                                            ic2 = ic1 + 1
                                        else
                                            ic2 = ic1 + 6
                                        end
                                        il1 += 1
                                        il2 += 1
                                        label = L"b, 2^{nd} order"
                                        ppa = plot!(dtvec, RDKing33vec2[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                                    ylabel=ylabel,yscale=:log10,
                                                    xlabel=xlabel)
                                        # ic2 += 1
                                        label = L"b, 3^{th} order"
                                        ppa = plot!(dtvec, RDKing33vec3[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                    ylabel=ylabel,yscale=:log10,
                                                    xlabel=xlabel)
                                    else
                                        if 3 ≤ ordera ≤ 4
                                            ic1 += 1
                                            if ic2type == 1
                                                ic2 = ic1 + 1
                                            else
                                                ic2 = ic1 + 6
                                            end
                                            il1 += 1
                                            il2 += 1
                                            label = L"b, 3^{th} order"
                                            ppa = plot!(dtvec, RDKing33vec3[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                                        ylabel=ylabel,yscale=:log10,
                                                        xlabel=xlabel)
                                            1
                                            label = L"b, 4^{th} order"
                                            ppa = plot!(dtvec, RDKing33vec4[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                        ylabel=ylabel,yscale=:log10,
                                                        xlabel=xlabel)
                                        else
                                            sdfgbn
                                        end
                                    end
                                end
                            else
                                # label = L"1^{st} order"
                                # plot!(dtvec, RDKing33vec1[vecdt,1],label=label,line=(wlinecl,:auto))
                                label = L"2^{nd} order"
                                plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,:auto))
                            end
                        end
                        # # display(plot(ppa))
                        # # savefig(string(file_fig_file,"_RDT2th.png"))
                        # display(plot(ppa))
                        # title = string("nτ",nτ)
                        # if Cnm_name == :C0
                        #     savefig(string(file_fig_file,"_RDKing2C0",title,".png"))
                        # elseif Cnm_name == :C2
                        #     savefig(string(file_fig_file,"_RDKing2C0",title,".png"))
                        # end
            
                        # display(plot(ppa))
                        # savefig(string(file_fig_file,"_RDKing2th.png"))
                        # display(plot(ppa))
                        if Cnm_name == :C0
                            title = string(L"C0",NCase)
                        elseif Cnm_name == :C2
                            title = string(L"C2",NCase)
                        end
                        savefig(string(file_fig_file,"_RDKing123th",title,".png"))
                    else
                        methodvv = 2
                        if is_save_nvG_NCase
                            if methodvv == 1
                                # RDKing33vec[:,isp] = reverse(abs.(RDKingvec[ivh,:,isp]))
                                RDKing33vec[:,isp] = (abs.(RDKingvec[ivh,:,isp]))
                                orderavec = order_converg(RDKing33vec[:,isp])
                            
                                # RDKing33vec[:,iFv] = reverse(abs.(RDKingvec[ivh,:,iFv]))
                                RDKing33vec[:,iFv] = (abs.(RDKingvec[ivh,:,iFv]))
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            elseif methodvv == 2
                                RDKing33vec[:,isp] = RDKing00vec[:,isp]
                                RDKing33vec[:,iFv] = RDKing00vec[:,iFv]
        
                                orderavec = order_converg(RDKing33vec[:,isp])
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = L"\Delta f_l^0"
                            elseif methodvv == 3
                                RDKing33vec[:,isp] = abs.(RDKingvec[ivh,1:end,isp] .- RDKingvec[ivh,end,isp])
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[ivh,1:end,iFv] .- RDKingvec[ivh,end,iFv])
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            else
                                RDKing33vec[:,isp] = abs.(RDKingvec[ivh,1:end,isp] .- RDKingvec[ivh,end,isp]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[ivh,1:end,iFv] .- RDKingvec[ivh,end,iFv]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            end
                        else
                            if methodvv == 1
                                # RDKing33vec[:,isp] = reverse(abs.(RDKingvec[1,:,isp]))
                                RDKing33vec[:,isp] = (abs.(RDKingvec[1,:,isp]))
                                orderavec = order_converg(RDKing33vec[:,isp])
                            
                                # RDKing33vec[:,iFv] = reverse(abs.(RDKingvec[1,:,iFv]))
                                RDKing33vec[:,iFv] = (abs.(RDKingvec[1,:,iFv]))
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            elseif methodvv == 2
                                RDKing33vec[:,isp] = RDKing00vec[:,isp]
                                RDKing33vec[:,iFv] = RDKing00vec[:,iFv]
        
                                orderavec = order_converg(RDKing33vec[:,isp])
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = L"\Delta f_l^0"
                            elseif methodvv == 3
                                RDKing33vec[:,isp] = abs.(RDKingvec[1,1:end,isp] .- RDKingvec[1,end,isp])
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[1,1:end,iFv] .- RDKingvec[1,end,iFv])
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            else
                                RDKing33vec[:,isp] = abs.(RDKingvec[1,1:end,isp] .- RDKingvec[1,end,isp]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[1,1:end,iFv] .- RDKingvec[1,end,iFv]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            end
                        end
                    
                        ordera = sum(orderavec) / length(orderavec)
                        orderb = sum(orderbvec) / length(orderbvec)
                    
                        RDKing33vec1 = deepcopy(RDKing33vec)
                        RDKing33vec2 = deepcopy(RDKing33vec)
                        # RDKing33vec3 = deepcopy(RDKing33vec)
                        # RDKing33vec4 = deepcopy(RDKing33vec)
                        for iii in 1:NCase-1
                            RDKing33vec1[iii,:] = RDKing33vec1[1,:] / (2^1) .^ (iii-1)
                            RDKing33vec2[iii,:] = RDKing33vec1[1,:] / (2^2) .^ (iii-1)
                            # RDKing33vec3[iii,:] = RDKing33vec1[1,:] / (2^3) .^ (iii-1)
                            # RDKing33vec4[iii,:] = RDKing33vec1[1,:] / (2^4) .^ (iii-1)
                        end
                        xlabel = L"\Delta t"
                        il1 = 1
                        il2 = 2
                        ic1 = 1
                        ic2type = 2
                        if ic2type == 1
                            ic2 = ic1 + 1
                        else
                            ic2 = ic1 + 6
                        end
                        vecdt = 1:length(Nτ_fixvec)-1
                        dtvec = 1 ./ Nτ_fixvec[vecdt]
                    
                        if methodvv == 1
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"a,\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = L"a,L1"
                                elseif  Lmode == :L2
                                    label = L"a,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[1],L",\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[1],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[1],L",L2")
                                end
                            end
                            # ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,linetypes[il1]),
                            #             ylabel=ylabel)
                            ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel)
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"b,\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = L"b,L1"
                                elseif  Lmode == :L2
                                    label = L"b,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[2],L",\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[2],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[2],L",L2")
                                end
                            end
                            # ppb = plot(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,linetypes[il2]),
                            ppb = plot(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel,
                                        xlabel=xlabel,xscale=:log10,legend=legendbR)
                        else
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"a,\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = L"a,L1"
                                elseif  Lmode == :L2
                                    label = L"a,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[1],L",\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[1],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[1],L",L2")
                                end
                            end
                            ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,:auto))
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"b,\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = L"b,L1"
                                elseif  Lmode == :L2
                                    label = L"b,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[2],L",\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[2],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[2],L",L2")
                                end
                            end
                            ppb = plot!(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel,xscale=:log10,legend=legendbR)
                        end
                        
                        # order
                        if 2 == 1
                            if ordera ≤ 2
                                ic1 += 1
                                if ic2type == 1
                                    ic2 = ic1 + 1
                                else
                                    ic2 = ic1 + 6
                                end
                                il1 += 1
                                il2 += 1
                                label = L"a, 1^{th} order"
                                plot!(dtvec, RDKing33vec1[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                if 1 ≤ ordera
                                    label = L"a, 2^{nd} order"
                                    plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                end
                            else
                                if 2 ≤ ordera ≤ 3
                                    ic1 += 1
                                    if ic2type == 1
                                        ic2 = ic1 + 1
                                    else
                                        ic2 = ic1 + 6
                                    end
                                    il1 += 1
                                    il2 += 1
                                    label = L"a, 2^{nd} order"
                                    plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                    label = L"a, 3^{th} order"
                                    plot!(dtvec, RDKing33vec3[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                else
                                    if 3 ≤ ordera ≤ 4
                                        ic1 += 1
                                        if ic2type == 1
                                            ic2 = ic1 + 1
                                        else
                                            ic2 = ic1 + 6
                                        end
                                        il1 += 1
                                        il2 += 1
                                        label = L"a, 3^{th} order"
                                        plot!(dtvec, RDKing33vec3[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                        label = L"a, 4^{th} order"
                                        plot!(dtvec, RDKing33vec4[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                    else
                                        @show ordera
                                        sdfgbn
                                    end
                                end
                            end
                            
                            if orderb ≤ 2
                                ic1 += 1
                                if ic2type == 1
                                    ic2 = ic1 + 1
                                else
                                    ic2 = ic1 + 6
                                end
                                il1 += 1
                                il2 += 1
                                label = L"b, 1^{th} order"
                                ppa = plot!(dtvec, RDKing33vec1[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                            ylabel=ylabel,yscale=:log10,
                                            xlabel=xlabel)
                                if 1 ≤ ordera
                                    label = L"b, 2^{nd} order"
                                    ppa = plot!(dtvec, RDKing33vec2[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                ylabel=ylabel,yscale=:log10,
                                                xlabel=xlabel)
                                end
                            else
                                if 2 ≤ ordera ≤ 3
                                    ic1 += 1
                                    if ic2type == 1
                                        ic2 = ic1 + 1
                                    else
                                        ic2 = ic1 + 6
                                    end
                                    il1 += 1
                                    il2 += 1
                                    label = L"b, 2^{nd} order"
                                    ppa = plot!(dtvec, RDKing33vec2[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                                ylabel=ylabel,yscale=:log10,
                                                xlabel=xlabel)
                                    # ic2 += 1
                                    label = L"b, 3^{th} order"
                                    ppa = plot!(dtvec, RDKing33vec3[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                ylabel=ylabel,yscale=:log10,
                                                xlabel=xlabel)
                                else
                                    if 3 ≤ ordera ≤ 4
                                        ic1 += 1
                                        if ic2type == 1
                                            ic2 = ic1 + 1
                                        else
                                            ic2 = ic1 + 6
                                        end
                                        il1 += 1
                                        il2 += 1
                                        label = L"b, 3^{th} order"
                                        ppa = plot!(dtvec, RDKing33vec3[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                                    ylabel=ylabel,yscale=:log10,
                                                    xlabel=xlabel)
                                        1
                                        label = L"b, 4^{th} order"
                                        ppa = plot!(dtvec, RDKing33vec4[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                    ylabel=ylabel,yscale=:log10,
                                                    xlabel=xlabel)
                                    else
                                        sdfgbn
                                    end
                                end
                            end
                        else
                            # label = L"1^{st} order"
                            # plot!(dtvec, RDKing33vec1[vecdt,1],label=label,line=(wlinecl,:auto))
                            label = L"2^{nd} order"
                            plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,:auto))
                        end
                        # # display(plot(ppa))
                        # # savefig(string(file_fig_file,"_RDT2th.png"))
                        # display(plot(ppa))
                        # if Cnm_name == :C0
                        #     title = string("C0nτ",nτ)
                        # elseif Cnm_name == :C2
                        #     title = string("C2nτ",nτ)
                        # end
                        # savefig(string(file_fig_file,"_RDKing2",title,".png"))
                    end
                end
    
                if methodvv == 1
    
                    if is_plot_Convergence
                        if Cnm_name == :C0
                            ppab = plot(ppa,ppb,layout=(1,2))
                            display(plot(pRDKingb,ppab,layout=(2,1)))
                            plot(pRDKingb,ppab,layout=(2,1))
                            savefig(string(file_fig_file,"_RDKingC4C0.png"))
                        elseif Cnm_name == :C2
                            if is_weight_DKing
                                display(plot(pKingb,pRDKingb,ppa,ppb,layout=(2,2)))
                                plot(pKingb,pRDKingb,ppa,ppb,layout=(2,2))
                                savefig(string(file_fig_file,"_RDKingC4WC2.png"))
                            else
                                display(plot(pKingb,pRDKingb,ppa,ppb,layout=(2,2)))
                                plot(pKingb,pRDKingb,ppa,ppb,layout=(2,2))
                                savefig(string(file_fig_file,"_RDKingC4C2.png"))
                            end
                        end
                    else
                        ppab = plot(ppa,ppb,layout=(1,2))
                        display(plot(pRDKingb,ppab,layout=(2,1)))
                        plot(pRDKingb,ppab,layout=(2,1))
                        if Cnm_name == :C0
                            savefig(string(file_fig_file,"_RDKing3C0.png"))
                        elseif Cnm_name == :C2
                            if is_weight_DKing
                                savefig(string(file_fig_file,"_RDKing3WC2.png"))
                            else
                                savefig(string(file_fig_file,"_RDKing3C2.png"))
                            end
                        end
                    end
                else
                    # display(plot(pRDKingb,ppa,layout=(2,1)))
                    # plot(pRDKingb,ppa,layout=(2,1))
                    # if Cnm_name == :C0
                    #     savefig(string(file_fig_file,"_RDKing2C0.png"))
                    # elseif Cnm_name == :C2
                    #     savefig(string(file_fig_file,"_RDKing2C2.png"))
                    # end
    
                    # display(plot(pKingab,pRDKingb,ppa,layout=(3,1)))
                    # plot(pKingab,pRDKingb,ppa,layout=(3,1))
                    if Cnm_name == :C0
                        savefig(string(file_fig_file,"_RDKing3C0.png"))
                    elseif Cnm_name == :C2
                        if is_weight_DKing
                            savefig(string(file_fig_file,"_RDKing3WC2.png"))
                        else
                            savefig(string(file_fig_file,"_RDKing3C2.png"))
                        end
                    end
    
                    display(plot(pKinga,pKingb,pRDKingb,ppa,layout=(2,2)))
                    plot(pKinga,pKingb,pRDKingb,ppa,layout=(2,2))
                    title = string("Δtₖ=",1/Nτ_fix)
                    if Cnm_name == :C0
                        savefig(string(file_fig_file,"_RDKing4C0.png"))
                    elseif Cnm_name == :C2
                        if is_weight_DKing
                            savefig(string(file_fig_file,"_RDKing4WC2.png"))
                        else
                            savefig(string(file_fig_file,"_RDKing4C2.png"))
                        end
                    end
                end
            end
        end
    end
    if is_plot_DflKing && is_plot_King_convergence
        if 1 == 2
            DKing = fvL0k1 - KvL0k1
            Dfing = (fvLc0k1 - fvLc0k)
        else
            KvLc0k1 = deepcopy(KvL0k1)
            for isp in 1:2
                KvLc0k1[isp] *= (nak1[isp] / vathk1[isp]^3)
            end
            DKing = fvLc0k1 - KvLc0k1
            Dfing = (fvLc0k1 - fvLc0k)
        end
        if is_plot_dfln_thoery
            DKingk = (KvLc0k1 - fvLc0k)
            isp = 1
            iFv = 2
            cfF3 = (4 / pi^2 * nak1[isp] / vathk1[isp]^3 * nak1[iFv] / vathk1[iFv]^3)
            
            ivhn = 57

            is_dtf_theory_A = false
            vha5 = vhek1[isp] .< vh5
            if is_dtf_theory_A
                mM = ma[isp] / ma[iFv]
                vabth = vathk1[isp] / vathk1[iFv]
                dtfLn = dtfMab(mM,vabth,vhkk1[isp][nvlevel0[isp][nvlevele0[isp]]])
                dfth = cfF3 * dtfLn[vha5]
                af = (Dfing[isp][vha5,1] ./ dfth)
                aK = (DKingk[isp][vha5,1] ./ dfth)
                # @show af[1:3]
                # @show aK[1:3]

                xlabel = string("v̂")
                label = string("a0,fk1-fk,nnv=",nnv[isp])
                pafa = plot(vhek1[isp][vha5][2:end],af[2:end] / af[3] .- 1,label=label,xlabel=xlabel)
                label = string("a0,Kk1-fk,Δt=",1 / Nτ_fix)
                paKa = plot!(vhek1[isp][vha5][2:end],aK[2:end] / aK[3] .- 1,label=label)
                ygfdrgf
            else
                dfth = dtfvLc0k[isp][vha5,1]
                af = (Dfing[isp][vha5,1] / dtk ./ dfth .- 1)
                aK = (DKingk[isp][vha5,1] / dtk ./ dfth .- 1)
                # @show dtk, dfth[1:3]
                # @show Dfing[isp][1:3,1] / dtk
                # @show DKingk[isp][1:3,1] / dtk
                # @show af[1:3]
                # @show aK[1:3]

                xlabel = string("v̂")
                ylabel = string("(fk1-fk)/dt / dtfk -1")
                if spices[1] == spices[2]
                    label = string("a0,nnv=",nnv[isp])
                else
                    label = string(spices[isp],"0,nnv=",nnv[isp])
                end
                pafa = plot(vhek1[isp][vha5][2:end],af[2:end],label=label,ylabel=ylabel)

                ylabel = string("(Kk1-fk)/dt / dtfk -1")
                if spices[1] == spices[2]
                    label = string("a0,nnv=",nnv[isp])
                else
                    label = string(spices[isp],"0,nnv=",nnv[isp])
                end
                paKa = plot(vhek1[isp][vha5][2:end],aK[2:end],label=label,ylabel=ylabel,xlabel=xlabel)
            end
            
            vha5 = vhek1[iFv] .< vh5
            if is_dtf_theory_A
                mM = ma[iFv] / ma[isp]
                vabth = vathk1[iFv] / vathk1[isp]
                dtfLn = dtfMab(mM,vabth,vhkk1[iFv][nvlevel0[iFv][nvlevele0[iFv]]])
                dfth = cfF3 * dtfLn[vha5]
                af = (Dfing[iFv][vha5,1] ./ dfth)
                aK = (DKingk[iFv][vha5,1] ./ dfth)
                label = string("b0,fk1-fk,nnv=",nnv[iFv])
                pafb = plot(vhek1[iFv][vha5][2:end],af[2:end] / af[3] .- 1,label=label,xlabel=xlabel)
                label = string("b0,Kk1-fk,Δt=",1 / Nτ_fix)
                paKb = plot!(vhek1[iFv][vha5][2:end],aK[2:end] / aK[3] .- 1,label=label)
            else
                dfth = dtfvLc0k[iFv][vha5,1]
                af = (Dfing[iFv][vha5,1] / dtk ./ dfth .- 1)
                aK = (DKingk[iFv][vha5,1] / dtk ./ dfth .- 1)
                # @show dtk, dfth[1:3]
                # @show Dfing[iFv][1:3,1] / dtk
                # @show DKingk[iFv][1:3,1] / dtk
                # @show af[1:3]
                # @show aK[1:3]

                if spices[1] == spices[2]
                    label = string("b0,dt=",1/Nτ_fix)
                else
                    label = string(spices[iFv],"0,dt=",1/Nτ_fix)
                end
                pafb = plot(vhek1[iFv][vha5][2:end],af[2:end],label=label)
                if spices[1] == spices[2]
                    label = string("b0,dt=",1/Nτ_fix)
                else
                    label = string(spices[iFv],"0,dt=",1/Nτ_fix)
                end
                paKb = plot(vhek1[iFv][vha5][2:end],aK[2:end],label=label,xlabel=xlabel)
            end
            # @show vhek1[isp][vha5][2:end]
            # @show af[2:end]
            display(plot(vhek1[isp][vha5][2:end],af[2:end]))
            # display(pafa)
            # display(pafb)
            # display(paKa)
            # display(paKb)
            display(plot(pafa,pafb,paKa,paKb,layout=(2,2)))
        end
        
        # isp = 1
        # iFv = 2

        title = string("iter=",0,",Δtₖ=",1/Nτ_fix)
    
        ylabel = L"\Delta K_l^0"
        isp = 1
        label = L"l=0"
        label = string("a, Δt=",1 / Nτ_fix)
        xlabel = L"\hat{v}"
        pDfinga = plot(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                        ylabel=ylabel)
        isp = 2
        # ylabel = "ΔF̂ₗ⁰"
        # label = string(spices[isp])
        label = string("b, Δt=",1 / Nτ_fix)
        pDfingb = plot!(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                        ylabel=ylabel,yscale=:log10,xlabel=xlabel)
        # display(plot(pDfinga,pDfingb,layout=(2,1)))
    
        ylabel = L"\Delta f_l^0"
        isp = 1
        label = L"l=0"
        label = string("a, Δt=",1 / Nτ_fix)
        xlabel = L"\hat{v}"
        pDKinga = plot(vhek1[isp], abs.(Dfing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                        ylabel=ylabel)
        isp = 2
        # ylabel = "ΔFₗ⁰"
        # label = string(spices[isp])
        label = string("b, Δt=",1 / Nτ_fix)
        pDKingb = plot!(vhek1[isp], abs.(Dfing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                        ylabel=ylabel,yscale=:log10,xlabel=xlabel)
        # display(plot(pDKinga,pDKingb,layout=(2,1)))
    
        # RDKing = (KvL0k1 - fvL0k1) ./ Dfing
        isp = 1
        ylabel = "RΔf̂ₗ⁰"
        if Cnm_name == :C0
            DKing[isp] ./= Dfing[isp]
        elseif Cnm_name == :C2
            DKing[isp] ./= fvLc0k1[isp] * (2^(iCase-1))
        end
        # label = string(spices[isp])
        # label = string("a")
        label = string("a, Δt=",1 / Nτ_fix)
        pRDKinga = plot(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                        ylabel=ylabel,yscale=:log10,title=title)
        isp = 2
        # ylabel = "RΔF̂ₗ⁰"
        if Cnm_name == :C0
            DKing[isp] ./= Dfing[isp]
        elseif Cnm_name == :C2
            DKing[isp] ./= fvLc0k1[isp] * (2^(iCase-1))
        end
        # label = string(spices[isp])
        label = string("b, Δt=",1 / Nτ_fix)
        pRDKingb = plot!(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                        ylabel=ylabel,yscale=:log10,xlabel=xlabel)
        if is_display_King
            display(plot(pDKinga,pDfinga,pRDKingb,layout=(3,1)))
        end
        
        if Cnm_name == :C0
            title = string("C0,Δtₖ=",1/Nτ_fix)
        elseif Cnm_name == :C2
            title = string("C2,Δtₖ=",1/Nτ_fix)
        end
        plot(pDKinga,pDfinga,pRDKingb,layout=(3,1))
        savefig(string(file_fig_file,"_RDKing",title,".png"))
    
        # display(plot(pDKinga,pDKingb,pRDKinga,pRDKingb,layout=(2,2)))


        isp = 1
        iFv = 2

        if is_save_nvG_NCase
            RDKingvec[1:nvG[isp],iCase,isp] = deepcopy(abs.(DKing[isp][:,1]).+epsT)
            RDKingvec[1:nvG[iFv],iCase,iFv] = deepcopy(abs.(DKing[iFv][:,1]).+epsT)
        else
            RDKingvec[1,iCase,isp] = deepcopy(abs.(DKing[isp][ivh,1]).+epsT)
            RDKingvec[1,iCase,iFv] = deepcopy(abs.(DKing[iFv][ivh,1]).+epsT)
        end
        
        if iCase == NCase
            if Cnm_name == :C0
                title = string(L"C0,\hat{v} =",fmtf2(vhe[isp][ivh]))
            elseif Cnm_name == :C2
                title = string(L"C2,\hat{v} =",fmtf2(vhe[isp][ivh]))
            end
            ylabel = "max(RΔf̂ₗ⁰)"
            if spices[1] == spices[2]
                if Cnm_name == :C0
                    label = L"a,C0"
                elseif Cnm_name == :C2
                    label = L"a,C2"
                end
            else
                if Cnm_name == :C0
                    label = string(spices[1],L",C0")
                elseif Cnm_name == :C2
                    label = string(spices[1],L",C2")
                end
            end
            if MultiType == :dt
                xlabel = L"\Delta t"
                if is_save_nvG_NCase
                    pRDfamax = plot(1 ./ Nτ_fixvec, RDKingvec[ivh,:,isp], 
                    # pRDfamax = plot(1 ./ Nτ_fixvec, RDKingvec[ivh,:,isp] .- 1, 
                                    ylabel=ylabel,
                                    xlabel=xlabel,xscale=:log10,
                                    label=label,title=title)
                    if spices[1] == spices[2]
                        label = string(L"b, \Delta t=",fmtf2(1 / Nτ_fix))
                    else
                        label = string(spices[2],L",\Delta t=",fmtf2(1 / Nτ_fix))
                    end
                    pRDfbmax = plot(1 ./ Nτ_fixvec, RDKingvec[ivh,:,iFv], 
                    # pRDfbmax = plot(1 ./ Nτ_fixvec, RDKingvec[ivh,:,iFv] .- 1, 
                                    ylabel=ylabel,
                                    xlabel=xlabel,xscale=:log10,
                                    label=label)
            
                    if is_display_RDfamax
                        display(plot(pRDfamax,pRDfbmax,layout=(2,1)))
                    end
                else
                    pRDfamax = plot(1 ./ Nτ_fixvec, RDKingvec[1,:,isp], 
                    # pRDfamax = plot(1 ./ Nτ_fixvec, RDKingvec[1,:,isp] .- 1, 
                                    ylabel=ylabel,
                                    xlabel=xlabel,xscale=:log10,
                                    label=label,title=title)
                    if spices[1] == spices[2]
                        label = string(L"b, \Delta t=",fmtf2(1 / Nτ_fix))
                    else
                        label = string(spices[2],L",\Delta t=",fmtf2(1 / Nτ_fix))
                    end
                    pRDfbmax = plot(1 ./ Nτ_fixvec, RDKingvec[1,:,iFv], 
                    # pRDfbmax = plot(1 ./ Nτ_fixvec, RDKingvec[1,:,iFv] .- 1, 
                                    ylabel=ylabel,
                                    xlabel=xlabel,xscale=:log10,
                                    label=label)
            
                    if is_display_RDfamax
                        display(plot(pRDfamax,pRDfbmax,layout=(2,1)))
                    end
                end
            elseif MultiType == :nnv
                xlabel = L"n_2"
                if is_save_nvG_NCase
                    pRDfamax = plot(nnvocpM, RDKingvec[ivh,:,isp], 
                    # pRDfamax = plot(nnvocpM, RDKingvec[ivh,:,isp] .- 1, 
                                    ylabel=ylabel,
                                    xlabel=xlabel,xscale=:log10,
                                    label=label,title=title)
                    if spices[1] == spices[2]
                        label = string(L"b, \Delta t=",fmtf2(1 / Nτ_fix))
                    else
                        label = string(spices[2],L",\Delta t=",fmtf2(1 / Nτ_fix))
                    end
                    pRDfbmax = plot(nnvocpM, RDKingvec[ivh,:,iFv], 
                    # pRDfbmax = plot(nnvocpM, RDKingvec[ivh,:,iFv] .- 1, 
                                    ylabel=ylabel,
                                    xlabel=xlabel,xscale=:log10,
                                    label=label)
            
                    if is_display_RDfamax
                        display(plot(pRDfamax,pRDfbmax,layout=(2,1)))
                    end
                    plot(pRDfamax,pRDfbmax,layout=(2,1))
                    if Cnm_name == :C0
                        savefig(string(file_fig_file,"_RDKing0MaxC0.png"))
                    elseif Cnm_name == :C2
                        savefig(string(file_fig_file,"_RDKing0MaxC0.png"))
                    end
                else
                    pRDfamax = plot(nnvocpM, RDKingvec[1,:,isp], 
                    # pRDfamax = plot(nnvocpM, RDKingvec[1,:,isp] .- 1, 
                                    ylabel=ylabel,
                                    xlabel=xlabel,xscale=:log10,
                                    label=label,title=title)
                    if spices[1] == spices[2]
                        label = string(L"b, \Delta t=",fmtf2(1 / Nτ_fix))
                    else
                        label = string(spices[2],L",\Delta t=",fmtf2(1 / Nτ_fix))
                    end
                    pRDfbmax = plot(nnvocpM, RDKingvec[1,:,iFv], 
                    # pRDfbmax = plot(nnvocpM, RDKingvec[1,:,iFv] .- 1, 
                                    ylabel=ylabel,
                                    xlabel=xlabel,xscale=:log10,
                                    label=label)
            
                    if is_display_RDfamax
                        display(plot(pRDfamax,pRDfbmax,layout=(2,1)))
                    end
                end
                plot(pRDfamax,pRDfbmax,layout=(2,1))
                if Cnm_name == :C0
                    savefig(string(file_fig_file,"_RDKing0MaxC0n2.png"))
                elseif Cnm_name == :C2
                    savefig(string(file_fig_file,"_RDKing0MaxC0n2.png"))
                end
            end
        end
    
        # rtgjhjnm
    end
    fvL0k1[:] = deepcopy(KvL0k1)
    return dtCFL
end

function fvLck1integral!(dtfvLc0k1::Vector{Matrix{T}},fvLc0k1::Vector{Matrix{T}},
    fvLc0k::Vector{Matrix{T}},dtfvLc0k::Vector{Matrix{T}},
    fvL0k1::Vector{Matrix{T}},fvL0k::Vector{Matrix{T}},
    dtnIKs::AbstractArray{T,N2},edtnIKTs::AbstractArray{T,N2},
    Mhck1::Vector{Matrix{T}},errMhc::Vector{Matrix{T}},
    nvG::Vector{Int64},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},
    ve::Vector{StepRangeLen},vk::Vector{AbstractVector{T}},
    vhek1::Vector{StepRangeLen},vhkk1::Vector{AbstractVector{T}},
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}},LMk1::Vector{Int64},LM1k1::Int64,
    naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},vthik::Vector{AbstractVector{T}},
    vhek::Vector{StepRangeLen},vhkk::Vector{AbstractVector{T}},LMk::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    nak1::AbstractVector{T},vathk::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},
    nModk1::Vector{Int64},nMjMs::Vector{Int64},vathk1i::AbstractVector{T},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},vathk1::AbstractVector{T},
    Nspan_optim_nuTi::AbstractVector{T},dtk::T,tk::T;orderRK::Int64=2,
    Nspan_nuTi_max::AbstractVector{T}=[1.05,1.2],nsk1::Int64=2,
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),is_Jacobian::Bool=true,
    is_δtfvLaa::Int=1,is_boundaryv0::Bool=false,is_fit_f::Bool=false,

    is_extrapolate_FLn::Bool=true,
    is_enforce_errdtnIKab::Bool=false,is_norm_error::Bool=true,
    is_plot_DflKing=false,is_plot_dfln_thoery::Bool=false,iter::Int64=1) where{T,N2}
    
    # ρk1 = ma .* nak1
    if orderRK == 1
        fvLck1integral0!(fvLc0k1,fvLc0k,dtfvLc0k1,nsk1,dtk)
    
        dtnIKsc!(dtnIKs,edtnIKTs,dtfvLc0k1,ve,ma,nak1,vathk1i,nsk1;
                atol_nIK=atol_nIK,is_norm_error=is_norm_error)
        if is_enforce_errdtnIKab
            dtnIKposteriorC!(dtnIKs,edtnIKTs)
        end
        # # Computing the `n, I, K`
        # nIKsc!(nIKs,fvLc0k1[:,1:2,:],ve,ma,nsk1;errnIKc=errnIKc)
        Iak1[:] = Iak + dtk * dtnIKs[2,:]
        Kak1[:] = Kak + dtk * dtnIKs[3,:]
    elseif orderRK == 2
        if 1 == 1
            dtfvLc0k1[:,:,:] = (dtfvLc0k + dtfvLc0k1) / 2
            fvLck1integral0!(fvLc0k1,fvLc0k,dtfvLc0k1,nsk1,dtk)
        
            dtnIKsc!(dtnIKs,edtnIKTs,dtfvLc0k1,ve,ma,nak1,(vathk + vathk1i) / 2,nsk1;
                    atol_nIK=atol_nIK,is_norm_error=is_norm_error)
            if is_enforce_errdtnIKab
                dtnIKposteriorC!(dtnIKs,edtnIKTs)
            end
            # # Computing the `n, I, K`
            # nIKsc!(nIKs,fvLc0k1[:,1:2,:],ve,ma,nsk1;errnIKc=errnIKc)
            Iak1[:] = Iak + dtk * dtnIKs[2,:]
            Kak1[:] = Kak + dtk * dtnIKs[3,:]
        else
            # dtfvLc0k1[:,:,:] = (dtfvLc0k + dtfvLc0k1) / 2
            fvLck1integral0!(fvLc0k1,fvLc0k,(dtfvLc0k + dtfvLc0k1) / 2,nsk1,dtk)
    
            dtnIKsk = deepcopy(dtnIKs)
            dtnIKsc!(dtnIKsk,edtnIKTs,dtfvLc0k,ve,ma,nak1,vathk,nsk1;
                    atol_nIK=atol_nIK,is_norm_error=is_norm_error)
            if is_enforce_errdtnIKab
                dtnIKposteriorC!(dtnIKsk,edtnIKTs)
            end
        
            dtnIKsc!(dtnIKs,edtnIKTs,dtfvLc0k1,ve,ma,nak1,vathk1i,nsk1;
                    atol_nIK=atol_nIK,is_norm_error=is_norm_error)
            if is_enforce_errdtnIKab
                dtnIKposteriorC!(dtnIKs,edtnIKTs)
            end
            # # Computing the `n, I, K`
            # nIKsc!(nIKs,fvLc0k1[:,1:2,:],ve,ma,nsk1;errnIKc=errnIKc)
            Iak1[:] = Iak + dtk / 2 * (dtnIKsk[2,:] + dtnIKs[2,:])
            Kak1[:] = Kak + dtk / 2 * (dtnIKsk[3,:] + dtnIKs[3,:])
        end
    else
        sdcv
    end

    # # Computing the `dtn, dtI, dtK`
    # if gridv_type == :uniform
    #     dtMcsd2l!(aa,err_Rc,dtfvLc0k1,vhe,nMjMs,ma.*nak1,vathk1i,LMk1,nsk1;is_renorm=is_renorm)
    # elseif gridv_type == :chebyshev
    #     # if is_dtf_MS_IK
    #     #     dtMcsd2l!(aa,err_Rc,dtfvLc0k1,vhe,nvG[1],nMjMs,ma.*nak1,vathk,LMk,ns;
    #     #               is_renorm=is_renorm,is_norm_error=is_norm_error)
    #     # else
    #     #     dtMcsd2l!(aa,err_Rc,dtfvLc0k1,vhe,nvG,nMjMs,ma.*nak1,vathk,LMk,ns;
    #     #               is_renorm=is_renorm,is_norm_error=is_norm_error)
    #     # end
    # end
    # @show dtk * dtnIKs[3,:] ./ Kak

    vthnIK!(vathk1,ma,nak1,Iak1,Kak1,nsk1)

    cf3 = zeros(T,nsk1)
    for isp in 1:nsk1
        cf3[isp] = (nak1[isp] / vathk1[isp]^3)
        fvL0k1[isp] = fvLc0k1[isp] / cf3[isp]
        vhek1[isp] = ve[isp] / vathk1[isp]
        vhkk1[isp] = vk[isp] / vathk1[isp]
    end

    # Computing the normalized kinetic moments
    RhnnEvens!(Mhck1, errMhc, fvL0k1, vhek1, nMjMs, LMk1, nsk1; is_renorm=is_renorm, L_Mh_limit=L_Mh_limit)

    # # Updating the FP collision terms according to the `FPS` operators.
    # dtfvLc0k1 = dtfvL0k1
    if prod(nModk1) == 1
        # # Calculate the parameters `nai, uai, vthi`
        submoment!(naik, uaik, vthik, Mhck1, nsk1)

        # # Updating the normalized distribution function `KvL0k1` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        # KvL0k1 = deepcopy(fvL0k1)
        KvL0k1 = Vector{Matrix{T}}(undef,nsk1)
        for isp in 1:nsk1
            KvL0k1[isp] = zeros(T,nvG[isp],LMk1[isp] + 3)
        end
        LM1k1, KvL0k1 = fvLDMz!(KvL0k1, vhek1, LMk1, 2, [uaik[1][1],uaik[2][1]]; 
                        L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)
        Ccol = zeros(T,2)
        if iter == 2
            is_plot_dfln_thoery7 = is_plot_dfln_thoery
        else
            is_plot_dfln_thoery7 = false
        end
        
        if is_Mhck_reconstruct
            FP0D2Vab2!(dtfvLc0k1,KvL0k1,vhkk1,nvG,nc0,nck,ocp,
                    nvlevele0,nvlevel0,LMk1,LM1k1,
                    CΓ,εᵣ,ma,Zq,spices,Ccol,nak1,[uaik[1][1],uaik[2][1]],vathk1;
                    is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs, 
                    is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                    is_extrapolate_FLn=is_extrapolate_FLn,
                    is_plot_dfln_thoery=is_plot_dfln_thoery7)
        else
            FP0D2Vab2!(dtfvLc0k1,fvL0k1,vhkk1,nvG,nc0,nck,ocp,
                    nvlevele0,nvlevel0,LMk1,LM1k1,
                    CΓ,εᵣ,ma,Zq,spices,Ccol,nak1,[uaik[1][1],uaik[2][1]],vathk1;
                    is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs, 
                    is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                    is_extrapolate_FLn=is_extrapolate_FLn,
                    is_plot_dfln_thoery=is_plot_dfln_thoery7)
        end
    else
        # # Calculate the parameters `nai, uai, vthi` from the re-normalized moments `ℳ̂ⱼ,ₗ⁰`
        submoment!(naik, uaik, vthik, nModk1, ns, Mhck1,
                Nspan_optim_nuTi,dtk;Nspan_nuTi_max=Nspan_nuTi_max,
                NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
                optimizer=optimizer,factor=factor,autodiff=autodiff,
                is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol)

        # # Updating the normalized distribution function `fvL0k1` according to the new parameters `nai`, `uai` and `vthi` on grids `vhe`
        KvL0k1 = deepcopy(fvL0k1)
        LM1k1, KvL0k1 = fvLDMz!(KvL0k1, vhek1, nvG, LMk1, ns, naik, uaik, vthik, nModk1; 
                         L_limit=L_limit, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM)

        1
        if is_Mhck_reconstruct
            FP0D2Vab2!(dtfvLc0k1,KvL0k1,vhkk1,nvG,nc0,nck,ocp,
                   nvlevele0,nvlevel0,LMk1,LM1k1,naik, uaik, vthik,
                   CΓ,εᵣ,ma,Zq,spices,nak1,vathk1,nModk1;
                   is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                   autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                   p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                   is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                   is_extrapolate_FLn=is_extrapolate_FLn)
        else
            FP0D2Vab2!(dtfvLc0k1,fvL0k1,vhkk1,nvG,nc0,nck,ocp,
                   nvlevele0,nvlevel0,LMk1,LM1k1,naik, uaik, vthik,
                   CΓ,εᵣ,ma,Zq,spices,nak1,vathk1,nModk1;
                   is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                   autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                   p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                   is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f,
                   is_extrapolate_FLn=is_extrapolate_FLn)
        end
    end

    for isp in 1:nsk1
        dtfvLc0k1[isp] *= cf3[isp]
    end
    
    if is_plot_DflKing

        if 1 == 2
            DKing = fvL0k1 - KvL0k1
            Dfing = (fvLc0k1 - fvLc0k)
        else
            KvLc0k1 = deepcopy(KvL0k1)
            for isp in 1:2
                KvLc0k1[isp] *= (nak1[isp] / vathk1[isp]^3)
            end
            DKing = fvLc0k1 - KvLc0k1
            Dfing = (fvLc0k1 - fvLc0k)
        end
        if MultiType == :dt
            xlabel = L"\Delta t"

            if is_L2dtfl0
                isppp = 1
                fl0L2dtv[:,iCase] = deepcopy(fvLc0k1[isppp][:,1])         # For the exact species
    
                if iCase == NCase
                    fl0L2dtv1 = ((sum((fl0L2dtv[:,1:end-1] ./ fl0L2dtv[:,end] .- 1) .^2;dims=1)) .^0.5)[:]
                    fl0L2order = fl0L2dtv1[1] ./ (2^2) .^(0:NCase-2)
        
                    xlabel = L"\Delta t"
                    ylabel = L"L_2^{\Delta t}"
                    label = L"Numerical"
                    pL2dt = plot(1 ./ Nτ_fixvec[1:end-1], fl0L2dtv1,line=(2,:auto),label=label,
                                ylabel=ylabel,yscale=:log10,
                                xlabel=xlabel,xscale=:log10)
                    label = L"2^{nd} order"
                    pL2dt = plot!(1 ./ Nτ_fixvec[1:end-1], fl0L2order,line=(2,:auto),label=label)
                    savefig(string(file_fig_file,"_L2dt",NCase,".png"))
                    display(pL2dt)
                end
            end
    
            if is_L2dtKing
                isppp = 1
                KingL2dtv[:,iCase] = deepcopy(KvLc0k1[isppp][:,1])         # For the exact species
    
                if iCase == NCase
                    KingL2dtv1 = ((sum((KingL2dtv[:,1:end-1] ./ KingL2dtv[:,end] .- 1) .^2;dims=1)) .^0.5)[:]
                    KingL2order = KingL2dtv1[1] ./ (2^2) .^(0:NCase-2)
        
                    xlabel = L"\Delta t"
                    ylabel = L"L_2^{\Delta t}"
                    label = L"Numerical"
                    pL2dt = plot(1 ./ Nτ_fixvec[1:end-1], KingL2dtv1,line=(2,:auto),label=label,
                                ylabel=ylabel,yscale=:log10,legend=legendbR,
                                xlabel=xlabel,xscale=:log10)
                    label = L"2^{nd} order"
                    pL2dt = plot!(1 ./ Nτ_fixvec[1:end-1], KingL2order,line=(2,:auto),label=label)
                    savefig(string(file_fig_file,"_L2dt",NCase,".png"))
                    display(pL2dt)
                    if is_CPUtime== true
                        pCPUtL2dt = plot(pL2dt,pCPUt,layout=(2,1))
                        savefig(string(file_fig_file,"_CPUL2dt",NCase,".png"))
                        display(plot(pCPUtL2dt))
                    end
                end
            end
            if is_L2dtDKing
                isppp = 1
                # DKingL2dtv[:,iCase] = deepcopy(DKing[isppp][:,1])
                DKingL2dtv[:,iCase] = DKing[isppp][:,1] ./ fvLc0k1[isppp][:,1]
    
                if iCase == NCase
                    DKingL2dtv[:,NCase+1] = deepcopy(vhek1[isppp])
    
                    vvvv4 = DKingL2dtv[:,NCase+1] .< 0.3
                    DKingL2dtv1 = ((sum((DKingL2dtv[vvvv4,1:NCase]) .^2;dims=1)) .^0.5)[:]
                    DKingL1order = DKingL2dtv1[1] ./ (2^1) .^(0:NCase-1)
                    DKingL2order = DKingL2dtv1[1] ./ (2^2) .^(0:NCase-1)
        
                    xlabel = L"\Delta t"
                    ylabel = L"L_2^{\Delta t}"
                    label = L"Numerical"
                    pL2dtDKing = plot(1 ./ Nτ_fixvec[1:end], DKingL2dtv1,line=(2,:auto),label=label,
                                ylabel=ylabel,yscale=:log10,
                                xlabel=xlabel,xscale=:log10,legend=legendbR)
                    label = L"1^{nd} order"
                    pL2dtDKing = plot!(1 ./ Nτ_fixvec, DKingL1order,line=(2,:auto),label=label)
                    label = L"2^{nd} order"
                    pL2dtDKing = plot!(1 ./ Nτ_fixvec, DKingL2order,line=(2,:auto),label=label)
                    savefig(string(file_fig_file,"_L2dtDKing",NCase,".png"))
                    display(pL2dtDKing)
                    if is_L2dtKing
                        pL2dtL2dtDKing = plot(pL2dt,pL2dtDKing,layout=(2,1))
                        display(pL2dtL2dtDKing)
                    end
                end
            end
        end

        
        # @show isp, δtf[isp][1:3,1]
        # @show 41, a[2:6]
        if  MultiType == :dt && is_plot_King_convergence
            if is_plot_dfln_thoery && iter == 2
                DKingk = (KvLc0k1 - fvLc0k)
                isp = 1
                iFv = 2
                cfF3 = (4 / pi^2 * nak1[isp] / vathk1[isp]^3 * nak1[iFv] / vathk1[iFv]^3)
                
                ivhn = 57
    
                is_dtf_theory_A = false
                vha5 = vhek1[isp] .< vh5
                if is_dtf_theory_A
                    mM = ma[isp] / ma[iFv]
                    vabth = vathk1[isp] / vathk1[iFv]
                    dtfLn = dtfMab(mM,vabth,vhkk1[isp][nvlevel0[isp][nvlevele0[isp]]])
                    dfth = cfF3 * dtfLn[vha5]
                    af = (Dfing[isp][vha5,1] ./ dfth)
                    aK = (DKingk[isp][vha5,1] ./ dfth)
                    # @show af[1:3]
                    # @show aK[1:3]
    
                    xlabel = string("v̂")
                    label = string("a0,fk1-fk,nnv=",nnv[isp])
                    pafa = plot(vhek1[isp][vha5][2:end],af[2:end] / af[3] .- 1,label=label,xlabel=xlabel)
                    label = string("a0,Kk1-fk,Δt=",1 / Nτ_fix)
                    paKa = plot!(vhek1[isp][vha5][2:end],aK[2:end] / aK[3] .- 1,label=label)
                    ygfdrgf
                else
                    dfth = dtfvLc0k[isp][vha5,1]
                    af = (Dfing[isp][vha5,1] / dtk ./ dfth .- 1)
                    aK = (DKingk[isp][vha5,1] / dtk ./ dfth .- 1)
                    # @show dtk, dfth[1:3]
                    # @show Dfing[isp][1:3,1] / dtk
                    # @show DKingk[isp][1:3,1] / dtk
                    # @show af[1:3]
                    # @show aK[1:3]
    
                    xlabel = string("v̂")
                    ylabel = string("(fk1-fk)/dt / dtfk -1")
                    if spices[1] == spices[2]
                        label = string("a",iter,",nnv=",nnv[isp])
                    else
                        label = string(spices[isp],iter,",nnv=",nnv[isp])
                    end
                    pafa = plot(vhek1[isp][vha5][2:end],af[2:end],label=label,ylabel=ylabel)
    
                    ylabel = string("(Kk1-fk)/dt / dtfk -1")
                    if spices[1] == spices[2]
                        label = string("a",iter,",nnv=",nnv[isp])
                    else
                        label = string(spices[isp],iter,",nnv=",nnv[isp])
                    end
                    paKa = plot(vhek1[isp][vha5][2:end],aK[2:end],label=label,ylabel=ylabel,xlabel=xlabel)
                end
                
                vha5 = vhek1[iFv] .< vh5
                if is_dtf_theory_A
                    mM = ma[iFv] / ma[isp]
                    vabth = vathk1[iFv] / vathk1[isp]
                    dtfLn = dtfMab(mM,vabth,vhkk1[iFv][nvlevel0[iFv][nvlevele0[iFv]]])
                    dfth = cfF3 * dtfLn[vha5]
                    af = (Dfing[iFv][vha5,1] ./ dfth)
                    aK = (DKingk[iFv][vha5,1] ./ dfth)
                    label = string("b",iter,",fk1-fk,nnv=",nnv[iFv])
                    pafb = plot(vhek1[iFv][vha5][2:end],af[2:end] / af[3] .- 1,label=label,xlabel=xlabel)
                    label = string("b0,Kk1-fk,Δt=",1 / Nτ_fix)
                    paKb = plot!(vhek1[iFv][vha5][2:end],aK[2:end] / aK[3] .- 1,label=label)
                else
                    dfth = dtfvLc0k[iFv][vha5,1]
                    af = (Dfing[iFv][vha5,1] / dtk ./ dfth .- 1)
                    aK = (DKingk[iFv][vha5,1] / dtk ./ dfth .- 1)
                    # @show dtk, dfth[1:3]
                    # @show Dfing[iFv][1:3,1] / dtk
                    # @show DKingk[iFv][1:3,1] / dtk
                    # @show af[1:3]
                    # @show aK[1:3]
    
                    if spices[1] == spices[2]
                        label = string("b",iter,",dt=",1/Nτ_fix)
                    else
                        label = string(spices[iFv],iter,",dt=",1/Nτ_fix)
                    end
                    pafb = plot(vhek1[iFv][vha5][2:end],af[2:end],label=label)
                    if spices[1] == spices[2]
                        label = string("b",iter,",dt=",1/Nτ_fix)
                    else
                        label = string(spices[iFv],iter,",dt=",1/Nτ_fix)
                    end
                    paKb = plot(vhek1[iFv][vha5][2:end],aK[2:end],label=label,xlabel=xlabel)
                end
                display(plot(pafa,pafb,paKa,paKb,layout=(2,2)))
            end
            
            if iter < 0
                DKing = fvLc0k1 - KvLc0k1
                Dfing = (fvLc0k1 - fvLc0k)
    
                isp = 1
                label = string("fl0ak,Δt=",1/Nτ_fix)
                pKinga = plot(vhek1[isp],fvLc0k[isp][:,1],line=(2,:auto),label=label)
                label = string("fl0ak1,t=",fmtf6(tk))
                pKinga = plot!(vhek1[isp],fvLc0k1[isp][:,1],line=(2,:auto),label=label)
                label = string("Kl0ak1,iter=",iter)
                pKinga = plot!(vhek1[isp],KvLc0k1[isp][:,1],line=(2,:auto),label=label)
    
                isp = 2
                label = string("fl0bk,vh=",fmtf4(vhek1[isp][ivh]))
                pKingb = plot(vhek1[isp],fvLc0k[isp][:,1],line=(2,:auto),label=label)
                label = string("fl0bk,vh=",fmtf4(vhek1[isp][ivh]*vathk1[isp]))
                pKingb = plot!(vhek1[isp],fvLc0k1[isp][:,1],line=(2,:auto),label=label)
                label = "Kl0bk1"
                pKingb = plot!(vhek1[isp],KvLc0k1[isp][:,1],line=(2,:auto),label=label)
                pKingab = plot(pKinga,pKingb,layout=(1,2))
                if is_display_King
                    display(plot(pKinga,pKingb,layout=(1,2)))
                end
            end
            if iter > 0
                DKing = fvLc0k1 - KvLc0k1
                Dfing = (fvLc0k1 - fvLc0k)
    
                isp = 1
                xlabel = L"\hat{v}"
                ylabel = L"f_l^0"
                label = L"f_l^0(t_k)"
                pKinga = plot(vhek1[isp],fvLc0k[isp][:,1],line=(2,:auto),label=label,xlabel=xlabel,ylabel=ylabel)
                label = L"f_l^{0*}(t_{k+1})"
                pKinga = plot!(vhek1[isp],fvLc0k1[isp][:,1],line=(2,:auto),label=label)
                label = L"f_l^0(t_{k+1})"
                pKinga = plot!(vhek1[isp],KvLc0k1[isp][:,1],line=(2,:auto),label=label)
    
                isp = 2
                label = L"F_l^0(t_k)"
                pKingb = plot(vhek1[isp],fvLc0k[isp][:,1],line=(2,:auto),label=label,xlabel=xlabel)
                label = L"F_l^{0*}(t_{k+1})"
                pKingb = plot!(vhek1[isp],fvLc0k1[isp][:,1],line=(2,:auto),label=label)
                label = L"F_l^0(t_{k+1})"
                pKingb = plot!(vhek1[isp],KvLc0k1[isp][:,1],line=(2,:auto),label=label)
                pKingab = plot(pKinga,pKingb,layout=(1,2))
                if is_display_King
                    display(plot(pKinga,pKingb,layout=(1,2)))
                end
            end
    
            if Cnm_name == :C0
                title = string("C0,Δtₖ=",1/Nτ_fix,",iter=",iter)
            elseif Cnm_name == :C2
                title = string("C2,Δtₖ=",1/Nτ_fix,",iter=",iter)
            end
    
            if 1 == 1
        
                isp = 1
                ylabel = L"\Delta K_l^0"
                label = L"l=0"
                label = string("a, Δt=",1 / Nτ_fix)
                xlabel = L"\hat{v}"
                pDfinga = plot(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel,title=title)
                isp = 2
                # ylabel = "ΔF̂ₗ⁰"
                # label = string(spices[isp])
                label = string("b, Δt=",1 / Nτ_fix)
                pDfingb = plot!(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel,yscale=:log10,xlabel=xlabel)
                # display(plot(pDfinga,pDfingb,layout=(2,1)))
            
                isp = 1
                ylabel = L"\Delta f_l^0"
                label = L"l=0"
                # label = string("a, Δt=",1 / Nτ_fix)
                label = string("a")
                xlabel = L"\hat{v}"
                pDKinga = plot(vhek1[isp], abs.(Dfing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel)
                isp = 2
                # ylabel = L"\Delta F_l^0"
                # label = string(spices[isp])
                # label = string("b, Δt=",1 / Nτ_fix)
                label = string("b")
                pDKingb = plot!(vhek1[isp], abs.(Dfing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel,yscale=:log10,xlabel=xlabel)
                # display(plot(pDKinga,pDKingb,layout=(2,1)))
            
                # RDKing = (KvL0k1 - fvL0k1) ./ Dfing
                isp = 1
                ylabel = L"\delta f_l^0"
                if Cnm_name == :C0
                    DKing[isp] ./= Dfing[isp]
                elseif Cnm_name == :C2
                    DKing[isp] ./= fvLc0k1[isp] * (2^(iCase-1))
                end
                # label = string(spices[isp])
                # label = string("a")
                label = string("a, Δt=",1 / Nτ_fix)
                pRDKinga = plot(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel,yscale=:log10)
                isp = 2
                # ylabel = L"\delta F_l^0"
                if Cnm_name == :C0
                    DKing[isp] ./= Dfing[isp]
                elseif Cnm_name == :C2
                    DKing[isp] ./= fvLc0k1[isp] * (2^(iCase-1))
                end
                # label = string(spices[isp])
                label = string("b, Δt=",1 / Nτ_fix)
                pRDKingb = plot!(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(3,:auto),
                                ylabel=ylabel,yscale=:log10,xlabel=xlabel)
        
        
                isp = 1
                iFv = 2
    
                if is_save_nvG_NCase
                    RDKingvec[1:nvG[isp],iCase,isp] = deepcopy(abs.(DKing[isp][:,1]).+epsT)
                    RDKingvec[1:nvG[iFv],iCase,iFv] = deepcopy(abs.(DKing[iFv][:,1]).+epsT)
                else
                    RDKingvec[1,iCase,isp] = deepcopy(abs.(DKing[isp][ivh,1]).+epsT)
                    RDKingvec[1,iCase,iFv] = deepcopy(abs.(DKing[iFv][ivh,1]).+epsT)
                end
                if is_display_King
                    display(plot(pDKinga,pDfinga,pRDKingb,layout=(3,1)))
                end
                plot(pDKinga,pDfinga,pRDKingb,layout=(3,1))
                title = string("Δtₖ=",1/Nτ_fix,"iter=",iter)
                if Cnm_name == :C0
                    savefig(string(file_fig_file,"_RDKingC0",title,".png"))
                elseif Cnm_name == :C2
                    savefig(string(file_fig_file,"_RDKingC2",title,".png"))
                end
            end
    
            if is_save_nvG_NCase
                if Lmode == :L0
                    RDKing00vec[iCase,isp] = abs.(RDKingvec[ivh,iCase,isp])
                    RDKing00vec[iCase,iFv] = abs.(RDKingvec[ivh,iCase,iFv])
                elseif Lmode == :L1
                    if is_weight_DKing
                        vec05 = vhek1[isp] .< vh5
                        RDKing00vec[iCase,isp] = sum(abs.(RDKingvec[1:nvG[isp],iCase,isp][vec05]) .* KvL0k1[isp][vec05,1])[1] / sum(vec05)
                        vec05 = vhek1[iFv] .< vh5
                        RDKing00vec[iCase,iFv] = sum(abs.(RDKingvec[1:nvG[iFv],iCase,iFv][vec05]) .* KvL0k1[iFv][vec05,1])[1] / sum(vec05)   
                    else
                        vec05 = vhek1[isp] .< vh5
                        RDKing00vec[iCase,isp] = sum(abs.(RDKingvec[1:nvG[isp],iCase,isp][vec05]))[1] / sum(vec05)
                        vec05 = vhek1[iFv] .< vh5
                        RDKing00vec[iCase,iFv] = sum(abs.(RDKingvec[1:nvG[iFv],iCase,iFv][vec05]))[1] / sum(vec05)
                    end
                else
                    dfbvbfdff
                end
            else
                if Lmode == :L0
                    RDKing00vec[iCase,isp] = abs.(RDKingvec[1,iCase,isp])
                    RDKing00vec[iCase,iFv] = abs.(RDKingvec[1,iCase,iFv])
                elseif Lmode == :L1
                    if is_weight_DKing
                        vec05 = vhek1[isp] .< vh5
                        RDKing00vec[iCase,isp] = sum(abs.(RDKingvec[1,iCase,isp][vec05]) .* KvL0k1[isp][vec05,1])[1] / sum(vec05)
                        vec05 = vhek1[iFv] .< vh5
                        RDKing00vec[iCase,iFv] = sum(abs.(RDKingvec[1,iCase,iFv][vec05]) .* KvL0k1[iFv][vec05,1])[1] / sum(vec05)   
                    else
                        vec05 = vhek1[isp] .< vh5
                        RDKing00vec[iCase,isp] = sum(abs.(RDKingvec[1,iCase,isp][vec05]))[1] / sum(vec05)
                        vec05 = vhek1[iFv] .< vh5
                        RDKing00vec[iCase,iFv] = sum(abs.(RDKingvec[1,iCase,iFv][vec05]))[1] / sum(vec05)
                    end
                else
                    dfbvbfdff
                end
            end
    
            isp = 1
            iFv= 2
            wlinecl = 2
            if is_output_convergence
                
                wlinecl = 2
                if 1 == 1
                    if 2 == 1
                        DKing = fvL0k1 - KvL0k1
                        Dfing = (fvLc0k1 - fvLc0k)
                    else
                        KvLc0k1 = deepcopy(KvL0k1)
                        for isp in 1:2
                            KvLc0k1[isp] *= (nak1[isp] / vathk1[isp]^3)
                        end
                        DKing = fvLc0k1 - KvLc0k1
                        Dfing = (fvLc0k1 - fvLc0k)
                    end
            
                    isp = 1
                    ylabel = L"\Delta f_l^0"
                    label = L"l=0"
                    label = string(L"a, \Delta t=",1 / Nτ_fix)
                    label = L"a"
                    xlabel = L"\hat{v}"
                    pDfinga = plot(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(wlinecl,:auto),
                                    ylabel=ylabel)
                    isp = 2
                    # ylabel = "ΔF̂ₗ⁰"
                    # label = string(spices[isp])
                    if spices[1] == spices[2]
                        label = string(L"b, \Delta t=",fmtf2(1 / Nτ_fix))
                    else
                        label = string(spices[2],L"\Delta t=",fmtf2(1 / Nτ_fix))
                    end
                    pDfingb = plot!(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(wlinecl,:auto),
                                    ylabel=ylabel,yscale=:log10)
                    # display(plot(pDfinga,pDfingb,layout=(2,1)))
            
                    isp = 1
                    ylabel = L"\delta f_l^0"
                    if Cnm_name == :C0
                        DKing[isp] ./= Dfing[isp]
                    elseif Cnm_name == :C2
                        DKing[isp] ./= fvLc0k1[isp] * (2^(iCase-1))
                    end
                    # label = string(spices[isp])
                    # label = string(L"a, \Delta t=",1 / Nτ_fix)
                    if spices[1] == spices[2]
                        if Cnm_name == :C0
                            label = L"a,C0"
                        elseif Cnm_name == :C2
                            label = L"a,C2"
                        end
                    else
                        if Cnm_name == :C0
                            label = string(spices[1],L",C0")
                        elseif Cnm_name == :C2
                            label = string(spices[1],L",C2")
                        end
                    end
                    pRDKinga = plot(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(wlinecl,:auto),
                                    ylabel=ylabel,yscale=:log10,legend=legendbR)
                    isp = 2
                    if Cnm_name == :C0
                        DKing[isp] ./= Dfing[isp]
                    elseif Cnm_name == :C2
                        DKing[isp] ./= fvLc0k1[isp] * (2^(iCase-1))
                    end
                    if spices[1] == spices[2]
                        label = string(L"b, \Delta t=",fmtf2(1 / Nτ_fix))
                    else
                        label = string(spices[2],L",\Delta t=",fmtf2(1 / Nτ_fix))
                    end
                    pRDKingb = plot!(vhek1[isp], abs.(DKing[isp][:,1]).+epsT,label=label,line=(wlinecl,:auto),
                                    ylabel=ylabel,yscale=:log10,
                                    xlabel=xlabel)
                           
                end
    
                isp = 1
                iFv = 2
                if Cnm_name == :C0
                    title = string(L"C0,\hat{v}=", fmtf2(vhe[1][ivh]))
                elseif Cnm_name == :C2
                    title = string(L"C2,\hat{v}=", fmtf2(vhe[1][ivh]))
                end
                if length(Nτ_fixvec) ≥ 2
                    # DKing *= Nτ_fix
            
                    wlinecl = 2
            
                    ######################################################## Convergence
    
                    if is_plot_Convergence
                        methodvv = 1
                        if is_save_nvG_NCase
                            if methodvv == 2
                                # RDKing33vec[:,isp] = reverse(abs.(RDKingvec[ivh,:,isp]))
                                RDKing33vec[:,isp] = (abs.(RDKingvec[ivh,:,isp]))
                                orderavec = order_converg(RDKing33vec[:,isp])
                            
                                # RDKing33vec[:,iFv] = reverse(abs.(RDKingvec[ivh,:,iFv]))
                                RDKing33vec[:,iFv] = (abs.(RDKingvec[ivh,:,iFv]))
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            elseif methodvv == 1
                                RDKing33vec[:,isp] = RDKing00vec[:,isp]
                                RDKing33vec[:,iFv] = RDKing00vec[:,iFv]
        
                                orderavec = order_converg(RDKing33vec[:,isp])
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = L"\Delta f_l^0"
                            elseif methodvv == 3
                                RDKing33vec[:,isp] = abs.(RDKingvec[ivh,1:end,isp] .- RDKingvec[ivh,end,isp])
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[ivh,1:end,iFv] .- RDKingvec[ivh,end,iFv])
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            else
                                RDKing33vec[:,isp] = abs.(RDKingvec[ivh,1:end,isp] .- RDKingvec[ivh,end,isp]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[ivh,1:end,iFv] .- RDKingvec[ivh,end,iFv]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            end
                        else
                            if methodvv == 2
                                # RDKing33vec[:,isp] = reverse(abs.(RDKingvec[1,:,isp]))
                                RDKing33vec[:,isp] = (abs.(RDKingvec[1,:,isp]))
                                orderavec = order_converg(RDKing33vec[:,isp])
                            
                                # RDKing33vec[:,iFv] = reverse(abs.(RDKingvec[1,:,iFv]))
                                RDKing33vec[:,iFv] = (abs.(RDKingvec[1,:,iFv]))
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            elseif methodvv == 1
                                RDKing33vec[:,isp] = RDKing00vec[:,isp]
                                RDKing33vec[:,iFv] = RDKing00vec[:,iFv]
        
                                orderavec = order_converg(RDKing33vec[:,isp])
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = L"\Delta f_l^0"
                            elseif methodvv == 3
                                RDKing33vec[:,isp] = abs.(RDKingvec[1,1:end,isp] .- RDKingvec[1,end,isp])
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[1,1:end,iFv] .- RDKingvec[1,end,iFv])
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            else
                                RDKing33vec[:,isp] = abs.(RDKingvec[1,1:end,isp] .- RDKingvec[1,end,isp]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[1,1:end,iFv] .- RDKingvec[1,end,iFv]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            end
                        end
                        if is_display_King
                            display(plot(pDKinga,pDfinga,pRDKingb,layout=(3,1)))
                        end
                    
                        ordera = sum(orderavec) / length(orderavec)
                        orderb = sum(orderbvec) / length(orderbvec)
                    
                        RDKing33vec1 = deepcopy(RDKing33vec)
                        RDKing33vec2 = deepcopy(RDKing33vec)
                        # RDKing33vec3 = deepcopy(RDKing33vec)
                        # RDKing33vec4 = deepcopy(RDKing33vec)
                        for iii in 1:NCase-1
                            RDKing33vec1[iii,:] = RDKing33vec1[1,:] / (2^1) .^ (iii-1)
                            RDKing33vec2[iii,:] = RDKing33vec1[1,:] / (2^2) .^ (iii-1)
                            # RDKing33vec3[iii,:] = RDKing33vec1[1,:] / (2^3) .^ (iii-1)
                            # RDKing33vec4[iii,:] = RDKing33vec1[1,:] / (2^4) .^ (iii-1)
                        end
                        xlabel = L"\Delta t"
                        il1 = 1
                        il2 = 2
                        ic1 = 1
                        ic2type = 2
                        if ic2type == 1
                            ic2 = ic1 + 1
                        else
                            ic2 = ic1 + 6
                        end
                        vecdt = 1:length(Nτ_fixvec)-1
                        dtvec = 1 ./ Nτ_fixvec[vecdt]
                    
                        if methodvv == 1
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"a,\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = L"a,L1"
                                elseif  Lmode == :L2
                                    label = L"a,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[1],L",\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[1],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[1],L",L2")
                                end
                            end
                            # ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,linetypes[il1]),
                            #             ylabel=ylabel)
                            ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel,xscale=:log10)
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"b,\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = L"b,L1"
                                elseif  Lmode == :L2
                                    label = L"b,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[2],L",\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[2],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[2],L",L2")
                                end
                            end
                            # ppb = plot(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,linetypes[il2]),
                            ppb = plot(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel,xscale=:log10,legend=legendbR)
                        else
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"a,\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = L"a,L1"
                                elseif  Lmode == :L2
                                    label = L"a,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[1],L",\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[1],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[1],L",L2")
                                end
                            end
                            ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,:auto))
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"b,\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = L"b,L1"
                                elseif  Lmode == :L2
                                    label = L"b,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[2],L",\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[2],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[2],L",L2")
                                end
                            end
                            ppb = plot!(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel,xscale=:log10,legend=legendbR)
                        end
                        
                        # order
                        if methodvv ≠ 1
                            if 2 == 1
                                if ordera ≤ 2
                                    ic1 += 1
                                    if ic2type == 1
                                        ic2 = ic1 + 1
                                    else
                                        ic2 = ic1 + 6
                                    end
                                    il1 += 1
                                    il2 += 1
                                    label = L"a, 1^{th} order"
                                    plot!(dtvec, RDKing33vec1[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                    if 1 ≤ ordera
                                        label = L"a, 2^{nd} order"
                                        plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                    end
                                else
                                    if 2 ≤ ordera ≤ 3
                                        ic1 += 1
                                        if ic2type == 1
                                            ic2 = ic1 + 1
                                        else
                                            ic2 = ic1 + 6
                                        end
                                        il1 += 1
                                        il2 += 1
                                        label = L"a, 2^{nd} order"
                                        plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                        label = L"a, 3^{th} order"
                                        plot!(dtvec, RDKing33vec3[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                    else
                                        if 3 ≤ ordera ≤ 4
                                            ic1 += 1
                                            if ic2type == 1
                                                ic2 = ic1 + 1
                                            else
                                                ic2 = ic1 + 6
                                            end
                                            il1 += 1
                                            il2 += 1
                                            label = L"a, 3^{th} order"
                                            plot!(dtvec, RDKing33vec3[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                            label = L"a, 4^{th} order"
                                            plot!(dtvec, RDKing33vec4[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                        else
                                            @show ordera
                                            sdfgbn
                                        end
                                    end
                                end
                                
                                if orderb ≤ 2
                                    ic1 += 1
                                    if ic2type == 1
                                        ic2 = ic1 + 1
                                    else
                                        ic2 = ic1 + 6
                                    end
                                    il1 += 1
                                    il2 += 1
                                    label = L"b, 1^{th} order"
                                    ppa = plot!(dtvec, RDKing33vec1[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                                ylabel=ylabel,yscale=:log10,
                                                xlabel=xlabel)
                                    if 1 ≤ ordera
                                        label = L"b, 2^{nd} order"
                                        ppa = plot!(dtvec, RDKing33vec2[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                    ylabel=ylabel,yscale=:log10,
                                                    xlabel=xlabel)
                                    end
                                else
                                    if 2 ≤ ordera ≤ 3
                                        ic1 += 1
                                        if ic2type == 1
                                            ic2 = ic1 + 1
                                        else
                                            ic2 = ic1 + 6
                                        end
                                        il1 += 1
                                        il2 += 1
                                        label = L"b, 2^{nd} order"
                                        ppa = plot!(dtvec, RDKing33vec2[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                                    ylabel=ylabel,yscale=:log10,
                                                    xlabel=xlabel)
                                        # ic2 += 1
                                        label = L"b, 3^{th} order"
                                        ppa = plot!(dtvec, RDKing33vec3[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                    ylabel=ylabel,yscale=:log10,
                                                    xlabel=xlabel)
                                    else
                                        if 3 ≤ ordera ≤ 4
                                            ic1 += 1
                                            if ic2type == 1
                                                ic2 = ic1 + 1
                                            else
                                                ic2 = ic1 + 6
                                            end
                                            il1 += 1
                                            il2 += 1
                                            label = L"b, 3^{th} order"
                                            ppa = plot!(dtvec, RDKing33vec3[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                                        ylabel=ylabel,yscale=:log10,
                                                        xlabel=xlabel)
                                            1
                                            label = L"b, 4^{th} order"
                                            ppa = plot!(dtvec, RDKing33vec4[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                        ylabel=ylabel,yscale=:log10,
                                                        xlabel=xlabel)
                                        else
                                            sdfgbn
                                        end
                                    end
                                end
                            else
                                # label = L"1^{st} order"
                                # plot!(dtvec, RDKing33vec1[vecdt,1],label=label,line=(wlinecl,:auto))
                                label = L"2^{nd} order"
                                plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,:auto))
                            end
                        end
                        # # display(plot(ppa))
                        # # savefig(string(file_fig_file,"_RDT2th.png"))
                        # display(plot(ppa))
                        # title = string("nτ",nτ)
                        # if Cnm_name == :C0
                        #     savefig(string(file_fig_file,"_RDKing2C0",title,".png"))
                        # elseif Cnm_name == :C2
                        #     savefig(string(file_fig_file,"_RDKing2C0",title,".png"))
                        # end
            
                        # display(plot(ppa))
                        # savefig(string(file_fig_file,"_RDKing2th.png"))
                        # display(plot(ppa))
                        if Cnm_name == :C0
                            title = string(L"C0",NCase)
                        elseif Cnm_name == :C2
                            title = string(L"C2",NCase)
                        end
                        savefig(string(file_fig_file,"_RDKing123th",title,".png"))
                    else
                        methodvv = 2
                        if is_save_nvG_NCase
                            if methodvv == 1
                                # RDKing33vec[:,isp] = reverse(abs.(RDKingvec[ivh,:,isp]))
                                RDKing33vec[:,isp] = (abs.(RDKingvec[ivh,:,isp]))
                                orderavec = order_converg(RDKing33vec[:,isp])
                            
                                # RDKing33vec[:,iFv] = reverse(abs.(RDKingvec[ivh,:,iFv]))
                                RDKing33vec[:,iFv] = (abs.(RDKingvec[ivh,:,iFv]))
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            elseif methodvv == 2
                                RDKing33vec[:,isp] = RDKing00vec[:,isp]
                                RDKing33vec[:,iFv] = RDKing00vec[:,iFv]
        
                                orderavec = order_converg(RDKing33vec[:,isp])
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = L"\Delta f_l^0"
                            elseif methodvv == 3
                                RDKing33vec[:,isp] = abs.(RDKingvec[ivh,1:end,isp] .- RDKingvec[ivh,end,isp])
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[ivh,1:end,iFv] .- RDKingvec[ivh,end,iFv])
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            else
                                RDKing33vec[:,isp] = abs.(RDKingvec[ivh,1:end,isp] .- RDKingvec[ivh,end,isp]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[ivh,1:end,iFv] .- RDKingvec[ivh,end,iFv]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            end
                        else
                            if methodvv == 1
                                # RDKing33vec[:,isp] = reverse(abs.(RDKingvec[1,:,isp]))
                                RDKing33vec[:,isp] = (abs.(RDKingvec[1,:,isp]))
                                orderavec = order_converg(RDKing33vec[:,isp])
                            
                                # RDKing33vec[:,iFv] = reverse(abs.(RDKingvec[1,:,iFv]))
                                RDKing33vec[:,iFv] = (abs.(RDKingvec[1,:,iFv]))
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            elseif methodvv == 2
                                RDKing33vec[:,isp] = RDKing00vec[:,isp]
                                RDKing33vec[:,iFv] = RDKing00vec[:,iFv]
        
                                orderavec = order_converg(RDKing33vec[:,isp])
                                orderbvec = order_converg(RDKing33vec[:,iFv])
                                ylabel = L"\Delta f_l^0"
                            elseif methodvv == 3
                                RDKing33vec[:,isp] = abs.(RDKingvec[1,1:end,isp] .- RDKingvec[1,end,isp])
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[1,1:end,iFv] .- RDKingvec[1,end,iFv])
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            else
                                RDKing33vec[:,isp] = abs.(RDKingvec[1,1:end,isp] .- RDKingvec[1,end,isp]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderavec = order_converg(RDKing33vec[1:end-1,isp])
                            
                                RDKing33vec[:,iFv] = abs.(RDKingvec[1,1:end,iFv] .- RDKingvec[1,end,iFv]) .* Nτ_fixvec / Nτ_fixvec[1]
                                orderbvec = order_converg(RDKing33vec[1:end-1,iFv])
                                ylabel = string(L"\Delta f_l^0","v̂=(",fmtf2(vhe[isp][ivh]),")")
                            end
                        end
                    
                        ordera = sum(orderavec) / length(orderavec)
                        orderb = sum(orderbvec) / length(orderbvec)
                    
                        RDKing33vec1 = deepcopy(RDKing33vec)
                        RDKing33vec2 = deepcopy(RDKing33vec)
                        # RDKing33vec3 = deepcopy(RDKing33vec)
                        # RDKing33vec4 = deepcopy(RDKing33vec)
                        for iii in 1:NCase-1
                            RDKing33vec1[iii,:] = RDKing33vec1[1,:] / (2^1) .^ (iii-1)
                            RDKing33vec2[iii,:] = RDKing33vec1[1,:] / (2^2) .^ (iii-1)
                            # RDKing33vec3[iii,:] = RDKing33vec1[1,:] / (2^3) .^ (iii-1)
                            # RDKing33vec4[iii,:] = RDKing33vec1[1,:] / (2^4) .^ (iii-1)
                        end
                        xlabel = L"\Delta t"
                        il1 = 1
                        il2 = 2
                        ic1 = 1
                        ic2type = 2
                        if ic2type == 1
                            ic2 = ic1 + 1
                        else
                            ic2 = ic1 + 6
                        end
                        vecdt = 1:length(Nτ_fixvec)-1
                        dtvec = 1 ./ Nτ_fixvec[vecdt]
                    
                        if methodvv == 1
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"a,\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = L"a,L1"
                                elseif  Lmode == :L2
                                    label = L"a,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[1],L",\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[1],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[1],L",L2")
                                end
                            end
                            # ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,linetypes[il1]),
                            #             ylabel=ylabel)
                            ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel)
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"b,\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = L"b,L1"
                                elseif  Lmode == :L2
                                    label = L"b,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[2],L",\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[2],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[2],L",L2")
                                end
                            end
                            # ppb = plot(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,linetypes[il2]),
                            ppb = plot(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel,
                                        xlabel=xlabel,xscale=:log10,legend=legendbR)
                        else
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"a,\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = L"a,L1"
                                elseif  Lmode == :L2
                                    label = L"a,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[1],L",\hat{v}=", fmtf2(vhek1[1][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[1],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[1],L",L2")
                                end
                            end
                            ppa = plot(dtvec, RDKing33vec[vecdt,1],label=label,line=(wlinecl,:auto))
                            if spices[1] == spices[2]
                                if Lmode == :L0
                                    label = string(L"b,\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = L"b,L1"
                                elseif  Lmode == :L2
                                    label = L"b,L2"
                                end
                            else
                                if Lmode == :L0
                                    label = string(spices[2],L",\hat{v}=", fmtf2(vhek1[2][ivh]))
                                elseif  Lmode == :L1
                                    label = string(spices[2],L",L1")
                                elseif  Lmode == :L2
                                    label = string(spices[2],L",L2")
                                end
                            end
                            ppb = plot!(dtvec, RDKing33vec[vecdt,2],label=label,line=(wlinecl,:auto),
                                        ylabel=ylabel,yscale=:log10,
                                        xlabel=xlabel,xscale=:log10,legend=legendbR)
                        end
                        
                        # order
                        if 2 == 1
                            if ordera ≤ 2
                                ic1 += 1
                                if ic2type == 1
                                    ic2 = ic1 + 1
                                else
                                    ic2 = ic1 + 6
                                end
                                il1 += 1
                                il2 += 1
                                label = L"a, 1^{th} order"
                                plot!(dtvec, RDKing33vec1[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                if 1 ≤ ordera
                                    label = L"a, 2^{nd} order"
                                    plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                end
                            else
                                if 2 ≤ ordera ≤ 3
                                    ic1 += 1
                                    if ic2type == 1
                                        ic2 = ic1 + 1
                                    else
                                        ic2 = ic1 + 6
                                    end
                                    il1 += 1
                                    il2 += 1
                                    label = L"a, 2^{nd} order"
                                    plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                    label = L"a, 3^{th} order"
                                    plot!(dtvec, RDKing33vec3[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                else
                                    if 3 ≤ ordera ≤ 4
                                        ic1 += 1
                                        if ic2type == 1
                                            ic2 = ic1 + 1
                                        else
                                            ic2 = ic1 + 6
                                        end
                                        il1 += 1
                                        il2 += 1
                                        label = L"a, 3^{th} order"
                                        plot!(dtvec, RDKing33vec3[vecdt,1],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]))
                                        label = L"a, 4^{th} order"
                                        plot!(dtvec, RDKing33vec4[vecdt,1],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]))
                                    else
                                        @show ordera
                                        sdfgbn
                                    end
                                end
                            end
                            
                            if orderb ≤ 2
                                ic1 += 1
                                if ic2type == 1
                                    ic2 = ic1 + 1
                                else
                                    ic2 = ic1 + 6
                                end
                                il1 += 1
                                il2 += 1
                                label = L"b, 1^{th} order"
                                ppa = plot!(dtvec, RDKing33vec1[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                            ylabel=ylabel,yscale=:log10,
                                            xlabel=xlabel)
                                if 1 ≤ ordera
                                    label = L"b, 2^{nd} order"
                                    ppa = plot!(dtvec, RDKing33vec2[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                ylabel=ylabel,yscale=:log10,
                                                xlabel=xlabel)
                                end
                            else
                                if 2 ≤ ordera ≤ 3
                                    ic1 += 1
                                    if ic2type == 1
                                        ic2 = ic1 + 1
                                    else
                                        ic2 = ic1 + 6
                                    end
                                    il1 += 1
                                    il2 += 1
                                    label = L"b, 2^{nd} order"
                                    ppa = plot!(dtvec, RDKing33vec2[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                                ylabel=ylabel,yscale=:log10,
                                                xlabel=xlabel)
                                    # ic2 += 1
                                    label = L"b, 3^{th} order"
                                    ppa = plot!(dtvec, RDKing33vec3[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                ylabel=ylabel,yscale=:log10,
                                                xlabel=xlabel)
                                else
                                    if 3 ≤ ordera ≤ 4
                                        ic1 += 1
                                        if ic2type == 1
                                            ic2 = ic1 + 1
                                        else
                                            ic2 = ic1 + 6
                                        end
                                        il1 += 1
                                        il2 += 1
                                        label = L"b, 3^{th} order"
                                        ppa = plot!(dtvec, RDKing33vec3[vecdt,2],label=label,line=(wlinecl,linetypes[il1],linecolors[ic1]),
                                                    ylabel=ylabel,yscale=:log10,
                                                    xlabel=xlabel)
                                        1
                                        label = L"b, 4^{th} order"
                                        ppa = plot!(dtvec, RDKing33vec4[vecdt,2],label=label,line=(wlinecl,linetypes[il2],linecolors[ic2]),
                                                    ylabel=ylabel,yscale=:log10,
                                                    xlabel=xlabel)
                                    else
                                        sdfgbn
                                    end
                                end
                            end
                        else
                            # label = L"1^{st} order"
                            # plot!(dtvec, RDKing33vec1[vecdt,1],label=label,line=(wlinecl,:auto))
                            label = L"2^{nd} order"
                            plot!(dtvec, RDKing33vec2[vecdt,1],label=label,line=(wlinecl,:auto))
                        end
                        # # display(plot(ppa))
                        # # savefig(string(file_fig_file,"_RDT2th.png"))
                        # display(plot(ppa))
                        # if Cnm_name == :C0
                        #     title = string("C0nτ",nτ)
                        # elseif Cnm_name == :C2
                        #     title = string("C2nτ",nτ)
                        # end
                        # savefig(string(file_fig_file,"_RDKing2",title,".png"))
                    end
                end
    
                if methodvv == 1
    
                    if is_plot_Convergence
                        if Cnm_name == :C0
                            ppab = plot(ppa,ppb,layout=(1,2))
                            display(plot(pRDKingb,ppab,layout=(2,1)))
                            plot(pRDKingb,ppab,layout=(2,1))
                            savefig(string(file_fig_file,"_RDKingC4C0.png"))
                        elseif Cnm_name == :C2
                            if is_weight_DKing
                                display(plot(pKingb,pRDKingb,ppa,ppb,layout=(2,2)))
                                plot(pKingb,pRDKingb,ppa,ppb,layout=(2,2))
                                savefig(string(file_fig_file,"_RDKingC4WC2.png"))
                            else
                                display(plot(pKingb,pRDKingb,ppa,ppb,layout=(2,2)))
                                plot(pKingb,pRDKingb,ppa,ppb,layout=(2,2))
                                savefig(string(file_fig_file,"_RDKingC4C2.png"))
                            end
                        end
                    else
                        ppab = plot(ppa,ppb,layout=(1,2))
                        display(plot(pRDKingb,ppab,layout=(2,1)))
                        plot(pRDKingb,ppab,layout=(2,1))
                        if Cnm_name == :C0
                            savefig(string(file_fig_file,"_RDKing3C0.png"))
                        elseif Cnm_name == :C2
                            if is_weight_DKing
                                savefig(string(file_fig_file,"_RDKing3WC2.png"))
                            else
                                savefig(string(file_fig_file,"_RDKing3C2.png"))
                            end
                        end
                    end
                else
                    # display(plot(pRDKingb,ppa,layout=(2,1)))
                    # plot(pRDKingb,ppa,layout=(2,1))
                    # if Cnm_name == :C0
                    #     savefig(string(file_fig_file,"_RDKing2C0.png"))
                    # elseif Cnm_name == :C2
                    #     savefig(string(file_fig_file,"_RDKing2C2.png"))
                    # end
    
                    # display(plot(pKingab,pRDKingb,ppa,layout=(3,1)))
                    # plot(pKingab,pRDKingb,ppa,layout=(3,1))
                    if Cnm_name == :C0
                        savefig(string(file_fig_file,"_RDKing3C0.png"))
                    elseif Cnm_name == :C2
                        if is_weight_DKing
                            savefig(string(file_fig_file,"_RDKing3WC2.png"))
                        else
                            savefig(string(file_fig_file,"_RDKing3C2.png"))
                        end
                    end
    
                    display(plot(pKinga,pKingb,pRDKingb,ppa,layout=(2,2)))
                    plot(pKinga,pKingb,pRDKingb,ppa,layout=(2,2))
                    title = string("Δtₖ=",1/Nτ_fix)
                    if Cnm_name == :C0
                        savefig(string(file_fig_file,"_RDKing4C0.png"))
                    elseif Cnm_name == :C2
                        if is_weight_DKing
                            savefig(string(file_fig_file,"_RDKing4WC2.png"))
                        else
                            savefig(string(file_fig_file,"_RDKing4C2.png"))
                        end
                    end
                end
            end
        end
    end
    fvL0k1[:] = deepcopy(KvL0k1)
end

"""
  The `iᵗʰ` iteration of with Euler method or Trapezoidal method: 

  Inputs:

  Outputs:
    fvLck1integral0!(fvLc0k1,fvLc0k,dtfvLc0k1,nsk1,dtk)
"""

# [], alg_embedded::Symbol ∈ [:ExEuler, :ImEuler, :Trapz]
function fvLck1integral0!(fvLc0k1::Vector{Matrix{T}},fvLc0k::Vector{Matrix{T}},
    dtfvLc0k1::Vector{Matrix{T}},nsk1::Int64,dtk::T) where{T}
    
    for isp in 1:nsk1
        fvLc0k1[isp] = fvLc0k[isp]  + dtk * dtfvLc0k1[isp]
    end
end



