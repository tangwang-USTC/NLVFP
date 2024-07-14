
"""
  The ODE form of the Fokker-Planck collision equations.
  
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

  Level of the algorithm
    k: the time step level
    s: the stage level in a single time step
    i: the inner iteration level during a single stage
    
  Inputs:
    is_δtfvLaa = 0          # [-1,    0,     1   ]
                            # [dtfaa, dtfab, dtfa]

  Outputs:
    Mck1integral_ode!!(Rck, Mck, psvec, t)

"""

function Mck1integral_ode!!(Rck::AbstractArray{T,N}, Mck::AbstractArray{T,N}, ps::AbstractVector{Any}, t) where{T,N}
    
    nstep = ps[1]
    # tk = ps[2]
    dtk = t - ps[2]                # ps[3]
    # Nt_save = ps[4]
    # count_save = ps[5]
 
    nsk = ps[6]
    mak = ps[7]
    Zqk = ps[8]
    nk = ps[9]
    Ik = ps[10]
    Kk = ps[11]
    vthk = ps[12]
    
    nnv = ps[13]
    nc0, nck, ocp = ps[14], ps[15], ps[16]
    vGdom = ps[17]

    vhk = ps[18]
    vhe = ps[19]
    nvlevel0, nvlevele0 = ps[20], ps[21]          # nvlevele = nvlevel0[nvlevele0]

    nModk = ps[22]
    naik = ps[23]
    uaik = ps[24]
    vthik = ps[25]

    LMk = ps[26]
    muk, Mμk, Munk, Mun1k, Mun2k = ps[27], ps[28], ps[29], ps[30], ps[31]
    # vthk1 = ps[32]
    # w3k, err_dtnIK, DThk = ps[33], ps[34], ps[35]         # w3k = vₜₕ⁻¹∂ₜvₜₕ
    nMjMs = ps[36]
    RMcsk = ps[37]
    Mhck = ps[38]
    si = ps[39]                                  # "si", the iteration number of stage and inner iteration
    err_Rck = deepcopy(Rck[1:njMs, :, :])

    ########### meshgrids 
    nvG = 2 .^nnv .+ 1
    LM1k = maximum(LMk) + 1
    LM1k_copy = deepcopy(LM1k)
    
    println()
    println("**************------------******************------------*********")
    
    # Computing the values of `Rck` .= 0.0, `Mck` may be updated according to the M-theorems
    # parameter such as `vth` will be updated according to `Mck`
    dtk = Rck_update!(Rck, err_Rck, Mhck, vhk, nvG, ocp, vGdom, 
        nvlevele0, nvlevel0, LMk, LM1k, naik, uaik, vthik, CΓ, εᵣ, 
        mak, Zqk, nk, vthk, nsk, nModk, nMjMs, DThk, RMcsk, Ik, Kk, Mck, dtk;
        NL_solve=NL_solve, is_normal=is_normal,
        restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
        optimizer=optimizer, factor=factor, is_Jacobian=is_Jacobian,
        is_δtfvLaa=is_δtfvLaa, is_boundaryv0=is_boundaryv0,
        is_check_conservation_dtM=is_check_conservation_dtM, is_fit_f=is_fit_f,
        L_shape=L_shape, eps_fup=eps_fup, eps_flow=eps_flow, jMax=jMax_vGmax,
        maxiter_vGm=maxiter_vGm, is_out_shape=is_out_shape,
        abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels,
        Msj_adapt=Msj_adapt, orderVconst=orderVconst, vGm_limit=vGm_limit,
        is_corrections=is_corrections,is_vth_ode=is_vth_ode)
    @show dtk, fmtf2.(DThk)
    
    LM1k = maximum(LMk) + 1

    # Updating the parameters `ps`
    # ps[9] = nk1
    # ps[10] = Ik1
    # ps[11] = Kk1
    # ps[12] = vthk1
    
    ps[13] = nnv
    nc0, nck, ocp = ps[14], ps[15], ps[16]
    vGdom = ps[17]

    ps[18] = vhk
    ps[19] = vhe
    ps[20], ps[21] = nvlevel0, nvlevele0

    ps[22] = nModk
    ps[23] = naik
    ps[24] = uaik
    ps[25] = vthik

    if LM1k ≠ LM1k_copy
        ps[26] = LMk
        ps[27], ps[28], ps[29], ps[30], ps[31] = muk, Mμk, Munk, Mun1k, Mun2k
    end
    ps[36] = nMjMs
    # ps[37] = RMcsk
    # ps[38] = Mhck
    
    if t ≠ ps[2]
        printstyled("nstep = ", nstep, color=:green,"\n")
        @show si,fmtf2(t),fmtf2(dtk)

        ps[1] = nstep + 1 
        ps[2] = t
        ps[10] = deepcopy(Ik)
        ps[11] = deepcopy(Kk)
        ps[37] = deepcopy(RMcsk)
        ps[38] = deepcopy(Mhck)
        ps[39] = 0

        # if count_save == Nt_save
        if ps[5] == ps[4]
            data_nIKT_Mh_saving(ps;is_moments_out=is_moments_out,is_MjMs_max=is_MjMs_max)
            ps[5] = 1           
        else
            ps[5] += 1       # count_save += 1
        end
    else
        @show si,fmtf2(t),fmtf2(dtk)
        ps[39] += 1 
    end
end



