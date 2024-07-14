
"""
  The ODE form of the Fokker-Planck collision equations.
  
  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

  Inputs:
    orders=order_dvδtf
    is_δtfvLaa = 0          # [-1,    0,     1   ]
                            # [dtfaa, dtfab, dtfa]

  Outputs:
    FP0D2VcExplicitRKEuler_fM!(fvL0c, pst0)

"""

function FP0D2VcExplicitRKEuler_fM!(fvL0c::AbstractArray{T,N}, ps::Dict{String,Any}, Nstep::Int64;
    is_optimδtfvL::Bool=false,is_corrections::Vector{Bool}=[true,false,false],NL_solve::Symbol=:NLsolve,
    residualMethod_FP0D::Int64=1,is_δtfvLaa::Int=1,is_normδtf::Bool=false,is_boundaryv0::Bool=false,
    is_normal::Bool=true,restartfit::Vector{Int}=[0,0,100],maxIterTR::Int64=1000,maxIterKing::Int64=1000,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::T=1e-18,f_tol::T=1e-18,g_tol::T=1e-18,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_full_fvL::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,p_noise_rel::T=0e-3,p_noise_abs::T=0e-15,
    orders::Int64=2,is_boundv0::Vector{Bool}=[true,false,false], Nsmooth::Int=3, 
    order_smooth::Int64=3,order_smooth_itp::Int64=1,order_nvc_itp::Int64=4,
    abstol_Rdy::AbstractVector{T}=[0.45,0.45,0.45],k_δtf::Int64=2,Nitp::Int64=10,
    nvc0_limit::Int64=4,L1nvc_limit::Int64=3,order_RK::Int64=4) where{T,N}

    for k in 1:Nstep
        FP0D2VcExplicitRKEuler_fM!(fvL0c, ps;is_optimδtfvL=is_optimδtfvL,
                        is_corrections=is_corrections,NL_solve=NL_solve,
                        residualMethod_FP0D=residualMethod_FP0D,is_δtfvLaa=is_δtfvLaa,
                        is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0,
                        is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,maxIterKing=maxIterKing,
                        autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                        rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM,is_full_fvL=is_full_fvL,
                        optimizer=optimizer,factor=factor,is_Jacobian=is_Jacobian,
                        p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs,
                        orders=orders,is_boundv0=is_boundv0,Nsmooth=Nsmooth, 
                        order_smooth=order_smooth,order_smooth_itp=order_smooth_itp,order_nvc_itp=order_nvc_itp,
                        abstol_Rdy=abstol_Rdy,k_δtf=k_δtf,Nitp=Nitp,
                        nvc0_limit=nvc0_limit,L1nvc_limit=L1nvc_limit,order_RK=order_RK)
    end
end



function FP0D2VcExplicitRKEuler_fM!(fvLc0k::AbstractArray{T,N}, ps::Dict{String,Any};
    is_optimδtfvL::Bool=true,residualMethod_FP0D::Int64=1,
    is_corrections::Vector{Bool}=[true,false,false],NL_solve::Symbol=:NLsolve,
    is_δtfvLaa::Int=1,is_normδtf::Bool=false,is_boundaryv0::Bool=false,
    is_normal::Bool=true,restartfit::Vector{Int}=[0,0,100],maxIterTR::Int64=1000,maxIterKing::Int64=1000,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::T=1e-18,f_tol::T=1e-18,g_tol::T=1e-18,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_full_fvL::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,p_noise_rel::T=0e-3,p_noise_abs::T=0e-15,
    orders::Int64=2,is_boundv0::Vector{Bool}=[true,false,false], Nsmooth::Int=3, 
    order_smooth::Int64=3,order_smooth_itp::Int64=1,order_nvc_itp::Int64=4,
    abstol_Rdy::AbstractVector{T}=[0.45,0.45,0.45],k_δtf::Int64=2,Nitp::Int64=10,
    nvc0_limit::Int64=4,L1nvc_limit::Int64=3,order_RK::Int64=4) where{T,N}
    
    nstep = ps["nstep"]
    tk = ps["tk"]
    dtk = ps["dt"]
    Nt_save = ps["Nt_save"]
    count_save = ps["count_save"]
    nsk = ps["ns"]
    mak = ps["ma"]
    Zqk = ps["Zq"]
    nk = ps["nk"]
    # Ik = ps["Ik"]
    # Kk = ps["Kk"]
    vthk = ps["vthk"]
    nModk = ps["nModk"]
    naik = ps["nai"]
    uaik = ps["uai"]
    vthik = ps["vthi"]
    # nvc3k = ps["nvc3k"]         # zeros(Int64, 2(order_smooth + 1), LM1k1, nsk)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
    
    LMk = ps["LMk"]
    nnv = ps["nnv"]
    # vGmax = ps["vGmax"]
    nc0, nck, ocp = ps["nc0"], ps["nck"], ps["ocp"]
    nvlevel0, nvlevele0 = ps["nvlevel0"], ps["nvlevele0"]          # nvlevele = nvlevel0[nvlevele0]
    # vGk = ps["vGk"]
    # vGe = ps["vGe"]

    ########### meshgrids 
    nvG = 2^nnv + 1
    LM1k = maximum(LMk) + 1
    
    println()
    println("**************------------******************------------*********")
    @show nstep,tk,dtk,LMk

    # # # Updating the conservative momentums `n, I, K`
    nIKs = zeros(3,nsk)
    nIKsc!(nIKs,fvLc0k[:,1:2,:],ve,ma,vthk,ns;errnIKc=errnIKc)

    if is_corrections[1] == false
        nk1 = nIKs[1,:]
    else
        nk1 = nk
    end
    ρa = mak .* nk1
    Ik1 = nIKs[2,:]
    Kk1 = nIKs[3,:]
    vthk1 = (2/3 * (2 * Kk1 ./ ρa - (Ik1 ./ ρa).^2)).^0.5

    Rvthk1 = vthk1 ./ vthk
    # @show Rvthk1 .- 1
    # @show sum(Kk1) - Kab0

    # # Updating the amplitude function of normalized distribution functions `f̂ₗᵐ(v̂)`  at the `kᵗʰ` step.
    nModk1 = copy(nModk)
    naik1 = copy(naik)      # `n̂a = naᵢ / na`
    uaik1 = copy(uaik)      # `ûa = uaᵢ / vthk1`         `uhk1 = uaik * vthk / vthk1`
    vthik1 = copy(vthik)    # `v̂th = vathᵢ / vthk1`
    LMk1 = LMk
    LM1k1 = LM1k
    muk1, Mμk1, Munk1, Mun1k1, Mun2k1 = ps["muk"], ps["Mμk"], ps["Munk"], ps["Mun1k"], ps["Mun2k"]

    @time dtfvL0k1, fvLc0k1, err_dtnIKk1, nIKTh, naik1, uaik1, vthik1, 
            LMk1, LM1k1, nModk1  = dtfvLSplineab(fvLc0k, vGk, vGe, nvG, nc0, nck, 
            ocp, nvlevele0, nvlevel0, muk1, Mμk1, Munk1, Mun1k1, Mun2k1, CΓ, εᵣ, 
            mak, Zqk, nk1, vthk1, naik1, uaik1, vthik1, LMk1, LM1k1, nsk, nModk1;
            NL_solve=NL_solve, is_normal=is_normal, 
            restartfit=restartfit, maxIterTR=maxIterTR, maxIterKing=maxIterKing,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, 
            rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_full_fvL=is_full_fvL,
            optimizer=optimizer, factor=factor,is_Jacobian=is_Jacobian,
            is_δtfvLaa=is_δtfvLaa,is_normδtf=is_normδtf, is_boundaryv0=is_boundaryv0, 
            is_check_conservation_dtM=false,is_update_nuTi=true)
    errThMM = nIKTh[4, :]

    # Updating the amplitude of normalized distribution function at the new step with `Explicit RK` method
    fvLc0k[:,:,:] = fvLc0k1 + dtk * dtfvL0k1   # the effective amplitude at the new step for `moments` is `fₗᵐ(tk1)` owing to the meshgrids `ve`
        
    # # # # Updating the conservative momentums `n, I, K`
    # nIKsk1 = zeros(3,nsk)
    # # nIKsc!(nIKsk1,fvLc0k[:,1:2,:],vGe,ma,vthk1,ns;errnIKc=errnIKc)

    # Updating the parameters `ps` when `t = 0.0`
    ps["nk"] = nk1
    ps["Ik"] = Ik1
    ps["Kk"] = Kk1
    ps["vthk"] = vthk1
    ps["nModk"] = nModk1
    ps["nai"] = naik1
    ps["uai"] = uaik1
    ps["vthi"] = vthik1
    ps["LMk"] = LMk1
    ps["tk"] = tk + dtk
    ps["nstep"] = nstep + 1 
    if count_save == Nt_save
        ps["count_save"] = 1
        println(idnIK,fmtf4(ps["tk"]),", ",nk1[1],", ",nk1[2],", ",Ik1[1],", ",Ik1[2],", ",Kk1[1],", ",Kk1[2],", ",errThMM[1],", ",errThMM[2])
        isp = 1
        if nModk1[isp] == 1
            println(idnModa,fmtf4(ps["tk"]),", ",LMk1[isp],", ",nModk1[isp],", ",naik1[isp][1],", ",uaik1[isp][1],", ",vthik1[isp][1])
        elseif nModk1[isp] == 2
            println(idnModa,fmtf4(ps["tk"]),", ",LMk1[isp],", ",nModk1[isp],", ",naik1[isp][1],", ",naik1[isp][2],", ",uaik1[isp][1],", ",uaik1[isp][2],", ",vthik1[isp][1],", ",vthik1[isp][2])
        elseif nModk1[isp] == 3
            println(idnModa,fmtf4(ps["tk"]),", ",LMk1[isp],", ",nModk1[isp],", ",naik1[isp][1],", ",naik1[isp][2],", ",naik1[isp][3],
                                  ", ",uaik1[isp][1],", ",uaik1[isp][2],", ",uaik1[isp][3],", ",vthik1[isp][1],", ",vthik1[isp][2],", ",vthik1[isp][3])
        else
            erbgjj
        end
        isp = 2
        if nModk1[isp] == 1
            println(idnModb,fmtf4(ps["tk"]),", ",LMk1[isp],", ",nModk1[isp],", ",naik1[isp][1],", ",uaik1[isp][1],", ",vthik1[isp][1])
        elseif nModk1[isp] == 2
            println(idnModb,fmtf4(ps["tk"]),", ",LMk1[isp],", ",nModk1[isp],", ",naik1[isp][1],", ",naik1[isp][2],", ",uaik1[isp][1],", ",uaik1[isp][2],", ",vthik1[isp][1],", ",vthik1[isp][2])
        elseif nModk1[isp] == 3
            println(idnModb,fmtf4(ps["tk"]),", ",LMk1[isp],", ",nModk1[isp],", ",naik1[isp][1],", ",naik1[isp][2],", ",naik1[isp][3],
                                  ", ",uaik1[isp][1],", ",uaik1[isp][2],", ",uaik1[isp][3],", ",vthik1[isp][1],", ",vthik1[isp][2],", ",vthik1[isp][3])
        else
            erbgjj
        end
    else
        ps["count_save"] = count_save + 1
    end
end
