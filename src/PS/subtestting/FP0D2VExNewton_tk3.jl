
# tk3: Applying the Explicit Newton method to solve the ODE equations of `f̂vL(v̂)`
k += 1
if k ≥ 3
    tkk += dt
    fvL0k3 = fvLk2 + dt * dtfvL0k2

    # Updating the thermal velocity `vₜₕ = vthk3` in single step
    vthk3M = vthk2 .* Rvthk3M
    println("------------------------------------------------")
    println("------------------------------------------------")
    println("------------------------------------------------")
    @show (k, tkk), Rvthk3M

    # # # Updating the normalized conservative momentums `nh, Ih, Kh`
    if 1 == 1
        is_renormM = false
        M000k3 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k3, fvL0k3[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M110k3 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k3, fvL0k3[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M200k3 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k3, fvL0k3[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)

        nhk3M = M000k3[1, :]
        Ihk3M = M110k3[1, :] / 3 ./ Rvthk3M
        Khk3M = M200k3[1, :] ./ Rvthk3M .^ 2

        # Checking the constraint: `K̂a = 3/2 + ûa²`
        errKIhk3M = 2 / 3 * (Khk3M - Ihk3M .^ 2) .- 1
        norm(errKIhk3M) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k3M = 3/2 + ûa²`", errKIhk3M)
    end
    if is_corrections[1] == false
        nak3M = nak2 .* nhk3M
    else
        nak3M = nak2
    end
    ρa = ma .* nak3M

    uhk3M = Ihk3M
    Ik3M = ρa .* vthk3M .* Ihk3M             # = ρa .* vthk2 .* M110k3[1,:] / 3
    Kk3M = 1 / 2 * ρa .* vthk3M .^ 2 .* Khk3M    # = 1/2 * ρa .* vthk2.^2 .* M200k3[1,:]
    println()
    @show sum(Ik3M) - sum(Ik2M)
    @show sum(Kk3M) - sum(Kk2M)

    uhk3M1 = 1.5^0.5 * Ik3M ./ (2ρa .* Kk3M - Ik3M .^2).^0.5
    vthk3M1 = (2/3 * (2Kk3M ./ ρa - (Ik3M ./ ρa).^2)).^0.5
    @show uhk3M1 - uhk3M
    @show vthk3M1 ./ vthk3M .- 1

    Dnhk2, DIk2, DKk2 = nhk3M .- 1, Ik3M - Ik2M, Kk3M - Kk2M
    RDIk2 = DIk2 ./ Ik2M
    RDKk2 = DKk2 ./ Kk2M
    RerrDnhk2 = nhk3M .- 1
    errDIk2 = abs(DIk2[1] - DIk2[2])
    if errDIk2 ≤ epsT
        RerrDIk2 = abs(sum(DIk2))
    else
        RerrDIk2 = abs(sum(DIk2)) / errDIk2
    end
    errDKk2 = abs(DKk2[1] - DKk2[2])
    if errDKk2 ≤ epsT
        RerrDKk2 = abs(sum(DKk2))
    else
        RerrDKk2 = abs(sum(DKk2)) / errDKk2
    end
    if RerrDIk2 > epsT100
        @warn("Which denotes the meshgrids `nvG` or the time step `dt` may not be the best values, please checking the convergence!", RerrDIk2)
    end
    if RerrDKk2 > epsT100
        @warn("Which denotes the meshgrids `nvG` or the time step `dt` may not be the best values, please checking the convergence!", RerrDKk2)
    end
    @show fmtf2.(sum(DKk2)), fmtf2.(sum(DIk2))
    @show fmtf2.(DKk2), fmtf2.(DIk2)
    @show fmtf2.(RDIk2), fmtf2.(RDKk2)
    @show fmtf2.(RerrDnhk2)
    @show fmtf2.(RerrDIk2)
    @show fmtf2.(RerrDKk2)

    nModk = copy(nMod)

    # Corrections to satisfy the conservation laws by applying a posterior analysis.
    if is_corrections[1] == false
        nhk3 = nhk3M
    else
        nhk3 = ones(ns)
    end
    if is_corrections[2]
        if RerrDIk2 > epsT
            if residualMethod_FP0D == 1
                if RerrDIk2 > epsT
                    DIk2 .-= sum(DIk2) / 2
                    Ik3 = Ik2M + DIk2
                else
                    Ik3 = Ik3M
                end
            elseif residualMethod_FP0D == 2
                ertyui
            end
        else
            Ik3 = Ik3M
        end
    else
        Ik3 = Ik3M
    end
    if is_corrections[3]
        if RerrDKk2 > epsT
            if residualMethod_FP0D == 1
                if RerrDKk2 > epsT
                    DKk2 .-= sum(DKk2) / 2
                    Kk3 = Kk2M + DKk2
                else
                    Kk3 = Kk3M
                end
            elseif residualMethod_FP0D == 2
            end
        else
            Kk3 = Kk3M
        end
    else
        Kk3 = Kk3M
    end

    # Updating the parameters `nhk3, uhk3, vthk3` according to the first three moments `n, I, K` at `(k+1)ᵗʰ` step
    if is_corrections[1] == false
        nak3 = nak2 .* nhk3
    else
        nak3 = nak2
    end
    ρa = ma .* nak3
    uhk3 = 1.5^0.5 * Ik3 ./ (2ρa .* Kk3 - Ik3 .^ 2) .^ 0.5
    Khk3 = 3 / 2 .+ uhk3 .^ 2                                     # `Khk3 = Khk3M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!()`. 
    vthk3 = (2 / 3 * (2 * Kk3 ./ ρa - (Ik3 ./ ρa) .^ 2)) .^ 0.5
    Rvthk3 = vthk3 ./ vthk2

    @show uhk3 ./ uhk3M .- 1
    @show vthk3 ./ vthk3M .- 1
    @show Rvthk3 - Rvthk3M

    # Updating the characteristic parameters of `f̂ₗᵐ(v̂)`: `x0k3 = [naik3,uaik3,vthik3]`
    nModk3 = copy(nModk)
    nMjMs01 = ceil.(Int, 3 / 2 * nModk3)
    naik3 = copy(naik2)      # `n̂a = naᵢ / na`
    uaik3 = copy(uaik2)      # `ûa = uaᵢ / vthk3`
    vthik3 = copy(vthik2)    # `v̂th = vathᵢ / vthk3`
    if prod(nModk3) == 1
        submoment!(naik3, uaik3, vthik3, fvL0k3, vGe, ns; Rvth=Rvthk3, is_renorm=false)
        # submoment!(naik3,uaik3,vthik3,nhk3,uhk3,ones(ns),ns)
        # submoment!(naik3,uaik3,vthik3,nhk3,uhk3.*Rvthk3,Rvthk3,ns;Rvth=Rvthk3)
        @show [naik3[isp][1] - 1 for isp in 1:ns]
        @show [uaik3[isp][1] - uhk3[isp] for isp in 1:ns]
        @show [vthik3[isp][1] - 1 for isp in 1:ns]
    else
        # submoment!(naik3, uaik3, vthik3, nModk3, fvL0k3, vGe, nhk3, uhk3, Khk3 / CjLL2(2), ns;
        #     is_nai_const=is_nai_const, Rvth=Rvthk3, is_renorm=true,
        #     optimizer=optimizer, factor=factor, autodiffs=autodiffs,
        #     is_Jacobian=is_Jacobian, show_trace=show_trace,
        #     maxIterKing=maxIterKing, p_tol=p_tol, f_tol=f_tol, g_tol=g_tol,
        #     p_noise_rel=p_noise_rel, p_noise_abs=p_noise_abs)
        submoment!(naik3,uaik3,vthik3,nModk3,fvL0k3,vGe,ns;
                   is_nai_const=is_nai_const,Rvth=Rvthk3,is_renorm=true,
                   optimizer=optimizer,factor=factor,autodiffs=autodiffs,
                   is_Jacobian=is_Jacobian,show_trace=show_trace,
                   maxIterKing=maxIterKing,p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                   p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
    end
    nhk3sum = [sum(naik3[k]) for k in 1:ns]
    Ihk3sum, vhthk3sum = zeros(ns), zeros(ns)
    nuTsNorm!(Ihk3sum, vhthk3sum, naik3, uaik3, vthik3)
    uhk3sum = Ihk3sum
    Khk3sum = 3 / 2 .+ uhk3sum .^ 2

    # Checking the constraint: `K̂a = 3/2 + ûa²`
    errKuhk3sum = 2 / 3 * (Khk3M - Ihk3sum .^ 2) .- 1
    norm(errKuhk3sum) ≤ epsT100 || @warn("Number of meshgrids `nvG` or the time step `dt` may be not enough to satisfy the convergence of `K̂a_k3sum = 3/2 + ûa²`", errKuhk3sum)

    @show 33333, Rvthk3, errKuhk3sum

    println()
    @show nhk3sum .- 1
    @show Ihk3sum - uhk3M
    @show vhthk3sum .- 1

    Ik3sum = ρa .* vthk3 .* Ihk3sum                  # = ρa .* vthk .* M110k3[1,:] / 3
    Kk3sum = 1 / 2 * ρa .* vthk3 .^ 2 .* Khk3sum     # = 1/2 * ρa .* vthk.^2 .* M200k3[1,:]
    println()
    @show sum(Ik3sum) - sum(Ik3M)
    @show sum(Kk3sum) - sum(Kk3M)

    println()
    @show [naik3[k] ./ naik2[k] .- 1 for k in 1:ns]
    @show [uaik3[k] ./ uaik2[k] .- 1 for k in 1:ns]
    @show [vthik3[k] ./ vthik2[k] .- 1 for k in 1:ns]

    # # Updating the amplitude function of normalized distribution functions `f̂ₗᵐ(v̂)`  at the `kᵗʰ` step according parameters `naik3,uaik3,vthik3`.
    LMk3 = 0LMk2
    fvLk3 = zeros(nvG, LM1k2 + 1, ns)
    @show naik3 - nai
    LMk3, fvLk3 = fvLDMz(fvLk3, vGe, nvG, LMk3, ns, naik3, uaik3, vthik3, nModk3; L_limit=L_limit,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_full_fvL=is_full_fvL)
    LM1k3 = maximum(LMk3) + 1

    # Checking the conservation laws of the renormalized distribution function `fvLk3`
    vthk3MM = vthk3
    if 1 == 1
        is_renormM = false
        M000k3M = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k3M, fvLk3[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M110k3M = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k3M, fvLk3[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M200k3M = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k3M, fvLk3[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)

        # Checking the conservations
        nhk3MM = M000k3M[1, :]
        Ihk3MM = M110k3M[1, :] / 3
        Khk3MM = M200k3M[1, :]

        # Checking the constraint: `K̂a = 3/2 + ûa²`
        errKIhk3MM = 2 / 3 * (Khk3MM - Ihk3MM .^ 2) .- 1
        norm(errKIhk3MM) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k3M = 3/2 + ûa²`", errKIhk3MM)

        @show 333, Khk3MM, errKIhk3MM
    end

    if is_corrections[1] == false
        nak3MM = nak2 .* nhk3MM
    else
        nak3MM = nak2
    end
    ρa = ma .* nak3MM
    uhk3MM = Ihk3MM
    Ik3MM = ρa .* vthk3M .* Ihk3MM
    Kk3MM = 1 / 2 * ρa .* vthk3M .^ 2 .* Khk3MM
    println()
    @show sum(Ik3MM) - sum(Ik3M)
    @show sum(Kk3MM) - sum(Kk3M)
    
    uhk3MMc = 1.5^0.5 * Ik3MM ./ (2ρa .* Kk3MM - Ik3MM .^ 2) .^ 0.5
    Khk3MMc = 3 / 2 .+ uhk3 .^ 2                                      # `Khk3 = Khk3M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!(⋯)`.
    vthk3MM = (2 / 3 * (2 * Kk3MM ./ ρa - (Ik3MM ./ ρa) .^ 2)) .^ 0.5
    Rvthk3MM = vthk3MM ./ vthk2

    @show uhk3MM ./ uhk3M .- 1
    @show vthk3MM ./ vthk3M .- 1
    @show Rvthk3MM - Rvthk3M

    pfk3(isp3, L1) = plot(vGe, [fvL0e[:, L1, isp3] fvL0k3[:, L1, isp3]],
        line=(2, :auto), label=string("isp,L=", (isp3, L1 - 1)))
    pDtfk3(isp3, L1) = plot(vGe, fvL0k3[:, L1, isp3] - fvL0e[:, L1, isp3], line=(2, :auto), label=string("ΔₜfvL"))

    pfk3up(isp3, L1) = plot(vGe, [fvL0e[:, L1, isp3] fvLk3[:, L1, isp3]],
        line=(2, :auto), label=string("fvL_up"))
    pDtfk3up(isp3, L1) = plot(vGe, fvLk3[:, L1, isp3] - fvL0e[:, L1, isp3], line=(2, :auto), label=string("Δₜfᵏ⁺¹"))

    pfDtfk3(L1) = display(plot(pfk3(1, L1), pfk3(2, L1), pDtfk3(1, L1), pDtfk3(2, L1),
        pfk3up(1, L1), pfk3up(2, L1), pDtfk3up(1, L1), pDtfk3up(2, L1), layout=(4, 2)))
    pfDtfk3.(L1:LM1k3)

    # # Updating the FP collision terms according to the `FPS` operators.
    err_dtnIKk3 = 0.0
    fvL4k3 = copy(fvLk3)
    dtfvL0k3 = zero.(fvLk3)
    if is_dtfvLaa === 2
        @time dtfvL0k3, dtfvL0aak3, fvL4k3, err_dtnIKk3 = dtfvLSplineab(dtfvL0k3, fvL4k3, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak3, vthk3, naik3, uaik3, vthik3, LMk3, LM1k3, ns, nMod;
            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_dtfvLaa=is_dtfvLaa,
            is_normdtf=is_normdtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    else
        @time dtfvL0k3, fvL4k3, err_dtnIKk3 = dtfvLSplineab(dtfvL0k3, fvL4k3, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak3, vthk3, naik3, uaik3, vthik3, LMk3, LM1k3, ns, nMod;
            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_dtfvLaa=is_dtfvLaa,
            is_normdtf=is_normdtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    end

    #########  Optimizations according to `y(v̂→0) → Cₗᵐ` and `Rd1y(v̂→0) → 0`
    if is_optimdtfvL
        dtfvL0k3opt = copy(dtfvL0k3)
        nvc3k3 = deepcopy(nvc3k2)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
        optimdtfvL0e!(nvc3k3, dtfvL0k3opt, fvL4k3, vGe, nvG, ns, LMk3;
            orders=order_dvδtf, is_boundv0=is_boundv0,
            Nsmooth=Nsmooth, order_smooth=order_smooth, abstol_Rdy=abstol_Rdy,
            k=k_dtf, Nitp=Nitp, order_smooth_itp=order_smooth_itp, order_nvc_itp=order_nvc_itp,
            nvc0_limit=nvc0_limit, L1nvc_limit=L1nvc_limit)
        dtMsnnE3opt = zeros(datatype, njMs, LM1k3, ns)
        dtMsnnE3opt = MsnnEvens(dtMsnnE3opt, dtfvL0k3opt, vGe, njMs, LMk3, LM1k3, ns; is_renorm=is_renorm)
    else
        dtfvL0k3opt = dtfvL0k3
    end
    
    # dtMsnk3
    if 1 == 1
        is_renormM = false
        R000k3 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(R000k3, dtfvL0k3opt[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        R110k3 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(R110k3, dtfvL0k3opt[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        R200k3 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(R200k3, dtfvL0k3opt[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)

        dtnhk3M = R000k3[1, :]
        RdtIk3M = R110k3[1, :] / 3    # = dtuhk3M + uhk3 .* Rdtvthk3M
        RdtKk3M = R200k3[1, :]        # = (2 * uhk3 .* dtuhk3M + (3 .+ 2 * uhk3.^2) .* Rdtvthk3M)

        # wk3M = R200k3[1,:] - 2 / 3 * uhk3 .* R110k3[1,:]
        wk3M = RdtKk3M - 2 * uhk3 .* RdtIk3M
        Rdtvthk3M = wk3M / 3
        dtvthk3M = vthk3 .* wk3M / 3               # wk3M = - (Rdtn - 3Rdtvth) = 3Rdtvth

        dtuhk3M = RdtIk3M - uhk3 .* Rdtvthk3M       # 

        # Checking the conservations according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
        errdtKuhk3M = RdtKk3M - (2 * uhk3 .* dtuhk3M + (3 .+ 2 * uhk3 .^ 2) .* Rdtvthk3M)
        norm(errdtKuhk3M) ≤ epsT1 || @warn("Number of meshgrids may be not enough to satisfy 
                     the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`", errdtKuhk3M)
        # @show  RdtKk3M ./ RdtIk3M.^2
    end
    dtIk3M = ρa .* vthk3 .* RdtIk3M
    dtKk3M = ρa .* vthk3 .^ 2 .* RdtKk3M / 2
    RerrdtKk3M = fmtf2(sum(dtKk3M) / abs(dtKk3M[1] - dtKk3M[2]))
    errdtIk3M = abs(dtIk3M[1] - dtIk3M[2])
    if errdtIk3M ≤ epsT
        RerrdtIk3M = fmtf2(errdtIk3M)
    else
        RerrdtIk3M = fmtf2(sum(dtIk3M) / errdtIk3M)
    end
    @show nnv, nvG
    @show fmtf2.(dtnhk3M)
    @show dtIk3M, RerrdtIk3M
    @show dtKk3M, RerrdtKk3M

    Rvthk3M = (1 .- 2 / 3 * dt * (dt * (dtuhk3M + uhk3 .* Rdtvthk3M) .^ 2 - wk3M)) .^ 0.5
    @show tkk, Rvthk3M

    # @show norm(dtfvL0k3 - dtfvL0e)
    # @show norm(dtfvL0k3opt - dtfvL0e)
end