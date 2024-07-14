"""
  Applying the fixed time step explicit Newton method to solve the ODE problems.
    The fixed time step is decided by the characteristic time scale of the relaxation process and
    the relative rate of change of the moments respect to time.

    `RdtM = M⁻¹Δₜ∂ₜM ≪ 1` and is `≤ 1e-3` defaultly.
  
  Updating `vth` and `ûa` according to the values of `Ka` and `Ia` as:

    vth = √(2/3 * (2Ka / ρa - (Ia / ρa)^2))
    ûa = √(3/2) × Ia / √(2ρa * Ka - Ia^2)

  Constraints:

    K̂a = 3/2 + ûa²

  1

  Inputs:
    residualMethod_FP0D::Int ∈ [1,2]. When `residualMethod_FP0D=1` denotes absorbing the residuals by using the "dichotomy method"; or else
                                     `residualMethod_FP0D=2` denotes absorbing the residuals with a geometric ratio  等比残差吸收
    1
  Outouts:

"""
# is_nai_const = true   # (=false, default) whether keep the values of `nai` to be constants.

# `is_corrections::Vector{Bool} = [true, true, true]` for `[is_corrections_n, is_corrections_I, is_corrections_K]`
# is_corrections[2:3] .= true
is_corrections = [true, false, false]
is_optimdtfvL = false

# tk
k = 1
tkk = t0
Rvthk = ones(ns) # The true values is decided by a iteration at the initial step and the previous step at the follow steps.
dt = dt0      # (=1e-2 default), Which is normalized by `td ~ 1[s]` for MCF plasma.
               # `1e0 ~ 1e-5` is proposed to be used for most situation.
println("------------------------------------------------")
println("------------------------------------------------")
@show (k,tkk,dt)
# Msnk
vthkM = copy(vth)
if 1 == 1
    is_renormM = false
    M000k = zeros(2,ns)
    j, L = 0, 0
    MsnnEvens!(M000k, fvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)
    M110k = zeros(2,ns)
    j, L = 1, 1
    MsnnEvens!(M110k, fvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)
    M200k = zeros(2,ns)
    j, L = 2, 0
    MsnnEvens!(M200k, fvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)

    # Checking the conservations
    nhkM = M000k[1,:]
    IhkM = M110k[1,:] / 3 ./ Rvthk
    KhkM = M200k[1,:] ./ Rvthk.^2
    errKIhkM = 2/3 * (KhkM - IhkM.^2) .- 1
    norm(errKIhkM) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_kM = 3/2 + ûa²`",errKIhkM)
end
if is_corrections[1] == false
    nak = na .* nhkM
else
    nak = na 
end
ρa = ma .* nak

uhkM = IhkM
IkM = ρa .* vthkM .* uhkM
KkM = 1/2 * ρa .* vthkM.^2 .* KhkM
# uhkMM = 1.5^0.5 * IkM ./ (2ρa .* KkM - IkM .^2).^0.5

# Updating the parameters `nhk, uhk, vthk` according to the first three moments `n, I, K` at `kᵗʰ` step
uhk = 1.5^0.5 * IkM ./ (2ρa .* KkM - IkM .^2).^0.5
vthk = (2/3 * (2KkM ./ ρa - (IkM ./ ρa).^2)).^0.5
@show uhkM ./ uhk .- 1
@show vth ./ vthk .- 1

# dtMsnk
if 1 == 1
    is_renormM = false
    R000k = zeros(2,ns)
    j, L = 0, 0
    MsnnEvens!(R000k, dtfvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)
    R110k = zeros(2,ns)
    j, L = 1, 1
    MsnnEvens!(R110k, dtfvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)
    R200k = zeros(2,ns)
    j, L = 2, 0
    MsnnEvens!(R200k, dtfvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)

    dtnhkM = R000k[1,:]
    RdtIkM = R110k[1,:] / 3    # = dtuhkM + uhk .* RdtvthkM
    RdtKkM = R200k[1,:]        # = (2 * uhk .* dtuhkM + (3 .+ 2 * uhk.^2) .* RdtvthkM)

    # wkM = R200k[1,:] - 2 / 3 * uhk .* R110k[1,:]
    wkM = RdtKkM - 2 * uhk .* RdtIkM
    RdtvthkM = wkM / 3
    dtvthkM = vthk .* wkM / 3               # wkM = - (Rdtn - 3Rdtvth) = 3Rdtvth
    
    dtuhkM = RdtIkM - uhk .* RdtvthkM       # 

    # Checking the conservations according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
    errdtKuhkM = RdtKkM - (2 * uhk .* dtuhkM + (3 .+ 2 * uhk.^2) .* RdtvthkM)
    norm(errdtKuhkM) ≤ epsT1 || @warn("Number of meshgrids may be not enough to satisfy 
                 the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`",errdtKuhkM)
    # @show  RdtKkM ./ RdtIkM.^2
end
dtIkM = ρa .* vthk .* RdtIkM
dtKkM = ρa .* vthk.^2 .* RdtKkM / 2
RerrdtKkM = fmtf2(sum(dtKkM) / abs(dtKkM[1] - dtKkM[2]))
errdtIkM = abs(dtIkM[1] - dtIkM[2])
if errdtIkM ≤ epsT
    RerrdtIkM = fmtf2(errdtIkM)
else
    RerrdtIkM = fmtf2(sum(dtIkM) / errdtIkM)
end
@show nnv,nvG
@show fmtf2.(dtnhkM)
@show dtIkM,RerrdtIkM
@show dtKkM,RerrdtKkM

# The optimized `dtfvL`
if L1nvc_limit ≤ 2
end

Rvthk1M = (1 .- 2/3 * dt * (dt * (dtuhkM + uhk .* RdtvthkM).^2 - wkM)).^0.5 
@show tkk, Rvthk1M

# tk1: Applying the Explicit Newton method to solve the ODE equations of `f̂vL(v̂)`
k += 1
if k ≥ 2
    tkk += dt
    fvL0k1 = fvL0 + dt * dtfvL0e

    # Updating the thermal velocity `vₜₕ = vthk1` in single step
    vthk1Mdt = vthk + dt * dtvthkM
    vthk1M = vthk .* Rvthk1M
    println("------------------------------------------------")
    println("------------------------------------------------")
    @show (k, tkk), Rvthk1M

    # # # Updating the normalized conservative momentums `nh, Ih, Kh`
    if 1 == 1
        is_renormM = false
        M000k1 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k1, fvL0k1[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M110k1 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k1, fvL0k1[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M200k1 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k1, fvL0k1[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)

        nhk1M = M000k1[1, :]
        Ihk1M = M110k1[1, :] / 3 ./ Rvthk1M
        Khk1M = M200k1[1, :] ./ Rvthk1M .^ 2

        # Checking the constraint: `K̂a = 3/2 + ûa²`
        errKIhk1M = 2 / 3 * (Khk1M - Ihk1M .^ 2) .- 1
        norm(errKIhk1M) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k1M = 3/2 + ûa²`", errKIhk1M)
    end
    if is_corrections[1] == false
        nak1M = nak .* nhk1M
    else
        nak1M = nak
    end
    ρa = ma .* nak1M

    uhk1M = Ihk1M
    Ik1M = ρa .* vthk1M .* Ihk1M                 # = ρa .* vthk .* M110k1[1,:] / 3
    Kk1M = 1 / 2 * ρa .* vthk1M .^ 2 .* Khk1M    # = 1/2 * ρa .* vthk.^2 .* M200k1[1,:]
    println()
    @show sum(Ik1M) - sum(IkM)
    @show sum(Kk1M) - sum(KkM)

    uhk1M1 = 1.5^0.5 * Ik1M ./ (2ρa .* Kk1M - Ik1M .^2).^0.5
    vthk1M1 = (2/3 * (2Kk1M ./ ρa - (Ik1M ./ ρa).^2)).^0.5
    @show uhk1M1 - uhk1M
    @show vthk1M1 ./ vthk1M .- 1

    Dnhk, DIk, DKk = nhk1M .- 1, Ik1M - IkM, Kk1M - KkM
    RDIk = DIk ./ IkM
    RDKk = DKk ./ KkM
    RerrDnhk = nhk1M .- 1
    errDIk = abs(DIk[1] - DIk[2])
    if errDIk ≤ epsT
        RerrDIk = abs(sum(DIk))
    else
        RerrDIk = abs(sum(DIk)) / errDIk
    end
    errDKk = abs(DKk[1] - DKk[2])
    if errDKk ≤ epsT
        RerrDKk = abs(sum(DKk))
    else
        RerrDKk = abs(sum(DKk)) / errDKk
    end
    if RerrDIk > epsT100
        @warn("Which denotes the meshgrids `nvG` or the time step `dt` may not be the best values, please checking the convergence!", RerrDIk)
    end
    if RerrDKk > epsT100
        @warn("Which denotes the meshgrids `nvG` or the time step `dt` may not be the best values, please checking the convergence!", RerrDKk)
    end
    @show fmtf2.(sum(DKk)), fmtf2.(sum(DIk))
    @show fmtf2.(DKk), fmtf2.(DIk)
    @show fmtf2.(RDIk), fmtf2.(RDKk)
    @show fmtf2.(RerrDnhk)
    @show fmtf2.(RerrDIk)
    @show fmtf2.(RerrDKk)

    nModk = copy(nMod)

    # Corrections to satisfy the conservation laws by applying a posterior analysis.
    is_corrections .= false
    if is_corrections[1] == false
        nhk1 = nhk1M
    else
        nhk1 = ones(ns)
    end
    if is_corrections[2]
        if RerrDIk > epsT
            if residualMethod_FP0D == 1
                if RerrDIk > epsT
                    DIk .-= sum(DIk) / 2
                    Ik1 = IkM + DIk
                else
                    Ik1 = Ik1M
                end
            elseif residualMethod_FP0D == 2
                ertyui
            end
        else
            Ik1 = Ik1M
        end
    else
        Ik1 = Ik1M
    end
    if is_corrections[3]
        if RerrDKk > epsT
            if residualMethod_FP0D == 1
                if RerrDKk > epsT
                    DKk .-= sum(DKk) / 2
                    Kk1 = KkM + DKk
                else
                    Kk1 = Kk1M
                end
            elseif residualMethod_FP0D == 2
            end
        else
            Kk1 = Kk1M
        end
    else
        Kk1 = Kk1M
    end

    # Updating the parameters `nhk1, uhk1, vthk1` according to the first three moments `n, I, K` at `(k+1)ᵗʰ` step
    if is_corrections[1] == false
        nak1 = nak .* nhk1
    else
        nak1 = nak
    end
    ρa = ma .* nak1
    uhk1 = 1.5^0.5 * Ik1 ./ (2ρa .* Kk1 - Ik1 .^ 2) .^ 0.5
    Khk1 = 3 / 2 .+ uhk1 .^ 2                                      # `Khk1 = Khk1M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!(⋯)`.
    vthk1 = (2 / 3 * (2 * Kk1 ./ ρa - (Ik1 ./ ρa) .^ 2)) .^ 0.5
    Rvthk1 = vthk1 ./ vthk

    @show uhk1 ./ uhk1M .- 1
    @show vthk1 ./ vthk1M .- 1
    @show Rvthk1 - Rvthk1M

    # Updating the characteristic parameters of `f̂ₗᵐ(v̂)`: `x0k1 = [naik1,uaik1,vthik1]`
    nModk1 = copy(nModk)
    nMjMs01 = ceil.(Int, 3 / 2 * nModk1)
    naik1 = copy(nai)      # `n̂a = naᵢ / na`
    uaik1 = copy(uai)      # `ûa = uaᵢ / vthk1`
    vthik1 = copy(vthi)    # `v̂th = vathᵢ / vthk1`
    if prod(nModk1) == 1
        submoment!(naik1, uaik1, vthik1, fvL0k1, vGe, ns; Rvth=Rvthk1, is_renorm=false)
        # submoment!(naik1,uaik1,vthik1,nhk1,uhk1,ones(ns),ns)
        # submoment!(naik1,uaik1,vthik1,nhk1,uhk1.*Rvthk1,Rvthk1,ns;Rvth=Rvthk1)
        @show [naik1[isp][1] - 1 for isp in 1:ns]
        @show [uaik1[isp][1] - uhk1[isp] for isp in 1:ns]
        @show [vthik1[isp][1] - 1 for isp in 1:ns]
    else
        # submoment!(naik1, uaik1, vthik1, nModk1, fvL0k1, vGe, nhk1, uhk1, Khk1 / CjLL2(2), ns;
        #     is_nai_const=is_nai_const, Rvth=Rvthk1, is_renorm=true,
        #     optimizer=optimizer, factor=factor, autodiffs=autodiffs,
        #     is_Jacobian=is_Jacobian, show_trace=show_trace,
        #     maxIterKing=maxIterKing, p_tol=p_tol, f_tol=f_tol, g_tol=g_tol,
        #     p_noise_rel=p_noise_rel, p_noise_abs=p_noise_abs)
        submoment!(naik1,uaik1,vthik1,nModk1,fvL0k1,vGe,ns;
                   is_nai_const=is_nai_const,Rvth=Rvthk1,is_renorm=true,
                   optimizer=optimizer,factor=factor,autodiffs=autodiffs,
                   is_Jacobian=is_Jacobian,show_trace=show_trace,
                   maxIterKing=maxIterKing,p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                   p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
    end
    nhk1sum = [sum(naik1[k]) for k in 1:ns]
    Ihk1sum, vhthk1sum = zeros(ns), zeros(ns)
    nuTsNorm!(Ihk1sum, vhthk1sum, naik1, uaik1, vthik1)
    uhk1sum = Ihk1sum
    Khk1sum = 3 / 2 .+ uhk1sum .^ 2

    # Checking the constraint: `K̂a = 3/2 + ûa²`
    errKIhk1sum = 2 / 3 * (Khk1sum - Ihk1sum .^ 2) .- 1
    norm(errKIhk1sum) ≤ epsT100 || @warn("Number of meshgrids `nvG` or the time step `dt` may be not enough to satisfy the convergence of `K̂a_k1sum = 3/2 + ûa²`", errKIhk1sum)
    
    @show 2222, Rvthk1, errKIhk1sum

    println()
    @show nhk1sum .- 1
    @show Ihk1sum - Ihk1M
    @show vhthk1sum .- 1

    Ik1sum = ρa .* vthk1 .* Ihk1sum                  # = ρa .* vthk .* M110k1[1,:] / 3
    Kk1sum = 1 / 2 * ρa .* vthk1 .^ 2 .* Khk1sum     # = 1/2 * ρa .* vthk.^2 .* M200k1[1,:]
    println()
    @show sum(Ik1sum) - sum(Ik1M)
    @show sum(Kk1sum) - sum(Kk1M)

    println()
    @show [naik1[k] ./ nai[k] .- 1 for k in 1:ns]
    @show [uaik1[k] ./ uai[k] .- 1 for k in 1:ns]
    @show [vthik1[k] ./ vthi[k] .- 1 for k in 1:ns]

    # # Updating the amplitude function of normalized distribution functions `f̂ₗᵐ(v̂)`  at the `kᵗʰ` step according parameters `naik1,uaik1,vthik1`.
    LMk1 = 0LM
    fvLk1 = zeros(nvG, LM1 + 1, ns)
    @show naik1 - nai
    LMk1, fvLk1 = fvLDMz(fvLk1, vGe, nvG, LMk1, ns, naik1, uaik1, vthik1, nModk1; L_limit=L_limit,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_full_fvL=is_full_fvL)
    LM1k1 = maximum(LMk1) + 1

    # Checking the conservation laws of the renormalized distribution function `fvLk1`
    vthk1MM = vthk1
    if 1 == 1
        is_renormM = false
        M000k1M = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k1M, fvLk1[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M110k1M = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k1M, fvLk1[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M200k1M = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k1M, fvLk1[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)

        # Checking the conservations
        nhk1MM = M000k1M[1, :]
        Ihk1MM = M110k1M[1, :] / 3
        Khk1MM = M200k1M[1, :]

        # Checking the constraint: `K̂a = 3/2 + ûa²`
        errKIhk1MM = 2 / 3 * (Khk1MM - Ihk1MM .^ 2) .- 1
        norm(errKIhk1MM) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k1M = 3/2 + ûa²`", errKIhk1MM)

        @show 333, Khk1MM, errKIhk1MM
    end

    if is_corrections[1] == false
        nak1MM = nak .* nhk1MM
    else
        nak1MM = nak
    end
    ρa = ma .* nak1MM
    uhk1MM = Ihk1MM
    Ik1MM = ρa .* vthk1MM .* Ihk1MM
    Kk1MM = 1 / 2 * ρa .* vthk1MM .^ 2 .* Khk1MM
    println()
    @show sum(Ik1MM) - sum(Ik1M)
    @show sum(Kk1MM) - sum(Kk1M)
    
    uhk1MMc = 1.5^0.5 * Ik1MM ./ (2ρa .* Kk1MM - Ik1MM .^ 2) .^ 0.5
    Khk1MMc = 3 / 2 .+ uhk1 .^ 2                                      # `Khk1 = Khk1M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!(⋯)`.
    vthk1MM = (2 / 3 * (2 * Kk1MM ./ ρa - (Ik1MM ./ ρa) .^ 2)) .^ 0.5
    Rvthk1MM = vthk1MM ./ vthk

    @show uhk1MM ./ uhk1M .- 1
    @show vthk1MM ./ vthk1M .- 1
    @show Rvthk1MM - Rvthk1M

    pfk1(isp3, L1) = plot(vGe, [fvL0e[:, L1, isp3] fvL0k1[:, L1, isp3]],
        line=(2, :auto), label=string("isp,L=", (isp3, L1 - 1)))
    pDtfk1(isp3, L1) = plot(vGe, fvL0k1[:, L1, isp3] - fvL0e[:, L1, isp3], line=(2, :auto), label=string("ΔₜfvL"))

    pfk1up(isp3, L1) = plot(vGe, [fvL0e[:, L1, isp3] fvLk1[:, L1, isp3]],
        line=(2, :auto), label=string("fvL_up"))
    pDtfk1up(isp3, L1) = plot(vGe, fvLk1[:, L1, isp3] - fvL0e[:, L1, isp3], line=(2, :auto), label=string("Δₜfᵏ⁺¹"))

    pfDtfk1(L1) = display(plot(pfk1(1, L1), pfk1(2, L1), pDtfk1(1, L1), pDtfk1(2, L1),
        pfk1up(1, L1), pfk1up(2, L1), pDtfk1up(1, L1), pDtfk1up(2, L1), layout=(4, 2)))
    pfDtfk1.(L1:LM1k1)

    # # Updating the FP collision terms according to the `FPS` operators.
    err_dtnIKk1 = 0.0
    fvL4k1 = copy(fvLk1)
    dtfvL0k1 = zero.(fvLk1)
    if is_dtfvLaa === 2
        @time dtfvL0k1, dtfvL0aak1, fvL4k1, err_dtnIKk1 = dtfvLSplineab(dtfvL0k1, fvL4k1, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak1, vthk1, naik1, uaik1, vthik1, LMk1, LM1k1, ns, nMod;
            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_dtfvLaa=is_dtfvLaa,
            is_normdtf=is_normdtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    else
        @time dtfvL0k1, fvL4k1, err_dtnIKk1 = dtfvLSplineab(dtfvL0k1, fvL4k1, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak1, vthk1, naik1, uaik1, vthik1, LMk1, LM1k1, ns, nMod;
            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_dtfvLaa=is_dtfvLaa,
            is_normdtf=is_normdtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    end

    #########  Optimizations according to `y(v̂→0) → Cₗᵐ` and `Rd1y(v̂→0) → 0`
    if is_optimdtfvL
        dtfvL0k1opt = copy(dtfvL0k1)
        nvc3k1 = zeros(Int64, 2(order_smooth + 1), LM1k1, ns)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
        optimdtfvL0e!(nvc3k1, dtfvL0k1opt, fvL4k1, vGe, nvG, ns, LMk1;
            orders=order_dvδtf, is_boundv0=is_boundv0,
            Nsmooth=Nsmooth, order_smooth=order_smooth, abstol_Rdy=abstol_Rdy,
            k=k_dtf, Nitp=Nitp, order_smooth_itp=order_smooth_itp, order_nvc_itp=order_nvc_itp,
            nvc0_limit=nvc0_limit, L1nvc_limit=L1nvc_limit)
        dtMsnnE3opt = zeros(datatype, njMs, LM1k1, ns)
        dtMsnnE3opt = MsnnEvens(dtMsnnE3opt, dtfvL0k1opt, vGe, njMs, LMk1, LM1k1, ns; is_renorm=is_renorm)
    else
        dtfvL0k1opt = dtfvL0k1
    end

    # dtMsnk1
    if 1 == 1
        is_renormM = false
        R000k1 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(R000k1, dtfvL0k1opt[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        R110k1 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(R110k1, dtfvL0k1opt[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        R200k1 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(R200k1, dtfvL0k1opt[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)

        dtnhk1M = R000k1[1, :]
        RdtIk1M = R110k1[1, :] / 3    # = dtuhk1M + uhk1 .* Rdtvthk1M
        RdtKk1M = R200k1[1, :]        # = (2 * uhk1 .* dtuhk1M + (3 .+ 2 * uhk1.^2) .* Rdtvthk1M)

        # wk1M = R200k1[1,:] - 2 / 3 * uhk1 .* R110k1[1,:]
        wk1M = RdtKk1M - 2 * uhk1 .* RdtIk1M
        Rdtvthk1M = wk1M / 3
        dtvthk1M = vthk1 .* wk1M / 3               # wk1M = - (Rdtn - 3Rdtvth) = 3Rdtvth

        dtuhk1M = RdtIk1M - uhk1 .* Rdtvthk1M       # 

        # Checking the conservations according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
        errdtKuhk1M = RdtKk1M - (2 * uhk1 .* dtuhk1M + (3 .+ 2 * uhk1 .^ 2) .* Rdtvthk1M)
        norm(errdtKuhk1M) ≤ epsT1 || @warn("Number of meshgrids may be not enough to satisfy 
                     the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`", errdtKuhk1M)
        # @show  RdtKk1M ./ RdtIk1M.^2
    end
    dtIk1M = ρa .* vthk1 .* RdtIk1M
    dtKk1M = ρa .* vthk1 .^ 2 .* RdtKk1M / 2
    RerrdtKk1M = fmtf2(sum(dtKk1M) / abs(dtKk1M[1] - dtKk1M[2]))
    errdtIk1M = abs(dtIk1M[1] - dtIk1M[2])
    if errdtIk1M ≤ epsT
        RerrdtIk1M = fmtf2(errdtIk1M)
    else
        RerrdtIk1M = fmtf2(sum(dtIk1M) / errdtIk1M)
    end
    @show nnv, nvG
    @show fmtf2.(dtnhk1M)
    @show dtIk1M, RerrdtIk1M
    @show dtKk1M, RerrdtKk1M

    Rvthk2M = (1 .- 2 / 3 * dt * (dt * (dtuhk1M + uhk1 .* Rdtvthk1M) .^ 2 - wk1M)) .^ 0.5
    @show tkk, Rvthk2M
end

# tk2: Applying the Explicit Newton method to solve the ODE equations of `f̂vL(v̂)`
k += 1
if k ≥ 3
    tkk += dt
    fvL0k2 = fvLk1 + dt * dtfvL0k1

    # Updating the thermal velocity `vₜₕ = vthk2` in single step
    vthk2M = vthk1 .* Rvthk2M
    println("------------------------------------------------")
    println("------------------------------------------------")
    println("------------------------------------------------")
    @show (k, tkk), Rvthk2M

    # # # Updating the normalized conservative momentums `nh, Ih, Kh`
    if 1 == 1
        is_renormM = false
        M000k2 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k2, fvL0k2[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M110k2 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k2, fvL0k2[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M200k2 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k2, fvL0k2[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)

        nhk2M = M000k2[1, :]
        Ihk2M = M110k2[1, :] / 3 ./ Rvthk2M
        Khk2M = M200k2[1, :] ./ Rvthk2M .^ 2

        # Checking the constraint: `K̂a = 3/2 + ûa²`
        errKIhk2M = 2 / 3 * (Khk2M - Ihk2M .^ 2) .- 1
        norm(errKIhk2M) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k2M = 3/2 + ûa²`", errKIhk2M)
    end
    if is_corrections[1] == false
        nak2M = nak1 .* nhk2M
    else
        nak2M = nak1
    end
    ρa = ma .* nak2M

    uhk2M = Ihk2M
    Ik2M = ρa .* vthk2M .* Ihk2M                 # = ρa .* vthk1 .* M110k2[1,:] / 3
    Kk2M = 1 / 2 * ρa .* vthk2M .^ 2 .* Khk2M    # = 1/2 * ρa .* vthk1.^2 .* M200k2[1,:]
    println()
    @show sum(Ik2M) - sum(Ik1M)
    @show sum(Kk2M) - sum(Kk1M)

    uhk2Mc = 1.5^0.5 * Ik2M ./ (2ρa .* Kk2M - Ik2M .^2).^0.5
    vthk2Mc = (2/3 * (2Kk2M ./ ρa - (Ik2M ./ ρa).^2)).^0.5
    @show uhk2Mc - uhk2M
    @show vthk2Mc ./ vthk2M .- 1

    Dnhk1, DIk1, DKk1 = nhk2M .- 1, Ik2M - Ik1M, Kk2M - Kk1M
    RDIk1 = DIk1 ./ Ik1M
    RDKk1 = DKk1 ./ Kk1M
    RerrDnhk1 = nhk2M .- 1
    errDIk1 = abs(DIk1[1] - DIk1[2])
    if errDIk1 ≤ epsT
        RerrDIk1 = abs(sum(DIk1))
    else
        RerrDIk1 = abs(sum(DIk1)) / errDIk1
    end
    errDKk1 = abs(DKk1[1] - DKk1[2])
    if errDKk1 ≤ epsT
        RerrDKk1 = abs(sum(DKk1))
    else
        RerrDKk1 = abs(sum(DKk1)) / errDKk1
    end
    if RerrDIk1 > epsT100
        @warn("Which denotes the meshgrids `nvG` or the time step `dt` may not be the best values, please checking the convergence!", RerrDIk1)
    end
    if RerrDKk1 > epsT100
        @warn("Which denotes the meshgrids `nvG` or the time step `dt` may not be the best values, please checking the convergence!", RerrDKk1)
    end
    @show fmtf2.(sum(DKk1)), fmtf2.(sum(DIk1))
    @show fmtf2.(DKk1), fmtf2.(DIk1)
    @show fmtf2.(RDIk1), fmtf2.(RDKk1)
    @show fmtf2.(RerrDnhk1)
    @show fmtf2.(RerrDIk1)
    @show fmtf2.(RerrDKk1)

    nModk = copy(nMod)

    # Corrections to satisfy the conservation laws by applying a posterior analysis.
    if is_corrections[1] == false
        nhk2 = nhk2M
    else
        nhk2 = ones(ns)
    end
    if is_corrections[2]
        if RerrDIk1 > epsT
            if residualMethod_FP0D == 1
                if RerrDIk1 > epsT
                    DIk1 .-= sum(DIk1) / 2
                    Ik2 = Ik1M + DIk1
                else
                    Ik2 = Ik2M
                end
            elseif residualMethod_FP0D == 2
                ertyui
            end
        else
            Ik2 = Ik2M
        end
    else
        Ik2 = Ik2M
    end
    if is_corrections[3]
        if RerrDKk1 > epsT
            if residualMethod_FP0D == 1
                if RerrDKk1 > epsT
                    DKk1 .-= sum(DKk1) / 2
                    Kk2 = Kk1M + DKk1
                else
                    Kk2 = Kk2M
                end
            elseif residualMethod_FP0D == 2
            end
        else
            Kk2 = Kk2M
        end
    else
        Kk2 = Kk2M
    end

    # Updating the parameters `nhk2, uhk2, vthk2` according to the first three moments `n, I, K` at `(k+1)ᵗʰ` step
    if is_corrections[1] == false
        nak2 = nak1 .* nhk2
    else
        nak2 = nak1
    end
    ρa = ma .* nak2
    uhk2 = 1.5^0.5 * Ik2 ./ (2ρa .* Kk2 - Ik2 .^ 2) .^ 0.5
    Khk2 = 3 / 2 .+ uhk2 .^ 2                                     # `Khk2 = Khk2M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!()`. 
    vthk2 = (2 / 3 * (2 * Kk2 ./ ρa - (Ik2 ./ ρa) .^ 2)) .^ 0.5
    Rvthk2 = vthk2 ./ vthk1

    @show uhk2 ./ uhk2M .- 1
    @show vthk2 ./ vthk2M .- 1
    @show Rvthk2 - Rvthk2M

    # Updating the characteristic parameters of `f̂ₗᵐ(v̂)`: `x0k2 = [naik2,uaik2,vthik2]`
    nModk2 = copy(nModk)
    nMjMs01 = ceil.(Int, 3 / 2 * nModk2)
    naik2 = copy(naik1)      # `n̂a = naᵢ / na`
    uaik2 = copy(uaik1)      # `ûa = uaᵢ / vthk2`
    vthik2 = copy(vthik1)    # `v̂th = vathᵢ / vthk2`
    if prod(nModk2) == 1
        submoment!(naik2, uaik2, vthik2, fvL0k2, vGe, ns; Rvth=Rvthk2, is_renorm=false)
        # submoment!(naik2,uaik2,vthik2,nhk2,uhk2,ones(ns),ns)
        # submoment!(naik2,uaik2,vthik2,nhk2,uhk2.*Rvthk2,Rvthk2,ns;Rvth=Rvthk2)
        @show [naik2[isp][1] - 1 for isp in 1:ns]
        @show [uaik2[isp][1] - uhk2[isp] for isp in 1:ns]
        @show [vthik2[isp][1] - 1 for isp in 1:ns]
    else
        # submoment!(naik2, uaik2, vthik2, nModk2, fvL0k2, vGe, nhk2, uhk2, Khk2 / CjLL2(2), ns;
        #     is_nai_const=is_nai_const, Rvth=Rvthk2, is_renorm=true,
        #     optimizer=optimizer, factor=factor, autodiffs=autodiffs,
        #     is_Jacobian=is_Jacobian, show_trace=show_trace,
        #     maxIterKing=maxIterKing, p_tol=p_tol, f_tol=f_tol, g_tol=g_tol,
        #     p_noise_rel=p_noise_rel, p_noise_abs=p_noise_abs)
        submoment!(naik2,uaik2,vthik2,nModk2,fvL0k2,vGe,ns;
                   is_nai_const=is_nai_const,Rvth=Rvthk2,is_renorm=true,
                   optimizer=optimizer,factor=factor,autodiffs=autodiffs,
                   is_Jacobian=is_Jacobian,show_trace=show_trace,
                   maxIterKing=maxIterKing,p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                   p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
    end
    nhk2sum = [sum(naik2[k]) for k in 1:ns]
    Ihk2sum, vhthk2sum = zeros(ns), zeros(ns)
    nuTsNorm!(Ihk2sum, vhthk2sum, naik2, uaik2, vthik2)
    uhk2sum = Ihk2sum
    Khk2sum = 3 / 2 .+ uhk2sum .^ 2

    # Checking the constraint: `K̂a = 3/2 + ûa²`
    errKuhk2sum = 2 / 3 * (Khk2sum - Ihk2sum .^ 2) .- 1
    norm(errKuhk2sum) ≤ epsT100 || @warn("Number of meshgrids `nvG` or the time step `dt` may be not enough to satisfy the convergence of `K̂a_k2sum = 3/2 + ûa²`", errKuhk2sum)

    @show 33333, Rvthk2, errKuhk2sum

    println()
    @show nhk2sum .- 1
    @show Ihk2sum - uhk2M
    @show vhthk2sum .- 1

    Ik2sum = ρa .* vthk2 .* Ihk2sum                  # = ρa .* vthk .* M110k2[1,:] / 3
    Kk2sum = 1 / 2 * ρa .* vthk2 .^ 2 .* Khk2sum     # = 1/2 * ρa .* vthk.^2 .* M200k2[1,:]
    println()
    @show sum(Ik2sum) - sum(Ik2M)
    @show sum(Kk2sum) - sum(Kk2M)

    println()
    @show [naik2[k] ./ naik1[k] .- 1 for k in 1:ns]
    @show [uaik2[k] ./ uaik1[k] .- 1 for k in 1:ns]
    @show [vthik2[k] ./ vthik1[k] .- 1 for k in 1:ns]

    # # Updating the amplitude function of normalized distribution functions `f̂ₗᵐ(v̂)`  at the `kᵗʰ` step according parameters `naik2,uaik2,vthik2`.
    LMk2 = 0LMk1
    fvLk2 = zeros(nvG, LM1k1 + 1, ns)
    @show naik2 - nai
    LMk2, fvLk2 = fvLDMz(fvLk2, vGe, nvG, LMk2, ns, naik2, uaik2, vthik2, nModk2; L_limit=L_limit,
        rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_full_fvL=is_full_fvL)
    LM1k2 = maximum(LMk2) + 1

    # Checking the conservation laws of the renormalized distribution function `fvLk2`
    vthk2MM = vthk2
    if 1 == 1
        is_renormM = false
        M000k2M = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k2M, fvLk2[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M110k2M = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k2M, fvLk2[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        M200k2M = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k2M, fvLk2[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)

        # Checking the conservations
        nhk2MM = M000k2M[1, :]
        Ihk2MM = M110k2M[1, :] / 3
        Khk2MM = M200k2M[1, :]

        # Checking the constraint: `K̂a = 3/2 + ûa²`
        errKIhk2MM = 2 / 3 * (Khk2MM - Ihk2MM .^ 2) .- 1
        norm(errKIhk2MM) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k2M = 3/2 + ûa²`", errKIhk2MM)

        @show 333, Khk2MM, errKIhk2MM
    end

    if is_corrections[1] == false
        nak2MM = nak1 .* nhk2MM
    else
        nak2MM = nak1
    end
    ρa = ma .* nak2MM
    uhk2MM = Ihk2MM
    Ik2MM = ρa .* vthk2M .* Ihk2MM
    Kk2MM = 1 / 2 * ρa .* vthk2M .^ 2 .* Khk2MM
    println()
    @show sum(Ik2MM) - sum(Ik2M)
    @show sum(Kk2MM) - sum(Kk2M)
    
    uhk2MMc = 1.5^0.5 * Ik2MM ./ (2ρa .* Kk2MM - Ik2MM .^ 2) .^ 0.5
    Khk2MMc = 3 / 2 .+ uhk2 .^ 2                                      # `Khk2 = Khk2M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!(⋯)`.
    vthk2MM = (2 / 3 * (2 * Kk2MM ./ ρa - (Ik2MM ./ ρa) .^ 2)) .^ 0.5
    Rvthk2MM = vthk2MM ./ vthk1

    @show uhk2MM ./ uhk2M .- 1
    @show vthk2MM ./ vthk2M .- 1
    @show Rvthk2MM - Rvthk2M

    pfk2(isp3, L1) = plot(vGe, [fvL0e[:, L1, isp3] fvL0k2[:, L1, isp3]],
        line=(2, :auto), label=string("isp,L=", (isp3, L1 - 1)))
    pDtfk2(isp3, L1) = plot(vGe, fvL0k2[:, L1, isp3] - fvL0e[:, L1, isp3], line=(2, :auto), label=string("ΔₜfvL"))

    pfk2up(isp3, L1) = plot(vGe, [fvL0e[:, L1, isp3] fvLk2[:, L1, isp3]],
        line=(2, :auto), label=string("fvL_up"))
    pDtfk2up(isp3, L1) = plot(vGe, fvLk2[:, L1, isp3] - fvL0e[:, L1, isp3], line=(2, :auto), label=string("Δₜfᵏ⁺¹"))

    pfDtfk2(L1) = display(plot(pfk2(1, L1), pfk2(2, L1), pDtfk2(1, L1), pDtfk2(2, L1),
        pfk2up(1, L1), pfk2up(2, L1), pDtfk2up(1, L1), pDtfk2up(2, L1), layout=(4, 2)))
    pfDtfk2.(L1:LM1k2)

    # # Updating the FP collision terms according to the `FPS` operators.
    err_dtnIKk2 = 0.0
    fvL4k2 = copy(fvLk2)
    dtfvL0k2 = zero.(fvLk2)
    if is_dtfvLaa === 2
        @time dtfvL0k2, dtfvL0aak2, fvL4k2, err_dtnIKk2 = dtfvLSplineab(dtfvL0k2, fvL4k2, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak2, vthk2, naik2, uaik2, vthik2, LMk2, LM1k2, ns, nMod;
            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_dtfvLaa=is_dtfvLaa,
            is_normdtf=is_normdtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    else
        @time dtfvL0k2, fvL4k2, err_dtnIKk2 = dtfvLSplineab(dtfvL0k2, fvL4k2, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak2, vthk2, naik2, uaik2, vthik2, LMk2, LM1k2, ns, nMod;
            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_dtfvLaa=is_dtfvLaa,
            is_normdtf=is_normdtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    end

    #########  Optimizations according to `y(v̂→0) → Cₗᵐ` and `Rd1y(v̂→0) → 0`
    if is_optimdtfvL
        dtfvL0k2opt = copy(dtfvL0k2)
        nvc3k2 = deepcopy(nvc3k1)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
        optimdtfvL0e!(nvc3k2, dtfvL0k2opt, fvL4k2, vGe, nvG, ns, LMk2;
            orders=order_dvδtf, is_boundv0=is_boundv0,
            Nsmooth=Nsmooth, order_smooth=order_smooth, abstol_Rdy=abstol_Rdy,
            k=k_dtf, Nitp=Nitp, order_smooth_itp=order_smooth_itp, order_nvc_itp=order_nvc_itp,
            nvc0_limit=nvc0_limit, L1nvc_limit=L1nvc_limit)
        dtMsnnE3opt = zeros(datatype, njMs, LM1k2, ns)
        dtMsnnE3opt = MsnnEvens(dtMsnnE3opt, dtfvL0k2opt, vGe, njMs, LMk2, LM1k2, ns; is_renorm=is_renorm)
    else
        dtfvL0k2opt = dtfvL0k2
    end
    
    # dtMsnk2
    if 1 == 1
        is_renormM = false
        R000k2 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(R000k2, dtfvL0k2opt[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        R110k2 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(R110k2, dtfvL0k2opt[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)
        R200k2 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(R200k2, dtfvL0k2opt[:, L+1, :], vGe, j, L, ns; is_renorm=is_renormM)

        dtnhk2M = R000k2[1, :]
        RdtIk2M = R110k2[1, :] / 3    # = dtuhk2M + uhk2 .* Rdtvthk2M
        RdtKk2M = R200k2[1, :]        # = (2 * uhk2 .* dtuhk2M + (3 .+ 2 * uhk2.^2) .* Rdtvthk2M)

        # wk2M = R200k2[1,:] - 2 / 3 * uhk2 .* R110k2[1,:]
        wk2M = RdtKk2M - 2 * uhk2 .* RdtIk2M
        Rdtvthk2M = wk2M / 3
        dtvthk2M = vthk2 .* wk2M / 3               # wk2M = - (Rdtn - 3Rdtvth) = 3Rdtvth

        dtuhk2M = RdtIk2M - uhk2 .* Rdtvthk2M       # 

        # Checking the conservations according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
        errdtKuhk2M = RdtKk2M - (2 * uhk2 .* dtuhk2M + (3 .+ 2 * uhk2 .^ 2) .* Rdtvthk2M)
        norm(errdtKuhk2M) ≤ epsT1 || @warn("Number of meshgrids may be not enough to satisfy 
                     the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`", errdtKuhk2M)
        # @show  RdtKk2M ./ RdtIk2M.^2
    end
    dtIk2M = ρa .* vthk2 .* RdtIk2M
    dtKk2M = ρa .* vthk2 .^ 2 .* RdtKk2M / 2
    RerrdtKk2M = fmtf2(sum(dtKk2M) / abs(dtKk2M[1] - dtKk2M[2]))
    errdtIk2M = abs(dtIk2M[1] - dtIk2M[2])
    if errdtIk2M ≤ epsT
        RerrdtIk2M = fmtf2(errdtIk2M)
    else
        RerrdtIk2M = fmtf2(sum(dtIk2M) / errdtIk2M)
    end
    @show nnv, nvG
    @show fmtf2.(dtnhk2M)
    @show dtIk2M, RerrdtIk2M
    @show dtKk2M, RerrdtKk2M

    Rvthk3M = (1 .- 2 / 3 * dt * (dt * (dtuhk2M + uhk2 .* Rdtvthk2M) .^ 2 - wk2M)) .^ 0.5
    @show tkk, Rvthk3M

    # @show norm(dtfvL0k2 - dtfvL0e)
    # @show norm(dtfvL0k2opt - dtfvL0e)
end
# dtfvL0k2

1