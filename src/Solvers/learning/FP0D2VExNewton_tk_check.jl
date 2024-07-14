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

  ∂ₜf̂ₗᵐ = ℭ̂ₗᵐ + (3vth⁻¹∂ₜvth - na⁻¹∂ₜna)

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
is_optimδtfvL = false

Nt = 1e1
# tk
k = 1
tkk = t0
nModk = copy(nMod)
Rvthk = ones(ns) # The true values is decided by a iteration at the initial step and the previous step at the follow steps.
dt = Nt * dt0  # (=1e-2 default), Which is normalized by `td ~ 1[s]` for MCF plasma.
               # `1e0 ~ 1e-5` is proposed to be used for most situation.
println("------------------------------------------------")
println("------------------------------------------------")
@show (k,tkk,dt)
# Msnk
vthkM = copy(vth)
if 1 == 1
    M000k = zeros(2,ns)
    j, L = 0, 0
    MsnnEvens!(M000k, fvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=true)
    M110k = zeros(2,ns)
    j, L = 1, 1
    MsnnEvens!(M110k, fvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=true)
    M200k = zeros(2,ns)
    j, L = 2, 0
    MsnnEvens!(M200k, fvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=false)

    # Checking the constraint: `K̂a = 3/2 * T̂a + ûa²`
    nhkM = M000k[1,:]
    IhkM = M110k[1,:] #./ Rvthk
    KhkM = M200k[1,:] #./ Rvthk.^2
    ThkM = Rvthk.^2

    errThkM = 2/3 * (KhkM - IhkM.^2) .- ThkM
    norm(errThkM) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_kM = 3/2 + ûa²`",errThkM)
end
@show IhkM
@show KhkM
if is_corrections[1] == false
    nak = na .* nhkM
else
    nak = na
end
ρa = ma .* nak

IkM = ρa .* vthkM .* IhkM
KkM = 1/2 * ρa .* vthkM.^2 .* KhkM
println()
@show sum(IkM) - sum(Ia)
@show sum(KkM) - sum(Ka)

# Updating the parameters `nhk, Ihk, vthk` according to the first three moments `n, I, K` at `kᵗʰ` step
Ihk = 1.5^0.5 * IkM ./ (2ρa .* KkM - IkM .^2).^0.5
Khk = 3/2 .+ Ihk.^2
vthk = (2/3 * (2KkM ./ ρa - (IkM ./ ρa).^2)).^0.5

# Checking the constraint: `K̂a = 3/2 * T̂a + ûa²`
errThk = Khk - Ihk.^2 .- 1.5
norm(errThk) ≤ epsT100 || @warn("Number of meshgrids `nvG` or the time step `dtk` may be not enough to satisfy the convergence of `K̂a_k = 3/2 + ûa²`",errThk)
@show IhkM ./ Ihk .- 1
@show KhkM ./ Khk .- 1
@show vth ./ vthk .- 1

# Updating the the parameters:`naik, uaik, vthik`
nhksum = [sum(nai[k]) for k in 1:ns]
Ihksum, vhthksum = zeros(ns), zeros(ns)
nuTsNorm!(Ihksum, vhthksum, nai, uai, vthi)
Khksum = 3 / 2 .+ Ihksum .^ 2

# dtMsnk
if 1 == 1
    R000k = zeros(2,ns)
    j, L = 0, 0
    MsnnEvens!(R000k, δtfvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=true)
    R110k = zeros(2,ns)
    j, L = 1, 1
    MsnnEvens!(R110k, δtfvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=true)
    R200k = zeros(2,ns)
    j, L = 2, 0
    MsnnEvens!(R200k, δtfvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=false)

    dtnhkM = R000k[1,:]
    RdtIkM = R110k[1,:]        # = dtIhkM + Ihk .* RdtvthkM
    RdtKkM = R200k[1,:]        # = (2 * Ihk .* dtIhkM + (3 .+ 2 * Ihk.^2) .* RdtvthkM)

    # wkM = R200k[1,:] - 2 * Ihk .* (R110k[1,:] / 3)   # When `is_renorm == false`
    wkM = RdtKkM - 2 * Ihk .* RdtIkM
    RdtvthkM = wkM / 3
    # dtvthkM = vthk .* wkM / 3               # wkM = - (Rdtn - 3Rdtvth) = 3Rdtvth
    
    dtIhkM = RdtIkM - Ihk .* RdtvthkM       # 

    # Checking the conservations according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
    errdtKIhkM = RdtKkM - (2 * Ihk .* dtIhkM + (3 .+ 2 * Ihk.^2) .* RdtvthkM)
    norm(errdtKIhkM) ≤ epsT1 || @warn("Number of meshgrids may be not enough to satisfy 
                 the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`",errdtKIhkM)
    # @show  RdtKkM ./ RdtIkM.^2
end
dtIkM = ρa .* vthk .* RdtIkM
dtKkM = ρa .* vthk.^2 .* RdtKkM / 2
errdtIkM = abs(dtIkM[1] - dtIkM[2])
if errdtIkM ≤ epsT
    RerrdtIkM = fmtf2(errdtIkM)
else
    RerrdtIkM = fmtf2(sum(dtIkM) / errdtIkM)
end
RerrdtKkM = fmtf2(sum(dtKkM) / abs(dtKkM[1] - dtKkM[2]))
@show nnv,nvG
@show fmtf2.(dtnhkM)
@show dtIkM,RerrdtIkM
@show dtKkM,RerrdtKkM
@show errdtKIhkM

# The optimized `δtfvL`
if L1nvc_limit ≤ 2
end

Rvthk1M = (1 .- 2/3 * dt * (dt * (dtIhkM + Ihk .* RdtvthkM).^2 - wkM)).^0.5 

# tk1: Applying the Explicit Newton method to solve the ODE equations of `f̂vL(v̂)`
k += 1
if k ≥ 2
    tkk += dt
    fvL0k1 = fvL0e + dt * δtfvL0e

    # fvL0k1 = fvL0e + dt * (δtfvL0e + wkM * fvL0e)

    # fvL0k1 = dt * δtfvL0e
    # for isp in 1:ns
    #     fvL0k1[:,:,isp] += (1 + dt * wkM[isp]) * fvL0e[:,:,isp]
    # end

    # Updating the thermal velocity `vₜₕ = vthk1` in single step
    # vthk1Mdt = vthk + dt * dtvthkM
    vthk1M = vthk .* Rvthk1M          # The first inner iteration of `fvL0k1` is normalzied by  `vthk`.
    println("------------------------------------------------")
    println("------------------------------------------------")
    @show (k, tkk), Rvthk1M .- 1
    
    # # # Updating the normalized conservative momentums `nh, Ih, Kh`
    if 1 == 1
        M000k1 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k1, fvL0k1[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M110k1 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k1, fvL0k1[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M200k1 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k1, fvL0k1[:, L+1, :], vGe, j, L, ns; is_renorm=false)

        nhk1M = M000k1[1, :]
        Ihk1M = M110k1[1, :] #./ Rvthk1M
        Khk1M = M200k1[1, :] #./ Rvthk1M .^ 2
        @show M110k1[1, :] - M110k[1, :]
        Thk1M = Rvthk1M.^2

        # Checking the constraint: `K̂a = 3/2 + ûa²`
        errThk1M = 2 / 3 * (Khk1M -  Ihk1M .^ 2) .- Thk1M
        norm(errThk1M) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k1M = 3/2 + ûa²`", errThk1M)
    end
    @show Ihk1M
    @show Khk1M
    @show Ihk1M - Ihk
    @show Khk1M - Khk

    if is_corrections[1] == false
        nak1M = nak .* nhk1M
    else
        nak1M = nak
    end
    ρa = ma .* nak1M

    Ik1M = ρa .* vthk .* Ihk1M                 # = ρa .* vthk .* M110k1[1,:] / 3  # When `is_normal = false`
    Kk1M = 1 / 2 * ρa .* vthk .^ 2 .* Khk1M    # = 1/2 * ρa .* vthk.^2 .* M200k1[1,:]
    println()
    @show sum(Ik1M) - sum(IkM)
    @show sum(Kk1M) - sum(KkM)
    @show Kk1M
    @show KkM

    # Ihk1Mc = 1.5^0.5 * Ik1M ./ (2ρa .* Kk1M - Ik1M .^2).^0.5
    # vthk1Mc = (2/3 * (2Kk1M ./ ρa - (Ik1M ./ ρa).^2)).^0.5
    # Khk1Mc = 3/2 .+ Ihk1Mc.^2

    # @show Ihk1Mc - Ihk1M
    # @show Khk1Mc - Khk1M
    # @show vthk1Mc ./ vthk .- 1
    # @show vthk1Mc ./ vthk1M .- 1

    # Ik1M1 = ρa .* vthk1Mc .* Ihk1Mc
    # Kk1M1 = 1 / 2 * ρa .* vthk1Mc .^ 2 .* Khk1Mc
    # @show Ik1M1 - Ik1M
    # @show Kk1M1 ./ Kk1M .- 1
    
    if 1 == 1
        Dnhk, DIk, DKk = nhk1M .- 1, Ik1M - IkM, Kk1M - KkM
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
        RDIk = DIk ./ IkM
        RDKk = DKk ./ KkM
        RerrDnhk = nhk1M .- 1
        @show fmtf2.(sum(DKk)), fmtf2.(sum(DIk))
        @show fmtf2.(DKk), fmtf2.(DIk)
        @show fmtf2.(RDIk), fmtf2.(RDKk)
        @show fmtf2.(RerrDnhk)
        @show fmtf2.(RerrDIk)
        @show fmtf2.(RerrDKk)

        # Corrections to satisfy the conservation laws by applying a posterior analysis.
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

        # Updating the parameters `nhk1, Ihk1, vthk1` according to the first three moments `n, I, K` at `(k+1)ᵗʰ` step
        if is_corrections[1] == false
            nak1 = nak .* nhk1
        else
            nak1 = nak
        end
        ρa = ma .* nak1
    end
    Ihk1 = 1.5^0.5 * Ik1 ./ (2ρa .* Kk1 - Ik1 .^ 2) .^ 0.5
    Khk1 = 3 / 2 .+ Ihk1 .^ 2                                      # `Khk1 = Khk1M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!(⋯)`.
    vthk1 = (2 / 3 * (2 * Kk1 ./ ρa - (Ik1 ./ ρa) .^ 2)) .^ 0.5
    Rvthk1 = vthk1 ./ vthk

    @show Ihk1 ./ Ihk1M .- 1
    @show Khk1 ./ Khk1M .- 1
    @show vthk1 ./ vthk1M .- 1
    @show Rvthk1 - Rvthk1M
 
    # Updating the characteristic parameters of `f̂ₗᵐ(v̂)`: `x0k1 = [naik1,uaik1,vthik1]`
    nModk1 = copy(nModk)
    nMjMs01 = ceil.(Int, 3 / 2 * nModk1)
    naik1 = copy(nai)      # `n̂a = naᵢ / na`
    uaik1 = copy(uai)      # `ûa = uaᵢ / vthk1`
    vthik1 = copy(vthi)    # `v̂th = vathᵢ / vthk1`
    if prod(nModk1) == 1
        submoment!(naik1, uaik1, vthik1, fvL0k1, vGe, ns; Rvth=Rvthk1)
        # submoment!(naik1,uaik1,vthik1,nhk1,Ihk1,ones(ns),ns)
        # submoment!(naik1,uaik1,vthik1,nhk1,Ihk1.*Rvthk1,Rvthk1,ns;Rvth=Rvthk1)
        @show [naik1[isp][1] - 1 for isp in 1:ns]
        @show [uaik1[isp][1] - Ihk1[isp] for isp in 1:ns]
        @show [vthik1[isp][1] - 1 for isp in 1:ns]
    else
        # submoment!(naik1, uaik1, vthik1, nModk1, fvL0k1, vGe, nhk1, Ihk1, Khk1, ns;
        #     is_nai_const=is_nai_const, Rvth=Rvthk1, is_renorm=true,
        #     optimizer=optimizer, factor=factor, autodiff=autodiff,
        #     is_Jacobian=is_Jacobian, show_trace=show_trace,
        #     maxIterKing=maxIterKing, p_tol=p_tol, f_tol=f_tol, g_tol=g_tol,
        #     p_noise_rel=p_noise_rel, p_noise_abs=p_noise_abs)
        @show Rvthk1
        submoment!(naik1,uaik1,vthik1,nModk1,fvL0k1,vGe,ns;
                   is_nai_const=is_nai_const,Rvth=Rvthk1,
                   optimizer=optimizer,factor=factor,autodiff=autodiff,
                   is_Jacobian=is_Jacobian,show_trace=show_trace,
                   maxIterKing=maxIterKing,p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                   p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
    end
    nhk1sum = [sum(naik1[k]) for k in 1:ns]
    Ihk1sum, vhthk1sum = zeros(ns), zeros(ns)
    nuTsNorm!(Ihk1sum, vhthk1sum, naik1, uaik1, vthik1)
    Khk1sum = 3 / 2 .+ Ihk1sum .^ 2

    # Checking the constraint: `K̂a = 3/2 + ûa²`
    errThk1sum = 2 / 3 * (Khk1sum - Ihk1sum .^ 2) .- 1
    norm(errThk1sum) ≤ epsT100 || @warn("Number of meshgrids `nvG` or the time step `dt` may be not enough to satisfy the convergence of `K̂a_k1sum = 3/2 + ûa²`", errThk1sum)
    
    @show 2222, Rvthk1, errThk1sum

    println()
    @show nhk1sum .- 1
    @show Ihk1sum - Ihk1
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
        M000k1M = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k1M, fvLk1[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M110k1M = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k1M, fvLk1[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M200k1M = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k1M, fvLk1[:, L+1, :], vGe, j, L, ns; is_renorm=false)

        nhk1MM = M000k1M[1, :]
        Ihk1MM = M110k1M[1, :]
        Khk1MM = M200k1M[1, :]

        # Checking the constraint: `K̂a = 3/2 * T̂a + ûa²`
        errThk1MM = 2 / 3 * (Khk1MM - Ihk1MM .^ 2) .- 1
        norm(errThk1MM) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k1M = 3/2 + ûa²`", errThk1MM)

        @show 333, Khk1MM, errThk1MM
    end

    @show Ihk1MM ./ Ihk1M .- 1

    if is_corrections[1] == false
        nak1MM = nak .* nhk1MM
    else
        nak1MM = nak
    end
    ρa = ma .* nak1MM
    Ik1MM = ρa .* vthk1MM .* Ihk1MM
    Kk1MM = 1 / 2 * ρa .* vthk1MM .^ 2 .* Khk1MM
    println()
    @show sum(Ik1MM) - sum(Ik1M)
    @show sum(Kk1MM) - sum(Kk1M)
    
    Ihk1MMc = 1.5^0.5 * Ik1MM ./ (2ρa .* Kk1MM - Ik1MM .^ 2) .^ 0.5
    Khk1MMc = 3 / 2 .+ Ihk1 .^ 2                                      # `Khk1 = Khk1M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!(⋯)`.
    vthk1MM = (2 / 3 * (2 * Kk1MM ./ ρa - (Ik1MM ./ ρa) .^ 2)) .^ 0.5
    Rvthk1MM = vthk1MM ./ vthk

    @show vthk1MM ./ vthk1M .- 1
    @show Rvthk1MM - Rvthk1M

    @show Khk1MMc ./ Khk1M .- 1
    @show Ihk1MMc ./ Ihk1M .- 1

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
    δtfvL0k1 = zero.(fvLk1)
    if is_δtfvLaa === 2
        @time δtfvL0k1, δtfvL0aak1, fvL4k1, err_dtnIKk1 = dtfvLSplineab(δtfvL0k1, fvL4k1, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak1, vthk1, naik1, uaik1, vthik1, LMk1, LM1k1, ns, nModk1;
            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_δtfvLaa=is_δtfvLaa,
            is_normδtf=is_normδtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    else
        @time δtfvL0k1, fvL4k1, err_dtnIKk1 = dtfvLSplineab(δtfvL0k1, fvL4k1, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak1, vthk1, naik1, uaik1, vthik1, LMk1, LM1k1, ns, nModk1;
            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_δtfvLaa=is_δtfvLaa,
            is_normδtf=is_normδtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    end

    #########  Optimizations according to `y(v̂→0) → Cₗᵐ` and `Rd1y(v̂→0) → 0`
    if is_optimδtfvL
        δtfvL0k1opt = copy(δtfvL0k1)
        nvc3k1 = zeros(Int64, 2(order_smooth + 1), LM1k1, ns)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
        optimdtfvL0e!(nvc3k1, δtfvL0k1opt, fvL4k1, vGe, nvG, ns, LMk1;
            orders=order_dvδtf, is_boundv0=is_boundv0,
            Nsmooth=Nsmooth, order_smooth=order_smooth, abstol_Rdy=abstol_Rdy,
            k=k_δtf, Nitp=Nitp, order_smooth_itp=order_smooth_itp, order_nvc_itp=order_nvc_itp,
            nvc0_limit=nvc0_limit, L1nvc_limit=L1nvc_limit)
        dtMsnnE3opt = zeros(datatype, njMs, LM1k1, ns)
        dtMsnnE3opt = MsnnEvens(dtMsnnE3opt, δtfvL0k1opt, vGe, njMs, LMk1, LM1k1, ns; is_renorm=is_renorm)
    else
        δtfvL0k1opt = δtfvL0k1
    end

    # dtMsnk1
    if 1 == 1
        R000k1 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(R000k1, δtfvL0k1opt[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        R110k1 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(R110k1, δtfvL0k1opt[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        R200k1 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(R200k1, δtfvL0k1opt[:, L+1, :], vGe, j, L, ns; is_renorm=false)

        dtnhk1M = R000k1[1, :]
        RdtIk1M = R110k1[1, :]        # = dtIhk1M + Ihk1 .* Rdtvthk1M
        RdtKk1M = R200k1[1, :]        # = (2 * Ihk1 .* dtIhk1M + (3 .+ 2 * Ihk1.^2) .* Rdtvthk1M)

        wk1M = RdtKk1M - 2 * Ihk1 .* RdtIk1M
        Rdtvthk1M = wk1M / 3
        dtvthk1M = vthk1 .* wk1M / 3               # wk1M = - (Rdtn - 3Rdtvth) = 3Rdtvth

        dtIhk1M = RdtIk1M - Ihk1 .* Rdtvthk1M       # 

        # Checking the conservations according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
        errdtKIhk1M = RdtKk1M - (2 * Ihk1 .* dtIhk1M + (3 .+ 2 * Ihk1 .^ 2) .* Rdtvthk1M)
        norm(errdtKIhk1M) ≤ epsT1 || @warn("Number of meshgrids may be not enough to satisfy 
                     the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`", errdtKIhk1M)
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

    Rvthk2M = (1 .- 2 / 3 * dt * (dt * (dtIhk1M + Ihk1 .* Rdtvthk1M) .^ 2 - wk1M)) .^ 0.5
    @show tkk, Rvthk2M, Rvthk2M - Rvthk1M
end

fdgfgg
# tk2: Applying the Explicit Newton method to solve the ODE equations of `f̂vL(v̂)`
k += 1
if k ≥ 3
    tkk += dt
    fvL0k2 = fvLk1 + dt * δtfvL0k1

    # Updating the thermal velocity `vₜₕ = vthk2` in single step
    vthk2M = vthk1 .* Rvthk2M
    println("------------------------------------------------")
    println("------------------------------------------------")
    println("------------------------------------------------")
    @show (k, tkk), Rvthk2M

    # # # Updating the normalized conservative momentums `nh, Ih, Kh`
    if 1 == 1
        M000k2 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k2, fvL0k2[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M110k2 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k2, fvL0k2[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M200k2 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k2, fvL0k2[:, L+1, :], vGe, j, L, ns; is_renorm=false)

        nhk2M = M000k2[1, :]
        Ihk2M = M110k2[1, :] #./ Rvthk2M
        Khk2M = M200k2[1, :] #./ Rvthk2M .^ 2
        Thk2M = Rvthk2M.^2

        # Checking the constraint: `K̂a = 3/2 + ûa²`
        errThk2M = 2 / 3 * (Khk2M - Ihk2M .^ 2) .- Thk2M
        norm(errThk2M) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k2M = 3/2 + ûa²`", errThk2M)
    end
    @show Ihk2M - Ihk1
    @show Khk2M - Khk1

    if is_corrections[1] == false
        nak2M = nak1 .* nhk2M
    else
        nak2M = nak1
    end
    ρa = ma .* nak2M

    Ik2M = ρa .* vthk1 .* Ihk2M                 # = ρa .* vthk1 .* M110k2[1,:] / 3
    Kk2M = 1 / 2 * ρa .* vthk1 .^ 2 .* Khk2M    # = 1/2 * ρa .* vthk1.^2 .* M200k2[1,:]
    println()
    @show sum(Ik2M) - sum(Ik1M)
    @show sum(Kk2M) - sum(Kk1M)

    Ihk2Mc = 1.5^0.5 * Ik2M ./ (2ρa .* Kk2M - Ik2M .^2).^0.5
    vthk2Mc = (2/3 * (2Kk2M ./ ρa - (Ik2M ./ ρa).^2)).^0.5
    Khk2Mc = 3/2 .+ Ihk2Mc.^2

    @show Ihk2Mc - Ihk2M
    @show Khk2Mc - Khk2M
    @show vthk2Mc ./ vthk1 .- 1
    @show vthk2Mc ./ vthk2M .- 1

    Ik2M1 = ρa .* vthk2Mc .* Ihk2Mc
    Kk2M1 = 1 / 2 * ρa .* vthk2Mc .^ 2 .* Khk2Mc
    @show Ik2M1 - Ik2M
    @show Kk2M1 ./ Kk2M .- 1

    # Examining the deviations of the conservation moments  and taking corrections.
    if 1 == 1
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
    
        # Updating the parameters `nhk2, Ihk2, vthk2` according to the first three moments `n, I, K` at `(k+1)ᵗʰ` step
        if is_corrections[1] == false
            nak2 = nak1 .* nhk2
        else
            nak2 = nak1
        end
        ρa = ma .* nak2
    end
    Ihk2 = 1.5^0.5 * Ik2 ./ (2ρa .* Kk2 - Ik2 .^ 2) .^ 0.5
    Khk2 = 3 / 2 .+ Ihk2 .^ 2                                     # `Khk2 = Khk2M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!()`. 
    vthk2 = (2 / 3 * (2 * Kk2 ./ ρa - (Ik2 ./ ρa) .^ 2)) .^ 0.5
    Rvthk2 = vthk2 ./ vthk1

    # Checking the constraint: `K̂a = 3/2 * T̂a + ûa²`
    errThk2 = Khk2 - Ihk2.^2 .- 1.5
    norm(errThk2) ≤ epsT100 || @warn("Number of meshgrids `nvG` or the time step `dtk` may be not enough to satisfy the convergence of `K̂a_k1 = 3/2 + ûa²`",errThk1)
    

    @show Ihk2 ./ Ihk2M .- 1
    @show vthk2 ./ vthk2M .- 1
    @show Rvthk2 - Rvthk2M

    # Updating the characteristic parameters of `f̂ₗᵐ(v̂)`: `x0k2 = [naik2,uaik2,vthik2]`
    nModk2 = copy(nModk1)
    nMjMs01 = ceil.(Int, 3 / 2 * nModk2)
    naik2 = copy(naik1)      # `n̂a = naᵢ / na`
    uaik2 = copy(uaik1)      # `ûa = uaᵢ / vthk2`
    vthik2 = copy(vthik1)    # `v̂th = vathᵢ / vthk2`
    if prod(nModk2) == 1
        submoment!(naik2, uaik2, vthik2, fvL0k2, vGe, ns; Rvth=Rvthk2)
        # submoment!(naik2,uaik2,vthik2,nhk2,Ihk2,ones(ns),ns)
        # submoment!(naik2,uaik2,vthik2,nhk2,Ihk2.*Rvthk2,Rvthk2,ns;Rvth=Rvthk2)
        @show [naik2[isp][1] - 1 for isp in 1:ns]
        @show [uaik2[isp][1] - Ihk2[isp] for isp in 1:ns]
        @show [vthik2[isp][1] - 1 for isp in 1:ns]
    else
        # submoment!(naik2, uaik2, vthik2, nModk2, fvL0k2, vGe, nhk2, Ihk2, Khk2 / CjLL2(2), ns;
        #     is_nai_const=is_nai_const, Rvth=Rvthk2, 
        #     optimizer=optimizer, factor=factor, autodiff=autodiff,
        #     is_Jacobian=is_Jacobian, show_trace=show_trace,
        #     maxIterKing=maxIterKing, p_tol=p_tol, f_tol=f_tol, g_tol=g_tol,
        #     p_noise_rel=p_noise_rel, p_noise_abs=p_noise_abs)
        submoment!(naik2,uaik2,vthik2,nModk2,fvL0k2,vGe,ns;
                   is_nai_const=is_nai_const,Rvth=Rvthk2,
                   optimizer=optimizer,factor=factor,autodiff=autodiff,
                   is_Jacobian=is_Jacobian,show_trace=show_trace,
                   maxIterKing=maxIterKing,p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                   p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
    end
    nhk2sum = [sum(naik2[k]) for k in 1:ns]
    Ihk2sum, vhthk2sum = zeros(ns), zeros(ns)
    nuTsNorm!(Ihk2sum, vhthk2sum, naik2, uaik2, vthik2)
    Khk2sum = 3 / 2 .+ Ihk2sum .^ 2

    # Checking the constraint: `K̂a = 3/2 + ûa²`
    errThk2sum = 2 / 3 * (Khk2sum - Ihk2sum .^ 2) .- 1
    norm(errThk2sum) ≤ epsT100 || @warn("Number of meshgrids `nvG` or the time step `dt` may be not enough to satisfy the convergence of `K̂a_k2sum = 3/2 + ûa²`", errThk2sum)

    @show 33333, Rvthk2, errThk2sum

    println()
    @show nhk2sum .- 1
    @show Ihk2sum - Ihk2
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
        M000k2M = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k2M, fvLk2[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M110k2M = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k2M, fvLk2[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M200k2M = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k2M, fvLk2[:, L+1, :], vGe, j, L, ns; is_renorm=false)

        nhk2MM = M000k2M[1, :]
        Ihk2MM = M110k2M[1, :]
        Khk2MM = M200k2M[1, :]

        # Checking the constraint: `K̂a = 3/2 * T̂a + ûa²`
        errThk2MM = 2 / 3 * (Khk2MM - Ihk2MM .^ 2) .- 1
        norm(errThk2MM) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k2M = 3/2 + ûa²`", errThk2MM)

        @show 333, Khk2MM, errThk2MM
    end

    @show Ihk2MM ./ Ihk2M .- 1

    if is_corrections[1] == false
        nak2MM = nak1 .* nhk2MM
    else
        nak2MM = nak1
    end
    ρa = ma .* nak2MM
    Ik2MM = ρa .* vthk2M .* Ihk2MM
    Kk2MM = 1 / 2 * ρa .* vthk2M .^ 2 .* Khk2MM
    println()
    @show sum(Ik2MM) - sum(Ik2M)
    @show sum(Kk2MM) - sum(Kk2M)
    
    Ihk2MMc = 1.5^0.5 * Ik2MM ./ (2ρa .* Kk2MM - Ik2MM .^ 2) .^ 0.5
    Khk2MMc = 3 / 2 .+ Ihk2 .^ 2                                      # `Khk2 = Khk2M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!(⋯)`.

    # Updating the parameters `RvthkMM`
    vthk2MM = (2 / 3 * (2 * Kk2MM ./ ρa - (Ik2MM ./ ρa) .^ 2)) .^ 0.5
    Rvthk2MM = vthk2MM ./ vthk1

    @show Ihk2MMc ./ Ihk2M .- 1
    @show Khk2MMc ./ Khk2M .- 1

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
    δtfvL0k2 = zero.(fvLk2)
    if is_δtfvLaa === 2
        @time δtfvL0k2, δtfvL0aak2, fvL4k2, err_dtnIKk2 = dtfvLSplineab(δtfvL0k2, fvL4k2, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak2, vthk2, naik2, uaik2, vthik2, LMk2, LM1k2, ns, nModk2;
            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_δtfvLaa=is_δtfvLaa,
            is_normδtf=is_normδtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    else
        @time δtfvL0k2, fvL4k2, err_dtnIKk2 = dtfvLSplineab(δtfvL0k2, fvL4k2, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak2, vthk2, naik2, uaik2, vthik2, LMk2, LM1k2, ns, nModk2;
            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_δtfvLaa=is_δtfvLaa,
            is_normδtf=is_normδtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    end

    #########  Optimizations according to `y(v̂→0) → Cₗᵐ` and `Rd1y(v̂→0) → 0`
    if is_optimδtfvL
        δtfvL0k2opt = copy(δtfvL0k2)
        nvc3k2 = deepcopy(nvc3k1)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
        optimdtfvL0e!(nvc3k2, δtfvL0k2opt, fvL4k2, vGe, nvG, ns, LMk2;
            orders=order_dvδtf, is_boundv0=is_boundv0,
            Nsmooth=Nsmooth, order_smooth=order_smooth, abstol_Rdy=abstol_Rdy,
            k=k_δtf, Nitp=Nitp, order_smooth_itp=order_smooth_itp, order_nvc_itp=order_nvc_itp,
            nvc0_limit=nvc0_limit, L1nvc_limit=L1nvc_limit)
        dtMsnnE3opt = zeros(datatype, njMs, LM1k2, ns)
        dtMsnnE3opt = MsnnEvens(dtMsnnE3opt, δtfvL0k2opt, vGe, njMs, LMk2, LM1k2, ns; is_renorm=is_renorm)
    else
        δtfvL0k2opt = δtfvL0k2
    end
    
    # dtMsnk2
    if 1 == 1
        R000k2 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(R000k2, δtfvL0k2opt[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        R110k2 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(R110k2, δtfvL0k2opt[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        R200k2 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(R200k2, δtfvL0k2opt[:, L+1, :], vGe, j, L, ns; is_renorm=false)

        dtnhk2M = R000k2[1, :]
        RdtIk2M = R110k2[1, :]        # = dtIhk2M + Ihk2 .* Rdtvthk2M
        RdtKk2M = R200k2[1, :]        # = (2 * Ihk2 .* dtIhk2M + (3 .+ 2 * Ihk2.^2) .* Rdtvthk2M)

        # wk2M = R200k2[1,:] - 2 / 3 * Ihk2 .* R110k2[1,:]
        wk2M = RdtKk2M - 2 * Ihk2 .* RdtIk2M
        Rdtvthk2M = wk2M / 3
        dtvthk2M = vthk2 .* wk2M / 3               # wk2M = - (Rdtn - 3Rdtvth) = 3Rdtvth

        dtIhk2M = RdtIk2M - Ihk2 .* Rdtvthk2M       # 

        # Checking the conservations according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
        errdtKIhk2M = RdtKk2M - (2 * Ihk2 .* dtIhk2M + (3 .+ 2 * Ihk2 .^ 2) .* Rdtvthk2M)
        norm(errdtKIhk2M) ≤ epsT1 || @warn("Number of meshgrids may be not enough to satisfy 
                     the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`", errdtKIhk2M)
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

    Rvthk3M = (1 .- 2 / 3 * dt * (dt * (dtIhk2M + Ihk2 .* Rdtvthk2M) .^ 2 - wk2M)) .^ 0.5
    @show tkk, Rvthk3M

    # @show norm(δtfvL0k2 - δtfvL0e)
    # @show norm(δtfvL0k2opt - δtfvL0e)
end

# tk3: Applying the Explicit Newton method to solve the ODE equations of `f̂vL(v̂)`
k += 1
if k ≥ 4
    tkk += dt
    fvL0k3 = fvLk2 + dt * δtfvL0k2

    # Updating the thermal velocity `vₜₕ = vthk3` in single step
    vthk3M = vthk2 .* Rvthk3M
    println("------------------------------------------------")
    println("------------------------------------------------")
    println("------------------------------------------------")
    @show (k, tkk), Rvthk3M

    # # # Updating the normalized conservative momentums `nh, Ih, Kh`
    if 1 == 1
        M000k3 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k3, fvL0k3[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M110k3 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k3, fvL0k3[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M200k3 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k3, fvL0k3[:, L+1, :], vGe, j, L, ns; is_renorm=false)

        nhk3M = M000k3[1, :]
        Ihk3M = M110k3[1, :] #./ Rvthk3M
        Khk3M = M200k3[1, :] #./ Rvthk3M .^ 2
        Thk3M = Rvthk3M.^2

        # Checking the constraint: `K̂a = 3/2 + ûa²`
        errThk3M = 2 / 3 * (Khk3M - Ihk3M .^ 2) .- Thk3M
        norm(errThk3M) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k3M = 3/2 + ûa²`", errThk3M)
    end
    @show Ihk3M - Ihk2
    @show Khk3M - Khk2

    if is_corrections[1] == false
        nak3M = nak2 .* nhk3M
    else
        nak3M = nak2
    end
    ρa = ma .* nak3M

    Ik3M = ρa .* vthk2 .* Ihk3M                 # = ρa .* vthk2 .* M110k3[1,:] / 3
    Kk3M = 1 / 2 * ρa .* vthk2 .^ 2 .* Khk3M    # = 1/2 * ρa .* vthk2.^2 .* M200k3[1,:]
    println()
    @show sum(Ik3M) - sum(Ik2M)
    @show sum(Kk3M) - sum(Kk2M)

    Ihk3Mc = 1.5^0.5 * Ik3M ./ (2ρa .* Kk3M - Ik3M .^2).^0.5
    vthk3Mc = (2/3 * (2Kk3M ./ ρa - (Ik3M ./ ρa).^2)).^0.5
    Khk3Mc = 3/2 .+ Ihk3Mc.^2

    @show Ihk3Mc - Ihk3M
    @show Khk3Mc - Khk3M
    @show vthk3Mc ./ vthk2 .- 1
    @show vthk3Mc ./ vthk3M .- 1

    Ik3M1 = ρa .* vthk3Mc .* Ihk3Mc
    Kk3M1 = 1 / 2 * ρa .* vthk3Mc .^ 2 .* Khk3Mc
    @show Ik3M1 - Ik3M
    @show Kk3M1 ./ Kk3M .- 1

    # Examining the deviations of the conservation moments  and taking corrections.
    if 1 == 1
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
    
        # Updating the parameters `nhk3, Ihk3, vthk3` according to the first three moments `n, I, K` at `(k+1)ᵗʰ` step
        if is_corrections[1] == false
            nak3 = nak2 .* nhk3
        else
            nak3 = nak2
        end
        ρa = ma .* nak3
    end
    Ihk3 = 1.5^0.5 * Ik3 ./ (2ρa .* Kk3 - Ik3 .^ 2) .^ 0.5
    Khk3 = 3 / 2 .+ Ihk3 .^ 2                                     # `Khk3 = Khk3M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!()`. 
    vthk3 = (2 / 3 * (2 * Kk3 ./ ρa - (Ik3 ./ ρa) .^ 2)) .^ 0.5
    Rvthk3 = vthk3 ./ vthk2

    # Checking the constraint: `K̂a = 3/2 * T̂a + ûa²`
    errThk3 = Khk3 - Ihk3.^2 .- 1.5
    norm(errThk3) ≤ epsT100 || @warn("Number of meshgrids `nvG` or the time step `dtk` may be not enough to satisfy the convergence of `K̂a_k2 = 3/2 + ûa²`",errThk2)
    

    @show Ihk3 ./ Ihk3M .- 1
    @show vthk3 ./ vthk3M .- 1
    @show Rvthk3 - Rvthk3M

    # Updating the characteristic parameters of `f̂ₗᵐ(v̂)`: `x0k3 = [naik3,uaik3,vthik3]`
    nModk3 = copy(nModk2)
    nMjMs01 = ceil.(Int, 3 / 2 * nModk3)
    naik3 = copy(naik2)      # `n̂a = naᵢ / na`
    uaik3 = copy(uaik2)      # `ûa = uaᵢ / vthk3`
    vthik3 = copy(vthik2)    # `v̂th = vathᵢ / vthk3`
    if prod(nModk3) == 1
        submoment!(naik3, uaik3, vthik3, fvL0k3, vGe, ns; Rvth=Rvthk3)
        # submoment!(naik3,uaik3,vthik3,nhk3,Ihk3,ones(ns),ns)
        # submoment!(naik3,uaik3,vthik3,nhk3,Ihk3.*Rvthk3,Rvthk3,ns;Rvth=Rvthk3)
        @show [naik3[isp][1] - 1 for isp in 1:ns]
        @show [uaik3[isp][1] - Ihk3[isp] for isp in 1:ns]
        @show [vthik3[isp][1] - 1 for isp in 1:ns]
    else
        # submoment!(naik3, uaik3, vthik3, nModk3, fvL0k3, vGe, nhk3, Ihk3, Khk3 / CjLL2(2), ns;
        #     is_nai_const=is_nai_const, Rvth=Rvthk3, 
        #     optimizer=optimizer, factor=factor, autodiff=autodiff,
        #     is_Jacobian=is_Jacobian, show_trace=show_trace,
        #     maxIterKing=maxIterKing, p_tol=p_tol, f_tol=f_tol, g_tol=g_tol,
        #     p_noise_rel=p_noise_rel, p_noise_abs=p_noise_abs)
        submoment!(naik3,uaik3,vthik3,nModk3,fvL0k3,vGe,ns;
                   is_nai_const=is_nai_const,Rvth=Rvthk3,
                   optimizer=optimizer,factor=factor,autodiff=autodiff,
                   is_Jacobian=is_Jacobian,show_trace=show_trace,
                   maxIterKing=maxIterKing,p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
                   p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
    end
    nhk3sum = [sum(naik3[k]) for k in 1:ns]
    Ihk3sum, vhthk3sum = zeros(ns), zeros(ns)
    nuTsNorm!(Ihk3sum, vhthk3sum, naik3, uaik3, vthik3)
    Khk3sum = 3 / 2 .+ Ihk3sum .^ 2

    # Checking the constraint: `K̂a = 3/2 + ûa²`
    errThk3sum = 2 / 3 * (Khk3sum - Ihk3sum .^ 2) .- 1
    norm(errThk3sum) ≤ epsT100 || @warn("Number of meshgrids `nvG` or the time step `dt` may be not enough to satisfy the convergence of `K̂a_k3sum = 3/2 + ûa²`", errThk3sum)

    @show 33333, Rvthk3, errThk3sum

    println()
    @show nhk3sum .- 1
    @show Ihk3sum - Ihk3
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
        M000k3M = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(M000k3M, fvLk3[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M110k3M = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(M110k3M, fvLk3[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        M200k3M = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(M200k3M, fvLk3[:, L+1, :], vGe, j, L, ns; is_renorm=false)

        nhk3MM = M000k3M[1, :]
        Ihk3MM = M110k3M[1, :]
        Khk3MM = M200k3M[1, :]

        # Checking the constraint: `K̂a = 3/2 * T̂a + ûa²`
        errThk3MM = 2 / 3 * (Khk3MM - Ihk3MM .^ 2) .- 1
        norm(errThk3MM) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k3M = 3/2 + ûa²`", errThk3MM)

        @show 333, Khk3MM, errThk3MM
    end

    @show Ihk3MM ./ Ihk3M .- 1

    if is_corrections[1] == false
        nak3MM = nak2 .* nhk3MM
    else
        nak3MM = nak2
    end
    ρa = ma .* nak3MM
    Ik3MM = ρa .* vthk3M .* Ihk3MM
    Kk3MM = 1 / 2 * ρa .* vthk3M .^ 2 .* Khk3MM
    println()
    @show sum(Ik3MM) - sum(Ik3M)
    @show sum(Kk3MM) - sum(Kk3M)
    
    Ihk3MMc = 1.5^0.5 * Ik3MM ./ (2ρa .* Kk3MM - Ik3MM .^ 2) .^ 0.5
    Khk3MMc = 3 / 2 .+ Ihk3 .^ 2                                      # `Khk3 = Khk3M / CjLL2(2)` when `is_renorm=true` in procedure `submoment!(⋯)`.

    # Updating the parameters `RvthkMM`
    vthk3MM = (2 / 3 * (2 * Kk3MM ./ ρa - (Ik3MM ./ ρa) .^ 2)) .^ 0.5
    Rvthk3MM = vthk3MM ./ vthk2

    @show Ihk3MMc ./ Ihk3M .- 1
    @show Khk3MMc ./ Khk3M .- 1

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
    δtfvL0k3 = zero.(fvLk3)
    if is_δtfvLaa === 2
        @time δtfvL0k3, δtfvL0aak3, fvL4k3, err_dtnIKk3 = dtfvLSplineab(δtfvL0k3, fvL4k3, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak3, vthk3, naik3, uaik3, vthik3, LMk3, LM1k3, ns, nModk3;
            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_δtfvLaa=is_δtfvLaa,
            is_normδtf=is_normδtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    else
        @time δtfvL0k3, fvL4k3, err_dtnIKk3 = dtfvLSplineab(δtfvL0k3, fvL4k3, vGk,
            nvG, nc0, nck, ocp, nvlevele0, nvlevel0, mu, Mμ, Mun, Mun1, Mun2,
            CΓ, εᵣ, ma, Zq, nak3, vthk3, naik3, uaik3, vthik3, LMk3, LM1k3, ns, nModk3;
            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_δtfvLaa=is_δtfvLaa,
            is_normδtf=is_normδtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
    end

    #########  Optimizations according to `y(v̂→0) → Cₗᵐ` and `Rd1y(v̂→0) → 0`
    if is_optimδtfvL
        δtfvL0k3opt = copy(δtfvL0k3)
        nvc3k3 = deepcopy(nvc3k2)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
        optimdtfvL0e!(nvc3k3, δtfvL0k3opt, fvL4k3, vGe, nvG, ns, LMk3;
            orders=order_dvδtf, is_boundv0=is_boundv0,
            Nsmooth=Nsmooth, order_smooth=order_smooth, abstol_Rdy=abstol_Rdy,
            k=k_δtf, Nitp=Nitp, order_smooth_itp=order_smooth_itp, order_nvc_itp=order_nvc_itp,
            nvc0_limit=nvc0_limit, L1nvc_limit=L1nvc_limit)
        dtMsnnE3opt = zeros(datatype, njMs, LM1k3, ns)
        dtMsnnE3opt = MsnnEvens(dtMsnnE3opt, δtfvL0k3opt, vGe, njMs, LMk3, LM1k3, ns; is_renorm=is_renorm)
    else
        δtfvL0k3opt = δtfvL0k3
    end
    
    # dtMsnk3
    if 1 == 1
        R000k3 = zeros(2, ns)
        j, L = 0, 0
        MsnnEvens!(R000k3, δtfvL0k3opt[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        R110k3 = zeros(2, ns)
        j, L = 1, 1
        MsnnEvens!(R110k3, δtfvL0k3opt[:, L+1, :], vGe, j, L, ns; is_renorm=true)
        R200k3 = zeros(2, ns)
        j, L = 2, 0
        MsnnEvens!(R200k3, δtfvL0k3opt[:, L+1, :], vGe, j, L, ns; is_renorm=false)

        dtnhk3M = R000k3[1, :]
        RdtIk3M = R110k3[1, :]        # = dtIhk3M + Ihk3 .* Rdtvthk3M
        RdtKk3M = R200k3[1, :]        # = (2 * Ihk3 .* dtIhk3M + (3 .+ 2 * Ihk3.^2) .* Rdtvthk3M)

        # wk3M = R200k3[1,:] - 2 / 3 * Ihk3 .* R110k3[1,:]
        wk3M = RdtKk3M - 2 * Ihk3 .* RdtIk3M
        Rdtvthk3M = wk3M / 3
        dtvthk3M = vthk3 .* wk3M / 3               # wk3M = - (Rdtn - 3Rdtvth) = 3Rdtvth

        dtIhk3M = RdtIk3M - Ihk3 .* Rdtvthk3M       # 

        # Checking the conservations according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
        errdtKIhk3M = RdtKk3M - (2 * Ihk3 .* dtIhk3M + (3 .+ 2 * Ihk3 .^ 2) .* Rdtvthk3M)
        norm(errdtKIhk3M) ≤ epsT1 || @warn("Number of meshgrids may be not enough to satisfy 
                     the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`", errdtKIhk3M)
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

    Rvthk3M = (1 .- 2 / 3 * dt * (dt * (dtIhk3M + Ihk3 .* Rdtvthk3M) .^ 2 - wk3M)) .^ 0.5
    @show tkk, Rvthk3M

    # @show norm(δtfvL0k3 - δtfvL0e)
    # @show norm(δtfvL0k3opt - δtfvL0e)
end
# δtfvL0k3

1