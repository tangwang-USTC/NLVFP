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

k = 1
Rvthk = ones(ns) # The true values is decided by a iteration at the initial step and the previous step at the follow steps.
dt = 1e0      # (=1e-2 default), Which is normalized by `td ~ 1[s]` for MCF plasma.
               # `1e0 ~ 1e-5` is proposed to be used for most situation.
# tk
@show dt
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
Ik = ρa .* vthkM .* uhkM
Kk = 1/2 * ρa .* vthkM.^2 .* KhkM

# Updating the parameters `nhk, uhk, vthk` according to the first three moments `n, I, K` at `kᵗʰ` step
uhk = 1.5^0.5 * Ik ./ (2ρa .* Kk - Ik .^2).^0.5
vthk = (2/3 * (2Kk ./ ρa - (Ik ./ ρa).^2)).^0.5
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

# tk1: Applying the Explicit Newton method to solve the ODE equations of `f̂vL(v̂)`
fvL0k1 = fvL0 + dt * dtfvL0e
Rvthk1M = (1 .- 2/3 * dt * (dt * (dtuhkM + uhk .* RdtvthkM).^2 - wkM)).^0.5 

# Updating the thermal velocity `vₜₕ = vthk1` in single step
vthk1Mdt = vthk + dt * dtvthkM
vthk1M = vthk .* Rvthk1M

# # # Updating the normalized conservative momentums `nh, Ih, Kh`
if 1 == 1
    is_renormM = false
    M000k1 = zeros(2,ns)
    j, L = 0, 0
    MsnnEvens!(M000k1, fvL0k1[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)
    M110k1 = zeros(2,ns)
    j, L = 1, 1
    MsnnEvens!(M110k1, fvL0k1[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)
    M200k1 = zeros(2,ns)
    j, L = 2, 0
    MsnnEvens!(M200k1, fvL0k1[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)

    nhk1M = M000k1[1,:]
    Ihk1M = M110k1[1,:] / 3 ./ Rvthk1M
    Khk1M = M200k1[1,:] ./ Rvthk1M.^2

    # Checking the constraint: `K̂a = 3/2 + ûa²`
    errKIhk1M = 2/3 * (Khk1M - Ihk1M.^2) .- 1
    norm(errKIhk1M) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k1M = 3/2 + ûa²`",errKIhk1M)
end
if is_corrections[1] == false
    nak1M = nak .* nhk1M
else
    nak1M = nak
end
ρa = ma .* nak1M

uhk1M = Ihk1M
Ik1M = ρa .* vthk1M .* Ihk1M             # = ρa .* vthk .* M110k1[1,:] / 3
Kk1M = 1/2 * ρa .* vthk1M.^2 .* Khk1M    # = 1/2 * ρa .* vthk.^2 .* M200k1[1,:]

# uhk1M1 = 1.5^0.5 * Ik1M ./ (2ρa .* Kk1M - Ik1M .^2).^0.5
# vthk1M1 = (2/3 * (2Kk1M ./ ρa - (Ik1M ./ ρa).^2)).^0.5
# @show uhk1M1 - uhk1M
# @show vthk1M1 ./ vthk1M .- 1

Dnhk, DIk, DKk = nhk1M .- 1, Ik1M - Ik, Kk1M - Kk
RDIk = DIk ./ Ik
RDKk = DKk ./ Kk
RerrDnhk = nhkM .- 1
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
    @warn("Which denotes the meshgrids `nvG` or the time step `dt` may not be the best values, please checking the convergence!",RerrDIk)
end
if RerrDKk > epsT100
    @warn("Which denotes the meshgrids `nvG` or the time step `dt` may not be the best values, please checking the convergence!",RerrDKk)
end
@show fmtf2.(DKk), fmtf2.(DIk)
@show fmtf2.(RDIk), fmtf2.(RDKk)
@show fmtf2.(RerrDnhk)
# @show fmtf2.(RerrDIk)
# @show fmtf2.(RerrDKk)

nModk = copy(nMod)

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
                Ik1 = Ik + DIk
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
                Kk1 = Kk + DKk
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
uhk1 = 1.5^0.5 * Ik1 ./ (2ρa .* Kk1 - Ik1 .^2).^0.5
Khk1 = (3/2 .+ uhk1.^2) / CjLL2(2)    # `Khk1 = Khk1M / CjLL2(2)` when `is_renorm=true`.
vthk1 = (2/3 * (2 * Kk1 ./ ρa - (Ik1 ./ ρa).^2)).^0.5
Rvthk1 = vthk1 ./ vthk

@show uhk1 ./ uhk1M .- 1
@show vthk1 ./ vthk1M .- 1
@show Rvthk1 - Rvthk1M

# Updating the characteristic parameters of `f̂ₗᵐ(v̂)`: `x0k1 = [naik1,uaik1,vthik1]`
nModk1 = copy(nModk)
nMjMs01 = ceil.(Int,3 / 2 * nModk1)
naik1 = copy(nai)      # `n̂a = naᵢ / na`
uaik1 = copy(uai)      # `ûa = uaᵢ / vthk1`
vthik1 = copy(vthi)    # `v̂th = vathᵢ / vthk1`
if prod(nModk1) == 1
    submoment!(naik1,uaik1,vthik1,fvL0k1,vGe,ns;Rvth=Rvthk1,is_renorm=false)
    # submoment!(naik1,uaik1,vthik1,nhk1,uhk1,ones(ns),ns)
    # submoment!(naik1,uaik1,vthik1,nhk1,uhk1.*Rvthk1,Rvthk1,ns;Rvth=Rvthk1)
    @show [naik1[isp][1] - 1 for isp in 1:ns]
    @show [uaik1[isp][1] - uhk1[isp] for isp in 1:ns]
    @show [vthik1[isp][1] - 1 for isp in 1:ns]
else
    submoment!(naik1,uaik1,vthik1,nModk1,fvL0k1,vGe,nhk1,uhk1,Khk1,ns;
               is_nai_const=is_nai_const,Rvth=Rvthk1,is_renorm=true,
               optimizer=optimizer,factor=factor,autodiffs=autodiffs,
               is_Jacobian=is_Jacobian,show_trace=show_trace,
               maxIterKing=maxIterKing,p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
               p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
    # submoment!(naik1,uaik1,vthik1,nModk1,fvL0k1,vGe,ns;
    #            is_nai_const=is_nai_const,Rvth=Rvthk1,is_renorm=true,
    #            optimizer=optimizer,factor=factor,autodiffs=autodiffs,
    #            is_Jacobian=is_Jacobian,show_trace=show_trace,
    #            maxIterKing=maxIterKing,p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
    #            p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
end
nhk1sum = [sum(naik1[k]) for k in 1:ns]
uhk1sum, vhthk1sum = zeros(ns), zeros(ns)
nuTsNorm!(uhk1sum, vhthk1sum, naik1,uaik1,vthik1)

# Checking the constraint: `K̂a = 3/2 + ûa²`
errKuhk1sum = 2/3 * (Khk1M - uhk1sum.^2) .- 1
norm(errKuhk1sum) ≤ epsT100 || @warn("Number of meshgrids `nvG` or the time step `dt` may be not enough to satisfy the convergence of `K̂a_k1sum = 3/2 + ûa²`",errKuhk1sum)

println()
@show nhk1sum .- 1
@show uhk1sum - uhk1M
@show vhthk1sum .- 1

# println()
# if norm(uhsum) > epsT
#     @show uhk1sum ./ uhsum .- 1
# end
# @show vhthk1sum ./ vhthsum .- 1

# println()
@show [naik1[k] ./ nai[k] .- 1 for k in 1:ns]
@show [uaik1[k] ./ uai[k] .- 1 for k in 1:ns]
@show [vthik1[k] ./ vthi[k] .- 1 for k in 1:ns]

# # Updating the amplitude function of normalized distribution functions `f̂ₗᵐ(v̂)`  at the `kᵗʰ` step according parameters `naik1,uaik1,vthik1`.
LMk1 = 0LM
fvLk1 = zeros(nck,LM1+1,ns)
@show naik1 - nai
LMk1, fvLk1 = fvLDMz(fvLk1,vGk,nck,LMk1,ns,naik1,uaik1,vthik1,nModk1;L_limit=L_limit,
                 rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM,is_full_fvL=is_full_fvL) 
LM1k1 = maximum(LMk1) + 1

# Checking the conservation laws of the renormalized distribution function `fvLk1`
if 1 == 1
    is_renormM = false
    M000k1M = zeros(2,ns)
    j, L = 0, 0
    MsnnEvens!(M000k1M, fvLk1[nvlevele, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)
    M110k1M = zeros(2,ns)
    j, L = 1, 1
    MsnnEvens!(M110k1M, fvLk1[nvlevele, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)
    M200k1M = zeros(2,ns)
    j, L = 2, 0
    MsnnEvens!(M200k1M, fvLk1[nvlevele, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)

    # Checking the conservations
    nhk1MM = M000k1M[1,:]
    Ihk1MM = M110k1M[1,:] / 3
    Khk1MM = M200k1M[1,:]

    if is_corrections[1] == false
        nak1MM = nak .* nhk1MM
    else
        nak1MM = nak 
    end
    ρa = ma .* nak1MM

    uhk1MM = Ihk1MM
    Ik1MM = ρa .* vthk1M .* Ihk1MM
    Kk1MM = 1/2 * ρa .* vthk1M.^2 .* Khk1MM

    # Checking the constraint: `K̂a = 3/2 + ûa²`
    errKIhk1MM = 2/3 * (Khk1MM - Ihk1MM.^2) .- 1
    norm(errKIhk1MM) ≤ epsT100 || @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a_k1M = 3/2 + ûa²`",errKIhk1MM)
end

pfk1(isp3,L1) = plot(vGe,[fvL0e[:,L1,isp3] fvL0k1[:,L1,isp3]],
                    line=(2,:auto),label=string("isp,L=",(isp3,L1-1)))
pDtfk1(isp3,L1) = plot(vGe,fvL0k1[:,L1,isp3]-fvL0e[:,L1,isp3],line=(2,:auto),label=string("ΔₜfvL"))

pfk1up(isp3,L1) = plot(vGe,[fvL[nvlevele,L1,isp3]  fvLk1[nvlevele,L1,isp3]],
                    line=(2,:auto),label=string("fvL_up"))
pDtfk1up(isp3,L1)=plot(vGe,fvLk1[nvlevele,L1,isp3]-fvL0e[:,L1,isp3],line=(2,:auto),label=string("Δₜfᵏ⁺¹"))

pfDtfk1(L1) = display(plot(pfk1(1,L1),pfk1(2,L1),pDtfk1(1,L1),pDtfk1(2,L1),
                    pfk1up(1,L1),pfk1up(2,L1),pDtfk1up(1,L1),pDtfk1up(2,L1),layout=(4,2)))
pfDtfk1.(L1:LM1k1)

# # Updating the FP collision terms according to the `FPS` operators.
err_dtnIKk1 = 0.0
fvL4k1 = copy(fvL0k1)
dtfvL0k1 = zero.(fvL0k1)
if is_dtfvLaa === 2
    @time dtfvL0k1,dtfvL0aak1,fvL4k1,err_dtnIKk1 = dtfvLSplineab(dtfvL0k1,fvL4k1,vGk,
            nvG,nc0,nck,ocp,nvlevele0,nvlevel0,mu,Mμ,Mun,Mun1,Mun2,
            CΓ,εᵣ,ma,Zq,nak1,vthk1,naik1,uaik1,vthik1,LMk1,LM1k1,ns,nMod;
            isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_dtfvLaa=is_dtfvLaa,
            is_normdtf=is_normdtf,is_boundaryv0=is_boundaryv0,is_resetv0=is_resetv0)
else
    @time dtfvL0k1,fvL4k1,err_dtnIKk1 = dtfvLSplineab(dtfvL0k1,fvL4k1,vGk,
            nvG,nc0,nck,ocp,nvlevele0,nvlevel0,mu,Mμ,Mun,Mun1,Mun2,
            CΓ,εᵣ,ma,Zq,nak1,vthk1,naik1,uaik1,vthik1,LMk1,LM1k1,ns,nMod;
            isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_dtfvLaa=is_dtfvLaa,
            is_normdtf=is_normdtf,is_boundaryv0=is_boundaryv0,is_resetv0=is_resetv0)
end
@show norm(dtfvL0k1 - dtfvL0e)
# dtfvL0k1

1