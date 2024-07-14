"""
  Applying the fixed time step explicit Newton method to solve the ODE problems.
    The fixed time step is decided by the characteristic time scale of the relaxation process and
    the relative rate of change of the moments respect to time.

    `RdtM = M⁻¹Δₜ∂ₜM ≪ 1` and is `≤ 1e-3` defaultly.
  
  Updating `vth` and `ûa` according to the values of `Ka` and `Ia` as:

    vth = √(2/3 * (2Ka / ρa - (Ia / ρa)^2))
    ûa = √(3/2) × Ia / √(2ρa * Ka - Ia^2)

  Constraints:

    K̂a = 3/2 + ûa

  1

  Inputs:
    residualMethod_FP0D::Int ∈ [1,2]. When `residualMethod_FP0D=1` denotes absorbing the residuals by using the "dichotomy method"; or else
                                     `residualMethod_FP0D=2` denotes absorbing the residuals with a geometric ratio  等比残差吸收
    1
  Outouts:

"""
is_remesh_vG = false

# tk
nhk, Ik, Kk = nIKs(fvL0e[:,1:2,:],vGe,ma,na,vth,ns)
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

    
    nhkM = M000k[1,:]
    uhkM = M110k[1,:] / 3
    KhkM = M200k[1,:]
    @show KhkM ./ (3/2 .+ uhkM.^2) .- 1
end
if is_corrections[1] == false
    nak = na .* nhk
else
    nak = na 
end
# dtMsn
if 1 == 2
    dtnhk, dtIk, dtKk = nIKs(δtfvL0[:,1:2,:],vGe,ma,na,vth,ns)
else
    is_renormM = false
    R000k = zeros(2,ns)
    j, L = 0, 0
    MsnnEvens!(R000k, δtfvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)
    R110k = zeros(2,ns)
    j, L = 1, 1
    MsnnEvens!(R110k, δtfvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)
    R200k = zeros(2,ns)
    j, L = 2, 0
    MsnnEvens!(R200k, δtfvL0e[:, L+1,:], vGe, j, L, ns; is_renorm=is_renormM)

    dtnhkM = R000k[1,:]
    dtuhkM = R110k[1,:] / 3
    dtKhkM = R200k[1,:]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    ΔKhkM = dtKhkM - (2 * uhkM + dt * dtuhkM) .* dtuhkM
    ΔKhkM = dtKhkM - (2 * uhkM + Dt .* dtuhkM) .* dtuhkM


    @show  dtKhkM ./ dtuhkM.^2
    @show  dtKhkM ./ (2 * uhkM .* dtuhkM)

    wk = R200k[1,:] - 2 / 3 * uh .* R110k[1,:]
    Rdtvthk = wk / 3
    DDDt = 1e-6
    RvthDt = 1 .- 2/3 *DDDt * (DDDt * (dtuhkM + uhk .* wk / 3).^2 - wk)
    DDt = wk ./ (dtuhkM + uhk .* wk / 3) .^2       # DDt → ∞
    dtvthk = vth .* wk / 3                         # Wk = - (Rdtn - 3Rdtvth) = 3Rdtvth
    # Checking the conservations
    ρa = ma .* nak
    dtnhk = R000k[1,:]
    dtIk = ρa .* vth .* R110k[1,:] * (1 / 3)
    dtKk = na .* Ta .* R200k[1,:]
end
RerrdtKk = fmtf2(sum(dtKk) / abs(dtKk[1] - dtKk[2]))
errdtIk = abs(dtIk[1] - dtIk[2])
if errdtIk ≤ epsT
    RerrdtIk = fmtf2(errdtIk)
else
    RerrdtIk = fmtf2(sum(dtIk) / errdtIk)
end
# Updating the parameters `nh, uh, vth` according to the first three moments `n, I, K` at `k` step

uhk = 1.5^0.5 * Ik ./ (2ρa .* Kk - Ik .^2).^0.5
vthk = (2/3 * (2Kk ./ ρa - (Ik ./ ρa).^2)).^0.5
@show nnv,nvG
@show fmtf2.(dtnhk)
@show dtIk,RerrdtIk
@show dtKk,RerrdtKk
@show uh ./ uhk .- 1
@show vth ./ vthk .- 1
# The optimized `δtfvL`
if L1nvc_limit ≤ 2
end

# tk1: Applying the Explicit Newton method to solve the ODE equations of `f̂vL(v̂)`
dt = 1e-2      # (=1e-2 default), Which is normalized by `td ~ 1[s]` for MCF plasma.
               # `1e0 ~ 1e-5` is proposed to be used for most situation.
fvL0k1 = fvL0 + dt * δtfvL0e

# Updating the thermal velocity `vₜₕ = vth` in single step
vthk1dt = vth + dt * dtvthk

# # # Updating the conservative momentums 
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
    uhk1M0 = M110k1[1,:] / 3
    Khk1M0 = M200k1[1,:]
    # Khk1M = 3/2 .+ uhk1M.^2
    RvthM = (3/2 ./ (Khk1M0 - uhk1M0.^2)).^0.5

    dRvthM = RvthM .- 1
    uhk1M = uhk1M0 .* RvthM
    Khk1M = Khk1M0 .* RvthM.^2
    vthk1M = vthk .* RvthM

    @show dRvthM
    @show (2/3 * (Khk1M - uhk1M.^2)).^0.5 .- 1



    vthk1Mdt = copy(vthk1dt)


    # Checking the conservations
    nhk11 = copy(nak)
    # nhk11 = nhk1M
    ρa = ma .* nhk11

    Ik11 = ρa .* vthk1M .* uhk1M
    Kk11 = 1/2 * ρa .* vthk1M.^2 .* Khk1M

    uhk11 = 1.5^0.5 * Ik11 ./ (2ρa .* Kk11 - Ik11 .^2).^0.5
    vthk11 = (2/3 * (2Kk11 ./ ρa - (Ik11 ./ ρa).^2)).^0.5

    @show uhk11 - uhk1M
    @show vthk1M ./ vthk11 .- 1
    @show vthk1dt ./ vthk11 .- 1
end
nhk1, Ik1, Kk1 = nIKs(fvL0k1[:,1:2,:],vGe,ma,na,vthk11,ns)
Dnhk, DIk, DKk = nhk1 .- 1, Ik1 - Ik, Kk1 - Kk
RDKk = DKk ./ Kk
RerrDnhk = nhk .- 1
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
@show dt
@show fmtf2.(RDKk)
@show fmtf2.(DKk)
@show fmtf2.(DIk)
@show fmtf2.(RerrDnhk)
@show fmtf2.(RerrDIk)
@show fmtf2.(RerrDKk)

nModk = copy(nMod)
sdfghjm
# # Corrections to satisfy the conservation laws by applying a posterior analysis.
# # `is_corrections::Vector{Bool} = [true, true, true]` for `[is_corrections_n, is_corrections_I, is_corrections_K]`
# is_corrections[2:3] .= false
# if is_corrections[1] == false
#     nak1 = nak .* nhk1
# else
#     nak1 = nak
# end
# if is_corrections[2]
#     if RerrDIk > epsT
#         if residualMethod_FP0D == 1
#             if RerrDIk > epsT
#                 DIk .-= sum(DIk) / 2
#                 Ik1 = Ik + DIk
#             end
#         elseif residualMethod_FP0D == 2
#         end
#     end
# end
# if is_corrections[3]
#     if RerrDKk > epsT
#         if residualMethod_FP0D == 1
#             if RerrDKk > epsT
#                 DKk .-= sum(DKk) / 2
#                 Kk1 = Kk + DKk
#             end
#         elseif residualMethod_FP0D == 2
#         end
#     end
# end

# Updating the parameters `nh, uh, vth` according to the first three moments `n, I, K` at `k1` step
ρa = ma .* nak1
uhk1 = 1.5^0.5 * Ik1 ./ (2ρa .* Kk1 - Ik1 .^2).^0.5
vthk12 = (2/3 * (2Kk1 ./ ρa - (Ik1 ./ ρa).^2)).^0.5

@show uhk1 ./ uhk .- 1
@show vthk12 ./ vthk .- 1
# Updating the characteristic parameters of `f̂ₗᵐ(v̂)`: `x0k1 = [naik1,uaik1,vthik1]`
nModk1 = copy(nModk)
nMjMs01 = ceil.(Int,3 / 2 * nModk1)
naik1 = Vector((undef),ns)     # `n̂a = naᵢ / na`
uaik1 = Vector((undef),ns)     # `ûa = uaᵢ / vth`
vthik1 = Vector((undef),ns)    # `v̂th = vathᵢ / vth`
if prod(nModk1) == 1
    naik1,uaik1,vthik1 = submoment(naik1,uaik1,vthik1,fvL0k1,vGe,ns)
else
    naik1,uaik1,vthik1 = submoment(naik1,uaik1,vthik1,nModk1,fvL0k1,vGe,ns;is_renorm=is_renorm,
                    optimizer=optimizer,factor=factor,autodiffs=autodiffs,
                    is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
end

# # Re-mesh the grid on the velocity axis direction
# if is_remesh_vG
#     vGk
#     nc0, nck = 1
#     nvlevel0, nvlevele0 = 1
# end

# # Updating the first two amplitude functions to satisfy the conservation laws.
# LMk1 = 0LM
# LMk1,fvLk1 = fvLDMz(fvLk1,vGk,LMk1,ns,naik1,uaik1,vthik1;L_limit=L_limit,
#                     rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)

# # Updating the amplitude function `f̂ₗᵐ(v̂)` according parameters `naik1,uaik1,vthik1`
# LM1k1 = maximum(LMk1)

# # Updating the FP collision terms according to the `FPS` operators.
# err_dtnIKk1 = 0.0
# δtfvL0k1 = zero.(fvL0k1)
# @time dtfvLSplineab!(δtfvL0k1, fvL0k1, err_dtnIKk1, vGk, nc0, nck, ocp, nvlevel0,
#     CΓ, εᵣ, ma, Zq, na, vth, nai, uai, vthi, LM, LM1, ns, nMod;
#     isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
#     autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
#     p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs, is_δtfvLaa=is_δtfvLaa,
#     is_normδtf=is_normδtf, is_boundaryv0=is_boundaryv0, is_resetv0=is_resetv0)
# R200k1 = zeros(2,ns)

# δtfvL0k1
    