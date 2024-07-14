




"""
    Dimensionless of MVFP system with Practical system of Units (PU)
"""

mDa = maSI / Dₐ
# The characteristic time for thermal equilibrium models
ispt = 1
iFvt = 2
# # When `mDa, n0, T0` are normalized by`Da, n20, Tk` respectively.
tau_ab0 = tau_fM_Tk(mDa, Zq, spices0, n0, T0; tau_scale=:max)
tau_ab02 = tau_fM_Tk(mDa, Zq, spices0, n0, T0; tau_scale=:minmax)

include(joinpath(pathroot, "src/PS/nuT/nuTs_d.jl"))

Id = md * nd * vd
Kd = md * nd * vd^2           # nd * Td

Td = md * vd^2   # `Tk` or `Dₐ * vd^2 / 2`
# Td = deepcopy(Tk)
@show Td / Tk;

# The Alfven times
wpd = √(nd * qd / ε₀ * (qd / md))
τₚ = 1 / wpd

tdd_scale = 1
# tdd = 1e6
# tdd = 1.0           # (= 1[s], default) Which denotes the time is dimensionless by unit `1[s]`.
# tdd = τₚ             # Which denotes the time is dimensionless by the Alfven times scale.
τ₀ = tau_ab0
tdd = τ₀ / tdd_scale  # Which denotes the time is dimensionless by the thermodynamic relation time scale .

kpd = vd * tdd   # The skin depth of the reference plasma
rd = deepcopy(kpd)
td = rd / vd

# # τ⁻¹ ~ (nd) / (md)^0.5 / (Td)^(3/2)
# # τ⁻¹ = (md * nd * e^4) / (md * Td)^(3/2)
# τp = (Dₐ * n20 * e^4) / (Dₐ * Tk)^(3/2)
# τ⁻¹ = (qd^2 / md)^2 * (nd / vd^3)
# Ratio_Rt = τp / τ⁻¹
# Ratio_Rt = (nd/n20) / (Dₐ/md)^0.5 / (Tk/Td)^1.5

CΓ = (tdd * wpd) * ((wpd / c₀)^3 / nd)
## The dimensionless moments
ma = maSI / md
Zq
na = naSI0 / nd
ua = uaSI0 / vd
vth = vthSI0 / vd
ρa = ma .* na
#
Ta = TaSI0 / Td
Eka = EkaSI0 / Td
Ia = IaSI0 / Id
Ka = KaSI0 / (nd * Td)
sa = ones(ns)

if ns == 33
    taueD = zeros(2)
    ii1, ii2 = 1, 2
    vec32 = [ii1, ii2]
    nai02 = Vector{AbstractVector{Float64}}(undef,2)
    vthi02 = Vector{AbstractVector{Float64}}(undef,2)
    nai02[1] = deepcopy(nai0[ii1])
    nai02[2] = deepcopy(nai0[ii2])
    vthi02[1] = deepcopy(nai0[ii1])
    vthi02[2] = deepcopy(nai0[ii2])
    tau_fM!(taueD, ma[vec32], Zq[vec32], spices0[vec32], na[vec32], vth[vec32], Coeff_tau, nai02, vthi02, nMod[vec32])
    
    taueA = zeros(2)
    ii1, ii2 = 1, 3
    vec32 = [ii1, ii2]
    nai02 = Vector{AbstractVector{Float64}}(undef,2)
    vthi02 = Vector{AbstractVector{Float64}}(undef,2)
    nai02[1] = deepcopy(nai0[ii1])
    nai02[2] = deepcopy(nai0[ii2])
    vthi02[1] = deepcopy(nai0[ii1])
    vthi02[2] = deepcopy(nai0[ii2])
    tau_fM!(taueA, ma[vec32], Zq[vec32], spices0[vec32], na[vec32], vth[vec32], Coeff_tau, nai02, vthi02, nMod[vec32])
    
    tauDA = zeros(2)
    ii1, ii2 = 2, 3
    vec32 = [ii1, ii2]
    nai02 = Vector{AbstractVector{Float64}}(undef,2)
    vthi02 = Vector{AbstractVector{Float64}}(undef,2)
    nai02[1] = deepcopy(nai0[ii1])
    nai02[2] = deepcopy(nai0[ii2])
    vthi02[1] = deepcopy(nai0[ii1])
    vthi02[2] = deepcopy(nai0[ii2])
    tau_fM!(tauDA, ma[vec32], Zq[vec32], spices0[vec32], na[vec32], vth[vec32], Coeff_tau, nai02, vthi02, nMod[vec32])
    @show taueD
    @show taueA
    
    @show tauDA
    @show tau_ab0
end

#
ua0 = deepcopy(ua)
Eka0 = deepcopy(Eka)
Ia0 = deepcopy(Ia)
Ka0 = deepcopy(Ka)
μEk0 = deepcopy(μEk)

Kab0 = sum(Ka)
Iab0 = sum(Ia)
sab0 = sum(sa)

nS0 = sum_kbn(na)
nhS0 = na / nS0
ρS0 = sum_kbn(ma .* na)
mS0 = ρS0 / nS0
mhS0 = ma / mS0
ρhS0 = ρa / ρS0
IS0 = sum_kbn(Ia)
uS0 = IS0 ./ ρS0
KS0 = sum_kbn(Ka)
vSth0 = (2 / 3 * (2KS0 ./ ρS0 - uS0^2))^0.5
TS0 = mS0 * vSth0^2 / 2

# 
εᵣ = ε / ε₀
μᵣ = μ / μ₀

t_unit = (τ₀ / td)
m_unit = Dₐ / md
abs(m_unit - 1) ≤ epsT10 || error("The scheme of Coulomb logarithm limit `m_unit = 1` in this version. Please uniform the algorithm!")
n_unit = n20 / nd
abs(n_unit - 1) ≤ epsT10 || error("The scheme of Coulomb logarithm limit `n_unit = 1` in this version. Please uniform the algorithm!")
v_unit = Mms / vd
T_unit = Tk / Td
T_unit1e3 = Tk / Td / 1000
logT_unit = log(T_unit)            # For `log(T_unit)` in codes to calculate Coulomb logarithm.
I_unit = (Dₐ * n20 * Mms) / Id
K_unit = (n20 * Tk) / (nd * Td)
s_unit = 1.0

Coeff_tau = 4.41720911682e2 * (m_unit^0.5 * T_unit^1.5 / n_unit)
τab = tau_fM(ma, Zq, spices0, na, Ta, Coeff_tau; tau_scale=:min)
# τab = tau_fM(ma, Zq, spices0, na, Ta,T_unit,logT_unit; tau_scale=:min)
@show τab - τ₀


isp3t = 2
iFv3t = 1

mamb = ma[isp3t] * ma[iFv3t]
Zqab = Zq[isp3t] * Zq[iFv3t]
tab_22 = zeros(2)
tau_fM!(tab_22, mamb, Zqab, spices0, na, vth, Coeff_tau)

uaub = abs(ua[isp3t] - ua[iFv3t])
if uaub ≥ epsT1000
    lnA15 = 15.0
    tau_ua = mamb / (Coeff_tau * na[iFv3t] * Zqab^2 * lnA15) / (1 + ma[iFv3t] / ma[isp3t]) * (ua[isp3t]^3 / erf((ua[isp3t] / vth[iFv3t])^2))
  
    tau_ub = mamb / (Coeff_tau * na[isp3t] * Zqab^2 * lnA15) / (1 + ma[isp3t] / ma[iFv3t]) * (ua[iFv3t]^3 / erf((ua[iFv3t] / vth[isp3t])^2))
    
    tau_uab = mamb / (Coeff_tau * na[isp3t] * Zqab^2 * lnA15) / (1 + ma[isp3t] / ma[iFv3t]) * (uaub^3 / erf((uaub / vth[isp3t])^2))
    tau_uba = mamb / (Coeff_tau * na[iFv3t] * Zqab^2 * lnA15) / (1 + ma[iFv3t] / ma[isp3t]) * (uaub^3 / erf((uaub / vth[iFv3t])^2))

    tau_u = mamb / (Coeff_tau * 2^1.5 * Zqab^2 * lnA15) / (1 + ma[iFv3t] / ma[isp3t]) * ((ua[isp3t]^2 + ua[iFv3t]^2)^1.5 / erf((ua[isp3t] / vth[iFv3t])^2 + (ua[iFv3t] / vth[isp3t])^2))
    
    @show [tau_ua, tau_ub, tau_u, tau_uab, tau_uba]
end

nS0 /= n_unit
ρS0 /= (m_unit * n_unit)
mS0 /= m_unit
IS0 /= I_unit
uS0 /= v_unit
KS0 /= K_unit
vSth0 /= v_unit
TS0 /= T_unit

@show uS0, TS0



