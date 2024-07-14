## parameters: Experimental moments in practical units.
"""
  `mD0`: is the normalized mass by `Dₐ`, except for electron which is normalized by `mₑ`.
  `Zq`: is the charge number, equivalent to `qa / e`.
  `n0`: is the normalized total number density of `spices[i]` which is normalized by `n20`.
  `u0`: is the normalized effective average velocity of `spices[i]` which is normalized by `Mms`.
  `Ek0`: is the normalized effective kinetic energy density of `spices[i] (of a single particle) which is normalized by `Tk`.
  `T0`: is the normalized effective (Experimental) thermal temperature of `spices[i] which is normalized by `Tk`.
  `K0`: is the normalized total energy density of `spices[i] which is normalized by `n20 * Tk`.

  Non-relativistic models:

    uₑ < 1.87e8 (EKe ~ 100 keV)
    vₑₜₕ < 1e8 (Te ~ 30 keV)
    Kα = 3 MeV -- (uα ~ 1.2e7 ~ ue)  -- Ke = 0.4 keV
"""

"""
  Inputs:
    n0_c:
"""

################# Reducing the ignorable spices
ns = length(spices0)
is_spices0 = zeros(Int64,ns)
is_spices0[1] = 1     # ne
for i = 1:ns
    n0[i] > n0_c ? is_spices0[i] = 1 : is_spices0[i] = 0
end
# re-arrange the spices by deletting the spices with n[i]=0
ns_on = is_spices0 .== 1
spices0 = spices0[ns_on]
ns = length(spices0)
mD0 = float(mD0[ns_on])
Zq = Zq[ns_on]
n0 = n0[ns_on]
if is_Ek0
    Ek0 = Ek0[ns_on]
    μEk = μEk[ns_on]
else
    u0= u0[ns_on]
end
T0 = T0[ns_on]
if is_qconservation
    n0[1] = sum(Zq[2:ns].*n0[2:ns])  # ne = sum(Zq*n0)
end
