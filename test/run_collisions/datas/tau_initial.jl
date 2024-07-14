
# if nMod[isp3t] ≥ 2
#     τab_min, tab_max = tau_fM(ma[isp3t],Zq[isp3t], na[isp3t], vth[isp3t], Coeff_tau, nai0[isp3t], vthi0[isp3t], nMod[isp3t];tau_scale=:minmax)
#     tab_m = zeros(2)
#     tau_fM!(tab_m,ma[isp3t],Zq[isp3t], na[isp3t], vth[isp3t], Coeff_tau, nai0[isp3t], vthi0[isp3t], nMod[isp3t])
# end

# tab_s = zeros(2)
# tau_fM!(tab_s,mamb,Zqab, na[isp3t] * nai0[isp3t], vth[isp3t] * vthi0[isp3t], na[iFv3t] * nai0[iFv3t], vth[iFv3t] * vthi0[iFv3t], Coeff_tau, nMod)

if is_fixed_timestep == false && maximum(nMod0) ≤ 4
    tau = zeros(2)       # [tau_min, tau_max]
    nai2 = Vector{AbstractVector{datatype}}(undef,2)
    vthi2 = Vector{AbstractVector{datatype}}(undef,2)
    tau_fM!(tau, ma, Zq, spices0, na, vth, Coeff_tau, 
            nai2, vthi2, nai0, vthi0, nMod, ns)
end

# if ns == 2
#     # tau_fM!(tau, ma, Zq, spices0, na, vth, Coeff_tau, nai0, vthi0, nMod)
#     nai2 = Vector{AbstractVector{datatype}}(undef,2)
#     vthi2 = Vector{AbstractVector{datatype}}(undef,2)
#     tau = tau_fM!(tau, ma, Zq, spices0, na, vth, Coeff_tau, 
#             nai2, vthi2, nai0, vthi0, nMod, ns)
# else
#     nai2 = Vector{AbstractVector{datatype}}(undef,2)
#     vthi2 = Vector{AbstractVector{datatype}}(undef,2)
#     tau_fM!(tau, ma, Zq, spices0, na, vth, Coeff_tau, 
#             nai2, vthi2, nai0, vthi0, nMod, ns)
#     @show tau
# end

# @show tau[2] / tau[1]
