

# nMjMs = zeros(Int64, ns)
# njMs_fix = njMs
# is_MjMs_max = false
# is_LM1_full = true
# L_Mh_limit = 0               # (=0, default) limit of `L1` for `Mh`

Mhc2 = Vector{Matrix{datatype}}(undef, ns)
Mhinitial_fDM!(Mhc2, nMjMs, LM, LM1, ns; is_LM1_full=is_LM1_full, L_Mh_limit=L_Mh_limit)

if 1 == 1
    errMhc = deepcopy(Mhc2)
    MhnnEvens!(Mhc2, errMhc, fvL0e, vhe, nMjMs, LM, ns; 
                is_renorm=is_renorm, is_err_renorm=is_err_renorm, L_Mh_limit=L_Mh_limit)
    if norm(errMhc) ≥ epsT1000
        @error("Error: The integrations of `Mjl` is not convergent!", errMhc)
    end
    Iha2 = zeros(ns)
    Iha2[1] = 1Mhc2[1][1, 2]
    Iha2[2] = 1Mhc2[2][1, 2]
    if norm(Iha2 - Iha) ≥ epsT1000
        @show Iha ./ Iha2 .- 1
        egdfvb
    end

    Mc2 = zeros(njMs+1,LM1,ns)
    aa = Mc2[1:njMs,:,:]
    MckMhck!(aa, Mhc2, nMjMs, LM, ρa, vth, ns; L_Mh_limit=L_Mh_limit)
    Mc2[1:njMs,:,:] = aa
    Mc2[njMs+1,1,:] = vth
    
    Ia2 = zeros(ns)
    Ia2[1] = 1Mc2[1, 2, 1]
    Ia2[2] = 1Mc2[1, 2, 2]

    Ka2 = zeros(ns)
    Ka2[1] = Mc2[2, 1, 1] * CMcKa 
    Ka2[2] = Mc2[2, 1, 2] * CMcKa 
    
    RMcs2 = zeros(njMs,LM1)
    Mtheorems_RMcs!(RMcs2,Mc2[1:njMs,:,:],ρa,ns)
else
    Mc2 = zeros(njMs,LM1,ns)
    errMc2 = deepcopy(Mc2)
    dtMcsd2l!(Mc2, errMc2, fvL0e, vhe, nMjMs, ρa, vth, LM, ns; is_renorm=is_renorm)
end
    
Ia2 = zeros(ns)
Ia2[1] = 1Mc2[1, 2, 1]
Ia2[2] = 1Mc2[1, 2, 2]

Ka2 = zeros(ns)
Ka2[1] = Mc2[2, 1, 1] * CMcKa 
Ka2[2] = Mc2[2, 1, 2] * CMcKa 

@show Ia0 - Ia2
@show Ka0 - Ka2
# wedsfgbn

aa = sum(nai0[1][1:nMod0[1]] .* uai0[1][1:nMod0[1]]) .* vth[1] - Ia2[1] / ρa[1]
@show 701, aa
bb = sum(nai0[2][1:nMod0[2]] .* uai0[2][1:nMod0[2]]) .* vth[2] - Ia2[2] / ρa[2]
@show 701, bb
if abs(aa) + abs(bb) > epsT1000
    sdnghghghg
end

isp3 = 1
DKha0 = sum(nai0[isp3][1:nMod[isp3]] .* vthi0[isp3][1:nMod[isp3]].^2) - 1
@show 701, DKha0
isp3 = 2
DKhb0 = sum(nai0[isp3][1:nMod[isp3]] .* vthi0[isp3][1:nMod[isp3]].^2) - 1
@show 701, DKhb0
if abs(DKha0) + abs(DKhb0) > epsT1000
    ddgffws
end
# Mhck10 = deepcopy(Mhc2)
# Mhck10 = MsnntL2fL0(Mhck10,nMjMs,LM,ns,nai0,uai0,vthi0,nMod;is_renorm=is_renorm)

# dsfrfdf
# 