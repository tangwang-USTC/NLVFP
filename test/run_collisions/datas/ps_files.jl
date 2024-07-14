

    # `w3k = Rdtvth = 𝒲 / 3`


if is_fvL_CP

    pst0_names = [
        "nstep",
        "tk",
        "dt",
        "tspan",
        "tauk",
        "Nt_save",
        "count_save",
    
        "ns",
        "ma",
        "Zq",
        "spices",
        
        "nk",
        "Ik",
        "Kk",
        "vthk",
        "vthk1",
        "w3k",

        "si",
        "sak",
        "dtsabk",
    
        "nnv",
        "nc0",
        "nck",
        "ocp",
        "vGm",
    
        "vhk",
        "vhe",
        "nvlevel0",
        "nvlevele0",
    
        "nModk",
        "nMjMs",
        "naik",
        "uaik",
        "vthik",
    
        "LMk",
        "muk",
        "Mμk",
        "Munk",
        "Mun1k",
        "Mun2k",
    
        "err_dtnIK",
        "DThk",
        "edtnIKTsk",
        "CRDnk",
        # "RMcsk",
        # "Mhck",
        "uCk",
    
        "is_fixed_timestep"
        ]
    pst0 = Dict(
        "nstep" => 1,
        "tk" => deepcopy(t0),
        "dt" => deepcopy(dt_initial),
        "tspan" => deepcopy(tspan),
        "tauk" => deepcopy(tau0),
        "Nt_save" => 1Nt_save,
        "count_save" => 1,
    
        "ns" => deepcopy(ns),              # 6
        "ma" => deepcopy(ma),
        "Zq" => deepcopy(Zq),
        "spices" => deepcopy(spices0),

        "nk" => deepcopy(n0),
        "Ik" => deepcopy(Iak10),
        "Kk" => deepcopy(Kak10),              # 11
        "vthk" => deepcopy(vth),
        "vthk1" => deepcopy(vth),
        "w3k" => zero.(vth),

        "si" => 0,
        "sak" => deepcopy(sak0),
        "dtsabk" => deepcopy(dtsabk0),       # 46
    
        "nnv" => deepcopy(nnv[1]),
        "nc0" => deepcopy(nc0[1]),
        "nck" => deepcopy(nck[1]),
        "ocp" => deepcopy(ocp[1]),         # 16
        "vGm" => deepcopy(vGdom[:,1]),
    
        "vhk" => deepcopy(vhk[1]),
        "vhe" => deepcopy(vhe[1]),
        "nvlevel0" => deepcopy(nvlevel0[1]),
        "nvlevele0" => deepcopy(nvlevele0[1]), # 21
    
        "nModk" => deepcopy(nModk10),
        "nMjMs" => deepcopy(nMjMs),         # 36
        "naik" => deepcopy(naik10), 
        "uaik" => deepcopy(uaik10),
        "vthik" => deepcopy(vthik10),
    
        "LMk" => deepcopy(LM),              # 26
        "muk" => deepcopy(mu),
        "Mμk" => deepcopy(Mμ),
        "Munk" => deepcopy(Mun),
        "Mun1k" => deepcopy(Mun1),
        "Mun2k" => deepcopy(Mun2),          # 31
    
        "err_dtnIK" => 0.0,
        "DThk" => zeros(ns),
        "edtnIKTsk" => zeros(4,ns),
        "CRDnk" => [0.0],
    
        # "RMcsk" => deepcopy(RMcs2),
        # "Mhck" => deepcopy(Mhc2),
        "uCk" => deepcopy(zeros(2)),
    
        "is_fixed_timestep" => is_fixed_timestep
        )
else
    pst0_names = [
        "nstep",
        "tk",
        "dt",
        "tspan",
        "tauk",
        "Nt_save",
        "count_save",
    
        "ns",
        "ma",
        "Zq",
        "spices",
        
        "nk",
        "Ik",
        "Kk",
        "vthk",
        "vthk1",
        "w3k",

        "si",
        "sak",
        "dtsabk",
    
        "nnv",
        "nc0",
        "nck",
        "ocp",
        "vGm",
    
        "vhk",
        "vhe",
        "nvlevel0",
        "nvlevele0",
    
        "nModk",
        "nMjMs",
        "naik",
        "uaik",
        "vthik",
    
        "LMk",
        "muk",
        "Mμk",
        "Munk",
        "Mun1k",
        "Mun2k",
    
        "err_dtnIK",
        "DThk",
        "edtnIKTsk",
        "CRDnk",
        "RDMck1",
        "Mhck",
        "errMhcop",
        "uCk",
    
        "is_fixed_timestep"
        ]
    
    pst0 = Dict(
        "nstep" => 1,
        "tk" => deepcopy(t0),
        "dt" => deepcopy(dt_initial),
        "tspan" => deepcopy(tspan),
        "tauk" => deepcopy(tau0),
        "Nt_save" => 1Nt_save,
        "count_save" => 1,
    
        "ns" => deepcopy(ns),              # 6
        "ma" => deepcopy(ma),
        "Zq" => deepcopy(Zq),
        "spices" => deepcopy(spices0),

        "nk" => deepcopy(n0),
        "Ik" => deepcopy(Iak10),
        "Kk" => deepcopy(Kak10),              # 11
        "vthk" => deepcopy(vth),
        "vthk1" => deepcopy(vth),
        "w3k" => zero.(vth),

        "si" => 0,
        "sak" => deepcopy(sak0),
        "dtsabk" => deepcopy(dtsabk0),       # 46
    
        "nnv" => deepcopy(nnv),
        "nc0" => deepcopy(nc0),
        "nck" => deepcopy(nck),
        "ocp" => deepcopy(ocp),         # 16
        "vGm" => deepcopy(vGdom),
    
        "vhk" => deepcopy(vhk),
        "vhe" => deepcopy(vhe),
        "nvlevel0" => deepcopy(nvlevel0),
        "nvlevele0" => deepcopy(nvlevele0), # 21
    
        "nModk" => deepcopy(nModk10),
        "nMjMs" => deepcopy(nMjMs),         # 36
        "naik" => deepcopy(naik10), 
        "uaik" => deepcopy(uaik10),
        "vthik" => deepcopy(vthik10),
    
        "LMk" => deepcopy(LM),              # 26
        "muk" => deepcopy(mu),
        "Mμk" => deepcopy(Mμ),
        "Munk" => deepcopy(Mun),
        "Mun1k" => deepcopy(Mun1),
        "Mun2k" => deepcopy(Mun2),          # 31
    
        "err_dtnIK" => 0.0,
        "DThk" => zeros(ns),
        "edtnIKTsk" => zeros(4,ns),
        "CRDnk" => [0.0],
    
        "RDMck1" => zero.(RDMck10),
        "Mhck" => deepcopy(Mhck10),
        "errMhcop" => deepcopy(errMhcop2),                    # The errors of renormalized kinetic moments in moment optimization step (solving the characteristic equations)
        "uCk" => deepcopy(zeros(2)),
    
        "is_fixed_timestep" => is_fixed_timestep
         )
end
pstk = deepcopy(pst0)
Npst0 = length(pstk)
psvec = Vector{Any}(undef,Npst0)
for k in 1:Npst0
    psvec[k] = deepcopy(pstk[pst0_names[k]])
end

@show nMod
if norm(u0) ≤ epsT1000
    DKha09 = abs(sum(nai[1][1:nMod[1]] .* vthi[1][1:nMod[1]].^2) - 1)
    DKhb09 = abs(sum(nai[2][1:nMod[2]] .* vthi[2][1:nMod[2]].^2) - 1)
    DKhak09 = abs(sum(naik10[1][1:nModk10[1]] .* vthik10[1][1:nModk10[1]].^2) - 1)
    DKhbk09 = abs(sum(naik10[2][1:nModk10[2]] .* vthik10[2][1:nModk10[2]].^2) - 1)
else
    sgfhnm9
end
DKha09 + DKha09 ≤ epsT1000 || error("DK: characteristic parameters are not converged!")
DKhak09 + DKhak09 ≤ epsT1000 || error("DKk: characteristic parameters are not converged!")

