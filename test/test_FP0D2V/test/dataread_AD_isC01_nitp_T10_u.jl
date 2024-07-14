
using DataFrames,CSV, Format,Plots
using LinearAlgebra

path = "G:\\atom\\julia\\FP0D2Vcon"
include(joinpath(path,"src\\Gammab.jl"))
cd(path)

##
# plotlyjs()
# plotly()
gr()                 # no LATeX
plot()
wline = 5              # width of lines
guidefontsize = 16
tickfontsize = 16
legendfontsize = 14
markersize = 3
## color
colors = [:black,:green,:red,:orange,:blue,:purple,:gray,:yellow]
# colors = permutedims(1:nRtol)
## line styles
linetypes = filter((s->begin s in Plots.supported_styles()
end), [:solid, :dash, :dot, :dashdot, :dashdotdot, :soliddot])
linetypes = reshape(linetypes, 1, length(linetypes))
nlt = length(linetypes)
markers = filter((m->begin m in Plots.supported_markers()
                  end), Plots._shape_keys)
markers = reshape(markers, 1, length(markers))
## Physics constants
const e = 1.602176634e-19  # [C]      Elementary charge
const c₀ = 2.99792458e8    # [m/s]    Speed of light in vacuum
const μ₀ = pi*4e-7         # [H/m]    Vacuum permeability  =1.2566370614e-16
const ε₀ = 1 /(μ₀ * c₀^2)  # [F/m]    Vacuum ermittivity   =8.8541878176e-12
    # Rinf: Rydberg constant
    # alpha0: fine structure constant
const me=9.10938356e-31    # [kg]     Electron mass, me=2*h0*Rinf/(c0*alpha0^2）
const Da=1.660539040e-27   # [kg]     dalton:unified atomic mass unit; 1Da=mC12/12
const mp=Da*1.007276466879 # [kg]     Proton mass      =1.672621898e-27;
const kB=1.38064852e-23    # [J/K]    Boltzamann constant
const h₀=6.626070040e-34   # [J*s]    Planck constant
const h̄ = h₀/(2*pi)

const n20 = 1e20            # [1/m^3]
const Mms = 1e6             # [m/s]
const Tk  = 1e3e            # [J] = [keV] = 1k * 1V * e
const K20k  = n20 * Tk      # []
const qma0 = e^2 / (4π * ε₀ * Da) # Coefficient of Γₐᵦ = ZₐZᵦ/m̂ₐ * qma0 where m̂ₐ = mₐ/Da
####      time parameters
t0 = 0.0
tmax = 1.5e-1
dt0 = 1e-7         # for dt for fix step
dt_intial = 1e-9   # for initial dt
ndtmax = 10        # for maximum time step
tspan = (t0 + dt_intial,tmax)  # start from the second step at t0 + dt_intial
# ts = range(t0,stop=tmax,length=101)
########## Arol and Rtol
Atol = 1e-5       # (=1e-3, default) Relative tolorence for ODE solver
Rtol = Atol         # Relati         ve tolerence for ODE solver
# Rtol = 1e-9
rtol_itersol = 1e-15 # Relative tolerence of IterativeSolvers
max_itersol = 180    # Maximum number of iterations
Refine_t = 1
## ###      velocity space parameters
##### velocity axis
nv_min0 = 6         # (=6,default) low limit for nv nv_min0 for GaussQuadrature iteration
dnv_gauss = 2       # increase number of nv for GaussQuadrature optimization
nv_limit0 = 60      # (=60,default) maximum of nv for v-grid, i = 0:nv_limit
                    # Usually =80 is suitable to atol_gauss = 1e-2
##### Remesh for v  #
nitp = 10            # (=6,default) dv/dv_new = [0,2N⁺], the multiple of remeshing for Gaussian grids
nitp_vth = 0        # = [0,2N⁺] , times for remesh the grids near v̂ = 1
n_vth = 3           # for remesh near v̂ = 1, [1/n_vth:1:n_vth]
                    # n_vth = 0 when nitp_vth = 0
nv0mim = 25         # = 27 (default), minimum number of remesh when v < v[1]
n50 = 15       # = min(vu) / max(vu) / vmin,
            # for minmum of velocity, v_min → 0
            # = 10 (default), n50 > 50 maybe cause cumulative errors
##### L-m spaces
L_limit0 = 20        # limit of L for all spices
                    # Lmax < 35 || numerical instability
R_dfLM = 1e-2       # maximum rate of change of f(v)= ∑fL(v) for ℓM
# axisymmetric in velocity spave
m = 0
nSf = 8    # m = 0 : , i = 1,2,3,5,6,8,9,13;
           # m > 0 : nSf =13, i = 1:13;
####  δfL/δt
nSdft = 1   # = 1 (default) the output the summation δf/δt = ∑ᵢSᵢ only
            # or else the nSf terms as δₜf[v,LM,nSf,nspices]
## model by spices according to n[i]
# spices
n0_c = 1e-10         # n[i] < n0_c means spices[i] is absent
is_PR = 0            # power of radiation
is_fus = 0           # fusion reactions
is_Lag = 0           # = 0 (default, mean u = u_Lag is off) for dfvL/dt
is_conservation = 1  # (=1, default means on) moments conservations
is_ncon = 1          # conservation only for number density, n
is_pcon = 0          # conservation only for momentum, p = m n u
is_Kcon = 1          # conservation only for total energy, K
is_cb = 1            # = 1 (default, callback is on )
Rtol_T = 1e-3      # (1e-3 default) Reltol tolerence for moments
Rate_uT = 1e-6      # (1e-3 default) Reltol tolerence for u[1] ≈ u[2]
Ratio_T = 1e-1     # (1e-1 default) Rate of change for moments
Rtol_ds = 1e-13    # (= 1e-6,or the end iteration will be too long) Reltol tolerence for relative entropy
             # i-e collision: maybe Rds > 0
nGite = 8    # (=8 default) Maximum iteration number for Gaussian collection
if is_conservation == 1
    is_ncon = 1
end
if is_ncon * is_pcon * is_Kcon == 1
    is_conservation = 1
end
if nitp_vth == 0
    n_vth = 0
end
## parameters

Atolvec = [1e-3,1e-4,1e-5,1e-6]
nv_limit0vec = [40,60,80,100,120]
nitpvec = [4,6,8,10,12,14]
Ek0vec = [0,2,20]
ne0vec = [0.0001,0.001,0.01,0.1,2,5]
ne0vec = [0.001]  # Rtol ?
T0vec = [1,10]

T0e = 20
ne0 = 1

Ek0vec = [100]
ne0vec = [.1]
nitpvec = [6]
T0vec = [20]
is_C = [1]
# Rtol_Tvec = [1e-1, 5e-2, 2e-2, 1e-2, 5e-3]
# Rtol_Tvec = [ 1e-3]
nTb = length(T0vec)
nub = length(Ek0vec)
nnb = length(ne0vec)
# nRtol = length(Rtol_Tvec)
# n = nRtol
# dn_max = zeros(n,nspices)
plotly()
# gr()
# pgfplotsx()
# inspectdr()
##
ma0 = zeros(Real,2)
ma0[1] = 2mp/me      # me
ma0[2] = 4   # ma is electron
for Ek0e in Ek0vec
    icolor = 0
    inlt = 0
for ne0 in ne0vec
for is_conservation in is_C
    iT = 0
# for nitp in nitpvec
# for nv_limit0 in nv_limit0vec
# for Atol in Atolvec
    # Rtol = Atol
    iT += 1
    if inlt ≥ nlt
        inlt = 1
    else
        inlt += 1
    end
    icolor += 1
    # println("ne=",ne0,",Ek=",Ek0e,",T0=",T0e,)
        spices =["e" "a" "b" "α" "beam" "impurity"]  # hcat() = cat(v,dims=2)
        mD0   = [ma0[1] ma0[2]   3   4      2      1   ]  # mp except mD0[1] = me
        Zq   = [ -1  1   1   2      1      1   ]  # e
        # variables: moments
        n0   = [ 1  ne0  0   0      0     0   ]  # n20
        Ek0  = [ 0  Ek0e  0   3e3      0     0   ]  # keV
        T0   = [ 1  T0e  10   10      0     0   ]  # keV
        is_qconservation = 0 # = 1 (default, conservation is on)
        # uₑ <1.87e8 (Eke ~ 100 keV)
        # vₑₜₕ < 1e8 (Te ~ 30 keV)
        # Ekα = 3 MeV -- (uα ~ 1.2e7 ~ ue)  -- Eke = 0.4 keV
        # ne0 = sum(ni0 * Zq) due to electric neutrality
        #################
        nspices = length(spices)
        is_spices = zeros(nspices)
        is_spices[1] = 1     # ne
        for i = 2:nspices
            n0[i] > n0_c ? is_spices[i] = 1 : is_spices[i] = 0
        end
        # re-arrange the spices by deletting the spices with n[i]=0
        ns_on = is_spices .== 1
        spices = spices[ns_on]
        nspices = length(spices)

        mD0 = float(mD0[ns_on])
        Zq = Zq[ns_on]
        n0 = n0[ns_on]
        Ek0= Ek0[ns_on]
        T0 = T0[ns_on]
        if is_qconservation == 1
            n0[1] = sum(Zq[2:nspices].*n0[2:nspices])  # ne = sum(Zq*n0)
        end
        # SI Units from practical units
        m0 = zeros(nspices)
        m0[1] = me * mD0[1];   m0[2:nspices] = mp * mD0[2:nspices]
        qZ = Zq * e

        Eka0 = Ek0 * Tk
        na0 = n0 * n20
        Ta0 = T0 * Tk
        ua0 = @. √(2Eka0 / m0)
        ûa0 = ua0 / Mms               # û = u/vth
        Ka0 = @. 1.5na0 * Ta0 + 0.5na0 * m0 * ua0^2
        vth0 = @. √(2Ta0/ m0)
        isp = 1
        iFv = 2
        isp = 1
        iFv = 2
        maD = m0/Da
        cT = (prod(maD))^0.5 / (maD[isp] * T0[iFv] + maD[iFv] * T0[isp])^1.5
        cTq = 4.41720911682e2 * cT * Zq[isp]^2 * Zq[iFv]^2 * lnA()
        νTab = n0[iFv] * cTq
        νTba = n0[isp] * cTq
        # τ₀ = norm([1 / νTab,1 / νTba])
        τ₀ = min(1 / νTab,1 / νTba)
        #####
        Eka = Eka0
        na = na0
        Ta = Ta0
        ua = ua0
        ûa = ûa0
        Ka = Ka0
        Ks0 = sum(Ka)
        vth = vth0
        û = ua./vth
        K̂a = (3//2 .+ û.^2)  #   [na .* Ta ]
        K̂20k = K̂a .* (na .* Ta ./K20k)
        ρa = m0 .* na
        Ia = ρa .* ua
        uinf = sum(Ia) / sum(ρa)
        Tinf = (Ks0 - sum(na/2 .* m0 .* uinf.^2)) / (3/2 * sum(na))
        println("τ₀=",τ₀,",τ₀,ab=",1 / νTab,",τ₀,ba=",1 / νTba)
        println("Is=",sum(Ia),",us0=",ûa,",u∞=",sum(uinf /Mms),",Ks0=",sum(K̂20k),",T∞=",Tinf /Tk)
        global sa0 = na .* (3/2 * (1 .+ log.(2π * m0)) - log.(na./Ta.^(3/2)))
        global s0 = sum(sa0)
        n_moments = 3
        moments = zeros(3,nspices)
        moments[3,:] = K̂20k'       # K̂a[K20k]
        moments[2,:] = ûa'          # ûa[Mms]
        moments[1,:] = n0'    # n̂a[n20]
        println("ûa = ",ûa0)
        fmtf = generate_formatter( "%1.1e" )   # "%e", "%'e'", "%1.1e", [d,f,s,e]
        fmtf2 = generate_formatter( "%1.2e" )
        if nspices == 2
            pathfile = string(path,"\\datas","\\FP0Dm",m,"_tnuKT","_nsp",nspices,"_m",fmtf(mD0[1]))
            filename = string("\\mb",fmtf(mD0[2]),"\\n",fmtf(n0[1]),"_",fmtf(n0[2]),
                        "_u",fmtf(ûa[1]),"_",fmtf(ûa[2]),"_T",fmtf(T0[1]),"_",fmtf(T0[2]),
                        "_Atol",fmtf(Atol),"_Rtol",fmtf(Rtol),"Rtol_T=",Rtol_T,"Ratio_T=",Ratio_T,
                        "_Rds",Rtol_ds,"_nvM",nv_limit0,"_nitp",nitp,"_nitpth",nitp_vth,"_nvth",n_vth,
                        "_isL",is_Lag,"_isC",is_conservation,"_isnC",is_ncon,".csv")
        elseif  nspices == 3
        elseif  nspices == 4
        else
        end
        pathdata = string(pathfile,filename)
        println("filename=",pathdata)
        if isfile(pathdata) == 1
            sol = CSV.File(read(pathdata)) |> DataFrame
            nt,~ = size(sol)
            if nt == 0
                println("File is empty!")
                # break
            else
                unique!(sol,1)         # due to t = x[1]
                dropmissing!(sol)
                # println("nt=", size(sol))
                # ## ################### Plots
                tvec = sol.t .< tmax
                tvec = sol.t .< 40τ₀
                titlenuT = string("n=",ne0,", Ek=",Ek0e,", Tk=",T0e)
                ######### Tab[nitp]
                # if is_conservation == 1
                #     label = "conservation"
                #     plot!(sol.t[tvec]/τ₀,sol.Ta[tvec],line = (:solid,colors[icolor],wline))
                #     pT = plot!(sol.t[tvec]/τ₀,sol.Tb[tvec],line = (:dot,colors[icolor],wline),
                #               legend=:topright);
                # else
                #     label = "no-conservation"
                #     plot!(sol.t[tvec]/τ₀,sol.Ta[tvec],line = (:solid,colors[icolor],wline))
                #     pT = plot!(sol.t[tvec]/τ₀,sol.Tb[tvec],line = (:solid,colors[icolor],wline),label=label,
                #               guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize,);
                # end
                # display(plot(pT))
                # #  title!(titlenuT)
                # xaxis!("t",widen=true)
                # yaxis!("T")
                ########### ΔT[nitp]
                # # Teff = (sol.na[tvec] .* sol.Ta[tvec] + sol.nb[tvec] .* sol.Tb[tvec]) / (sol.na[tvec] + sol.nb[tvec])
                # Teff = (sol.Ta[tvec] + sol.Tb[tvec])
                # # Teff = Teff[1]
                # ΔT = abs.(sol.Ta[tvec] - sol.Tb[tvec]) ./ Teff
                # if is_conservation == 1
                #     label = string("nitp=",nitpvec[iT])
                #     pT = plot!(sol.t[tvec]/τ₀,ΔT,line = (linetypes[inlt],colors[icolor],wline),label=label,legend=:topright,
                #               guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize);
                # else
                #     label = "no-conservation"
                #     pT = plot!(sol.t[tvec]/τ₀,ΔT,line = (linetypes[inlt],colors[icolor],wline),label=label,
                #               guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize);
                # end
                # display(plot(pT))
                # #  title!(titlenuT)
                # xaxis!("t",widen=true)
                # yaxis!("ΔT",yscale=:log)
                ######## Ka = nT(3/2+ û²)
                # Ks = sol.Ka[tvec] + sol.Kb[tvec]
                # ΔK = Ks/Ks[1] .-1
                # if is_conservation == 1
                #     label = string("conservation")
                #     pK = plot!(sol.t[tvec]/τ₀,ΔK,line = (linetypes[inlt],colors[icolor],wline),label=label,
                #               guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize,);
                #     # plot!(sol.t[tvec]/τ₀,sol.Ka[tvec])
                #     # pK = plot!(sol.t[tvec]/τ₀,sol.Kb[tvec],line = (linetypes[inlt],colors[icolor],wline),
                #     #           legend=:topright,label=label);
                # else
                #     label = string("nitp=",nitpvec[iT])
                #     # label = string("no-conservation")
                #     pK = plot!(sol.t[tvec]/τ₀,ΔK,line = (linetypes[inlt],colors[icolor],wline),label=label,
                #               guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize,);
                # end
                # display(plot(pK))
                # #  title!(titlenuT)
                # xaxis!("t",widen=true)
                # # yaxis!("ΔK",yscale=:log)
                # yaxis!("ΔK")
                ######################################
                ######### Tab
                # if is_conservation == 1
                #     # label = "conservation"
                #     # label = string("nitp=",nitpvec[iT])
                #     label = "    "
                #     plot!(sol.t[tvec]/τ₀,sol.Ta[tvec],line = (:solid,colors[icolor],wline),label=label)
                #     label = "    "
                #     pT = plot!(sol.t[tvec]/τ₀,sol.Tb[tvec],line = (:dot,colors[icolor+1],wline),label=label,legend=:topright,
                #                   guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize);
                # else
                #     label = "no-conservation"
                #     plot(sol.t[tvec]/τ₀,sol.Ta[tvec])
                #     pT = plot!(sol.t[tvec]/τ₀,sol.Tb[tvec],line = (:solid,colors[icolor],wline),label=label,
                #               guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize,);
                # end
                # #  title!(titlenuT)
                # xaxis!("t",widen=true)
                # yaxis!("T")
                # display(plot(pT))
                ######### ΔK
                if is_conservation == 1
                    label = "conservation"
                    Ks = sol.Ka[tvec] + sol.Kb[tvec]
                    ΔK = (Ks/Ks[1] .-1) / eps(Float64)
                    pK = plot!(sol.t[tvec]/τ₀,ΔK,line = (colors[icolor],wline),legend=false,
                             guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize,);
                else
                    label = "no-conservation"
                    Ks = sol.Ka[tvec] + sol.Kb[tvec]
                    ΔK = Ks/Ks[1] .-1
                    pK = plot!(sol.t[tvec]/τ₀,ΔK,line = (:solid,colors[icolor],wline),label=label,
                              guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize,);
                end
                #  title!(titlenuT)
                xaxis!("t [τ₀]",widen=true)
                yaxis!("ΔK [eps]")
                display(plot(pK))
                ####### ua[t]
                # if is_conservation == 1
                #     label = "conservation"
                #     label = "    "
                #     plot!(sol.t[tvec]/τ₀,sol.ua[tvec],line = (linetypes[inlt],colors[icolor],wline),label=label)
                #     label = "    "
                #     pu = plot!(sol.t[tvec]/τ₀,sol.ub[tvec],line = (linetypes[inlt+2],colors[icolor+1],wline),label=label,legend=:topright,
                #                   guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize);
                # else
                #     label = "no-conservation"
                #     plot!(sol.t[tvec]/τ₀,sol.ua[tvec],line = (linetypes[inlt],colors[icolor],wline),label=label)
                #     pu = plot!(sol.t[tvec]/τ₀,sol.ub[tvec],line = (linetypes[inlt],colors[icolor],wline),label=label,legend=:topright,
                #                   guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize);
                # end
                # # println("ua=",maximum(sol.ua),",ub=",maximum(sol.ub))
                # #  title!(titlenuT)
                # xaxis!("t",widen=true)
                # yaxis!("u")
                # # yaxis!("u",yscale=:log)
                # display(plot(pu))
                ####### p = mnu, Δp
                # if is_conservation == 1
                #     label = "conservation"
                #     ps = m0[1] * sol.na[tvec] .* sol.ua[tvec] + m0[2] * sol.nb[tvec] .* sol.ub[tvec]
                #     ps = n20 * Mms * ps
                #     ps = (ps./ps[1] .- 1) ./ eps(Float64)
                #     pp = plot!(sol.t[tvec]/τ₀,ps,line = (:dot,colors[icolor],wline),legend=false,
                #               guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize);
                # else
                #     label = "no-conservation"
                #     ps = m0[1] * sol.na[tvec] .* sol.ua[tvec] + m0[2] * sol.nb[tvec] .* sol.ub[tvec]
                #     ps = n20 * Mms * ps
                #     ps = (ps./ps[1] .- 1) ./ eps(Float64)
                #     pp = plot!(sol.t[tvec]/τ₀,ps,line = (:solid,colors[icolor],wline),label=label,
                #               guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize);
                # end
                # # println("ua=",maximum(sol.ua),",ub=",maximum(sol.ub))
                # #  title!(titlenuT)
                # xaxis!("t [τ₀]",widen=true)
                # yaxis!("ΔI [eps]")
                # # yaxis!("Δp",yscale=:log)
                # display(plot(pp))
                ####### sa =
                # isp = 1
                # iFv = 2
                # sa =  sol.na[tvec] .* (3/2 * (1 .+ log.(2π * m0[isp])) .- log.(sol.na[tvec]./sol.Ta[tvec].^(3/2)) )
                # sb =  sol.nb[tvec] .* (3/2 * (1 .+ log.(2π * m0[iFv])) .- log.(sol.nb[tvec]./sol.Tb[tvec].^(3/2)) )
                # ss = sa + sb
                # Rss = 1 .- ss/ss[1]
                # Rss = Rss .+ Rss[2]
                # if is_conservation == 1
                #     label = "conservation"
                #     ps = plot!(sol.t[tvec]/τ₀,Rss[tvec],line = (:dash,colors[icolor],wline),legend=false,
                #               guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize);
                # else
                #     label = "no-conservation"
                #     ps = plot!(sol.t[tvec]/τ₀,Rss[tvec],line = (:solid,colors[icolor],wline),label=label,
                #               guidefontsize = guidefontsize,tickfontsize = tickfontsize,legendfontsize=legendfontsize);
                # end
                # #  title!(titlenuT)
                # xaxis!("t",widen=true)
                # yaxis!("Δs")
                # # yaxis!("Δs",yscale=:log10)
                # display(plot(ps))
                ##### Entropy density
                # isp = 1
                # sa =  - sol.na[tvec] .* log.(n20 * sol.na[tvec]./(Tk * sol.Ta[tvec]).^(3/2))
                # iFv = 2
                # sb =  - sol.nb[tvec] .* log.(n20 * sol.nb[tvec]./(Tk * sol.Tb[tvec]).^(3/2))
                # sab = sa + sb
                # display(plot(sol.t[tvec], sab .- sab[1],label= "Entropy density"))
                #######
                # figname = string(figpath,figpatha,figpathb,"\\Ek",Ek0vec[1],"_t_T")
                # savefig(pT,figname)
                ########################################
                # plot(sol.t[tvec],sol.ua[tvec],label= "uₐ");
                # pu = plot!(sol.t[tvec],sol.ub[tvec],label="uᵦ",title = titlenuT);
                # display(plot(pu))
            end
        else
            println("Warning: File is inexistent!")
        end
        println("τ₀,ab=",τ₀,",τ₀,ab=",1 / νTab,",τ₀,ba=",1 / νTba)
        println("Is=",sum(Ia),",us0=",ûa,",u∞=",sum(uinf /Mms),",Ks0=",sum(K̂20k),",T∞=",Tinf /Tk)

    end
    # pT = plot!(sol.t[tvec],sol.Tb[tvec],label="Tᵦ",legend=:outertopright);
end
end
