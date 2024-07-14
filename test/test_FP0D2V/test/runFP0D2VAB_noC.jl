
# using OhMyREPL
using Plots
using Legendre, GaussQuadrature, SpecialFunctions, FiniteDifferences
using Dierckx, DataInterpolations, NumericalIntegration,SmoothingSplines
# Interpolations
using LinearAlgebra, LinearAlgebraX
using DifferentialEquations, OrdinaryDiffEq
using DataFrames, CSV, Format
using LocalFilters
# using SmoothingSplines
# Richardson
# using Query , DataFramesMeta, NumericIO
# using MatrixEquations, IterativeSolvers
# using SylvesterEquations
# using VFP2Ds

path = "H:\\BaiduNetdiskWorkspace\\VFP0D3V"
cd(path)
include(joinpath(path,"mathematics\\LaguerreP.jl"))
include(joinpath(path,"mathematics\\Matrix.jl"))
include(joinpath(path,"mathematics\\Itrapezoidal1D.jl"))
include(joinpath(path,"mathematics\\remeshvgs.jl"))
include(joinpath(path,"mathematics\\diffmatrix.jl"))
include(joinpath(path,"mathematics\\derivationCD.jl"))
include(joinpath(path,"mathematics\\gradiate.jl"))
include(joinpath(path,"src\\f\\initial\\fLDMu0.jl"))        # ξ < 0
include(joinpath(path,"src\\f\\initial\\premeshV.jl"))
include(joinpath(path,"src\\solver\\solverMethod.jl"))
include(joinpath(path,"src\\solver\\solverFP0D2V.jl"))
include(joinpath(path,"src\\solver\\solvernuT.jl"))
include(joinpath(path,"src\\solver\\iterations\\gmresJFa.jl"))
include(joinpath(path,"src\\Collision\\collisionsL.jl"))
include(joinpath(path,"src\\Collision\\HGnsp.jl"))
include(joinpath(path,"src\\Collision\\constraints.jl"))
include(joinpath(path,"src\\Collision\\Gammab.jl"))
# include(joinpath(path,"src\\Collision\\optimsns.jl"))
include(joinpath(path,"src\\Collision\\optimsF.jl"))
include(joinpath(path,"src\\Collision\\optimsH.jl"))
include(joinpath(path,"src\\Collision\\FP0D2Vfl0.jl"))
include(joinpath(path,"src\\Collision\\dfdt0D2Vab.jl"))
include(joinpath(path,"src\\Moments\\moments.jl"))
include(joinpath(path,"src\\Moments\\momentsNorm.jl"))
include(joinpath(path,"test\\run_collisions\\testdH.jl"))

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
const CmnuK = Da * Mms^2/(2K20k/n20) # for normalization
α = 0.0
fmtf = generate_formatter( "%1.1e" )   # "%e", "%'e'", "%1.1e", [d,f,s,e]
fmtf2 = generate_formatter( "%1.2e" )
fmtf4 = generate_formatter( "%1.4e" )
fmtf6 = generate_formatter( "%1.6e" )
fmtfn = generate_formatter( "%1.4e" )
# plotlyjs()
# plotly()
gr()                 # no LATeX
plot()
# legend = false, :outertopright, :topleft, :bottomright
## variables: algorithms, model and solver
####      time parameters
t0 = 0.0
tmax = 1e-4
dt0 = 1e-6         # for dt for fix step
dt_intial = 1e-9   # for initial dt
ndtmax = 10        # for maximum time step
nτ = 3e-6
# ts = range(t0,stop=tmax,length=101)
########## Arol and Rtol
Atol = 1e-4         # (=1e-3, default) Relative tolorence for ODE solver
Rtol = Atol         # Relati         ve tolerence for ODE solver
# Rtol = 1e-9
rtol_itersol = 1e-15 # Relative tolerence of IterativeSolvers
max_itersol = 180    # Maximum number of iterations
Refine_t = 1
datatype = Float64
## Newton convergencr
# atol = 1e-4
abstol = 1e-9
reltol = 1e-4
# maxiterN = 1
# maxiteriN = 1
# # ns01 = 4       # for global convergence during velocity mapping
# #                # s01 = 1 decays to be the lacal convergence method
# ϵ = 1e-4       # for Maxtrix-Vector products, Jv
# k = 3
# optimH = 0     # (=0/1 or false/trues), for optims of ∂ⁿHvL
# L1min = 1
# L1max = 2
# L1m = 1
# if L1m > 0
#     L1v = L1m:L1m
# else
#     L1v = L1min:L1max
# end
# is_mpoint = -1
## ###      velocity space parameters
##### velocity axis
nv_limit0 = 100      # (=60,default) maximum of nv for v-grid, i = 0:nv_limit
                    # Usually =80 is suitable to atol_gauss = 1e-2
##### Remesh for v  #
nitp = 0            # (=6,default) dv/dv_new = [0,2N⁺], the multiple of remeshing for Gaussian grids
s1 = 1.05           # (= 1.1 default which mean 24 grids will increase a level, × 10)  and limited [1.05, 3.3]
nv0min = 0          # (= 0 default which means no remeshing when v < vG[1])
n50 = 15            # vmin = min(vu) / max(vu) / vmin,
                    # for minmum of velocity, v_min → 0
                    # = 10 (default), n50 > 50 maybe cause cumulative errors
nvG1 = 4            # vmin = min(vmin, vG[1] / nvG1)
nitp_vth = 0        # = [0,2N⁺] , times for remesh the grids near v̂ = 1
n_vth = 3           # for remesh near v̂ = 1, [1/n_vth:1:n_vth]
                    # n_vth = 0 when nitp_vth = 0
##### L-m spaces
L_limit0 = 8        # limit of L for all spices
                    # Lmax < 35 || numerical instability
Rate_fLM = 1e-2     # maximum rate of change of f(v)= ∑fL(v) for ℓM
# axisymmetric in velocity spave
m = 0
nSf = 8             # m = 0 : , i = 1,2,3,5,6,8,9,13;
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
is_conservation = 0  # (=1, default means on) moments conservations
is_ncon = 1          # conservation only for number density, n
is_pcon = 0          # conservation only for momentum, p = m n u
is_Kcon = 1          # conservation only for total energy, K
is_cb = 0            # = 1 (default, callback is on )
Rtol_T = 1e-3       # (1e-2 default) Reltol tolerence for moments
Rate_uT = 1e-6      # (1e-3 default) Reltol tolerence for u[1] ≈ u[2]
Ratio_T = 1e-1      # (2e-1 default) Rate of change for moments
Rtol_ds = 1e-13     # (= 1e-6,or the end iteration will be too long) Reltol tolerence for relative entropy
             # i-e collision: maybe Rds > 0
nGite = 0    # (=8 default) Maximum iteration number for Gaussian collection
if is_conservation == 1
    is_ncon = 1
end
if is_ncon * is_pcon * is_Kcon == 1
    is_conservation = 1
end
if nitp_vth == 0
    n_vth = 0
end
is_write_flm = 0   # data file for flm
is_plot_nuT = 1    # for plotting of moments
## parameters
# # ne0vec = [5,2,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
# ne0vec = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,2,5]
# Ek0vec = [0,1e-4,1e-3,1e-2,1e-1,1,2,5,10,20,50,100]
# T0vec = [2,5,10,20,50,100]     #

#### All is ok
Atolvec = [1e-3,1e-4,1e-5,1e-6]
nv_limit0vec = [40,60,80,100,120]
nitpvec = [4,6,8,10,12,14]
Ek0vec = [0,2,20]
ne0vec = [0.0001,0.001,0.01,0.1,2,5]
# ne0vec = [0.001]  # Rtol ?
T0vec = [1,10]     #

Ek0e = 1
ne0 = 1
T0e = 10

Ek0vec = [100]
ne0vec = [0.1]
T0vec = [20]
ma0 = zeros(Real,2)
ma0[1] = mp/me      # me
ma0[2] = 1   # ma is electron
# [1e-3, 10,2]
# global nitp
# for nitp in nitpvec
# global nv_limit0
# for nv_limit0 in nv_limit0vec
# println("para=",nv_limit0)
# for Ek0e in Ek0vec
# for ne0 in ne0vec
# for T0e in T0vec
    println("ne=",ne0,",Ek=",Ek0e,",T0=",T0e)
        spices =["e" "a" "b" "α" "beam" "impurity"]  # hcat() = cat(v,dims=2)
        mD0   = [ma0[1] ma0[2]   3   4      2      1   ]  # mp except mD0[1] = me
        Zq   = [ 1  2   1   2      1      1   ]  # e
        # variables: moments
        n0   = [ 1  ne0  0   0      0     0   ]  # n20
        Ek0  = [ 0 Ek0e  0   3e3      0     0   ]  # keV
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
        ma = zeros(nspices)
        ma[1] = me * mD0[1];   ma[2:nspices] = mp * mD0[2:nspices]
        qZ = Zq * e
        Eka0 = Ek0 * Tk
        na0 = n0 * n20
        Ta0 = T0 * Tk
        ua0 = @. √(2Eka0 / ma)
        ûa0 = ua0 / Mms               # û = u/vth
        Ka0 = @. 1.5na0 * Ta0 + 0.5na0 * ma * ua0^2
        vth0 = @. √(2Ta0/ ma)
        isp2 = 1
        iFv2 = 2
        maD = ma/Da
        cT = (prod(maD))^0.5 / (maD[isp2] * T0[iFv2] + maD[iFv2] * T0[isp2])^1.5
        cTq = 4.41720911682e2 * cT * Zq[isp2]^2 * Zq[iFv2]^2 * lnA()
        νTab = n0[iFv2] * cTq
        νTba = n0[isp2] * cTq
        # τ₀ = norm([1 / νTab,1 / νTba])
        τ₀ = min(1 / νTab,1 / νTba)
        println("τ₀=",τ₀,",τ₀,ab=",1 / νTab,",τ₀,ba=",1 / νTba)
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
        ρa = ma .* na
        Ia = ρa .* ua
        Is0 = sum(Ia)
        uinf = sum(Ia) / sum(ρa)
        Tinf = (Ks0 - sum(na/2 .* ma .* uinf.^2)) / (3/2 * sum(na))
        println("Is=",sum(Ia),",us0=",ûa,",u∞=",sum(uinf /Mms),",Ks0=",sum(K̂20k),",T∞=",Tinf /Tk)
        global sa0 = na .* (3/2 * (1 .+ log.(2π * ma)) - log.(na./Ta.^(3/2)))
        global s0 = sum(sa0)
        n_moments = 3
        moments = zeros(1 + n_moments,nspices)
        # moments[1,:] = sa'         # point for break consition
        moments[2,:] = n0'         # n̂a[n20]
        moments[3,:] = ûa'         # ûa[Mms]
        moments[4,:] = K̂20k'       # K̂a[K20k]
        println("snuK = ",moments)
        # (ua[1]+vth[1]) < 1e8 || throw(ArgumentError("uₑ or vₑₜₕ > 1e8 lead to distribution not normali "))
        ############## First step, the initial distribution functions
        # dimensions: [x,y,z,v,L,m,t,spices]，[v,L,m,t,spices]
        nv, ℓM = nvLMnsp(nspices,ma,na,ua,Ta,α,nv_limit0,L_limit0,Rate_fLM,Rtol_T)
        L_limit = ℓM
        LM1 = ℓM + 1
        # println("LM1t0=",LM1 )
        if nv > 29
            # println("nv1 > 90 , inv(Mv) is diffcult now, let nv1 = 90")
            nv = nv_limit0
            # nv should be less than 90 with BigFloat datas
            # or else Lₙᵅ(v̂) may cause errors due to inv(Mv)
        end
        nv1 = nv + 1     # i = 0:nv
        vG, w1v = laguerre(nv1,α)
        w1 = w1v' |> collect
        # remesh vG with interpolations for integrals and derivatives
        vu = vth + ua
        vmin = minimum(vu)/maximum(vu) / n50     # for v̂ < vG[1]
        vmin = min(vmin,vG[1]/nvG1)
        nv0itp,nvremesh,vremesh,nvgauss = remeshvgs(nv1,vmin,vG;s1=s1,nv0min=nv0min,nitp=nitp)
        v12_max = fmtf2(maximum(vremesh[2:nvremesh]./vremesh[1:nvremesh-1]))
        println("s1=",s1,",nv0min=",nv0min,",nitp=",nitp,",nvG1=",nvG1,",nv0itp=",length(nv0itp),",nG=",nv1,",nvremesh=",nvremesh,",max(v[i]/v[j])=",v12_max)
        # println("v=",Float32.([vremesh[i] for i in nvgauss]))
        # println("v=",Float32.(v))
        # dd
        # println("v=",(vremesh[nvgauss]) .-v)
        ##################################
        ns = nspices
        nsp_vec = 1:nspices
        LM = zeros(Int,nspices)
        Datatype = Float64
        fvL = zeros(Datatype,nvremesh,LM1,nspices)  # function  f(vᵢ,μⱼ,isp) without cf
        ###### fvL[v,L,nspices] and f̂nL[n,L,nspices]
        # weight for Laguerre expansion , wlg = vᵅ e⁻ᵛ,
        if α == 0
            wlg = exp.(vG)
        else
            wlg = vG.^α .* exp.(vG)
        end
        for isp in nsp_vec
            nspF = nsp_vec[nsp_vec .≠ isp]
            if nspices == 2
                LM[isp],fL0 = FLa2(L_limit,na[isp],û[isp],vremesh,Rate_fLM) # fL0 = fL0 * exp(-v)
                fvL[:,1:LM[isp] + 1,isp] = fL0
            end
        end
        fvL,dfvL,ddfvL = dvdf012L(fvL,nspices,LM,vremesh)
        # isp2 = 2
        # label = string("isp2=",isp2)
        # pf0 = plot(log.(vremesh),fvL[:,:,isp2],line=(2,:auto),legend=false)
        # pf1 = plot(log.(vremesh),dfvL[:,:,isp2],line=(2,:auto),legend=false)
        # pf2 = plot(log.(vremesh),ddfvL[:,:,isp2],line=(2,:auto),label=label,legend=true)
        # display(plot(pf0,pf1,pf2,layout=(3,1)))
        # agasgf
        ℓM_new = maximum(LM)
        if ℓM_new < ℓM
            ℓM = ℓM_new
            LM1 = ℓM + 1
            fvL = fvL[:,1:LM1,:]
        end
        μ, DP, Mμ, Mun = Dμ(ℓM;datatype = datatype)
        ################################ whether
        fvL0 = 1fvL
        t = 0
        # fvL = dfdtcon(fvL,p,t)
        # γn = momentsGaussnn(nspices,nvgauss,v,w1,fvL)
        # # for isp in nsp_vec
        # #     fvL[:,:,isp] /= γn[isp]
        # # end
        # plot(log.(vremesh),fvL[:,1,:])
        # ddsssssa
        ##### Hvu[:,:,:,isp] and Gvu[:,:,:,isp] 、coefficients Ĥ(𝓋̂ ,μ) and ĜL(𝓋̂ ,μ)
        fvu = zeros(Datatype,nvremesh,LM1,nspices)  # function  inv(Mv) * f̂L(v) * inv(Mμ)
        Hvu = zeros(Datatype,nvremesh,LM1,nspices)
        Gvu = zeros(Datatype,nvremesh,LM1,nspices)
        HvL = zeros(Datatype,nvremesh,LM1,nspices)
        GvL = zeros(Datatype,nvremesh,LM1,nspices)
        for isp in nsp_vec
            nspF = nsp_vec[nsp_vec .≠ isp]
            if nspices == 2
                iFv = nspF[1]
                L1iFv = 1:LM[iFv] + 1
                mM = ma[isp] / ma[iFv]           # ma/mb
                # for Ĥ(𝓋̂ )    # 𝓋̂ = vₐ/vbth = v̂ᵦ * (vath / vbth) due to v̂ₐ = v̂ᵦ
                vabth = vth[isp] / vth[iFv]
                ûb = û[iFv]
                if ûb < 1e-7        # 0.5 + 1e-15
                    JLn1v0 = 1//2       # Limit value of JLn1FL(v → 0)
                else
                    JLn1v0 = (√π *(1+2ûb^2) / (2ûb) * erf(ûb) + exp(-ûb^2)) / 4
                end
                HvL[:,:,isp], GvL[:,:,isp] = HGnspva(fvL[:,L1iFv,iFv],LM1,LM[iFv],
                                      datatype,nvremesh,w1,vabth,vremesh,mM,JLn1v0)
                # fvu[:,:,isp] = fvL[:,:,isp] * Mun
                # va = vremesh * vabth
                # HvL[:,:,isp],dHvL,ddHvL = dvdH012L(HvL[:,:,isp],LM[isp],va)
                # RH!(RHvL,HvL[:,:,isp],dHvL,ddHvL,FvL,LM,va)
                # RH!(RHvL,HvL,dHvL,ddHvL,FvL,L1,va)
                # GvL[:,:,isp],dGvL,ddGvL = dvdG012L(GvL[:,:,isp],LM,va)
            else
            end
        end # isp
        GvL0 = deepcopy(GvL)
        HvL0 = deepcopy(HvL)
        FvL0 = dvdF0L(fvL,ns,LM,vremesh;optims=true)
        @time HvL1 = optimRH(HvL0,FvL0,vremesh,vth,LM1);
        #############
        # isp3 = 1
        # nspF = nsp_vec[nsp_vec .≠ isp3]
        # iFv3 = nspF[1]
        # vabth = vth[isp3] / vth[iFv3]
        # va = vremesh * vabth
        # vlog = log.(va)
        # x0 = HvL0[:,L1,:]
        # if optimH == 1
        #     k = 2
        # end
        # L1 = 1
        # HvL2,dHvL,ddHvL = dvdH012L(x0,ns,L1,vremesh;k=k,optim=optimH)
        # println("Hv=",HvL2[1,isp3]/va[1],",dH=",dHvL[1,isp3])
        # label = string("H,L=",L1-1)
        # pH = plot(vlog,HvL2,label=label)
        # label = string("dH")
        # pdH = plot(vlog,dHvL,label=label)
        # label = string("ddH")
        # pddH = plot(vlog,ddHvL,label=label)
        # display(plot(pH,pdH,pddH,layout=(3,1)))
        dddd
        # xn= 0 * x0
        # RH0 = RHvL(x0)
        # R0 = [norm(RH0[iv,:]) for iv in 1:nvremesh]
        #  pdHvL = plot(log.(va),dHvL[:,:,isp],line=(2,:auto),label=label)
        ##########
        # GvL,dGvL,ddGvL = dvdG012L(GvL,ns,LM,vremesh)
        # RGvL = 0 * GvL
        # RG!(RGvL,GvL,dGvL,ddGvL,HvL,LM,vremesh,vth,ns)
        # label = string("RG,isp=",isp)
        # pRG = plot(log.(va),RGvL[:,:,isp],line=(2,:auto),label=label)
        rwetneynh
        ## #############################
        vlog = log.(vremesh)
        dvth0 = 0.0                       # dvₜₕ = ∂ₜvₜₕ = ∂ₜTₐ / (mₐvₜₕ) = vₜₕ ∂ₜTₐ/2Tₐ
        p = Dict(
          "ns" => nspices,
          "datatype" => datatype,
          "tk" => t0,
          "ma" => ma,
          "Zq" => Zq,
          "na" => na0,
          "ua" => ua0,
          "Ka" => Ka0,
          "T0" => T0,
          "dvth" => dvth0,
          "LM" => LM,
          "nv_min0" => nv_min0,
          "nv1" => nv1,
          "LM1" => LM1,
          "L_limit" => L_limit,
          "is_Lag" => is_Lag,
        )
        p = deepcopy(parameters)
        ##  ########## Collision terms, δfₗₘ/δt which is normalized by coefficients cf (without)
        # # # Si = Mvn * fnl * Mun = fvL * inv(Mμ),
        if m == 0
            nSf = 8
            @time dfL = collisionL0(nspices,nSf,datatype,nvremesh,LM1,LM,ma,Zq,
                               na,vth,vremesh,μ,DP,fvL,HvL,GvL,Mμ,Mun)
        else
        end
        #########
        # vlog = log.(vremesh)
        # nvplot = vlog .< 5
        # isp2 = 1
        # xlabel = string("δₜfL,isp2=",isp2)
        # pdfL1 = plot(vlog[nvplot],dfL[nvplot,:,isp2],xlabel=xlabel,line=(2,"auto"))
        # isp2 = 2
        # xlabel = string("δₜfL,isp2=",isp2)
        # pdfL2 = plot(vlog[nvplot],dfL[nvplot,:,isp2],xlabel=xlabel,line=(2,"auto"))
        # display(plot(pdfL1,pdfL2,layout=(2,1)))
        # ## ########
        is_con = 0
        γn = momentsGaussnn(nspices,nvgauss,vG,w1,fvL)
        if is_con == 1
            # n̂a, ûa, K̂a = momentsGaussn(nspices,nvgauss,vG,w1,fvL)
            for isp in nsp_vec
                nspF = nsp_vec[nsp_vec .≠ isp]
                iFv = nspF[1]
                mM = ma[isp] / ma[iFv]
                nb = na[iFv]
                vbth = vth[iFv]
                qma = qma0 * Zq[isp] * Zq[iFv] / (ma[isp]/ Da)
                lnAab = lnA()
                Γa = 4π * qma^2 * lnAab
                γ0, γ02, γ10 = γLm1(isp,iFv,Cγ,ma,na,ûa,K̂a,vth,dvth,dTa,nvgauss,vG,w1,dfLF,dfLR1,dfLR2)
                println("isp=",isp,",γ=",[γ0, γ02, γ10])
                L1 = 1
                dfL[:,L1,isp] = Γa / vbth^3 * (mM * dfLF[:,L1,isp] + γ02 * (dfLR1[:,L1,isp] + γ0 * dfLR2[:,L1,isp]))
                L1 = 2
                dfL[:,L1,isp] = Γa / vbth^3 * (mM * dfLF[:,L1,isp] + γ10 * (dfLR1[:,L1,isp] + dfLR2[:,L1,isp]))
            end
        end
        δn̂, δû, δK̂ = momentsGaussnd(nspices,nvgauss, vG,w1,dfL) # [nₐ, vₜₕ, nₐTₐ]
        if ua[1] == 0 && T0[1] == 1.0
            if T0e == 50
                dTa = [583.041352, - 583.04138]    # Ta = [1, 50] , ma = mb = mp
                dvth = [1.2759706e8, - 1.8044951e7]
            elseif T0e == 20
                dTa = [844.31, - 844.31]    # Ta = [1, 20] , ma = mb = mp
                dvth = [1.84775e8, - 4.13169515e7]
            elseif T0e == 10
                dTa = [1058.11053, - 1058.11053]    # Ta = [1, 10]
                dvth = [2.31565e8, - 7.3227188e7]
            elseif T0e == 5
                dTa = [1204.098674404, - 1204.09867846]    # Ta = [1, 5]
                dvth = [2.63513816e8, - 1.17846961e8]
            elseif T0e == 2
                dTa = [800.98484324, - 800.98484457]    # Ta = [1, 2]
                dvth = [1.7529342e8, - 1.23951e8]
            elseif T0e == 1.1
                dTa = [148.455, - 148.455]    # Ta = [1, 1.1]
                dvth = [3.2489e7, - 3.0977e7]
            else
                dTa = [0.0, 0.0]    # Ta = [1, 1.1]
                dvth = [0.0, 0.0]
            end
        else
            dTa = [0.0, 0.0]    # Ta = [1, 1.1]
            dvth = [0.0, 0.0]
        end
    # ## Iₐ = mₐnₐIₐ, dIa = ∂ₜIₐ
        # ## K̂ₐ = (3/2 + ûₐ²)  [nₐTₐ]
    ## with conservation constraints
        dK1 = na .* (Ta .* δK̂ )
        dI1 = ma .* na .* (vth .* δû)
        dK = na .* (Ta .* δK̂ - 2Tk * K̂a .* dTa)
        dI = ma .* na .* (vth .* δû - ûa .* dvth)
        println("dK=",fmtf4.(dK1),",dI=",fmtf4.(dI1))
# return
# dd
        ############  Solve the FP0D2V problem
        alg = solverODEAlg()
        ##  condition controls
        dtmax0 = τ₀ / ndtmax   # Maximum time step
        tmax = max(tmax, nτ * τ₀)
        tmin = t0 + dt_intial
        tspan = (tmin,tmax)  # start from the second step at t0 + dt_intial
        ##### DiscreteCallback   affect!(integrator) = integrator.u[1] += 1
        condition(u,t,integrator) = u[1,1]  > 0
        affect!(integrator) = terminate!(integrator)
        cb = DiscreteCallback(condition, affect!,save_positions=(true,true))
        ##### ContinuousCallback
        ######  cb = ContinuousCallback(condition,nothing, affect!)           # for period
        ## Files
        if nspices == 2
            path_nuT = string(path,"\\datasADTe","\\FP0Dm",m,"_tnuKT","_nsp",nspices,"_m",fmtf(mD0[1]))
            path_flm = string(path,"\\datasADTe","\\FP0Dm",m,"_tflm","_nsp",nspices,"_m",fmtf(mD0[1]))
            filename = string("\\mb",fmtf(mD0[2]),"\\n",fmtf(n0[1]),"_",fmtf(n0[2]),
                        "_u",fmtf(ûa[1]),"_",fmtf(ûa[2]),"_T",fmtf(T0[1]),"_",fmtf(T0[2]),
                        "_Atol",fmtf(Atol),"_Rtol",fmtf(Rtol),"Rtol_T=",Rtol_T,"Ratio_T=",Ratio_T,
                        "_Rds",Rtol_ds,"_nvM",nv_limit0,"_nitp",nitp,"_nitpth",nitp_vth,"_nvth",n_vth,
                        "_isL",is_Lag,"_isC",is_conservation,"_isnC",is_ncon,".csv")
        elseif  nspices == 3
        elseif  nspices == 4
        else
        end
        println(path_nuT)
        println("filename=",filename)
        file_nuT = string(path_nuT,filename)
        idnuT = open(file_nuT,"a+")     # truncate/appand, read, write, create
        nuTt = DataFrame(t=t0,na=n0[1],nb=n0[2],ua=ûa0[1],ub=ûa0[2],Ta=T0[1],Tb=T0[2])
        CSV.write(idnuT,nuTt)
        # println(idnuT,t,n0,ûa0,T0)
        prob = ODEProblem(dfdt,fvL,tspan,parameters)   # dfLm/dt  # 趋势 ok,  τ is too small
        # prob = ODEProblem(dfdtcon,fvL,tspan,parameters)   # dfLm/dt
        alg = solverODEAlg()
        if is_cb == 0
            # @time solt = solve(prob)
            # solt = solve(prob,alg,dt= dt_intial,dtmax = dtmax0,reltol=Rtol,abstol=Atol)
            @time solt = solve(prob,alg,dt= dt_intial,dtmax = dtmax0,force_dtmin=true,reltol= Rtol,abstol= Atol)
        else
            @time solt = solve(prob,alg,dt= dt_intial,dtmax = dtmax0,callback=cb,force_dtmin=true,reltol= Rtol,abstol= Atol)
            # solt = solve(prob,alg,dt= dt_intial,dtmax = dtmax0,callback=cb,force_dtmin=true,progress=true,reltol= Rtol,abstol= Atol)
        end
        close(idnuT)
        # ddddtt
        ## #### Datas,  rm(file)
        if is_write_flm == 1
            file_flm = string(path_flm,filename)
            idsol = open(file_flm,"a+")
            sol = DataFrame(solt)
            nt , nm = size(sol)
            # sol = rename!(sol,[:t,:na,:ua,:Ka,:nb,:ub,:Kb])
            unique!(sol,1)            # due to t = x[1]
            dropmissing!(sol)
            nt = length(sol.timestamp)
            CSV.write(idsol,sol)
            close(idsol)
        end
        ## ################### Plots
        if is_plot_nuT == 1
            if isfile(file_nuT) == 1
                nuTt = CSV.File(read(file_nuT)) |> DataFrame
                dropmissing!(nuTt)
                unique!(nuTt,1)            # due to t = x[1]
                nt = length(nuTt.t)
                if nt == 0
                    println("Warning: File is empty!")
                else
                tvec = nuTt.t .< 0.1 * τ₀
                # tvec =  1e-9 .< nuTt.t .< 0.05 * τ₀
                # tvec = nuTt.t .< tmax
                titlenuT = string("n=",ne0,", Ek=",Ek0e,", Tk=",T0e)
                pTb = plot(nuTt.t[tvec]/τ₀,nuTt.Tb[tvec],label="Tᵦ",xlabel="t",ylabel="T [ Tₖ ]")
                pTa = plot(nuTt.t[tvec]/τ₀,nuTt.Ta[tvec],label="Tₐ",xlabel="t",ylabel="T [ Tₖ ]")
                pT = plot(nuTt.t[tvec]/τ₀,nuTt.Ta[tvec],label="Tₐ",xlabel="t",ylabel="T [ Tₖ ]")
                pT = plot!(nuTt.t[tvec]/τ₀,nuTt.Tb[tvec],label="Tᵦ",title = titlenuT)
                display(plot(pT))
                # ### u
                # plot(nuTt.t[tvec]/τ₀,nuTt.ua[tvec],label= "uₐ",xlabel="t",ylabel="u")
                # pu = plot!(nuTt.t[tvec]/τ₀,nuTt.ub[tvec],label="uᵦ",title = titlenuT)
                # display(plot(pu))
                ##### K
                # Ka = nuTt.na .* (1.5flmt.Ta + 0.5ma[1] * nuTt.ua.^2)
                # Kb = nuTt.nb .* (1.5flmt.Tb + 0.5ma[2] * nuTt.ub.^2)
                # Ks = Ka .+ Kb
                # pK = plot(nuTt.t, Ks/Ks[1] .- 1,xlabel="t",ylabel="dK",label= "Kₛ")
                # display(plot(pK))
                end
            else
                println("Warning: File is inexistent!")
            end
        end
        println("Is=",sum(Ia),",us0=",(ua),",u∞=",sum(uinf /Mms),",Ks0=",sum(K̂20k),",T∞=",Tinf /Tk)
       println("τ₀,ab=",τ₀,",τ₀,ab=",1 / νTab,",τ₀,ba=",1 / νTba)
        println(file_nuT)
#     end
#     # pT = plot!(sol.t[tvec]/τ₀,sol.Tb[tvec],label="Tᵦ",legend=:outertopright);
# end
# end
