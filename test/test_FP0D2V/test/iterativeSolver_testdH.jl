using IterativeSolvers

include(joinpath(path,"src\\Collision\\optimsns2.jl"))
# include(joinpath(path,"src\\Collision\\optimsns.jl"))
include(joinpath(path,"src\\solver\\iterations\\gmresJFs.jl"))

# plotly()
gr()

abstol = 1e-9
reltol = 1e-4
maxiterN = 92
maxiterV = 6
ns01 = 4     # for global convergence during velocity mapping
               # s01 = 1 decays to be the lacal convergence method
ϵ = 1e-4       # for Maxtrix-Vector products, Jv
k = 2
L1 = 6
optimH = 0     # (=0/1 or false/trues), for optims of ∂ⁿHvL
##
isp3 = 1
nspF = nsp_vec[nsp_vec .≠ isp3]
iFv3 = nspF[1]
vabth = vth[isp3] / vth[iFv3]
va = vremesh * vabth
vlog = log.(va)
vopt = -9 .< vlog .< 2
Nvopt = length(va[vopt])
println()
RdHp(x0) = RdHLp(x0; FvL = FvL[:,L1,:], optim=optimH,k=k,L1=L1,v1=vremesh,vth=vth,ns=ns)
RdH(x0) = RdHL(x0; FvL = FvL[:,L1,:], optim=optimH,k=k,L1=L1,v1=vremesh,vth=vth,ns=ns)
#######################
x0 = HvL0[:,L1,:]
if optimH == 1 && k ≤ 0
    k = 2
end
xn = 1 * x0
println("x01=",xn[73:75,:])
RH0 = RdH(x0)
# display(plot(vlog,RH0))
# R0 = [norm(RH0[iv,:]) for iv in 1:nvremesh]
x1 = gmresvnkg!(xn, RdH; ϵ = ϵ,log = false, ns01 = ns01,  reltol = reltol,maxiterV=maxiterV,maxiterN=maxiterN)

println("x02=",x1[73:75,:])
RHn = RdH(x1)
Rn = [norm(RHn[iv,:]) for iv in 1:nvremesh]
##
xlabel = string("L=",L1-1,",Atol=",abstol,",Rtol=",reltol,",ϵ=",ϵ)
label = "x0"
px0 = plot(vlog[vopt],x0[vopt,:],label=label,line=(2,:solid))
label = "xn"
px0 = plot!(vlog[vopt],x1[vopt,:],label=label,line=(2,:solid))
label = string("RH0,k=",k)
pRH = plot(vlog[vopt],RH0[vopt,:],label=label,line=(2,:solid))
xlabel = string("k=",k,",miN=",maxiterN,",miV=",maxiterV)
label = "min"
mbit = abs.(RHn) .> abs.(RH0)
p12 = plot(vlog, mbit,label=label,line=(2,:auto),xlabel=xlabel)
# # label = "RHn"
# # pRH2 = plot(vlog[vopt],RHn[vopt,:],label=label,line=(2,:auto),xlabel=xlabel)
# x1[mbit] = x0[mbit]
label = "RHn"
RHn = RdH(x1)
pRH2 = plot(vlog[vopt],RHn[vopt,:],label=label,line=(2,:solid),xlabel=xlabel)
display(plot(pRH,px0,pRH2,p12,layout=(2,2)))
##############
# HvL,dHvL = dvdH01L(x0,ns,L1,vremesh;k=k,optim=optimH)
# label = string("H,L=",L1-1)
# pH = plot(vlog,HvL,label=label)
# label = string("dH,isp=",isp3)
# pdH = plot(vlog,dHvL,label=label)
# display(plot(pH,pdH,layout=(2,1)))
# ddsgrr
# # isp0 = 2
# label = "H0"
# plot(vlog[vopt],dHvL[vopt,:],label=label,line=(2,:auto))
# isp0 = 2
# label = "Hn"
# HvL,dHvL,ddHvL = dvdH012L(xn,ns,L1,vremesh)
# pp2 = plot!(vlog[vopt],dHvL[vopt,:],label=label,line=(2,:auto))
