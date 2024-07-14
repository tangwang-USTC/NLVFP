using IterativeSolvers

include(joinpath(path,"src\\Collision\\optimsns2.jl"))
# include(joinpath(path,"src\\Collision\\optimsns.jl"))
include(joinpath(path,"src\\solver\\iterations\\gmresJFs.jl"))

abstol = 1e-12
reltol = 1e-5
maxiterN = 292
maxiterV = 1
ϵ = 1e-4       # for Maxtrix-Vector products, Jv
k = 1
L1 = 2
optimH = 1     # (=0/1 or false/trues), for optims of ∂ⁿHvL
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
if L1 == 1
    RHvL(x0) = RHL(x0;FvL=FvL[:,L1,:],optim=optimH,k=k,L1=L1,v1=vremesh,vth=vth,ns=ns)
else
    RHvL(x0) = RHL(x0;FvL=FvL[:,L1,:],optim=optimH,k=k,L1=L1,v1=vremesh,vth=vth,ns=ns)
end
if L1 == 1
    RHvLp(x0) = RHLp(x0;FvL=FvL[:,L1,:],optim=optimH,k=k,L1=L1,v1=vremesh,vth=vth,ns=ns)
else
    RHvLp(x0) = RHLp(x0;FvL=FvL[:,L1,:],optim=optimH,k=k,L1=L1,v1=vremesh,vth=vth,ns=ns)
end
# ggg
#######################
x0 = HvL0[:,L1,:]
if optimH == 1 && k ≤ 0
    k = 2
end
xn = 1 * x0
RH0 = RHvL(x0)
# R0 = [norm(RH0[iv,:]) for iv in 1:nvremesh]
x1, hist1 = gmresnv!(xn, RHvL; ϵ = ϵ,log = false, reltol = reltol,maxiterV=maxiterV,maxiterN=maxiterN)

RHn = RHvL(xn)
Rn = [norm(RHn[iv,:]) for iv in 1:nvremesh]
##
xlabel = string("L=",L1-1,",Atol=",abstol,",Rtol=",reltol,",ϵ=",ϵ)
label = "x0"
px0 = plot(vlog[vopt],x0[vopt,:],label=label,line=(2,:auto))
label = "xn"
px0 = plot!(vlog[vopt],xn[vopt,:],label=label,line=(2,:auto))
label = string("RH0,k=",k)
pRH = plot(vlog[vopt],RH0[vopt,:],label=label,line=(2,:auto))
xlabel = string("log(v),k=",k,",miN=",maxiterN,",miV=",maxiterV)
label = "min"
mbit = abs.(RHn) .> abs.(RH0)
p12 = plot(vlog, mbit,label=label,line=(2,:auto),xlabel=xlabel)
# label = "RHn"
# pRH2 = plot(vlog[vopt],RHn[vopt,:],label=label,line=(2,:auto),xlabel=xlabel)
xn[mbit] = x0[mbit]
label = "RHn"
RHn = RHvL(xn)
pRH2 = plot(vlog[vopt],RHn[vopt,:],label=label,line=(2,:auto),xlabel=xlabel)
display(plot(pRH,px0,pRH2,p12,layout=(2,2)))
##############
# HvL,dHvL,ddHvL = dvdH012L(x0,ns,L1,vremsh;k=k,optim=optimH)
# label = string("H")
# pH = plot(vlog,HvL,label=label)
# label = string("dH")
# pdH = plot(vlog,dHvL,label=label)
# label = string("ddH")
# pddH = plot(vlog,ddHvL,label=label)
# display(plot(pH,pdH,pddH,layout=(3,1)))
# ddsgrr
# # isp0 = 2
# label = "H0"
# plot(vlog[vopt],dHvL[vopt,:],label=label,line=(2,:auto))
# isp0 = 2
# label = "Hn"
# HvL,dHvL,ddHvL = dvdH012L(xn,ns,L1,vremesh)
# pp2 = plot!(vlog[vopt],dHvL[vopt,:],label=label,line=(2,:auto))
