using IterativeSolvers

include(joinpath(path,"src\\Collision\\optimsns2.jl"))
# include(joinpath(path,"src\\Collision\\optimsns.jl"))
include(joinpath(path,"src\\solver\\iterations\\gmresJFs.jl"))

abstol = 1e-12
reltol = 1e-5
maxiterN = 292
maxiterV = 5
ϵ = 1e-4       # for Maxtrix-Vector products, Jv
k = 2
L1 = 3
optimG = 1     # (=0/1 or false/trues), for optims of ∂ⁿGvL
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
    RGvL(x0) = RGL(x0;HvL=HvL[:,L1,:],optim=optimG,k=k,L1=L1,v1=vremesh,vth=vth,ns=ns)
else
    RGvL(x0) = RGL(x0;HvL=HvL[:,L1,:],optim=optimG,k=k,L1=L1,v1=vremesh,vth=vth,ns=ns)
end
if L1 == 1
    RGvLp(x0) = RGLp(x0;HvL=HvL[:,L1,:],optim=optimG,k=k,L1=L1,v1=vremesh,vth=vth,ns=ns)
else
    RGvLp(x0) = RGLp(x0;HvL=HvL[:,L1,:],optim=optimG,k=k,L1=L1,v1=vremesh,vth=vth,ns=ns)
end
# ggg
#######################
x0 = GvL0[:,L1,:]
if optimG == 1 && k ≤ 0
    k = 2
end
xn = 1 * x0
RG0 = RGvL(x0)
# R0 = [norm(RG0[iv,:]) for iv in 1:nvremesh]
x1, hist1 = gmresnv!(xn, RGvL; ϵ = ϵ,log = false, reltol = reltol,maxiterV=maxiterV,maxiterN=maxiterN)

RGn = RGvL(xn)
Rn = [norm(RGn[iv,:]) for iv in 1:nvremesh]
##
xlabel = string("L=",L1-1,",Atol=",abstol,",Rtol=",reltol,",ϵ=",ϵ)
label = "x0"
px0 = plot(vlog[vopt],x0[vopt,:],label=label,line=(2,:auto))
label = "xn"
px0 = plot!(vlog[vopt],xn[vopt,:],label=label,line=(2,:auto))
label = string("RG0,k=",k)
pRG = plot(vlog[vopt],RG0[vopt,:],label=label,line=(2,:auto))
xlabel = string("log(v),k=",k,",miN=",maxiterN,",miV=",maxiterV)
label = "min"
mbit = abs.(RGn) .> abs.(RG0)
p12 = plot(vlog, mbit,label=label,line=(2,:auto),xlabel=xlabel)
# label = "RGn"
# pRG2 = plot(vlog[vopt],RGn[vopt,:],label=label,line=(2,:auto),xlabel=xlabel)
xn[mbit] = x0[mbit]
label = "RGn"
RGn = RGvL(xn)
pRG2 = plot(vlog[vopt],RGn[vopt,:],label=label,line=(2,:auto),xlabel=xlabel)
display(plot(pRG,px0,pRG2,p12,layout=(2,2)))
##############
# GvL,dGvL,ddGvL = dvdG012L(x0,ns,L1,vremsh;k=k,optim=optimG)
# label = string("G")
# pG = plot(vlog,GvL,label=label)
# label = string("dG")
# pdG = plot(vlog,dGvL,label=label)
# label = string("ddG")
# pddG = plot(vlog,ddGvL,label=label)
# display(plot(pG,pdG,pddG,layout=(3,1)))
# ddsgrr
# # isp0 = 2
# label = "G0"
# plot(vlog[vopt],dGvL[vopt,:],label=label,line=(2,:auto))
# isp0 = 2
# label = "Gn"
# GvL,dGvL,ddGvL = dvdG012L(xn,ns,L1,vremesh)
# pp2 = plot!(vlog[vopt],dGvL[vopt,:],label=label,line=(2,:auto))
