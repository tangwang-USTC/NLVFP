ncv0 = zeros(Int64,2,LM1,ns)
# Applying the Explicit Newton method to solve the ODE equations
dt = 1e-0           # (=1e2 default) Which is normalized by `td ~ 1e10` for MCF plasma.
# `1e0 ~ 1e4` is proposed to be used for most situation.


L1m = 9
vmin = 0.1
vmax = 8.3
Diffmethod = 2

RdtfvL0k = dtfvL0[:,L1m,:] ./ fvL0[:,L1m,:]
RDfvL0k = dt * neps * RdtfvL0k
yy = copy(dtfvL0[:,L1m,isp3])
order_dvdtf = -1
RdvdtfvL0 = zeros(nvG)
RdvdtfvL0 = dvdtfvL0CDS(RdvdtfvL0,yy,nvG,dvGe,fvL0[:,L1m,isp3],L1m;orders=order_dvdtf)
order_dvdtf = 1
RdvdtfvL01 = zeros(nvG)
RdvdtfvL01 = dvdtfvL0CDS(RdvdtfvL01,yy,nvG,dvGe,fvL0[:,L1m,isp3],L1m;orders=order_dvdtf)
order_dvdtf = 2
RdvdtfvL02 = zeros(nvG)
RdvdtfvL02 = dvdtfvL0CDS(RdvdtfvL02,yy,nvG,dvGe,fvL0[:,L1m,isp3],L1m;orders=order_dvdtf)
[RdvdtfvL0 RdvdtfvL01 RdvdtfvL02]
if Diffmethod == 1
    DRdtfvL0k = diff(RdtfvL0k;dims=1) / dvGe * neps
elseif Diffmethod == 2
    DRdtfvL0k = diff(dtfvL0[:,L1m,:];dims=1) / dvGe ./ fvL0[2:end,L1m,:] * neps
else
    DRdtfvL0k = diff(dtfvL0[:,L1m,:];dims=1) / dvGe ./ fvL0[2:end,L1m,:] ./ RdtfvL0k[2:end,:] * neps
end
DDRdtfvL0k = diff(DRdtfvL0k;dims=1) ./ DRdtfvL0k[2:end,:] / dvGe

# datadtdRfvL0a = DataFrame(zeros(nvG-1,LM1+2),:auto)
# datadtdRfvL0a[:,1] = vGe[2:end]
# datadtdRfvL0a[:,2:end-1] = diff(RdtfvL0k[nvlevele0,:,isp3];dims=1)
# datadtdRfvL0a[:,end] = vGe[2:end]

vvec = vmin .≤ vG0 .< vmax
label = string("L = ",L1m-1)
plpfvL0k1a = plot(vG0[vvec],RDfvL0k[vvec,isp3],label=label,
    xlabel="v̂",ylabel="RΔfvLa [eps]",line=(2,:auto),legend=legendtR)
plpfvL0k1b = plot(vG0[vvec],RDfvL0k[vvec,iFv3],label=label,
    xlabel="v̂",ylabel="RΔfvLb [eps]",line=(2,:auto),legend=legendtR)
vvec2 = vmin .< vG0[2:end] .< vmax
if Diffmethod == 1
    plpDRdtfvL0k1a = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2,isp3], label=label,
        xlabel="v̂", ylabel="DRdtfvLa [eps]", line=(2, :auto), legend=false)
    plpDRdtfvL0k1b = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2,iFv3], label=label,
        xlabel="v̂", ylabel="DRdtfvLb [eps]", line=(2, :auto), legend=false)
elseif Diffmethod == 2
    plpDRdtfvL0k1a = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2,isp3], label=label,
        xlabel="v̂", ylabel="RDdtfvLa [eps]", line=(2, :auto), legend=false)
    plpDRdtfvL0k1b = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2,iFv3], label=label,
        xlabel="v̂", ylabel="RDdtfvLb [eps]", line=(2, :auto), legend=false)
else
    plpDRdtfvL0k1a = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2,isp3], label=label,
        xlabel="v̂", ylabel="RDdtfvLa [eps]", line=(2, :auto), legend=false)
    plpDRdtfvL0k1b = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2,iFv3], label=label,
        xlabel="v̂", ylabel="RDdtfvLb [eps]", line=(2, :auto), legend=false)
end
vvec3 = vmin .< vG0[3:end] .< vmax
plpDDRdtfvL0k1a = plot(vG0[3:end][vvec3],DDRdtfvL0k[vvec3,isp3],label=label,
xlabel="v̂",ylabel="DDRdtfvLa [eps]",line=(2,:auto),legend=false)
plpDDRdtfvL0k1b = plot(vG0[3:end][vvec3],DDRdtfvL0k[vvec3,iFv3],label=label,
xlabel="v̂",ylabel="DDRdtfvLb [eps]",line=(2,:auto),legend=false)
display(plot(plpfvL0k1a,plpfvL0k1b,plpDRdtfvL0k1a,plpDRdtfvL0k1b,
plpDDRdtfvL0k1a,plpDDRdtfvL0k1b,layout=(3,2)))

if 1 == 2
    nvec = 0.08 .≤ vG0 .≤ 10
    plot(vG0[nvec],RdtfvL0[nvec,:,isp3],line=(2,:auto))
    1
end