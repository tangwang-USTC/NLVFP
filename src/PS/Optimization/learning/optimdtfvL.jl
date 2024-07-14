ncv0 = zeros(Int64,2,LM1,ns)
# Applying the Explicit Newton method to solve the ODE equations
dt = 1e-0           # (=1e2 default) Which is normalized by `td ~ 1e10` for MCF plasma.
# `1e0 ~ 1e4` is proposed to be used for most situation.


Diffmethod = 1
vmin = 0.00
vmax = 1.3
L1min = 1
L1max = LM1 - 0
L1max = L1min + 3

RdtfvL0k = dtfvL0 ./ fvL0
RDfvL0k1 = dt * neps * RdtfvL0k
if Diffmethod == 1
    DRdtfvL0k = diff(RdtfvL0k;dims=1) / dvGe * neps
elseif Diffmethod == 2
    DRdtfvL0k = diff(dtfvL0;dims=1) / dvGe ./ fvL0[2:end,:,:] * neps
else
    DRdtfvL0k = diff(dtfvL0;dims=1) / dvGe ./ fvL0[2:end,:,:] ./ RdtfvL0k[2:end,:,:] * neps
end
DDRdtfvL0k = diff(DRdtfvL0k;dims=1) ./ DRdtfvL0k[2:end,:,:] / dvGe

datadtdRfvL0a = DataFrame(zeros(nvG-1,LM1+2),:auto)
datadtdRfvL0a[:,1] = vGe[2:end]
datadtdRfvL0a[:,2:end-1] = diff(RdtfvL0k[nvlevele0,:,isp3];dims=1)
datadtdRfvL0a[:,end] = vGe[2:end]

vvec = vmin .≤ vG0 .< vmax
label = (L1min:L1max)'
Lvec = L1min:L1max
plpfvL0k1a = plot(vG0[vvec],RDfvL0k1[vvec,Lvec,isp3],label=label,
    xlabel="v̂",ylabel="RΔfvLa [eps]",line=(2,:auto),legend=legendtR)
plpfvL0k1b = plot(vG0[vvec],RDfvL0k1[vvec,Lvec,iFv3],label=label,
    xlabel="v̂",ylabel="RΔfvLb [eps]",line=(2,:auto),legend=legendtR)
vvec2 = vmin .< vG0[2:end] .< vmax
if Diffmethod == 1
    plpDRdtfvL0k1a = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2, Lvec, isp3], label=label,
        xlabel="v̂", ylabel="DRdtfvLa [eps]", line=(2, :auto), legend=false)
    plpDRdtfvL0k1b = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2, Lvec, iFv3], label=label,
        xlabel="v̂", ylabel="DRdtfvLb [eps]", line=(2, :auto), legend=false)
elseif Diffmethod == 2
    plpDRdtfvL0k1a = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2, Lvec, isp3], label=label,
        xlabel="v̂", ylabel="RDdtfvLa [eps]", line=(2, :auto), legend=false)
    plpDRdtfvL0k1b = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2, Lvec, iFv3], label=label,
        xlabel="v̂", ylabel="RDdtfvLb [eps]", line=(2, :auto), legend=false)
else
    plpDRdtfvL0k1a = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2, Lvec, isp3], label=label,
        xlabel="v̂", ylabel="RDdtfvLa [eps]", line=(2, :auto), legend=false)
    plpDRdtfvL0k1b = plot(vG0[2:end][vvec2], DRdtfvL0k[vvec2, Lvec, iFv3], label=label,
        xlabel="v̂", ylabel="RDdtfvLb [eps]", line=(2, :auto), legend=false)
end
vvec3 = vmin .< vG0[3:end] .< vmax
plpDDRdtfvL0k1a = plot(vG0[3:end][vvec3],DDRdtfvL0k[vvec3,Lvec,isp3],label=label,
xlabel="v̂",ylabel="DDRdtfvLa [eps]",line=(2,:auto),legend=false)
plpDDRdtfvL0k1b = plot(vG0[3:end][vvec3],DDRdtfvL0k[vvec3,Lvec,iFv3],label=label,
xlabel="v̂",ylabel="DDRdtfvLb [eps]",line=(2,:auto),legend=false)
display(plot(plpfvL0k1a,plpfvL0k1b,plpDRdtfvL0k1a,plpDRdtfvL0k1b,
plpDDRdtfvL0k1a,plpDDRdtfvL0k1b,layout=(3,2)))

    