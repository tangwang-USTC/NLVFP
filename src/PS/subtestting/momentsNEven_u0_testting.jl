# When `u0 > 0`

################# General moments of `ℓᵗʰ`-order coefficient, `𝓜ⱼ(f̂ₗᵐ)`, `𝓜ⱼ(∂ᵥf̂ₗᵐ)`, `𝓜ⱼ(∂²ᵥf̂ₗᵐ)`
println()

Msnn = zeros(datatype,njMs)
Msnn = Msnnorm(Msnn,fvL[:,L1,isp3],vGk,nvlevel,nc0,nck,njMs,L;is_renorm=is_renorm)

Msnn3 = zeros(datatype,njMs,LM1,ns)
Msnn3 = Msnnorm(Msnn3,fvL,vGk,nvlevel,nc0,nck,njMs,LM,LM1,ns;is_renorm=is_renorm)

##
MsnnE = zeros(datatype,njMs)
MsnnE = MsnnEvens(MsnnE,fvL[nvlevele,L1,isp3],vGe,njMs,L;is_renorm=is_renorm)

MsnnEns = zeros(datatype,njMs,ns)
MsnnEns = MsnnEvens(MsnnEns,fvL[nvlevele,L1,:],vGe,njMs,L,ns;is_renorm=is_renorm)

MsnnE3 = zeros(datatype,njMs,LM1,ns)
MsnnE3 = MsnnEvens(MsnnE3,fvL[nvlevele,:,:],vGe,njMs,LM,LM1,ns;is_renorm=is_renorm)
ispa = isp3
MsnnE3na = MsnnE3[:,:,ispa] ./ (MsnnE3[1,:,ispa])'
ispb = iFv3
MsnnE3nb = MsnnE3[:,:,ispa] ./ (MsnnE3[1,:,ispa])'

##
DMsnnEn1 = MsnnE - Msnn
DMsnnEn3 = MsnnE3 - Msnn3
RDMsnnEn3 = DMsnnEn3 ./ Msnn3 *neps

RerrMsnnEn = (1 .- MsnnE ./ Msnn) *neps
maxRerrMsnn = maximum(abs.(RerrMsnnEn))

## Plotting
# title = string("vM,nnv,ocp,nvG,nc0,nck=",(vGmax,nnv,ocp,nvG,nc0,nck))
title = string("vM,nnv,ocp=",(vGmax,nnv,ocp))
label = string("RerrMsnnEn,L=",L1-1)
xlabel = string("j, maxRMsnn=",fmtf2(maxRerrMsnn),"[eps]",",nvG,nc0,nck=",(nvG,nc0,nck))
# ylabel = string("Relative error of moments `MsnnE`")
pRMsnnEn = plot(jvec,RerrMsnnEn,title=title,label=label)
xlabel!(xlabel)
# ylabel!(ylabel)

display((plot(pRMsnnEn)))
# display((plot(pRMsnn,pRMsnnE,pRMsnnEn,layout=(3,1))))
