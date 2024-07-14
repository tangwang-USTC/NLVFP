

################# General moments of `ℓᵗʰ`-order coefficient, `𝓜ⱼ(f̂ₗᵐ)`, `𝓜ⱼ(∂ᵥf̂ₗᵐ)`, `𝓜ⱼ(∂²ᵥf̂ₗᵐ)`
println()
jj, LL = 0, 0
# MsnnerrI00 = zeros(2)
# MsnnEvens!(MsnnerrI00,fvL[nvlevele,1,isp3],vGe,jj,LL;is_renorm=is_renorm)
Msnn0 = MsnnEvens(0.0,fvL[nvlevele,1,isp3],vGe,jj,LL;is_renorm=is_renorm)
# Msnn0s = zeros(ns)
# Msnn0s = MsnnEvens(Msnn0s,fvL[nvlevele,1,:],vGe,jj,LL;is_renorm=is_renorm)
jj, LL = 2, 0
# MsnnerrI20 = zeros(2)
# MsnnEvens!(MsnnerrI20,fvL[nvlevele,1,isp3],vGe,jj,LL;is_renorm=is_renorm)
Msnn2 = MsnnEvens(0.0,fvL[nvlevele,1,isp3],vGe,jj,LL;is_renorm=is_renorm)
# Msnn2s = zeros(ns)
# Msnn2s = MsnnEvens(Msnn2s,fvL[nvlevele,1,:],vGe,jj,LL;is_renorm=is_renorm)
jj, LL = 1, 1
# MsnnerrI11 = zeros(2)
# MsnnEvens!(MsnnerrI11,fvL[nvlevele,2,isp3],vGe,jj,LL;is_renorm=is_renorm)
Msnn1 = MsnnEvens(0.0,fvL[nvlevele,2,isp3],vGe,jj,LL;is_renorm=is_renorm)
# Msnn1s = zeros(ns)
# Msnn1s = MsnnEvens(Msnn1s,fvL[nvlevele,2,:],vGe,ma,na,vth,jj,1,ns;is_renorm=is_renorm)

jj, LL = 0, 0
MsnnerrI00 = zeros(2,ns)
MsnnEvens!(MsnnerrI00,fvL[nvlevele,LL+1,:],vGe,jj,LL,ns;is_renorm=is_renorm)

jj, LL = 2, 0
MsnnerrI20 = zeros(2,ns)
MsnnEvens!(MsnnerrI20,fvL[nvlevele,LL+1,:],vGe,jj,LL,ns;is_renorm=is_renorm)
jj, LL = 1, 1
MsnnerrI11 = zeros(2,ns)
MsnnEvens!(MsnnerrI11,fvL[nvlevele,LL+1,:],vGe,jj,LL,ns;is_renorm=is_renorm)

MsnnE = zeros(datatype,njMs)
MsnnE = MsnnEvens(MsnnE,fvL[nvlevele,L1,isp3],vGe,njMs,L;is_renorm=is_renorm)

MsnnEns = zeros(datatype,njMs,ns)
MsnnEns = MsnnEvens(MsnnEns,fvL[nvlevele,L1,:],vGe,njMs,L,ns;is_renorm=is_renorm)

MsnnE3 = zeros(datatype,njMs,LM1,ns)
MsnnE3 = MsnnEvens(MsnnE3,fvL[nvlevele,:,:],vGe,njMs,LM,LM1,ns;is_renorm=is_renorm)

RDMsnnE1 = MsnnE ./ Msnnt
DMsnnE1 = MsnnE - Msnnt
DMsnnEn1 = MsnnE - Msnn
DMsnn1
DMsnnE = MsnnE3[:,L1,isp3] - Msnnt

RerrMsnnE = (1 .- MsnnE ./ Msnnt)*neps
RerrMsnnEn = (1 .- MsnnE ./ Msnn)*neps
RerrMsnn
maxRerrMsnn = maximum(abs.(RerrMsnnE))
maxRerrMsnn = maximum(abs.(RerrMsnnEn))

## Plotting
# title = string("vM,nnv,ocp,nvG,nc0,nck=",(vGmax,nnv,ocp,nvG,nc0,nck))
label = string("RerrMsnnE")
xlabel = string("j, maxRMsnn=",fmtf2(maxRerrMsnn),"[eps]")
ylabel = string("Relative error of moments `MsnnE`")
pRMsnnE = plot(jvec,RerrMsnnE,label=label)
xlabel!(xlabel)
ylabel!(ylabel)

label = string("RerrMsnnE_n")
xlabel = string("j, maxRMsnn=",fmtf2(maxRerrMsnn),"[eps]",",nvG,nc0,nck=",(nvG,nc0,nck))
ylabel = string("Relative error of moments `MsnnE`")
pRMsnnEn = plot(jvec,RerrMsnnEn,label=label)
xlabel!(xlabel)
# ylabel!(ylabel)

# display((plot(pRMsnnE)))
display((plot(pRMsnn,pRMsnnE,pRMsnnEn,layout=(3,1))))
