

# include(joinpath(pathroot,"src\\subtestting\\momentsNEven_testting.jl"))
################# General moments of `ℓᵗʰ`-order coefficient, `𝓜ⱼ(f̂ₗᵐ)`, `𝓜ⱼ(∂ᵥf̂ₗᵐ)`, `𝓜ⱼ(∂²ᵥf̂ₗᵐ)`
println()

n̂aG, IaG, KaG = nIKsGauss(fvL[:,1:2,:],vGk,nvlevel,nc0,nck,ma,na,vth,ns)

n̂aE, IaE, KaE = nIKs(fvL[nvlevele,1:2,:],vGe,ma,na,vth,ns)
dIaE = IaE - Ia
dIaG = IaG - Ia

dKaE = KaE - Ka
dKaG = KaG - Ka

jj, LL = 0, 0
nhaE = zeros(2,ns)
MsnnEvens!(nhaE,fvL[nvlevele,LL+1,:],vGe,jj,LL,ns;is_renorm=false)
jj, LL = 1, 1
IhaE = zeros(2,ns)
MsnnEvens!(IhaE,fvL[nvlevele,LL+1,:],vGe,jj,LL,ns;is_renorm=false)
jj, LL = 2, 0
KhaE = zeros(2,ns)
MsnnEvens!(KhaE,fvL[nvlevele,LL+1,:],vGe,jj,LL,ns;is_renorm=false)

# nhaE[1,:] .*= na
IhaE[1,:] .*= (ma .* na .* vth / 3)
KhaE[1,:] .*= (ma .* na .* vth.^2 / 2)

RDnhaE = nhaE[1,:] ./ na .- 1
RDIhaE = IhaE[1,:] ./ Ia .- 1
RDKhaE = KhaE[1,:] ./ Ka .- 1

jj = 0
# Mn0 = MsrnEvens(0.0,fvL[nvlevele,1,isp3],vGe,ma[isp3],na[isp3],vth[isp3],jj,0;is_renorm=is_renorm)
Mn0s = zeros(ns)
Mn0s = MsrnEvens(Mn0s,fvL[nvlevele,1,:],vGe,ma,na,vth,jj,0,ns;is_renorm=is_renorm)
jj = 2
# Mn2 = MsrnEvens(0.0,fvL[nvlevele,1,isp3],vGe,ma[isp3],na[isp3],vth[isp3],jj,0;is_renorm=is_renorm)
Mn2s = zeros(ns)
Mn2s = MsrnEvens(Mn2s,fvL[nvlevele,1,:],vGe,ma,na,vth,jj,0,ns;is_renorm=is_renorm)
jj = 1
# Mn1 = MsrnEvens(0.0,fvL[nvlevele,2,isp3],vGe,ma[isp3],na[isp3],vth[isp3],jj,1;is_renorm=is_renorm)
Mn1s = zeros(ns)
Mn1s = MsrnEvens(Mn1s,fvL[nvlevele,2,:],vGe,ma,na,vth,jj,1,ns;is_renorm=is_renorm)

RDKaG = KaG ./ Ka .- 1
RDKaE = KaE ./ Ka .- 1
RDKaGE = KaG ./ KaE .- 1
RDMnE = Mn2s ./ Ka .- 1

MsrnE = zeros(datatype,njMs)
MsrnE = MsrnEvens(MsrnE,fvL[nvlevele,L1,isp3],vGe,ma[isp3],na[isp3],vth[isp3],njMs,L;is_renorm=is_renorm)

MsrnEns = zeros(datatype,njMs,ns)
MsrnEns = MsrnEvens(MsrnEns,fvL[nvlevele,L1,:],vGe,ma,na,vth,njMs,L,ns;is_renorm=is_renorm)

MsrnE3 = zeros(datatype,njMs,LM1,ns)
MsrnE3 = MsrnEvens(MsrnE3,fvL[nvlevele,:,:],vGe,ma,na,vth,njMs,LM,LM1,ns;is_renorm=is_renorm)

# DMsrnEn1 = MsrnE - Msn
#
# RerrMsrnEn = (1 .- MsrnE ./ Msn)*neps
# maxRerrMsnn = maximum(abs.(RerrMsrnEn))
