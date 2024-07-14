
# When `u0[isp3] > 0`

################# General moments of `ℓᵗʰ`-order coefficient, `𝓜ⱼ(f̂ₗᵐ)`, `𝓜ⱼ(∂ᵥf̂ₗᵐ)`, `𝓜ⱼ(∂²ᵥf̂ₗᵐ)`
println()

δtfvL02 = copy(δtfvL0)
# δtfvL02[1,:,:] = copy(δtfvL02[2,:,:])
dtn̂aE, dtIaE, dtKaE = nIKs(δtfvL02[:,1:2,:],vhe,ma,na,vth,ns)
DdtKaE = sum(dtKaE)
RDdtKaE = DdtKaE / dtKaE[1]
@show L,L1,RDdtKaE,dtn̂aE

dtMsnE = zeros(datatype,njMs)
dtMsnE = MsrnEvens(dtMsnE,δtfvL02[:,L1,isp3],vGe,
            ma[isp3],na[isp3],vth[isp3],njMs,L;is_renorm=is_renorm)

jj = L + 2
dtMsnEjL = zeros(datatype,ns)
dtMsnEjL = MsrnEvens(dtMsnEjL,δtfvL02[:,L1,:],vGe,ma,na,vth,jj,L,ns;is_renorm=is_renorm)

DdtMsnEjL = sum(dtMsnEjL)
RDdtMsnEjL = DdtMsnEjL / dtMsnEjL[1]
@show RDdtMsnEjL 

dtMsnEns = zeros(datatype,njMs,ns)
dtMsnEns = MsrnEvens(dtMsnEns,δtfvL02[:,L1,:],vGe,ma,na,vth,njMs,L,ns;is_renorm=is_renorm)


dtMsnE3 = zeros(datatype,njMs,LM1,ns)
dtMsnE3 = MsrnEvens(dtMsnE3,δtfvL02[:,:,:],vGe,ma,na,vth,njMs,LM,ns;is_renorm=is_renorm)

jtype = :n2
dj = 1
dtMsnE31 = zeros(datatype,njMs,LM1,ns)
dtMsnE31 = MsrnEvens(dtMsnE31,δtfvL02[:,:,:],vGe,ma,na,vth,njMs,LM,ns,jtype,dj;is_renorm=is_renorm)

ratdtMsnE31 = dtMsnE31[:,:,1] ./ dtMsnE31[:,:,2]
