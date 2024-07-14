
"""
  When `nMod ≥ 2`, the value of

  `dtMst`, `δtfvLt` and then `dtMsnEnt` and `dtKaEt`

  may be not properly computed.
"""

################# General moments of `ℓᵗʰ`-order coefficient, `𝓜ⱼ(f̂ₗᵐ)`, `𝓜ⱼ(∂ᵥf̂ₗᵐ)`, `𝓜ⱼ(∂²ᵥf̂ₗᵐ)`
println()
is_renorm = false
dtMst = zeros(datatype,njMs,ns)
if prod(nMod) == 1
    dtMst = dtMsrnt(dtMst,njMs,ma,Zq,na,vth,ns)
    # dtMst = dtMsrnt(dtMst,njMs,ma,Zq,na,vth,ns,nai,vthi,nMod;is_renorm=is_renorm)
else
    dtMst = dtMsrnt(dtMst,njMs,ma,Zq,na,vth,ns,nai,vthi,nMod;is_renorm=is_renorm)
    jj = 2
    dtKat = zeros(ns)
    dtKat = dtMsrnt(dtKat,jj,ma,Zq,na,vth,ns,nai,vthi,nMod;is_renorm=is_renorm)
    RdtKat = dtKat ./ dtMst[2,:] .- 1
    DdtKat = sum(dtKat) / dtKat[1]
end
DdtKat = sum(dtMst[2,:])
RDdtKat = DdtKat / dtMst[2,1]

δtfvL02 = copy(δtfvL0)

# δtfvL02[1,:,:] = copy(δtfvL02[2,:,:])

dtn̂aE, dtIaE, dtKaE = nIKs(δtfvL02[:,1:2,:],vGe,ma,na,vth,ns)
DdtKaE = sum(dtKaE)
RDdtKaE = DdtKaE / dtKaE[1]
if nSdtf == 1
    dtn̂aEt, dtIaEt, dtKaEt = nIKs(δtfvLt[:,1:2,:],vGe,ma,na,vth,ns) # ;is_renorm=is_renorm
    aaa = dtKaE ./ dtKaEt .- 1
    DdtM2t = dtMst[2,:] - dtKaEt
    DdtKaEt = sum(dtKaEt)
    RDdtKaEt = DdtKaEt / dtKaEt[1]
elseif nSdtf == 7
    dtn̂aEt7, dtIaEt7, dtKaEt7 = nIKs(δtfvLt7[:,:,:],vGe,ma,na,vth,ns;is_renorm=is_renorm)
    aaa7 = dtKaE ./ dtKaEt7
    DdtKaEt7 = sum(dtKaEt7)
    RDdtKaEt7 = DdtKaEt7 / dtKaEt7[1]
end

dtMsnE = zeros(datatype,njMs)
dtMsnE = MsrnEvens(dtMsnE,δtfvL02[:,L1,isp3],vGe,
            ma[isp3],na[isp3],vth[isp3],njMs,L;is_renorm=is_renorm)

jj = L + 2
dtMsnEjL = zeros(datatype,ns)
dtMsnEjL = MsrnEvens(dtMsnEjL,δtfvL02[:,L1,:],vGe,
            ma,na,vth,jj,L,ns;is_renorm=is_renorm)

DdtMsnEjL = sum(dtMsnEjL)
RDdtMsnEjL = DdtMsnEjL / dtMsnEjL[1]

dtMsnEnt = zeros(datatype,njMs,ns)
dtMsnEnt = MsrnEvens(dtMsnEnt,δtfvLt[:,L1,:],vGe,
            ma,na,vth,njMs,L,ns;is_renorm=is_renorm)

dtMsnEns = zeros(datatype,njMs,ns)
dtMsnEns = MsrnEvens(dtMsnEns,δtfvL02[:,L1,:],vGe,
            ma,na,vth,njMs,L,ns;is_renorm=is_renorm)

DdtMs = dtMst - dtMsnEns
DdtMst = dtMst - dtMsnEnt
DdtMsEnEt = dtMsnEnt - dtMsnEns

RDdtMs = DdtMs ./ dtMst
RDdtMst = DdtMst ./ dtMst
RDdtMsEnEt = DdtMsEnEt ./ dtMsnEnt

RDdtMs[1,:] = DdtMs[1,:]
RDdtMst[1,:] = DdtMst[1,:]
RDdtMsEnEt[1,:] = DdtMsEnEt[1,:]


dtMsnE3 = zeros(datatype,njMs,LM1,ns)
dtMsnE3 = MsrnEvens(dtMsnE3,δtfvL02[:,:,:],vGe,ma,na,vth,njMs,LM,ns;is_renorm=is_renorm)

jtype = :n2
dj = 1
dtMsnE31 = zeros(datatype,njMs,LM1,ns)
dtMsnE31 = MsrnEvens(dtMsnE31,δtfvL02[:,:,:],vGe,ma,na,vth,njMs,LM,ns,jtype,dj;is_renorm=is_renorm)

ratdtMsnE31 = dtMsnE31[:,:,1] ./ dtMsnE31[:,:,2]
