is_renorm = true

################# General moments of `ℓᵗʰ`-order coefficient, `𝓜ⱼ(f̂ₗᵐ)`, `𝓜ⱼ(∂ᵥf̂ₗᵐ)`, `𝓜ⱼ(∂²ᵥf̂ₗᵐ)`

# L1 = 1
# L = L1 - 1

dtMsnnE = zeros(datatype,njMs)
dtMsnnE = MsnnEvens(dtMsnnE,dtfvL0[isp3][:,L1],vhe[isp3],njMs,L;is_renorm=is_renorm)

# jj = L - 0
# dtMsnnEjL = zeros(datatype,ns)
# dtMsnnEjL = MsnnEvens(dtMsnnEjL,dtfvL0[:,L1,:],vhe[isp3],jj,L,ns;is_renorm=is_renorm)

# dtMsnnEns = zeros(datatype,njMs,ns)
# dtMsnnEns = MsnnEvens(dtMsnnEns,dtfvL0[:,L1,:],vhe,njMs,L,ns;is_renorm=is_renorm)

# ratdtMs = dtMsnnEns[:,1] ./ dtMsnnEns[:,2]

dtMsnnE3 = zeros(datatype,njMs,LM1,ns)
dtMsnnE3 = MsnnEvens(dtMsnnE3,dtfvL0,vhe,njMs,LM,LM1,ns;is_renorm=is_renorm)


# ratdtMsnnE3 = dtMsnnE3[:,:,1] ./ dtMsnnE3[:,:,2]

# jtype = :n2
# dj = 1
# dtMsnnE31 = zeros(datatype,njMs,LM1,ns)
# dtMsnnE31 = MsnnEvens(dtMsnnE31,dtfvL0,vhe,njMs,LM,LM1,ns,jtype,dj)

# ratdtMsnnE31 = dtMsnnE31[:,:,1] ./ dtMsnnE31[:,:,2]


@show "dtMsnnE3 = ";
dtMsnnE3