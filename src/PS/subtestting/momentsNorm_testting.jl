################# General moments of `ℓᵗʰ`-order coefficient, `𝓜ⱼ(f̂ₗᵐ)`, `𝓜ⱼ(∂ᵥf̂ₗᵐ)`, `𝓜ⱼ(∂²ᵥf̂ₗᵐ)`
println()

Msnn = zeros(datatype,njMs)
Msnn = Msnnorm(Msnn,fvL[:,L1,isp3],vGk,nvlevel,nc0,nck,njMs,L;is_renorm=is_renorm)

Msnn3 = zeros(datatype,njMs,LM1,ns)
Msnn3 = Msnnorm(Msnn3,fvL,vGk,nvlevel,nc0,nck,njMs,LM,LM1,ns;is_renorm=is_renorm)

RDMsnn1 = Msnn ./ Msnnt
DMsnn1 = Msnn - Msnnt
DMsnn = Msnn3[:,L1,isp3] - Msnnt
RerrMsnn = (1 .- Msnn ./ Msnnt)*neps
maxRerrMsnn = maximum(abs.(RerrMsnn))

## Plotting
title = string("vM,nnv,ocp=",(vGmax,nnv,ocp))
label = string("RerrMsnn,L=",L)
xlabel = string("j, maxRMsnn=",fmtf2(maxRerrMsnn),"[eps]")
# ylabel = string("Relative error of moments `Msnn`")
pRMsnn = plot(jvec,RerrMsnn,title=title,label=label)
xlabel!(xlabel)
# ylabel!(ylabel)
# display((plot(pRMsnn)))
