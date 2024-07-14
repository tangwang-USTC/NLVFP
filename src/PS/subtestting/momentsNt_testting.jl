################# General moments of `ℓᵗʰ`-order coefficient, `𝓜ⱼ(f̂ₗᵐ)`, `𝓜ⱼ(∂ᵥf̂ₗᵐ)`, `𝓜ⱼ(∂²ᵥf̂ₗᵐ)`
println()

Msnt = zeros(datatype,njMs)
Msnnt = zeros(datatype,njMs)
if nMod[isp3] == 1
    if uai0[isp3][1] == 0
        Msnt = MsntL2fM(Msnt,jvec,nai0[isp3][1],vthi0[isp3][1];is_renorm=is_renorm)      # The theoretical values of the moments of `f̂₀`
    end
    Msnnt = MsnntL2fL0(Msnnt,njMs,L,nai0[isp3][1],uai0[isp3][1],vthi0[isp3][1];is_renorm=is_renorm)
else
    Msnnt = MsnntL2fL0(Msnnt,njMs,L,nai0[isp3],uai0[isp3],vthi0[isp3],nMod[isp3];is_renorm=is_renorm)
end

Msnn3t = zeros(datatype,njMs,LM1,ns)
if prod(nMod) == 1
    Msnn3t = MsnntL2fL0(Msnn3t,njMs,LM,ns,nai0,uai0,vthi0,nMod;is_renorm=is_renorm)
    # Msnn3t = MsnntL2fL0(Msnn3t,njMs,LM,ns,nai0,uai0,vthi0;is_renorm=is_renorm)
else
    Msnn3t = MsnntL2fL0(Msnn3t,njMs,LM,ns,nai0,uai0,vthi0,nMod;is_renorm=is_renorm)
    # Msnn3t = MsnntL2fL0(Msnn3t,njMs,LM,ns,nai0,uai0,vthi0,nMod,jtype,dj;is_renorm=is_renorm)
end
