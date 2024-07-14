
nspvec = 1:ns
if e == e
    # 3D
    fvL0e = Vector{Matrix{datatype}}(undef,ns)
    for isp in nspvec
        fvL0e[isp] = zeros(nvG[isp],L_limit+1)
    end
    if prod(nMod0) == -1
        LM, fvL0e = fvLDMz(fvL0e,vhe,LM,ns,nai0,uai0,vthi0;
                         L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
    else
        LM, fvL0e = fvLDMz(fvL0e,vhe,nvG,LM,ns,nai0,uai0,vthi0,nMod0;L_limit=L_limit,
                         rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM,is_LM1_full=is_LM1_full)
    end
    LM1 = maximum(LM) + 1
end
fvLc0e = copy(fvL0e)
for isp in 1:ns
    fvLc0e[isp] *= (na[isp] / vth[isp]^3 / pi^1.5)
end

mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1 - 1)
# μ, w2, Mμ, Mun, Mun1, Mun2 = LegendreMμ012(LM1-1)
# mu = reshape(μ,1,LM1)

# μ, w2, Mμ, Mun, Mun1, Mun2 = LegendreMμ012(LM[isp3])
# mu = reshape(μ,1,LM[isp3]+1)
