
"""
  Computing `fvu = f̂(v̂,ℓ) = fvL * Mun`
"""

# fvu = fvL * Mun
function fvuTransformSH!(fvu,fvL,PL0,LM1)

    for L1 in 1:LM1
        fvu[:,L1] = fvL * PL0[L1,:]
    end
    return fvu
end

Lv = 0:LM1-1
# plmμ = Plm(Lv,Lv,μ)  # Plm(ℓ,m,μ) , ℓ=0:L, m=0:ℓ
plmμ = Plm(Lv,Lv,μ)
PL0 = Plm(Lv,0,μ)
fvu2 = zero.(fvL2)
fvuTransformSH!(fvu2,fvL2,PL0,LM1)
ddfvu2 = zero.(ddfvL2)
fvuTransformSH!(ddfvu2,ddfvL2,PL0,LM1)

a = (fvL20./vG0 * Mun - dfvL2 * Mun) .* (ddfvL2 * Mun)
a1 = (fvL20./vG0 - dfvL2) * Mun .* (ddfvL2 * Mun)
a2 = (fvL20./vG0 * Mun) .* (ddfvL2 * Mun) - dfvL2 * Mun .* (ddfvL2 * Mun)

aG20 = (GvL2[:,3]./vG0 - dGvL2[:,3])
a = (GvL2./vG0 - dGvL2)
aG0 = a[:,1] .* vG0
aG1 = a[:,2] ./ vG0.^2
aG2 = a[:,3] ./ vG0
aG3 = a[:,4] ./ vG0.^2
aG4 = a[:,5] ./ vG0.^3
