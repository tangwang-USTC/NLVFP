using Legendre, GaussQuadrature, SpecialFunctions
using Trapz
using Plots
"""
  F0(vᵦ) = 4/√π /vᵦₜₕ³ * exp((-vᵦ/vᵦₜₕ)^2)
  F̂0(v̂ᵦ) = 4/√π * exp((-v̂ᵦ)^2)

  I0(F0) = ∫₀ᵛ(vᵦ² * F0(vᵦ))dvᵦ
         = erf(𝓋̂ ) - 2/√π * 𝓋̂ * exp(-𝓋̂²)
"""

nv = 30
α = 0.0
n20 = 1e20
vbth = 2e6
vath = 1e6

v, w1 = laguerre(nv,α)
va = vb = v * vbth
𝓋 = va / vbth         # = v
va1 = v * vath        # = v * vath / vbth
𝓋1 = va1 / vbth

F0(v,vth) = 4/√π * n20 / vth^3 * exp(- (v/vth)^2)
F̂0(v̂) = 4/√π * n20 * exp(- v̂^2)
I0(𝓋) = erf(𝓋) - 2/√π * 𝓋 * exp(-𝓋^2)
##
Fb0 = F0.(vb,vbth)
F̂b0 = F̂0.(v)
Fa0 = F0.(va,vath)
F̂a0 = F̂0.(v)
na = I0.(𝓋)
na1 = I0.(𝓋1)
na2 = zeros(nv)
for i in 1:nv
    na2[i] = trapz(vb[1:i],vb[1:i].^2 .* Fb0[1:i]) / n20
end
na2 = zeros(nv)
for i in 1:nv
    na2[i] = trapz(vb[1:i],vb[1:i].^2 .* Fb0[1:i]) / n20
end
plot(𝓋,na)
plot!(𝓋1,na1)
plot(v,F̂b0)
plot(vb,Fb0)
plot!(v,F̂b0)
nv3 = vb .< 5max(vbth,vath)
plot(va[nv3],Fa0[nv3])
plot!(vb[nv3],Fb0[nv3])
plot(va[nv3]/vath,Fa0[nv3])
plot!(vb[nv3]/vbth,Fb0[nv3])
