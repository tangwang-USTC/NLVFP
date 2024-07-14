

"""
  Residuals of Possion equations which could expressed as:

   ∇²Ĥ(v̂,μ) = - 4πF̂(𝓋̂,μ)

  In spherical harmonics coordinate, the residuals could be written as:

   RĤL = ddĤL + 2 * dĤL /𝓋̂  + 4π * F̂L - L(L+1) * ĤL/𝓋̂²

   RdĤL = dĤL + 2 * ĤL /𝓋̂  - (dĤL + 2 * ĤL /𝓋̂)|ᵥ→₀ + ∫₀^𝓋̂ (4π * F̂L- (L(L+1)+2) * ĤL/𝓋̂²)d𝓋̂

   RdĤL2 = dĤL + (∫₀^𝓋̂ (4π * 𝓋̂² * F̂L - L(L+1) * ĤL)d𝓋̂ - (𝓋̂² * dĤL)|ᵥ→₀) / 𝓋̂²

   RdH is more better than RdH2 for all L^th order except for L = 1

                = ⥜      when L = 0              dᵥĤ = 0
  (HL ./ 𝓋̂)ᵥ→₀  = const  when L = 1             dᵥĤ = const
                = 0

  (HL ./ 𝓋̂)ᵥ→₀ = (dᵥĤ)ᵥ→₀ when L ≥ 1

  Inputs:
    HvLn: HvL[:,L1,isp]
    FLn: FvL[:,L1,isp]
    ratio_df: is different for variable `û` and `L1`, and will tend to be `1` when `L1 → 1` and `û → 0`

  Outputs:
    HvL1 = optimRH!(HvLn,FvLn,v1,vth,LM1;ϵ=ϵ,ns=ns,L1m=L1m,abstol=abstol,reltol=reltol,
               maxiterA=maxiterA,maxiterN=maxiterN,λ=λ,k=k,vcfilter=vcfilter,s=s)
"""

nc = 57
domain = [-1,1]
# vgc = chebyshev_nodes(nc,domain)
vgc = chebyshev_extrema(nc,domain)
# mapping to [0,∞] with domain [v_min,v_max]
vdom = [1e-2,100]
v1 = 0.5(vdom[1] - vdom[2]) * vgc .+ 0.5(vdom[1] + vdom[2])
Dc = chebyshevdiff(nc;M=2)
Dc2 = Dc[:,:,2]
Dc1 = Dc[:,:,1]
A = Dc2 + diagm(2 ./ vgc)
