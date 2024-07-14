"""
  The low frequency component of Ampere's circuital law:

  ∂ₜ𝐄 = - 𝐉
     = - Zₐvₜₕnₐû = - Zₐ/3 * 𝓜₁(𝐟̂₁)
     = - Zₐ/3 * ∫dv̂(4π * v̂³ * 𝐟̂₁)

  where
    𝓜ₗ(f̂ₗᵐ) = 4π * ∫dv̂(v̂^(L+2) * fₗᵐ) with v̂ ∈ [0,∞)
    f(x,v̂,t) is at the location of E[j+1] with length(f[:,v̂]) = nx.

  Meshgrids in x dimesion:

  ```
  1   3/2            j-1 j-0.5   j                      Nx-1 Nx-0.5  Nx
  |----o----|----o----|----o----| ... |----o----|----o----|----o----|
 E[1]     E[2]                E[j]                               E[Nx]  nE = nx
f[1,:]                       f[j,:]                             f[Nx,:] nf[:,1] = nx
     B[1+0.5]          B[j+0.5]                             B[Nx+0.5]   nB = nx -1
  ```

  Meshgrids in velocity space:
  v̂ = [0; [v̂_G]; ∞]
  nv = nG + 2
  ℓ = 0:ℓM
  nL = ℓM + 1
  m = -ℓM:1:ℓM
  nm = 2ℓ + 1
  nϕ = nm + 1 = 2ℓ + 2

   E⁺(i) = E⁻(i) - dt * Zₐ/3 * vₜₕ(i) * 𝓜₁(𝐟̂₁(i)) ,

   where
         Eₛ[i] = E(xᵢ)
         cL = (2L + 1)/(2L - 1)
         CfE = Za/ma * cL * Δt/2 /(v̂[j,α+1] * Δv̂[j,α+0.5])
         Δv̂[j,α+0.5] = v̂[j,α+1] - v̂[j,α]

  Inputs:
    E = [E1[i], E2[i], E3[i]]
    Zdt = Za * dt
    dimsx:
    v: [0; [v̂_G]; ∞]
    nv = nG + 2
    f1l0: f̂lm(L=1,m=0)  # f̂₁ᵐ(:,v̂) = {flm, for l = 1, m = 0:1}
    f1l1: f̂lm(L=1,m=1)
    nx = (nx, ny, nz)

  Output:
    𝐄 = 𝐄(x,y,z)

    update_EJ!(E,J,dimsx,dt,nx::Tuple)
"""

# E = Ex = Ey = Ez
function update_EJ!(E,J,dimsx,dt,nx::Tuple)
  if dimsx == 1
    for i = 1:nx[1]
      E[i] -= dt * J[i]
    end
  elseif dimsx == 2
    for j = 2:nx[2]-1
      for i = 2:nx[1]-1
        E[i,j] -= dt * J[i,j]
      end
    end
  else
  end
end
