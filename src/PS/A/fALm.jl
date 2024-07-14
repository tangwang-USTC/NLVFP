"""
  𝒜ₗᵐ (x,y,z,v̂) for advection term in physics space when m ≥ 0,
   especialy, 𝒜₀⁰ = 0 when LM = 0; otherwise, 𝒜₀⁰ ~ f̂₁.

   𝒜ₗᵐ(x[j],t[k]) due to f(x[j],v̂,t) is at the location of E[j,t] with length(fₗᵐ[:,v̂]) = nx.

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

   Alm =  v̂ * ∑ᵢ(ΓAᵢLm[j] - ΓAᵢLm[j-1]) where ΓAᵢLm[j] = ΓAᵢLm(xⱼ), i=1,2,3.
     ΓAᵢLm(xⱼ) = v̂ * vth * (fAiLm)
       fAxLm = -(L-m)/(2L-1) * flm(L-1,m) - (L+m+1)/(2L+3) * flm(L+1,m)
       fAyLm =
       fAzLm =

   With open boundaries, for example of 1D model,
     ΓALm[1] = ΓALm[2] = ΓALm[1/2]
     ΓALm[Nₓ] = ΓALm[Nₓ-1] = ΓALm[Nₓ-1/2]

  Inputs:
    ΓALm .= 0
    L:
    m:
    LM: the maximum order of M
    vth: vth[xⱼ]
    flm: f̂(L,m) = f̂ₗᵐ(i,v̂), the target order (ℓ,m)
    fl1m: f̂(L+1,m)
    f1lm: f̂(L-1,m)
    nx = (nx, ny, nz)

  Output:
    ΓAₗᵐ(x,y,v̂,L,m)
      = [ΓAₗᵐ[1,1,v̂,:,:], ΓAₗᵐ[1,2,v̂,:,:] ... ΓAₗᵐ[1,Ny-1,v̂,:,:]
         ΓAₗᵐ[2,1,v̂,:,:], ΓAₗᵐ[2,2,v̂,:,:] ... ΓAₗᵐ[2,Ny-1,v̂,:,:]
                              :
         ΓAₗᵐ[Nx-1,1,v̂,:,:], ΓAₗᵐ[Nx-1,2,v̂,:,:] ... ΓAₗᵐ[Nx-1,Ny-1,v̂,:,:]]

    ΓALmnD3V!(ΓALm,dimsx,flm,v,LM,vth,nx)

"""

function ΓALmnD3V!(ΓALm,dimsx,flm,v,LM,vth,nx::Tuple)
    if dimsx == 1
        for i in 2:nx[1] - 1
            vi = vth[i] * v
            flmi = flm[i]
            # L = 0, m = 0 ,
            L1 = 1
            # m1 = 1
            if LM == 0
                # ΓALm[i][:,L1,1] .= 0
            else
                fl1m = flmi[:,L1+1,1]     # f(L+1,m)
                ΓALm[i][:,1,1] = - vi / 3 .* fl1m
            end
            for L in 1:LM
                L1 = L + 1
                if L == LM
                    for m in 0:L
                        m1 = m + 1
                        if m == 0
                            f1lm = flmi[:,L1-1,m1]     # f(L-1,m)
                            ΓALm[i][:,L1,m1] = - L/(2L-1) * vi .* f1lm
                        elseif m == L
                            ΓALm[i][:,L1,m1] .=  0
                        else
                            f1lm = flmi[:,L1-1,m1]     # f(L-1,m)
                            ΓALm[i][:,L1,m1] = - (L-m)/(2L-1) * vi .* f1lm
                        end
                    end
                else
                    for m in 0:L
                        m1 = m + 1
                        if m == 0
                            f1lm = flmi[:,L1-1,m1]     # f(L-1,m)
                            fl1m = flmi[:,L1+1,m1]     # f(L+1,m)
                            ΓALm[i][:,L1,m1] = - vi .* ( L/(2L-1) * f1lm + L1/(2L+3) * fl1m )
                        elseif m == L
                            fl1m = flmi[:,L1+1,m1]     # f(L+1,m)
                            ΓALm[i][:,L1,m1] =  - (L+L1)/(2L+3) * vi .* fl1m
                        else
                            f1lm = flmi[:,L1-1,m1]     # f(L-1,m)
                            fl1m = flmi[:,L1+1,m1]     # f(L+1,m)
                            ΓALm[i][:,L1,m1] =  - vi .* ( (L-m)/(2L-1) * f1lm + (L+m1)/(2L+3) * fl1m )
                        end
                    end
                end
            end
        end
        # Open boundaries
        # i = 1
        for L in 0:LM
            L1 = L + 1
            for m in 0:L
                ΓALm[1][:,L1,m+1] = ΓALm[2][:,L1,m+1]
            end
        end
        # i = nx[1]
        for L in 0:LM
            L1 = L + 1
            for m in 0:L
                ΓALm[nx[1]][:,L1,m+1] = ΓALm[nx[1]-1][:,L1,m+1]
            end
        end
    elseif dimsx == 2
    else
    end
end

"""
  𝒜ₗᵐ = v̂ * ∑ᵢ ∂xᵢ Γ𝒜ᵐ[i], with Γ𝒜ᵐ[i] = vth * f̂𝒜ᵐ[i] where i = 1, vth = √(2Ta/ma).

    f̂𝒜ᵐ[1] = vth * (c1[1] * f̂[L-1,m] + c1[2] * f̂[L+1,m])
      with c1[1] = - (L-m)/(2L-1), c1[2] = - (L+1)/(2L+3)
      L = 0
      L = LM, | m = 0:
              | m = L:
              | m,     f̂𝒜ᵐ[1] = vth * c1[1] * f̂[L-1,m]
      L ,     | m = 0:
              | m = L: f̂𝒜ᵐ[1] =          vth *  c1[2] * f̂[L+1,m]
              | m,

  Inputs:
    vth = vth(xᵢ)
    flm = flm(xᵢ,v̂)

  Outputs:
    f̂𝒜ᵐ[1]  = ΓALm0D3V1(L,m,LM,f1lm,fl1m)
"""

function ΓALm0D3V1(L,m,LM,vth,f1lm,fl1m)
    if L == 0
        if LM == 0
            return 0
        else
            return -  vth / 3 * fl1m
        end
    elseif L == LM
        if m == 0
            return - vth * L*(2L-1) * f1lm
        elseif m == L
            return  0
        else
            return  - vth * (L-m)*(2L-1) * f1lm
        end
    else
        if m == 0
            return - vth * L*(2L-1) * f1lm - vth * (L+1)*(2L+1) * fl1m
        elseif m == L
            return  - vth * (L+m+1)*(2L+1) * fl1m
        else
            return  - vth * (L-m)*(2L-1) * f1lm - vth * (L+m+1)*(2L+1) * fl1m
        end
    end
end

"""
  𝒜ₗᵐ = v̂ * ∑ᵢ ∂xᵢ Γ𝒜ᵐ[i], with Γ𝒜ᵐ[i] = vth * f̂𝒜ᵐ[i] where i = 1:2, vth = √(2Ta/ma).

    f̂𝒜ᵐ[1] = vth * (c1[1] * f̂[L-1,m] + c1[2] * f̂[L+1,m])
      with c1[1] = - (L-m)/(2L-1), c1[2] = - (L+1)/(2L+3)
      L = 0
      L = LM, | m = 0:
              | m = L:
              | m,     f̂𝒜ᵐ[1] = vth * c1[1] * f̂[L-1,m]
      L ,     | m = 0:
              | m = L: f̂𝒜ᵐ[1] =          vth *  c1[2] * f̂[L+1,m]
              | m,

      where f̂𝒜ᵐ[1]  = ΓALm1D3V(L,m,LM,f1lm,fl1m)

    f̂𝒜ₗᵐ[2] =

  Inputs:
    vth = vth(xᵢ)
    flm = flm(xᵢ,v̂)

  Outputs:
    f̂𝒜ₗᵐ = [f̂𝒜ₗᵐ[1], f̂𝒜ₗᵐ[2]] = ΓALm0D3V2(L,m,LM,vth,f1lm,fl1m)
"""

function ΓALm0D3V2(L,m,LM,vth,f1lm,fl1m)
    if L == 0
        return -  vth / 3 * fl1m
    elseif L == LM
        if m == 0
            return - vth * L*(2L-1) * f1lm
        elseif m == L
            return  0
        else
            return  - vth * (L-m)*(2L-1) * f1lm
        end
    else
        if m == 0
            return - vth * L*(2L-1) * f1lm - vth * (L+1)*(2L+1) * fl1m
        elseif m == L
            return  - vth * (L+m+1)*(2L+1) * fl1m
        else
            return  - vth * (L-m)*(2L-1) * f1lm - vth * (L+m+1)*(2L+1) * fl1m
        end
    end
end


"""
  𝒜ₗᵐ = v̂ * ∑ᵢ ∂xᵢ Γ𝒜ᵐ[i], with Γ𝒜ᵐ[i] = vth * f̂𝒜ᵐ[i] where i = 1:3, vth = √(2Ta/ma).

    f̂𝒜ᵐ[1] = vth * (c1[1] * f̂[L-1,m] + c1[2] * f̂[L+1,m])
      with c1[1] = - (L-m)/(2L-1), c1[2] = - (L+1)/(2L+3)
      L = 0
      L = LM, | m = 0:
              | m = L:
              | m,     f̂𝒜ᵐ[1] = vth * c1[1] * f̂[L-1,m]
      L ,     | m = 0:
              | m = L: f̂𝒜ᵐ[1] =          vth *  c1[2] * f̂[L+1,m]
              | m,

      where f̂𝒜ᵐ[1]  = ΓALm1D3V(L,m,LM,f1lm,fl1m)

    f̂𝒜ₗᵐ[2] =

    f̂𝒜ₗᵐ[3] =

  Inputs:
    vth = vth(xᵢ)
    flm = flm(xᵢ,v̂)

  Outputs:
    f̂𝒜ₗᵐ = [f̂𝒜ₗᵐ[1], f̂𝒜ₗᵐ[2], f̂𝒜ₗᵐ[3]] = ΓALm0D3V3(L,m,LM,vth,f1lm,fl1m)
"""

function ΓALm0D3V(L,m,LM,vth,f1lm,fl1m)
    if L == 0
        return -  vth / 3 * fl1m
    elseif L == LM
        if m == 0
            return - vth * L*(2L-1) * f1lm
        elseif m == L
            return  0
        else
            return  - vth * (L-m)*(2L-1) * f1lm
        end
    else
        if m == 0
            return - vth * L*(2L-1) * f1lm - vth * (L+1)*(2L+1) * fl1m
        elseif m == L
            return  - vth * (L+m+1)*(2L+1) * fl1m
        else
            return  - vth * (L-m)*(2L-1) * f1lm - vth * (L+m+1)*(2L+1) * fl1m
        end
    end
end
