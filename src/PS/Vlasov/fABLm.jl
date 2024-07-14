"""
  ∂ₜf̂ₗᵐ = 𝒜ₗᵐ + 𝓑ₗᵐ; especially,
     𝓑ₗ₌₀⁰ = 0
     𝒜₀⁰ = 0 when LM = 0; otherwise, 𝒜₀⁰ ~ f̂₁.

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

"""


"""
   flm⁺[i] = flm⁻[i] + fABlm[i] where

    fABlm[i] = Δt * 𝓑ₗᵐ[i] + Δt / (x[i+0.5]-x[i-0.5]) * (ΓAlm[i+1] - ΓAlm[i-1])/2
             = Δt * 𝓑ₗᵐ[i] + Δt / (x[i+1]-x[i-1]) * (ΓAlm[i+1] - ΓAlm[i-1])

      Δt * 𝓑ₗᵐ = flm_B!(Blm,Zmdt,dimsx,flm,LM,B,nx)
      ΓAlm = ΓALmnD3V!(ΓALm,dimsx,flm,LM,vth,nx)

  Output:
    flm = update_flm_AB!(flm,Zma,dt,dimsx,v,LM,B,vth,nx)

"""
# xb = p["xb"]
function update_flm_AB!(flm,Zma,dt,dimsx,LM,B,vth,xb,nx::Tuple)
    Zmdt = Zma * dt
    ΓA = 0 * flm
    ΓALmnD3V!(ΓA,dimsx,flm,v,LM,vth,nx)
    # println("ΓA=",ΓA[1:4])
# display(plot!(v[v.<10],real(ΓA[2][v.<10,1,1]),xlabel="v",ylabel="ΓA00"))
# display(plot!(v[v.<10],real(ΓA[2][v.<10,2,1]),xlabel="v",ylabel="ΓA10"))
# display(plot!(v[v.<10],real(ΓA[2][v.<10,2,2]),xlabel="v",ylabel="ΓA11"))
# display(plot!(v[v.<10],imag(ΓA[2][v.<10,2,2]),xlabel="v",ylabel="ΓA1n1"))
# display(plot!(v[v.<10],real(ΓA[3][v.<10,3,1]-ΓA[1][v.<10,3,1]),xlabel="v",ylabel="ΓA20"))
# display(plot!(v[v.<10],real(ΓA[2][v.<10,3,3]),xlabel="v",ylabel="ΓA21"))
# display(plot!(v[v.<10],imag(ΓA[2][v.<10,3,2]),xlabel="v",ylabel="ΓA2n1"))
# display(plot!(v[v.<10],real(ΓA[2][v.<10,3,3]),xlabel="v",ylabel="ΓA22"))
# display(plot!(v[v.<10],imag(ΓA[2][v.<10,3,3]),xlabel="v",ylabel="ΓA2n2"))
        # Main procedure,  𝒜ₗᵐ + 𝓑ₗᵐ
    if dimsx == 1
        if LM == 0
            for i in 2:1:nx[1]-1
                flm[i] +=  dt / (xb[i+1]-xb[i-1]) * (ΓA[i+1] - ΓA[i-1])
            end
            # i = 1
            # flm[i] = flm[i]
            # i = nx[1]
        else
            Blm = 0 * flm
            flm_B!(Blm,Zmdt,dimsx,flm,LM,B,nx)
            for i in 2:1:nx[1]-1
                flm[i] += Blm[i] + dt / (xb[i+1]-xb[i-1]) * (ΓA[i+1] - ΓA[i-1])
            end
            i = 1
            flm[i] += Blm[i]
            i = nx[1]
            flm[i] += Blm[i]
        end
    elseif dimsx == 2
        for i in 1:nx[1]
        for j in 1:nx[2]
        end
        end
    else
    end
end


"""
   flm⁺[i] = flm⁻[i] + flm⁻[i-1] - flm⁺[i-1] + fABlm[i-0.5] where

    fABlm[i-0.5] = Δt * (𝓑ₗᵐ[i] + 𝓑ₗᵐ[i-1]) +
                   2Δt * 2 / (x[i]-x[i-1]) * (ΓAlm[i] - ΓAlm[i-1])

      Δt * 𝓑ₗᵐ = flm_B!(Blm,Zmdt,dimsx,flm,LM,B,nx)
      ΓAlm = ΓALmnD3V!(ΓALm,dimsx,flm,LM,vth,nx)

  Output:
    flm = update_flm_AB!(flm,Zma,dt,dimsx,v,LM,B,vth,nx)

"""
# xb = p["xb"]
function update_flm_AB2!(flm,Zma,dt,dimsx,LM,B,vth,xb,nx::Tuple)
    Zmdt = Zma * dt
    ΓA = 0 * flm
    # ΓALmnD3V!(ΓA,dimsx,flm,v,LM,vth,nx)
    if dimsx == 1
        if LM == 0
            # boundary
            i = 1
            # flmi1 = 0
            flmi = flm[i]
            flm[i] = flmi
            i = 2  #
            flmi1 = flmi
            flmi = flm[i]
            flm[i] = flmi + flmi1 - flm[i-1] +  2dt / (xb[i]-xb[i-1]) * (ΓA[i] - ΓA[i-1])
            # Main procedure
            # 𝒜ₗᵐ + 𝓑ₗᵐ , reuse ΓA(x,v,L,m) as ABlm(x,v,L,m)
            dt2 = 2dt
            for i in 3:1:nx[1]
                ΓA[i-2] =  dt2 / (xb[i]-xb[i-1]) * (ΓA[i] - ΓA[i-1])
            end
            for i in 3:1:nx[1]-1
                flmi1 = flmi
                flmi = flm[i]
                flm[i] = flmi + flmi1 -  flm[i-1] + ΓA[i-2]
            end
        else
            Blm = 0 * flm
            flm_B!(Blm,Zmdt,dimsx,flm,LM,B,nx)
            # boundary
            i = 1
            flmi1 = 0
            flmi = flm[i]
            flm[i] = flmi + Blm[i]
            println(size(flm[3]))
            dd
            i = 2  # ΓA[2] = ΓA[1]
            flmi1 = flmi
            flmi = flm[i]
            flm[i] = flmi + flmi1 -  flm[i-1] + (Blm[i] + Blm[i-1]) +  2dt / (xb[i]-xb[i-1]) * (ΓA[i] - ΓA[i-1])
            # Main procedure
            # 𝒜ₗᵐ + 𝓑ₗᵐ , reuse ΓA(x,v,L,m) as ABlm(x,v,L,m)
            dt2 = 2dt
            for i in 3:1:nx[1]
                ΓA[i-2] = (Blm[i] + Blm[i-1]) + dt2 / (xb[i]-xb[i-1]) * (ΓA[i] - ΓA[i-1])
            end
            for i in 3:1:nx[1]-1
                flmi1 = flmi
                flmi = flm[i]
                flm[i] = flmi + flmi1 - flm[i-1] + ΓA[i-2]
            end
        end
    elseif dimsx == 2
        for i in 1:nx[1]
        for j in 1:nx[2]
        end
        end
    else
    end
end
