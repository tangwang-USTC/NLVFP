
"""
  𝓑ₗᵐ (x,y,z,v̂) for magnetic field diffusion term when m ≥ 0 ;
    especially, 𝓑₀⁰ = 0.

   f(x[j],v̂,t) is at the location of E[j] with length(fₗᵐ[:,v̂]) = nx.

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

   when 1 ≤ ℓ ≤ ℓM - 1,
    𝓑ₗᵐ = - Za/ma * (im * m * B1 * f̂(L,m) +
         (im/2 * B2 + 1 /2 * B3) * f̂(L,m-1) +
         (im/2 * B2 - 1 /2 * B3) *  (L - m) * (L + m + 1) * f̂(L,m+1))

   Especially when m = 0 and m = L,
    𝓑ₗ⁰ = - Za/ma * L * (L + 1) * (B2 * 𝕴(f̂ₗ¹)+ B3 * 𝕽(f̂ₗ¹)), ~ f̂(L,1)
    𝓑ₗᴸ = - Za/ma * (im * L * B1 * f̂(L,L)) + 1/2 * (im * B2 + B3) * f̂(L,L-1)))

  Inputs:
    Blm .= 0
    L:
    m:
    Zmdt = Za/ma * dt  # dt = ht both for FDTD and Hsplit
    B = B[B1[i], B2[i], B3[i]] , Bs[i] = (Bs[i-1/2] + Bs[i+1/2]) / 2 with CDM.
    fl1m: f̂(L,m-1)
    flm0: f̂(L,m) = f̂ₗᵐ(i,v̂)
    flm1: f̂(L,m+1)

  Output:
    𝓑ₗᵐ(x,y,v̂,L,m)

    = [𝓑ₗᵐ[1,1,v̂,:,:], 𝓑ₗᵐ[1,2,v̂,:,:] ... 𝓑ₗᵐ[1,Ny-1,v̂,:,:]
       𝓑ₗᵐ[2,1,v̂,:,:], 𝓑ₗᵐ[2,2,v̂,:,:] ... 𝓑ₗᵐ[2,Ny-1,v̂,:,:]
                            :
       𝓑ₗᵐ[Nx-1,1,v̂,:,:], 𝓑ₗᵐ[Nx-1,2,v̂,:,:] ... 𝓑ₗᵐ[Nx-1,Ny-1,v̂,:,:]]

    = flm_B!(Blm,Zmdt,dimsx,flm,LM,B,nx)

"""

function flm_B!(Blm,Zmdt,dimsx,flm,LM,B,nx::Tuple)
    if dimsx == 1
        for i in 1:nx[1]
            if i == 1
                Bi = [B[1][i],B[2][i],B[3][i]]
            elseif i == nx[1]
                Bi = [B[1][i-1],B[2][i-1],B[3][i-1]]
            else
                Bi = 0.5 *[B[1][i] + B[1][i+1], B[2][i] + B[2][i+1], B[3][i] + B[3][i+1]]
            end
            # L = 0  Blm(L=0,m=0) .= 0  # L1 = 1  # m1 = 1
            # Blm[i][:,1,1] .= 0
            flmi = flm[i]
            for L in 1:LM
                L1 = L + 1
                if L == LM
                    for m in 0:L
                        m1 = m + 1
                        if m == 0
                            flm1 = flmi[:,L1,m1+1]     # f(L,m + 1)
                            Blm[i][:,L1,m1] = Blm0D3V(~,L,m,LM,Zmdt,Bi,~,flm1)
                        else
                            if m == L
                                fl1m = flmi[:,L1,m1-1]     # f(L,m - 1)
                                flm0 = flmi[:,L1,m1]       # f(L,m)
                                Blm[i][:,L1,m1] = Blm0D3V(flm0,L,m,LM,Zmdt,Bi,fl1m,~)
                            else
                                fl1m = flmi[:,L1,m1-1]     # f(L,m - 1)
                                flm0 = flmi[:,L1,m1]       # f(L,m)
                                Blm[i][:,L1,m1] = Blm0D3V(flm0,L,m,LM,Zmdt,Bi,fl1m,~)
                            end
                        end
                    end
                else
                    for m in 0:L
                        m1 = m + 1
                        if m == 0
                            flm1 = flmi[:,L1,m1+1]     # f(L,m + 1)
                            Blm[i][:,L1,m1] = Blm0D3V(~,L,m,LM,Zmdt,Bi,~,flm1)
                        else
                            if m == L
                                fl1m = flmi[:,L1,m1-1]     # f(L,m - 1)
                                flm0 = flmi[:,L1,m1]       # f(L,m)
                                Blm[i][:,L1,m1] = Blm0D3V(flm0,L,m,LM,Zmdt,Bi,fl1m,~)
                            else
                                fl1m = flmi[:,L1,m1-1]     # f(L,m - 1)
                                flm0 = flmi[:,L1,m1]       # f(L,m)
                                flm1 = flmi[:,L1,m1+1]     # f(L,m + 1)
                                Blm[i][:,L1,m1] = Blm0D3V(flm0,L,m,LM,Zmdt,Bi,fl1m,flm1)
                            end
                        end
                    end
                end
            end
        end
    elseif dimsx == 2
    else
    end
end

"""
  Inputs:
    L:
    m:
    Zmdt = Za/ma * dt  # dt = ht both for FDTD and Hsplit
    B = B[B1[i], B2[i], B3[i]] , Bs[i] = (Bs[i-1/2] + Bs[i+1/2]) / 2 with CDM.
    fl1m: f̂(L,m-1)
    flm0: f̂(L,m) = f̂ₗᵐ(i,v̂)
    flm1: f̂(L,m+1)

  Output:
    𝓑ₗᵐ[i,v̂] = 𝓑ₗᵐ(xᵢ,v̂) = Blm0D3V(flm0,L,m,LM,Zmdt,B,fl1m,flm1)

"""

function Blm0D3V(flm0,L,m,LM,Zmdt,B,fl1m,flm1)
    if L == 0
        # return 0
    else
        if m == 0
            return Zmdt * L * (L + 1) * ( B[2] * imag(flm1)+ B[3] * real(flm1) )
        else
            if m == L
                return - 0.5Zmdt * (im * 2m * B[1] * flm0 + (im * B[2] + B[3]) * fl1m )
                # return - 0.5Zmdt * (im * 2m * B[1] * flm0 + (im * B[2] ) * fl1m )
                # return - 0.5Zmdt * ( + ( + B[3]) * fl1m )
            else
                if L == LM
                    return - 0.5Zmdt * (im * 2m * B[1] * flm0 + (im * B[2] + B[3]) * fl1m )
                else
                    return - 0.5Zmdt * (im * 2m * B[1] * flm0 + (im * B[2] + B[3]) * fl1m + (L - m) * (L + m + 1) * (im * B[2] - B[3]) * flm1 )
                end
            end
        end
    end
end

"""
  when 1 ≤ ℓ ≤ ℓM - 1,
   𝓑ₗᵐ = - Za/ma * (im * m * B1 * f̂(L,m) +
        (im/2 * B2 + 1 /2 * B3) * f̂(L,m-1) +
        (im/2 * B2 - 1 /2 * B3) *  (L - m) * (L + m + 1) * f̂(L,m+1))

  Bs[i] = (Bs[i-1/2] + Bs[i+1/2]) / 2 with CDM. However what's the boundary scheme?

  Inputs:
    Zma = Za/ma
    B = B[B1[i], B2[i], B3[i]]

  Outputs:
    BlmnD3V!(flm0,L,m,Zma,B,fl1m,flm1,nx)
    𝓑ₗᵐ(x,y,z) = flm0

"""
#
# function BlmnD3V!(flm0,L,m,Zma,B,fl1m,flm1,nx::Tuple)
#     dimsx = length(nx)
#     if dimsx == 1
#         if m == 0
#             for i in 1:nx[1]
#                 flm0[i] = B[2][i] * imag(flm1[i])+ B[3][i] * real(flm1[i])
#             end
#             flm0 = Zma * L * (L + 1) * flm0
#         elseif m == L
#             for i in 1:nx[1]
#                 flm0[i] = - ( im * L * B[1][i] * flm0[i] + 1/2 * (im * B[2][i] + B[3][i]) * fl1m[i] )
#             end
#             flm0 = Zma * flm0
#         else
#             clm = (L - m) * (L - m + 1)
#             for i in 1:nx[1]
#                 flm0[i] = - 0.5 * (im * 2m * B[1][i] * flm0[i] + (im * B[2][i] + B[3][i]) * fl1m[i] + clm * (im * B[2][i] - B[3][i]) * flm1[i] )
#             end
#             flm0 = Zma * flm0
#         end
#     elseif dimsx == 2
#         if m == 0
#             for i in 1:nx[1]
#                 for j in 1:nx[2]
#                     flm0[i,j] = B[2][i,j] * imag(flm1[i,j])+ B[3][i,j] * real(flm1[i,j])
#                 end
#             end
#             flm0 = Zma * L * (L + 1) * flm0
#         elseif m == L
#             for i in 1:nx[1]
#                 for j in 1:nx[2]
#                     flm0[i,j] = - ( im * L * B[1][i,j] * flm0[i,j] + 1/2 * (im * B[2][i,j] + B[3][i,j]) * fl1m[i,j] )
#                 end
#             end
#             flm0 = Zma * flm0
#         else
#             clm = (L - m) * (L - m + 1)
#             for i in 1:nx[1]
#                 for j in 1:nx[2]
#                     flm0[i,j] = - 0.5 * (im * 2m * B[1][i,j] * flm0[i,j] + (im * B[2][i,j] + B[3][i,j]) * fl1m[i,j] + clm * (im * B[2][i,j] - B[3][i,j]) * flm1[i,j] )
#                 end
#             end
#             flm0 = Zma * flm0
#         end
#     else
#     end
# end
