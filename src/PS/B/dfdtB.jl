"""
  dataf = {f̂̂ₗᵐ(x,v̂)} with struct [nₓ,nᵥ,nL,2nL+ 1], the ϕ dimension is

  Diffusion due to electric field B = B(x,y,z) = [B1, B2, B3]
  m = -ℓ:1:ℓ
  v̂ = [0; [v̂_G]; ∞]
  nv = nG + 2
  ℓ = 0:ℓM
  nL = ℓM + 1
  nm = 2ℓ + 1
"""


function dfdtB1D3V(nx,nv,nL,f,p)
  # ℓ == 0
  if nL == 1
  elseif nL == 2
  else
  end
  for ℓ in 1:Nv + 1
  end
end


function dflmB1D3V(time,CB,f)
  for α in 2:Nv + 1
  end
end

"""
  m = 0
  fl1 = [nx,nv]
"""

function dfl0B1D3V(nx,nv,nL,dt,fl1)
  Bl0 = zeros(ComplexF64,nx,nv,nL)
  # L = 0
  # B00 .= 0
  for L in 1:LM
    L1 = L + 1
    Bl0[:,:,L1] = L * L1  * (By .* imag(fl1) + Bz .* real(fl1))
  end
end

"""
  m = ± ℓM
"""
function dflMB1D3V(dt,flm)
end





"""
                 |m=0----------------|
                 |m=1----------;m=-ℓM|
                 |m=2--------;m=-ℓM+1|
  dataf[i,α,β,:]=|         :         |
                 |m=+ℓM-2;-------m=-3|
                 |m=ℓM-1;--------m=-2|
                 |m=ℓM;----------m=-1|
"""
