
"""
   Calculate the high ElectroMagnetic wave (EMW) fields with Leapfog method (FD).
     * High frequency conponent of Ampere's circuital law:
        ∂ₜ𝐄 = ∇ × 𝐁
     * Faraday's law of induction
        ∂ₜ𝐁 = - ∇ × 𝐄

    Domain: Ω(x,y) = [x_bounds[1],x_bounds[2]] × [y_bounds[1],y_bounds[2]]
            with fix uniform meshgrids wuth steps Δx and Δy
    time step: Δt is decided by the CFL conditions where
          Δt = params["n_CFL"]  * √(Δx^2 + Δy^2 + ...)

  Inputs:
    Eᵢ = Eᵢ[x,y,z]:
    Bⱼ = Bⱼ[x,y,z]:
    ctx = 1 / dimsx * Δt / Δx

  Output:
    EB = update_Xdy!(Eᵢ, Bⱼ, ctx)
"""

"""
  Update Eᵢ due to  ∂Bⱼ/∂y:
        ∂ₜE1 = + ∂x₂B3,
        ∂ₜE3 = - ∂x₂B1
"""
function update_Edy!(Ex, Bz, ctx)
  dimsx = ndims(Ex)
  if dimsx == 1
    for i = 2:size(Ex,1)-1
      Ex[i] += ctx*(Bz[i] - Bz[i-1])
    end
  elseif dimsx == 2
    for j = 2:size(Ex,2)-1
      for i = 2:size(Ex,1)-1
        Ex[i,j] += ctx*(Bz[i,j] - Bz[i,j-1] + Bz[i-1,j] - Bz[i-1,j-1])
      end
    end
  else
  end
end

"""
  Update Eᵢ due to  ∂Bⱼ/∂x:
    ∂ₜE2 = - ∂x₁B3, the similar to
    ∂ₜE3 = + ∂x₁B2
"""

function update_Edx!(Ey, Bz, ctx)
  dimsx = ndims(Ey)
  if dimsx == 1
    for i = 2:size(Ey,1)-1
      Ey[i] += ctx*(Bz[i] - Bz[i-1])
    end
  elseif dimsx == 2
    for j = 2:size(Ey,2)-1
      for i = 2:size(Ey,1)-1
        Ey[i,j] += ctx*(Bz[i,j] - Bz[i-1,j] + Bz[i,j-1] - Bz[i-1,j-1])
      end
    end
  else
  end
end

"""
  Update Bᵢ due to  ∂Eⱼ/∂x:
    ∂ₜB2 = + ∂x₁E3, the similar to
    ∂ₜB3 = - ∂x₁E2
"""

function update_Bdx!(By, Ez, ctx)
  dimsx = ndims(Ez)
  if dimsx == 1
    for i = 1:size(By,1) - 1
      By[i] += ctx*(Ez[i+1] - Ez[i])
    end
  elseif dimsx == 2
    for j = 1:size(By,2) - 1
      for i = 1:size(By,1) - 1
        By[i,j] += ctx*(Ez[i+1,j+1] - Ez[i,j+1] + Ez[i+1,j] - Ez[i,j])
      end
    end
  else
  end
end

"""
  Update Bᵢ due to  ∂Eⱼ/∂y:
    ∂ₜB1 = + ∂x₂E3, the similar to
    ∂ₜB3 = - ∂x₂E1
"""
function update_Bdy!(Bx, Ez, ctx)
  dimsx = ndims(Ez)
  if dimsx == 1
    for i = 1:size(Bx,1) - 1
      Bx[i] += ctx*(Ez[i+1]- Ez[i])
    end
  elseif dimsx == 2
    for j = 1:size(Bx,2) - 1
      for i = 1:size(Bx,1) - 1
        Bx[i,j] += ctx*(Ez[i+1,j+1] - Ez[i+1,j] + Ez[i,j+1]- Ez[i,j])
      end
    end
  else
  end
end
