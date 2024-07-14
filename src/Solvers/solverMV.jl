
"""
   FDTD / Hsplit (Halmiton splitting) method for Maxwell-Vlasov equations。

   ∂ₜf̂ₗᵐ = 𝒜ₗᵐ + 𝓔ₗᵐ + 𝓑ₗᵐ; especially,
     𝓑ₗ₌₀⁰ = 𝓔₀⁰ = 0
     𝒜₀⁰ = 0 when LM = 0; otherwise, 𝒜₀⁰ ~ f̂₁.

    * FDTD scheme:
    * Hsplit scheme: Hamiltonian splitting method with three subsystems:
       Subsystem 1: High frequency conponent of Ampere's circuital law
                    ∂t/∂B = 0
                    ∂t/∂E = + ∇ × B(x,t)
       Subsystem 2: Faraday's law of induction
                    ∂t/∂E = 0
                    ∂t/∂B = - ∇ × E(x,t)
       Subsystem 3: The low frequency component of Ampere's circuital law
                    ∂t/∂B = 0
                    ∂t/∂E = - Jq
"""

"""
   Update E[Ex, Ey, Ez] and B[Bx, By, Bz] in 2D3V module with FDTD method.

      ∂ₜE1 = ∂x₂B3
      ∂ₜE2 =       - ∂x₁B3
      ∂ₜE3 = ∂x₁B2 - ∂x₂B1

      ∂ₜB1 = ∂x₂E3
      ∂ₜB2 =       - ∂x₁E3
      ∂ₜB3 = ∂x₁E2 - ∂x₂E1

      ∂ₜE1 = - Jq1
      ∂ₜE2 = - Jq2
      ∂ₜE3 = - Jq3

   Input:
     d: dataEB
     p: parameters
     coef = n_CFL * Δt / Δx, = p["ctx"]
     Zmdt = Zma * dt

   Output:
     dataEB = FDTD_step!(d,p,flm, Zma,dt,dimsx,v,nv,LM,E,B,nx)

"""

function FDTD_step(d,flm,Mom,Ta,w1,p,Zma,dt,dimsx,v,nv,LM,vth,nx...)
  if dimsx == 1
    update_Edx!(d["Ey"], d["Bz"], -p["ctx"])
    update_Edx!(d["Ez"], d["By"], +p["ctx"])
    generate_boundary_Efields!(d, p)
    E = [d["Ex"], d["Ey"],d["Ez"]]
    update_flm_E!(flm,Zma * dt,dimsx,v,nv,LM,E,nx)
    # momentsNG!(Mom,Ta,nv,v,w1,flm,nx)
    # println("Ka_E=",Mom[2][2])
    # display(plot!(v[v.<10],real(flm[2][v.<10,1,1]),xlabel="v",ylabel="f00"))
    # display(plot!(v[v.<10],real(flm[2][v.<10,2,1]),xlabel="v",ylabel="f10"))
    # display(plot!(v[v.<10],real(flm[2][v.<10,2,2]),xlabel="v",ylabel="f11"))
    display(plot!(v[v.<10],imag(flm[2][v.<10,2,2]),xlabel="v",ylabel="f1n1"))
    # display(plot!(v[v.<10],real(flm[2][v.<10,3,1]),xlabel="v",ylabel="f20"))
    # display(plot!(v[v.<10],real(flm[2][v.<10,3,3]),xlabel="v",ylabel="f21"))
    # display(plot!(v[v.<10],imag(flm[2][v.<10,3,2]),xlabel="v",ylabel="f2n1"))
    # display(plot!(v[v.<10],real(flm[2][v.<10,3,3]),xlabel="v",ylabel="f22"))
    # display(plot!(v[v.<10],imag(flm[2][v.<10,3,3]),xlabel="v",ylabel="f2n2"))
    update_Bdx!(d["By"], d["Ez"], +p["ctx"])
    update_Bdx!(d["Bz"], d["Ey"], -p["ctx"])
    generate_boundary_Bfields!(d, p)
    B = [d["Bx"], d["By"],d["Bz"]]
    update_flm_AB!(flm,Zma,dt,dimsx,LM,B,vth,p["xb"],nx)
    # momentsNG!(Mom,Ta,nv,v,w1,flm,nx)
    # println("Ka_AB=",Mom[3][1])
    J = 0 * E
    f10 = [flm[ix][:,2,1] for ix in 1:nx[1]]
    f11 = [flm[ix][:,2,2] for ix in 1:nx[1]]
    J!(J,vth,nv,v,w1,f10,f11,nx)
    # println("J=",fmtf2.(J[3]))
    update_EJ!(d["Ex"],J[1],dimsx,dt,nx)
    update_EJ!(d["Ey"],J[2],dimsx,dt,nx)
    update_EJ!(d["Ez"],J[3],dimsx,dt,nx)
    return d,flm
  elseif dimsx == 2
    update_Edy!(d["Ex"], d["Bz"], +p["ctx"]["y"])
    update_Edx!(d["Ey"], d["Bz"], -p["ctx"]["x"])
    update_Edx!(d["Ezx"], d["By"], +p["ctx"]["x"])
    update_Edy!(d["Ezy"], d["Bx"], -p["ctx"]["y"])
    d["Ez"] = d["Ezx"] + d["Ezy"]
    generate_boundary_Efields!(d, p)
    update_Bdy!(d["Bx"], d["Ez"], -p["ctx"]["y"])
    update_Bdx!(d["By"], d["Ez"], +p["ctx"]["x"])
    update_Bdx!(d["Bzx"], d["Ey"], -p["ctx"]["x"])
    update_Bdy!(d["Bzy"], d["Ex"], +p["ctx"]["y"])
    d["Bz"] = d["Bzx"] + d["Bzy"]
    generate_boundary_Bfields!(d,flm, p)
  else
  end
end



"""
   Update E[Ex, Ey, Ez] and B[Bx, By, Bz] in 2D module with Hsplit method.

   Subsystem 1:
     ∂ₜE = 0
     ∂ₜB1 =         - ∂x₂E3
     ∂ₜB2 = + ∂x₁E3
     ∂ₜB3 = + ∂x₂E1 - ∂x₁E2

   Subsystem 2: time step = Δt
     ∂ₜB = 0
     ∂ₜE1 = ∂x₂B3
     ∂ₜE2 =       - ∂x₁B3
     ∂ₜE3 = ∂x₁B2 - ∂x₂B1

   Subsystem 3: time step = Δt/2
     ∂ₜB = 0
     ∂ₜE1 = - Jq * ê₁
     ∂ₜE2 = - Jq * ê₂
     ∂ₜE3 = - Jq * ê₃

    ##################### in 1D module Ω = [x,0,0], the subsystems will be

    Subsystem 1:
      ∂ₜE = 0
      ∂ₜB1 = 0
      ∂ₜB2 = + ∂x₁E3
      ∂ₜB3 = - ∂x₁E2

    Subsystem 2: time step = Δt
      ∂ₜB = 0
      ∂ₜE1 = + ∂x₂B3
      ∂ₜE2 = - ∂x₁B3
      ∂ₜE3 = + ∂x₁B2

    Subsystem 3: time step = Δt/2
      ∂ₜB = 0
      ∂ₜE1 = - Jq * ê₁
      ∂ₜE2 = - Jq * ê₂
      ∂ₜE3 = - Jq * ê₃

   # A second-order symmetric composition method as:
     Step 1：update Sys. 1 with time step = Δt/2  → B = B(E,Δt/2)
     Step 2：update Sys. 2 with time step = Δt/2  → E = E(B,Δt/2)
     Step 3：update Sys. 3 with time step = Δt    → E = E(J,Δt)
     Step 4：update Sys. 2 with time step = Δt/2  → E = E(B,Δt/2)
     Step 5：update Sys. 1 with time step = Δt/2  → B = B(E,Δt/2)

   Symmetric, include boundary condtions

   Input:
     d: dataEB
     p: p_dict,  ctx = 1 / dimsx * Δt / Δx = p["ctx"]

   Output:
     dataEB = Hsplit_step(d,flm,p,Zma,dt,dimsx,v,nv,LM,vth,nx...)
"""

function Hsplit_step(d,flm,p,Zma,dt,dimsx,v,nv,LM,vth,nx...)
  dimsx = p["dims"][1]
  if dimsx == 1
    update_Edx!(d["Ey"], d["Bz"], -0.5p["ctx"])
    update_Edx!(d["Ez"], d["By"], +0.5p["ctx"])
    E = [d["Ex"], d["Ey"],d["Ez"]]
    update_flm_E!(flm,Zma * 0.5dt,dimsx,v,nv,LM,E,nx)
    update_Bdx!(d["By"], d["Ez"], +0.5p["ctx"])
    update_Bdx!(d["Bz"], d["Ey"], -0.5p["ctx"])
    generate_boundary_Bfields!(d, p)
    # System 3
    B = [d["Bx"], d["By"],d["Bz"]]
    update_flm_AB!(flm,Zma,dt,dimsx,LM,B,vth,p["xb"],nx)
    #
    #
    update_flm_E!(flm,Zma * 0.5dt,dimsx,v,nv,LM,E,nx)
    update_Bdx!(d["Bz"], d["Ey"], -0.5p["ctx"])
    update_Bdx!(d["By"], d["Ez"], +0.5p["ctx"])
    update_Edx!(d["Ez"], d["By"], +0.5p["ctx"])
    update_Edx!(d["Ey"], d["Bz"], -0.5p["ctx"])
    generate_boundary_Efields!(d, p)
    return d,flm
  elseif dimsx == 2
    update_Bdy!(d["Bx"], d["Ez"], -0.5p["ctx"]["y"])
    update_Bdx!(d["By"], d["Ez"], +0.5p["ctx"]["x"])
    update_Bdx!(d["Bzx"], d["Ey"], -0.5p["ctx"]["x"])
    update_Bdy!(d["Bzy"], d["Ex"], +0.5p["ctx"]["y"])
    d["Bz"] = d["Bzx"] + d["Bzy"]

    update_Edy!(d["Ex"], d["Bz"], +0.5p["ctx"]["y"])
    update_Edx!(d["Ey"], d["Bz"], -0.5p["ctx"]["x"])
    update_Edx!(d["Ezx"], d["By"], +0.5p["ctx"]["x"])
    update_Edy!(d["Ezy"], d["Bx"], -0.5p["ctx"]["y"])
    d["Ez"] = d["Ezx"] + d["Ezy"]
    generate_boundary_Bfields!(d, p)

    # System 3  +- 1 * p["ctx"]["y"]
    #
    update_Edy!(d["Ex"], d["Bz"], +0.5p["ctx"]["y"])
    update_Edx!(d["Ey"], d["Bz"], -0.5p["ctx"]["x"])
    update_Edx!(d["Ezx"], d["By"], +0.5p["ctx"]["x"])
    update_Edy!(d["Ezy"], d["Bx"], -0.5p["ctx"]["y"])
    d["Ez"] = d["Ezx"] + d["Ezy"]

    update_Bdy!(d["Bx"], d["Ez"], -0.5p["ctx"]["y"])
    update_Bdx!(d["By"], d["Ez"], +0.5p["ctx"]["x"])
    update_Bdx!(d["Bzx"], d["Ey"], -0.5p["ctx"]["x"])
    update_Bdy!(d["Bzy"], d["Ex"], +0.5p["ctx"]["y"])
    d["Bz"] = d["Bzx"] + d["Bzy"]
    generate_boundary_Efields!(d, p)
    return d,flm
  else
  end
end
