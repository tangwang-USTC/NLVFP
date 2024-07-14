
"""
   FDTD / Hsplit (Halmiton splitting) method for Maxwell equations。

    * FDTD scheme:
    * Hsplit scheme: Hamiltonian splitting method with three subsystems:
       Subsystem 1: ∂t/∂E = 0
                    ∂t/∂B = - ∇ × E(x,t)
       Subsystem 2: ∂t/∂B = 0
                    ∂t/∂E = + ∇ × B(x,t)
       Subsystem 3: ∂t/∂B = 0
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


   Input:
     d: dataEB
     p: parameters
     coef = n_CFL * Δt / Δx, = p["ctx"]

   Output:
     dataEB = FDTD_step!(d, p)

"""

function FDTD_step!(d, p)
  if p["dims"][1] == 1
    update_Edx!(d["Ey"], d["Bz"], -p["ctx"])
    update_Edx!(d["Ez"], d["By"], +p["ctx"])
    generate_boundary_Efields!(d, p)
    update_Bdx!(d["By"], d["Ez"], +p["ctx"])
    update_Bdx!(d["Bz"], d["Ey"], -p["ctx"])
    generate_boundary_Bfields!(d, p)
  elseif p["dims"][1] == 2
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
    generate_boundary_Bfields!(d, p)
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
     dataEB = Hsplit_step!(d, p)
"""

function Hsplit_step!(d, p)
  if p["dims"][1] == 1
    update_Edx!(d["Ey"], d["Bz"], -0.5p["ctx"])
    update_Edx!(d["Ez"], d["By"], +0.5p["ctx"])
    update_Bdx!(d["By"], d["Ez"], +0.5p["ctx"])
    update_Bdx!(d["Bz"], d["Ey"], -0.5p["ctx"])
    generate_boundary_Bfields!(d, p)
    # System 3
    #
    update_Bdx!(d["Bz"], d["Ey"], -0.5p["ctx"])
    update_Bdx!(d["By"], d["Ez"], +0.5p["ctx"])
    update_Edx!(d["Ez"], d["By"], +0.5p["ctx"])
    update_Edx!(d["Ey"], d["Bz"], -0.5p["ctx"])
    generate_boundary_Efields!(d, p)
  elseif p["dims"][1] == 2
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
  else
  end
end
