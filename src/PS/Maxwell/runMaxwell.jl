
"""
   Calculate parameters needed for simulation and return them as a dictionary.

    Domain: Ω(x,y) = [x_bounds[1],x_bounds[2]] × [y_bounds[1],y_bounds[2]]
            with fix uniform meshgrids wuth steps Δx and Δy
    time step: Δt is decided by the CFL conditions where
          Δt = 0.5 * √(Δx^2 + Δy^2)


    ε0 = 8.85418782e-12  # permittivity of free space [F/m] (F=Farad)
      ε = εr * ε0
      Permittivity is a constant of proportionality between electric displacement
      and electric field intensity in a given medium.
      force = charge * electric field
      F = Q * E
      ∇ × E = 0 : Curl of electric field (∇ is the del or nabla operator)
      (∇f is a vector field of the df/dx df/dy etc...)
      (∇ × E is a vector field representing rotational displacement)

    Electric flux density
      D = ε * E
      E = D/ε  (= −∇V)
      ∇ · D = ρv : Divergence of electric flux density
      (ρv: electric charge density [C/m3])
      (a scalar measure of strength of source or sink)
      ∇²V = −ρ/e
      Electrical conductivity: σ
      Perfect electric conductors (PECs) have σ close to infinity
      Current density J = σE (current in material, charges moving due to E field)
      Charge Q moving at speed v in field B (magnetic flux density):
      force = charge * speed cross-product magnetic flux density
      F = Q * (v × B)

    Magnetic Field
      H = B / µ     (ignore local effect of material on flux):
      B = μrμ0H = μH    (relative permeability and permeability of free space)
      µ0 = 4π * 1e-7  # permeability of free space [H/m] (H=Henry)
      µ = µr * µ0
      η0 = sqrt(µ0 / ε0)  # Characteristic impedance of free space = ~ 377.0

    Energy must no be able to propagate further than 1 spatial step: cΔt <= Δx
      c = 1 / sqrt(ε0 * µ0)  # speed of light in free space [m/s]
      Sc = 1  # Sc = c * Δt / Δx Courant number (we set it to 1, for now)
      More generally: Δt must be <= 1/(c*sqrt(1/Δx^2 + 1/Δy^2 + 1/Δz^2))

    Source: excitation function with a gaussian pulse
      peak_time = 30.0
      pulse_width = 10
      gaussian_pulse(t) = exp(-((t - peak_time) / pulse_width)^2)
      tvec = 0:0.1:80
      plot(tvec,gaussian_pulse.(tvec))
      source_function = gaussian_pulse

    Wave: (T, λ) with λ / T = λ * freq = ω / k = c₀. With (t, x, B) is normalized by (T, λ , 1/c₀)
       ω / k = 1
       E = E * sin(2π * (t - x))
       B = B * sin(2π * (t - x))
    Normalized Maxwell equations in 2D3V forms: Ω = Ω(x,y), A = A(x,y) = A[Ex, Ay, Az], A = {E, B, Jq}.
      ∂t/∂B = - ∇ × E(x,y)
      ∂t/∂E = + ∇ × B(x,y,) - Jq

    Poyinting vector: S = E × B
    Boundary:
        bounds_conds = ["ABC","PML", "PEC", "PMC"]
        ABC: Absorbing Boundary Condition is an open boundary conditions. The fields at the grid points
               have electric field values formulated using Engquist Majda one way wave equations where
               the boundaries give a sense of absorbing the total field incident on them and reflecting none back to the domain.
             1D is good but its quite difficult to make 2D ABC. The solution is to use PML.
        PML: Perfect Macthed Layer is an artificial absorbing layer, where the fields near the boundary
              are attenuated over a predetermined length of boundary width before they reach the boudary to a zero value
              at the boundary using a polynomially increasing electrical conductivity value over the boundary width
              with maximum at the boundary and also choosing a magnetic conductivity value at every point
              in the boundary width to avoid reflection at that point.
        PEC: Perfect Electric Conductor, which is in the sense that the boundary grid points
              have zero electric field values irrespective of the influence of external fields.
        PMC: Perfect Magnetic Conductor, which is  in the sense that the boundary grid points
              have zero magnetic field values irrespective of the influence of external fields.

  The 1D TEM wave is x-directed z-polarized TEM wave containing the y-directed magnetic field Hy and
z-directed electric field Ez. The time update in Yee Algorithm is done using Leapfrog time-stepping.
Here, the H fields are updated every half time-step and E fileds are updated every full time-step.
This is shown by two alternating vector updates spanning entire spatial grid inside a main for-loop
for time update spanning the entire time-grid. The vector updates span only a part of spatial grid
where the wave, starting from source, has reached at that particular time instant (exploiting sparse vectors)
avoiding field updates at all points in the grid which is unnecessary at that time instant.
The spatial and temporal parameters are not unit less and are given real values.

"""

"""
  Run the Maxwell-Vlasov simulations
    * FDTD scheme:
    * Hsplit scheme: Hamiltonian splitting method with three subsystems:
       Subsystem 1: ∂t/∂E = 0
                    ∂t/∂B = - ∇ × E(x,t)
       Subsystem 2: ∂t/∂B = 0
                    ∂t/∂E = + ∇ × B(x,t)
       Subsystem 3: ∂t/∂B = 0
                    ∂t/∂E = - Jq
  Inputs:
    dimsx: dimension of domian Ω[x,y,z]

  Output:
    dataEB: E and B at the last time

"""

function runMaxwell(init_params_dict)
  params_dict = init_params(init_params_dict)
  dataEB, dataEBt = init_dataEB(params_dict)
  nt_max = params_dict["time_steps"]      # Maximum time steps,
  dt0 = params_dict["time_step"]          # Initial time step
  nEBt = params_dict["EB_probe_time"][1]
  if params_dict["time_integral"] == "Leapfog" || params_dict["time_integral"] == ""
    for k = 1:nt_max
      time = k * dt0
      EBsource_fields!(dataEB, time, params_dict)
      FDTD_step!(dataEB, params_dict)
      push!(dataEBt["E"],(time,dataEB["Ex"][nEBt],dataEB["Ey"][nEBt],dataEB["Ez"][nEBt]))
      push!(dataEBt["B"],(time,dataEB["Bx"][nEBt],dataEB["By"][nEBt],dataEB["Bz"][nEBt]))
      output(dataEB, k, params_dict)
    end
  elseif params_dict["time_integral"] == "Hsplit"
    update_Bdx!(dataEB["By"], dataEB["Ez"], +0.5params_dict["ctx"])
    update_Bdx!(dataEB["Bz"], dataEB["Ey"], -0.5params_dict["ctx"])
    generate_boundary_Bfields!(dataEB, params_dict)
      # update_Edx!(dataEB["Ey"], dataEB["Bz"], -0.5params_dict["ctx"])
      # update_Edx!(dataEB["Ez"], dataEB["By"], +0.5params_dict["ctx"])
    for k = 1:nt_max
      time = k * dt0
      EBsource_fields!(dataEB, time, params_dict)
      Hsplit_step!(dataEB, params_dict)
      push!(dataEBt["E"],(time,dataEB["Ex"][nEBt],dataEB["Ey"][nEBt],dataEB["Ez"][nEBt]))
      push!(dataEBt["B"],(time,dataEB["Bx"][nEBt],dataEB["By"][nEBt],dataEB["Bz"][nEBt]))
      output(dataEB, k, params_dict)
    end
  end
  return dataEB,dataEBt
end
