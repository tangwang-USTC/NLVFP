
"""
   Calculate parameters needed for simulation and return them as a dictionary.

    Domain: Ω(x,y) = [x_bounds[1],x_bounds[2]] × [y_bounds[1],y_bounds[2]]
            with fix uniform meshgrids wuth steps Δx and Δy
    Meshgrids in x dimesion:

      ```
      1   3/2            j-1 j-0.5   j                      Nx-1 Nx-0.5  Nx
      |----o----|----o----|----o----| ... |----o----|----o----|----o----|
     E[1]     E[2]                E[j]                               E[Nx]  nE = nx
    f[1,:]                       f[j,:]                             f[Nx,:] nf[:,1] = nx
         B[1+0.5]          B[j+0.5]                             B[Nx+0.5]   nB = nx -1
      ```

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
       Subsystem 1: ∂t/∂f̂ₗᵐ = 𝓔ₗᵐ
                    ∂t/∂E = 0
                    ∂t/∂B = - ∇ × E(x,t)
       Subsystem 2: ∂t/∂f̂ₗᵐ = 𝓔ₗᵐ
                    ∂t/∂E = + ∇ × B(x,t)
                    ∂t/∂B = 0
       Subsystem 3: ∂t/∂f̂ₗᵐ = 𝒜ₗᵐ + 𝓑ₗᵐ
                    ∂t/∂E = - Jq
                    ∂t/∂B = 0
  Inputs:
    dimsx: dimension of domian Ω[x,y,z]

  Output:
    dataEB: E and B at the last time

"""

function runMV(params_correct)
  p_dict = init_params(params_correct)
  ####################     velocity
  if p_dict["models"]["is_f"] == 1
    dimsv = p_dict["dims"][2]
    nG = p_dict["nG"]
    nv = nG + 2
    vG, w1v = laguerre(nG, 0.0)
    w1 = w1v' |> collect
    vnGL = 2vG[nG] - vG[nG-1]
    v = [0.0; vG; vnGL]
    nL = p_dict["nL"]
    LM = nL - 1
    nϕ = 2nL
  end
  ####################     time
  t0 = 0.0
  nt_max = p_dict["time_steps"]        # Maximum time steps,
  dt = p_dict["time_step"]             # Initial time step
  #################33       parameters
  dimsx = p_dict["dims"][1]
  nx = p_dict["grid_size"]
  nEBt = p_dict["EB_probe_time"][1]
  dataEB, dataEBt = init_dataEB(p_dict)
  if p_dict["models"]["is_f"] == 1
    ma = p_dict["ma"]
    Zma = p_dict["Za"] / ma
    nuT0 = p_dict["nuT0"]
    T0 = nuT0[3]
    Momt,Ta = initial_Mom(dimsx,t0,nuT0,p_dict["xb"],nx)
    Mom = Momt.Mom[1]
    K0 = nuT0[1] * T0 * (1.5 + nuT0[2]^2)
    vth = (2Ta ./ ma).^0.5
    # println("Ta0=",Ta)
    println("Ka0=",K0)
    ##################      distribution function
    if dimsv == 2
      datatype = Float64
      μ, w2, Mμ, Mun = LegendreMμ0(LM,datatype = Float64)
    elseif dimsv == 3
      datatype = ComplexF64
      μ, w2, Mμ, Mun = LegendreMμ(LM, datatype = Float64)
      fup = initial_fup3V(datatype,dimsx,nv,nL,nϕ,nx)
      initial_fup3V!(fup,dimsx,nL,v,μ,ma,Mom[1],Mom[2],Ta,nx)
      flm = SPTLmnD3V(datatype,dimsx,fup, μ, w2, nx)
    end
  end
  # display(plot(v[v.<10],real(flm[2][v.<10,1,1]),xlabel="v",ylabel="f00"))
  # display(plot(v[v.<10],real(flm[2][v.<10,2,1]),xlabel="v",ylabel="f10"))
  # display(plot(v[v.<10],real(flm[2][v.<10,2,2]),xlabel="v",ylabel="f11"))
  # display(plot(v[v.<10],imag(flm[2][v.<10,2,2]),xlabel="v",ylabel="f1n1"))
  # display(plot(v[v.<10],real(flm[2][v.<10,3,1]),xlabel="v",ylabel="f20"))
  # display(plot(v[v.<10],real(flm[2][v.<10,3,2]),xlabel="v",ylabel="f21"))
  # display(plot(v[v.<10],imag(flm[2][v.<10,3,2]),xlabel="v",ylabel="f2n1"))
  # display(plot(v[v.<10],real(flm[2][v.<10,3,3]),xlabel="v",ylabel="f22"))
  # display(plot(v[v.<10],imag(flm[2][v.<10,3,3]),xlabel="v",ylabel="f2n2"))
  # display(plot(v[v.<10],real(flm[2][v.<10,4,1]),xlabel="v",ylabel="f30"))
  # display(plot(v[v.<10],real(flm[2][v.<10,4,2]),xlabel="v",ylabel="f31"))
  # display(plot(v[v.<10],imag(flm[2][v.<10,4,2]),xlabel="v",ylabel="f3n1"))
  # display(plot(v[v.<10],real(flm[2][v.<10,4,3]),xlabel="v",ylabel="f32"))
  # display(plot(v[v.<10],imag(flm[2][v.<10,4,3]),xlabel="v",ylabel="f3n2"))
  # display(plot(v[v.<10],real(flm[2][v.<10,4,4]),xlabel="v",ylabel="f33"))
  # display(plot(v[v.<10],imag(flm[2][v.<10,4,4]),xlabel="v",ylabel="f3n3"))
  # ddd
  ###################      Solver
  if dimsx == 1
    ### momentsNG!  momentsNT!
    if p_dict["time_integral"] == "Leapfog" || p_dict["time_integral"] == ""
      if p_dict["models"]["is_f"] == 1
        for k = 1:nt_max
          times = k * dt
          EBsource_fields!(dataEB, times, p_dict)
          println("////////////////")
          dataEB, flm = FDTD_step(dataEB,flm,Mom,Ta,w1, p_dict,Zma,dt,dimsx,v,nv,p_dict["nL"] - 1,vth,nx)
          push!(dataEBt["E"],(times,dataEB["Ex"][nEBt],dataEB["Ey"][nEBt],dataEB["Ez"][nEBt]))
          push!(dataEBt["B"],(times,dataEB["Bx"][nEBt],dataEB["By"][nEBt],dataEB["Bz"][nEBt]))
          Mom2 = Mom * 1
          momentsNG!(Mom2,Ta,nv,v,w1,flm,nx)
          ua2 = (Mom2[2][1]).^2 + (Mom2[2][2]).^2 + (Mom2[2][3]).^2
          Ta2 = 2/3 * ( Mom2[3] ./ Mom2[1]  - Ta .* ua2 )
          println("t=",fmtf2.(times),",dTa=",fmtf4.(Ta2 .- T0))
          # momentsNG!(Mom,Ta,nv,v,w1,flm,nx)
          # ua2 = (Mom[2][1]).^2 + (Mom[2][2]).^2 + (Mom[2][3]).^2
          # Ta = 2/3 * ( Mom[3] ./ Mom[1]  - Ta .* ua2 )
          # println("t=",fmtf2.(times),",dKa=",fmtf2.(Mom[3] .- K0))
          # println("t=",fmtf2.(times),",dKa=",fmtf2.(Mom[3]))
          # println("t=",fmtf2.(times),",dTa=",fmtf2.(Ta))
          # println("t=",fmtf2.(times),",n0=",fmtf2((sum(Mom[1] .- nuT0[1]))/nx),",na=",fmtf2.(Mom[1] .- nuT0[1]))
          # vth = (2Ta ./ ma).^0.5
          push!(Momt.t,times)
          push!(Momt.Mom,1*Mom)
          # if k == 27
          #   break
          # end
        end
        return dataEB,dataEBt,flm,fup,Momt
      else
        for k = 1:nt_max
          times = k * dt
          EBsource_fields!(dataEB, times, p_dict)
          FDTD_step!(dataEB, p_dict)
          push!(dataEBt["E"],(times,dataEB["Ex"][nEBt],dataEB["Ey"][nEBt],dataEB["Ez"][nEBt]))
          push!(dataEBt["B"],(times,dataEB["Bx"][nEBt],dataEB["By"][nEBt],dataEB["Bz"][nEBt]))
        end
        return dataEB,dataEBt
      end
    elseif p_dict["time_integral"] == "Hsplit"
      for k = 1:nt_max
        times = k * dt
        EBsource_fields!(dataEB, times, p_dict)
        vth = (2Mom[3] ./ ma).^0.5
        dataEB, flm = Hsplit_step(dataEB,flm, p_dict,Zma,dt,dimsx,v,nv,p_dict["nL"] - 1,vth,nx)
        push!(dataEBt["E"],(times,dataEB["Ex"][nEBt],dataEB["Ey"][nEBt],dataEB["Ez"][nEBt]))
        push!(dataEBt["B"],(times,dataEB["Bx"][nEBt],dataEB["By"][nEBt],dataEB["Bz"][nEBt]))
        momentsNG!(Mom,Ta,nv,v,w1,flm,nx)
        push!(Momt.t,times)
        push!(Momt.Mom,1*Mom)
      end
    end
  elseif dimsx == 2
    for k = 1:nt_max
      times = k * dt
      EBsource_fields!(dataEB, times, p_dict)
      Hsplit_step!(dataEB,flm, p_dict)
      push!(dataEBt["E"],(times,dataEB["Ex"][nEBt],dataEB["Ey"][nEBt],dataEB["Ez"][nEBt]))
      push!(dataEBt["B"],(times,dataEB["Bx"][nEBt],dataEB["By"][nEBt],dataEB["Bz"][nEBt]))
      output(dataEB, k, p_dict)
    end
  else
  end
end


  # xb = p_dict["xb"]
# println("tk=",k,",Ta=",Mom[3])
# println("tk=",k,",uax=",Mom[2][1])
# println("tk=",k,",uay=",Mom[2][2][1:4])
# println("tk=",k,",uaz=",Mom[2][3])
# println("tk=",k,",na=",Mom[1][1:5])
  # nv30 = v .< 10
  # pf0 = plot(v[nv30],real(flm[4][nv30,1,1]),ylabel="f₀⁰")
  # pf10 = plot(v[nv30],real(flm[4][nv30,2,1]),ylabel="f₁⁰",xlabel="v̂")
  # pf11 = plot(v[nv30],real(flm[4][nv30,2,2]),ylabel="f₁¹")
  # pf1n1 = plot(v[nv30],imag(flm[4][nv30,2,2]),ylabel="f₁⁻¹",xlabel="v̂")
  # display(plot(pf0,pf10,pf11,pf1n1,layout=(4,1)))


# output(dataEB, k, p_dict)
# output(dataEB,dataEBt,flm,fup,Momt, k, p_dict)
# if k == 7
#   break
# end
# println("tk=",k,",uay=",Momt.Mom[1][2][2][1:3])
