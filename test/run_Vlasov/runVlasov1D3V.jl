using Test, Plots
using DataFrames,CSV, Format
using FFTW

path = "G:\\atom\\julia\\MVFP2D3V"
include(joinpath(path,"src\\Maxwell\\Hsplit2d.jl"))
include(joinpath(path,"src\\Maxwell\\EBshapes.jl"))
include(joinpath(path,"src\\Maxwell\\EBsource.jl"))
include(joinpath(path,"src\\runMV.jl"))
include(joinpath(path,"src\\initialfields.jl"))

plotly()

fmtf = generate_formatter( "%1.1e" )   # "%e", "%'e'", "%1.1e", [d,f,s,e]
fmtf2 = generate_formatter( "%1.2e" )
fmtf4 = generate_formatter( "%1.4e" )
println("New Testing")
##
"""
  Domian: Ω(x,y) = bounds_x × bounds_y
  Grids numbers: grid_size[nx, ny]
  time_steps = nt: Maximum of time steps

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
  " ":

"""
# Defaults
E0 = [0.0, 0.0, 0.0]    # Initial amplitude of elctric field
B0 = [0.0, 0.0, 0.0]    # Initial amplitude of magnetic field
ω = 1
EMW_shapes = ["Gaussian", "TEM", "TE", "TM", "Static","Static+"]
bounds_conds = ["MurABC","MurABC","PML", "PEC", "PMC", ""]
time_integral = ["","leapfog", "Hsplit"] # the default scheme is "" = "leapfog"
# Models
is_A = 1      # Advection term
is_B = 1      # Magnetic field diffusion term
is_C = 1      # Collision term
is_E = 1      # Electric field diffusion term
is_f = 1      # distribution functions
## Parameters , times
times_tL = 0.7          #
time_step = 5e-3        # default time step if is_cfl == "off"
time_integral_MVFP = "Hsplit" # time integral method for MVFP euqations
# time_integral_MVFP = "Leapfog"

## Electromagnetic fields
if is_E == 1 || is_B == 1
  EMW_shapes_name = "Gaussian" # Electric and magnetic field shape
  EMW_shapes_name = "TEM" # Electric and magnetic field shape
  # E0 = [0.0, 0.0, 1.0]    # Initial amplitude of elctric field
  # B0 = [0.0, 1.0, 0.0]    # Initial amplitude of magnetic field
  E0 = [0.0, 1.0, 0.0]    # Initial amplitude of elctric field
  B0 = [0.0, 0.0, 1.0]    # Initial amplitude of magnetic field
  ω = 1                   # (= 10 default) normalized circular frequency
  ϕ₀ = 0.00 * 0.5π         # (= 0 default) intial phase of EMW
  EBsource_position_num = [3, 2, 2]
end
## x dimension
if is_A == 1
  dimsx = 1                # dimension of physics domain [x,y,z]
  Lx = 14                 # Length of Lx
  nw = 20                 # (=20 default) Number of grids in a wavelength
  bounds_x = (0.0, Lx/ω)  #
  bounds_y = (-4.0/ω, 4.0/ω) #
  num_grids_lamba = nw * ω      # Number of grids in a wavelength
  grid_size_x = num_grids_lamba * (bounds_x[2] - bounds_x[1]) |> Int
  grid_size_y = num_grids_lamba * (bounds_y[2] - bounds_y[1]) |> Int
end
## velocity spaces
if is_f == 1
  # velocity axis
  dimsv = 1           # dimension of velocity space
  nG_min0 = 6         # (=6,default) low limit for nG nG_min0 for GaussQuadrature iteration
  dnG_gauss = 2       # increase number of nG for GaussQuadrature optimization
  nG_lim = 60         # (=60,default) maximum of nG for v-grid, i = 0:nG_limit
                      # Usually =80 is suitable to atol_gauss = 1e-2
  nG = 35
  #### Remesh for v
  # L-m spaces
  L_lim = 20          # limit of L for all spices
                      # Lmax < 35 || numerical instability
  Rate_fLM = 1e-2       # maximum rate of change of f(v)= ∑fL(v) for ℓM
  nL = 3
end

## CFL conditions
n_CFL = 1                   # (=0.5 default) for CFL limit: n_CFL * Δt / Δx < 1
time_steps_limit = Lx * nw /n_CFL * times_tL  # (Int) for the entire domain
is_cfl = "on"                 # (= "on" default means CFL contral is on.)
##
if EMW_shapes_name == "Gaussian"
  grid_size_x = 100
  grid_size_y = 64
  time_steps_limit = 2*90
end
##
params_correct = Dict(
  "models" => Dict("is_A" => is_A,"is_B" => is_B,"is_C" => is_C,"is_E" => is_E,"is_f" => is_f),
  "time_steps" => time_steps_limit,
  "time_step" => time_step,
  "time_integral" => time_integral_MVFP,
  "dims" => Dict("x" => dimsx,"v" => dimsv),
  "bounds_x" => bounds_x,
  "bounds_y" => bounds_y,
  "grid_size" => Dict("x" => grid_size_x, "y" => grid_size_y),
  "nG" => nG,
  "nL" => nL,
  "nG_lim" => nG_lim,
  "L_lim" => L_lim,
  "n_CFL" => n_CFL,
  "is_cfl" => is_cfl,
  "EMW_shape" => EMW_shapes_name,
  "EB_source_xyz" => EBsource_position_num,
  "EB_probe_time" => [35,5,5],
  "bounds_conds" => "MurABS",
  "output" => Dict(
     "iteration_max" => 10,
     "directory_name" => "test"
  )
)
##
params_calculated = init_params(params_correct)
# Initial_data
data_empty = Dict()
datat_empty = Dict()
if params_calculated["dims"]["x"] == 1
  for name in ("Ex","Ey","Ez")
    data_empty[name] = zeros(Float64, params_calculated["grid_size"]["x"])
  end
  for name in ("Bx","By","Bz")
    data_empty[name] = zeros(Float64, params_calculated["grid_size"]["x"]-1)
  end
  datat_empty["EBprobes"] = [5,5,5]
  datat_empty["E"] = DataFrame(t = 0.0,Ex = 0.0, Ey = 0.0, Ez = 0.0)
  datat_empty["B"] = DataFrame(t = 0.0,Bx = 0.0, By = 0.0, Bz = 0.0)
elseif  params_calculated["dims"]["x"] == 2
  for name in ("Ex","Ey","Ezx","Ezy","Ez")
    data_empty[name] = zeros(Float64, params_calculated["grid_size"]["x"], params_calculated["grid_size"]["y"])
  end
  for name in ("Bx","By","Bzx","Bzy","Bz")
    data_empty[name] = zeros(Float64, params_calculated["grid_size"]["x"]-1, params_calculated["grid_size"]["y"]-1)
  end
else
end
## Boudnary or source function
if params_correct["EMW_shape"] == "Gaussian"
  if params_calculated["dims"]["x"] == 1
    println("EBW = 1D Gaussian")
    x_min = params_correct["bounds_x"][1]
    width = 1.0
    duration = 2.0
    params_correct["E_shape_y"] = laser_pulse_gauss(x_min, width, duration)
    params_correct["E_shape_z"] = laser_pulse_gauss(x_min, width, duration)
    params_correct["B_shape_y"] = laser_pulse_gauss(x_min, width, duration)
    params_correct["B_shape_z"] = laser_pulse_gauss(x_min, width, duration)
  elseif params_calculated["dims"]["x"] == 2
    println("EBW = 2D Gaussian")
    # Example202: Laser pulse shape in direction x and y
    x_min = params_correct["bounds_x"][1]
    width = 1.0
    duration = 2.0
    params_correct["E_shape_y"] = laser_pulse_gauss(x_min, width, duration)
    params_correct["E_shape_z"] = laser_pulse_gauss(x_min, width, duration)
    params_correct["B_shape_y"] = laser_pulse_gauss(x_min, width, duration)
    params_correct["B_shape_z"] = laser_pulse_gauss(x_min, width, duration)
  else
  end
elseif params_correct["EMW_shape"] == "TEM"
  if params_calculated["dims"]["x"] == 1
    println("EBW = 1D TEM")
    params_correct["E_shape_y"] = EMWplane(ω, E0[2], ϕ₀)
    params_correct["E_shape_z"] = EMWplane(ω, E0[3], ϕ₀)
    params_correct["B_shape_y"] = EMWplane(ω, B0[2], ϕ₀)
    params_correct["B_shape_z"] = EMWplane(ω, B0[3], ϕ₀)
  elseif params_calculated["dims"]["x"] == 2
    println("EBW = 2D TEM")
    params_correct["E_shape_y"] = EMWplane(ω, E0[2], ϕ₀)
    params_correct["E_shape_z"] = EMWplane(ω, E0[3], ϕ₀)
    params_correct["B_shape_y"] = EMWplane(ω, B0[2], ϕ₀)
    params_correct["B_shape_z"] = EMWplane(ω, B0[3], ϕ₀)
  else
  end
elseif params_correct["EMW_shape"] == "TE"
  if params_calculated["dims"]["x"] == 1
    println("EBW = 1D TE")
  elseif params_calculated["dims"]["x"] == 2
    println("EBW = 2D TE")
  else
  end
elseif params_correct["EMW_shape"] == "TM"
  if params_calculated["dims"]["x"] == 1
    println("EBW = 1D TM")
  elseif params_calculated["dims"]["x"] == 2
    println("EBW = 2D TM")
  else
  end
elseif params_correct["EMW_shape"] == "Static"
  println("EBW = 2D Static")
end
# @testset "Calculation of parameters" begin
#     @test params_calculated["box_size"]["x"] == 12.0
#     @test params_calculated["box_size"]["y"] == 8.0
#     @test params_calculated["space_step"]["x"] == 0.125
#     @test params_calculated["space_step"]["y"] == 0.125
#     @test params_calculated["time_step"] == sqrt(2.0)/16.0
#     @test params_calculated["n_CFL"]["x"] == 0.5*sqrt(0.5)
#     @test params_calculated["n_CFL"]["y"] == 0.5*sqrt(0.5)
# end
## Solve the problem
@time data, datat = runMV(params_correct)
# @benchmark data = runMV(params_correct)
## Plots
Lx = (params_correct["bounds_x"][2] - params_correct["bounds_x"][1])
Δx = Lx / (params_correct["grid_size"]["x"]-1)
xvec = params_correct["bounds_x"][1]:Δx:params_correct["bounds_x"][2] |> collect
xvec_cent = xvec[1:end-1] + ( xvec[2:end] - xvec[1:end-1])/2
if params_calculated["dims"]["x"] == 1
  if E0[2] ≠ 0 || B0[3] ≠ 0
    nameE = "Ey"
    nameB = "Bz"
  elseif E0[3] ≠ 0 || B0[2] ≠ 0
    nameE = "Ez"
    nameB = "By"
  else
  end
  title = string("nω = ",nw,", t =", Lx,", ",params_correct["time_integral"])
  S = data[nameE][1:end-1] .* data[nameB][1:end-1]
  nx = length(xvec)
  if Lx < 20
    nnn = 2
  elseif 20 < Lx < 100
    nnn = 5
  else
    nnn = 20
  end
  nEvec = 5:nx - nnn * nw
  nBvec =  nEvec[1:end-1]
  pE = plot(xvec[nEvec],data[nameE][nEvec],ylabel=nameE,title = title,legend= false)
  pB = plot(xvec[nBvec],data[nameB][nBvec],ylabel=nameB)
  pS = plot(xvec[nEvec],S[nEvec],xlabel="x",ylabel="Sx",legend= false)
  display(plot(pE,pB,pS,layout=(3,1)))
  E = datat["E"]
  B = datat["B"]
  if nameE == "Ex"
    Ei = E.Ex
  elseif nameE == "Ey"
    Ei = E.Ey
  else
    Ei = E.Ez
  end
  if nameB == "Bx"
    Bi = B.Bx
  elseif nameB == "By"
    Bi = B.By
  else
    Bi = B.Bz
  end
  pEt = plot(E.t, Ei,ylabel = nameE)
  pBt = plot(B.t, Bi,ylabel = nameB)
  pSt = plot(B.t, Ei .* Bi,ylabel = "S(t)",xlabel = "t",legend=false)
  display(plot(pEt,pBt,pSt,layout=(3,1)))
  ## ### FFT
  nEvec = params_correct["EB_source_xyz"][1] + 10:nx - nnn * nw
  nBvec = nEvec[1:end-1]
  nd = length(data[nameE][nEvec])
  nfft = nextpow(2,nd)
  da = zeros(nfft)
  da[1:length(nEvec)] = data[nameE][nEvec]
  Y = fft(da)
  E2fft = abs.(Y / Lx)
  pEfft = plot(E2fft,ylabel= nameE,title = title)
  ####
  nd = length(data[nameB][nBvec])
  nfft = nextpow(2,nd)
  da = zeros(nfft)
  da[1:length(nBvec)] = data[nameB][nBvec]
  Y = fft(da)
  B2fft = abs.(Y / Lx)
  pBfft = plot(B2fft,ylabel= nameB)
  ####
  nd = length(S[nEvec])
  nfft = nextpow(2,nd)
  da = zeros(nfft)
  da[1:length(nEvec)] = S[nEvec]
  Y = fft(da)
  S2fft = abs.(Y / Lx)
  pSfft = plot(S2fft,xlabel="frequency [Hz]",ylabel="Sx",legend= false)
  display(plot(pEfft,pBfft,pSfft,layout=(3,1)))
elseif params_calculated["dims"]["x"] == 2
  if E0[2] ≠ 0 || B0[3] ≠ 0
    nameE = "Ey"
    nameB = "Bz"
  elseif E0[3] ≠ 0 || B0[2] ≠ 0
    nameE = "Ez"
    nameB = "By"
  else
  end
  E = datat["E"]
  B = datat["B"]
  if nameE == "Ex"
    Ei = E.Ex
  elseif nameE == "Ey"
    Ei = E.Ey
  else
    Ei = E.Ez
  end
  if nameB == "Bx"
    Bi = B.Bx
  elseif nameB == "By"
    Bi = B.By
  else
    Bi = B.Bz
  end
  pEt = plot(E.t, Ei,ylabel = nameE)
  pBt = plot(B.t, Bi,ylabel = nameB)
  pSt = plot(B.t, Ei .* Bi,ylabel = "S(t)",xlabel = "t",legend=false)
  display(plot(pEt,pBt,pSt,layout=(3,1)))
  Δy = (params_correct["bounds_y"][2] - params_correct["bounds_y"][1]) /  (params_correct["grid_size"]["y"]-1)
  yvec = params_correct["bounds_y"][1]:Δy:params_correct["bounds_y"][2] |> collect
  ndata = 37
  Endata = data[nameE][:,ndata]
  Bndata = data[nameB][:,ndata]
      pE = plot(Endata,ylabel=nameE)
      pB = plot(Bndata,ylabel=nameB)
      pS = plot(Endata .* Bndata, xlabel="x", ylabel="S")
      display(plot(pE,pB,pS,layout=(3,1)))
  # heatmap(data[nameB],xlabel="x",ylabel="y")
  heatmap(data[nameE],xlabel="y",ylabel="x")
end
