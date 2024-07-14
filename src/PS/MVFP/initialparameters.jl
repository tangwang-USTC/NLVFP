
"""
  Calculate parameters needed for simulation and return them as a dictionary.
  Initial the parameters include [Lx, Δx, Δt, ctx, y]
"""
function init_params(parameters)
  params = deepcopy(parameters)
  dimsx = params["dims"][1]
  nx = params["grid_size"][1]
  if dimsx == 1
    params["grid_size"] = params["grid_size"][1]
    params["bounds"] = params["bounds"]["x"]
    params["box_size"] = Dict{String,Float64}()
    params["box_size"]["x"] = params["bounds"][2] - params["bounds"][1]

    # Δx = box_size["x"] / nx
    params["space_step"] = Dict{String,Float64}()
    params["space_step"]["x"] = params["box_size"]["x"] / params["grid_size"][1]

    ## CFL conditions, time step limited by  Δt ≤ n_CFL * Δx
    if params["is_cfl"] == "on"
      params["time_step"] = params["n_CFL"] * params["space_step"]["x"]
      # println("dt =", params["time_step"])
      # Coefficients = 1/dimsx * Δt / Δx
      params["ctx"] = Dict{String,Float64}()
      params["ctx"] = params["time_step"] / params["space_step"]["x"]
    else
      ctx = params["time_step"] / params["space_step"]["x"]
      if ctx > 0.99
        println("dt/dx =",ctx,", CFL condition may cause instablity.")
      end
      params["ctx"] = ctx
    end
    # Staggered meshgrids: for the E[xb] and (B[xc],f[xc])
      # boundary mesh: xb[:] = x0:Δx:x9 ,                length(x1) = nx     for E and f
      # central mesh: xc[:] = (x0 + Δx/2):Δx:(x9-Δx/2),  length(x2) = nx - 1 for B
    params["xb"] = range(params["bounds"][1], length= nx, step=params["space_step"]["x"]) |> collect
    params["xc"] = range(params["bounds"][1] + 0.5*params["space_step"]["x"], length= nx -1, step=params["space_step"]["x"]) |> collect

    # boundary condition parameters
    if params["bounds_conds"] == "MurABS"
      params["bounds_stepx"] = Dict()
      for name in ("Ex","Ey","Ez","Bx","By","Bz")
        params["bounds_stepx"][name] =  zeros(4)
      end
    end
  elseif  dimsx == 2
    ny = params["grid_size"][2]
    params["box_size"] = Dict{String,Float64}()
    params["box_size"]["x"] = params["bounds"]["x"][2] - params["bounds"]["x"][1]
    params["box_size"]["y"] = params["bounds"]["y"][2] - params["bounds"]["y"][1]

    # Δx = box_size["x"] / nx; Δy = box_size["y"] / ny
    params["space_step"] = Dict{String,Float64}()
    params["space_step"]["x"] = params["box_size"]["x"]/ nx
    params["space_step"]["y"] = params["box_size"]["y"]/ ny

    ## CFL conditions, time step limited by n_CFL * Δt < Δx
    if params["is_cfl"] == "on"
      params["time_step"] = params["n_CFL"] * sqrt(params["space_step"]["x"]^2 + params["space_step"]["y"]^2)
    else
      if params["time_step"] > sqrt(params["space_step"]["x"]^2 + params["space_step"]["y"]^2)
        println(" dt / dx > 1, CFL condition may cause instablity.")
      end
    end
    # Coefficients = 1/dimsx * Δt / Δx
    params["ctx"] = Dict{String,Float64}()
    params["ctx"]["x"] = 1/dimsx * params["time_step"] / params["space_step"]["x"]
    params["ctx"]["y"] = 1/dimsx * params["time_step"] / params["space_step"]["y"]

    # Staggered meshgrids: for the E[xb,yb] and (B[xc,yc],f[xc,yc])
      # boundary mesh: xb[:] = x0:Δx:x9 ,                length(x1) = nx     for E and f
      # central mesh: xc[:] = (x0 + Δx/2):Δx:(x9-Δx/2),  length(x2) = nx - 1 for B
    params["xb"] = range(params["bounds"]["x"][1], length= nx, step=params["space_step"]["x"]) |> collect
    params["xc"] = range(params["bounds"]["x"][1] + 0.5*params["space_step"]["x"], length= nx -1, step=params["space_step"]["x"]) |> collect

      # boundary mesh: yb[:] = y0:Δy:y9 ,                length(y1) = ny     for E
      # central mesh: yc[:] = (y0 + Δy/2):Δy:(y9-Δy/2),  length(y2) = ny - 1 for B = B[1:end-1] with B[end] ≡ 0
    params["yb"] = range(params["bounds"]["y"][1], length= ny, step=params["space_step"]["y"]) |> collect
    params["yc"] = range(params["bounds"]["y"][1] + 0.5*params["space_step"]["y"], length= ny -1, step=params["space_step"]["y"]) |> collect

    # boundary condition parameters
    if params["bounds_conds"] == "MurABS"
      params["bounds_stepx"] = Dict()
      params["bounds_stepy"] = Dict()
      for name in ("Ex","Ey","Ez","Bx","By","Bz")
        params["bounds_stepx"][name] =  zeros(ny,4)
        params["bounds_stepy"][name] =  zeros(nx,4)
      end
    end
  else
  end
  return params
end

"""
  Initialize dataEB
    dataEB = ["Ex","Ey","Ezx","Ezy","Ez","Bx","By","Bzx","Bzy","Bz"]
    E = E[Ex, Ey, Ez], where Ez = Ezx + Ezy due to ∂ₜE3 = ∂x₁B2 - ∂x₂B1 in 2D3V model
    B = B[Bx, By, Bz], where Bz = Bzx + Bzy due to ∂ₜB3 = ∂x₁E2 - ∂x₂E1 in 2D3V model
"""

function init_dataEB(params)
  dimsx = params["dims"][1]
  dataEB = Dict{String, Array{Float64,dimsx}}()
  dataEBt = Dict()
  if dimsx == 1
    for name in ("Ex","Ey","Ez","Bx","By","Bz")
      dataEB[name] = zeros(Float64, params["grid_size"][1])
    end
    dataEBt["E"] = DataFrame(t = 0.0,Ex = 0.0, Ey = 0.0, Ez = 0.0)
    dataEBt["B"] = DataFrame(t = 0.0,Bx = 0.0, By = 0.0, Bz = 0.0)
  elseif  dimsx == 2
    for name in ("Ex","Ey","Ezx","Ezy","Ez","Bx","By","Bzx","Bzy","Bz")
      dataEB[name] = zeros(Float64, params["grid_size"][1], params["grid_size"][2])
    end
    dataEBt["E"] = DataFrame(t = 0.0,Ex = 0.0, Ey = 0.0, Ez = 0.0)
    dataEBt["B"] = DataFrame(t = 0.0,Bx = 0.0, By = 0.0, Bz = 0.0)
  else
  end
  return dataEB,dataEBt
end

"""
  Moment parameters such as na(x...,t), ua(x...,t), Ta(x...,t)
"""

# function init_Mom(params)
#   dimsx = params["dims"][1]
#   dataM = Dict{String, Array{Float64,dimsx}}()
#   dataMt = Dict()
#   if dimsx == 1
#     for name in ("na","ua","Ta")
#       dataM[name] = zeros(Float64, params["grid_size"][1])
#     end
#     dataMt["na"] = DataFrame(t = 0.0, na = 0.0)
#     dataMt["Ta"] = DataFrame(t = 0.0, Ta = 0.0)
#     dataMt["ua"] = DataFrame(t = 0.0,ux = 0.0, uy = 0.0, uz = 0.0)
#   elseif  dimsx == 2
#   else
#   end
#   return dataM,dataMt
# end
