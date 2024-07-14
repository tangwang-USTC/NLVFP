
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

# # Initial_data
# data_empty = Dict()
# datat_empty = Dict()
# if params_calculated["dims"][1] == 1
#   for name in ("Ex","Ey","Ez")
#     data_empty[name] = zeros(Float64, params_calculated["grid_size"][1])
#   end
#   for name in ("Bx","By","Bz")
#     data_empty[name] = zeros(Float64, params_calculated["grid_size"][1]-1)
#   end
#   datat_empty["EBprobes"] = [5,5,5]
#   datat_empty["E"] = DataFrame(t = 0.0,Ex = 0.0, Ey = 0.0, Ez = 0.0)
#   datat_empty["B"] = DataFrame(t = 0.0,Bx = 0.0, By = 0.0, Bz = 0.0)
# elseif  params_calculated["dims"][1] == 2
#   for name in ("Ex","Ey","Ezx","Ezy","Ez")
#     data_empty[name] = zeros(Float64, params_calculated["grid_size"][1], params_calculated["grid_size"][2])
#   end
#   for name in ("Bx","By","Bzx","Bzy","Bz")
#     data_empty[name] = zeros(Float64, params_calculated["grid_size"][1]-1, params_calculated["grid_size"][2]-1)
#   end
# else
# end
