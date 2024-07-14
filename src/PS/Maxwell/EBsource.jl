
"""
  Source field：

  Generate fields at left boundary (x[1] = p["x_bounds"][1]) at time moment 'time'

"""

function EBsource_fields!(d, time, p)
  if p["dims"][1] == 1
      isource = p["EB_source_xyz"][1]    # abs(isource - bounds) ≥ 2
      x = p["bounds"][1] + isource * p["space_step"]["x"]
    d["Ey"][isource] += p["E_shape_y"].(time, x)
    d["Ez"][isource] += p["E_shape_z"].(time, x)
      t = time + 0.5*p["time_step"]
      x = p["bounds"][1] + (isource + 0.5)*p["space_step"]["x"]
    d["Bz"][isource] += p["B_shape_z"].(t, x)
    d["By"][isource] += p["B_shape_y"].(t, x)
  elseif  p["dims"][1] == 2
    ny = p["grid_size"][2]
    isource = p["EB_source_xyz"][1]    # abs(isource - bounds) ≥ 2
    x = p["bounds"]["x"][1] + isource * p["space_step"]["x"]
    d["Ey"][isource,:] += p["E_shape_y"].(time, x, p["y1"])
  d["Ezx"][isource,:] += p["E_shape_z"].(time, x, p["y1"])
  d["Ez"][isource,:] = d["Ezx"][isource,:] + d["Ezy"][isource,:]
  #
    t = time + 0.5*p["time_step"]
    x = p["bounds"]["x"][1] + (isource + 0.5)*p["space_step"]["x"]
  d["By"][isource,1:ny-1] += p["B_shape_y"].(t, x, p["y2"])
  d["Bzx"][isource,1:ny-1] += p["B_shape_z"].(t, x, p["y2"])
  d["Bz"][isource,:] = d["Bzx"][isource,:] + d["Bzy"][isource,:]
  else
  end
end
