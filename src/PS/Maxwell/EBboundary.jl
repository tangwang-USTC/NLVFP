"""
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

"""

"""
  Mur ABC boudary conditions
"""

function generate_boundary_Efields!(d, p)
  names = ("Ex","Ey","Ez")
  nx = p["grid_size"][1]
  if p["dims"][2] == 1
    ###                        Left boundary of x dimension
    if p["bounds_conds"] == "MurABS"
      for name in names
        d[name][1] = p["bounds_stepx"][name][2]
        p["bounds_stepx"][name][2] = p["bounds_stepx"][name][1]
        p["bounds_stepx"][name][1] = d[name][2]
      end
      #################   right boundary of x dimension
      for name in names
        d[name][nx-1] = p["bounds_stepx"][name][4]
        p["bounds_stepx"][name][4] = p["bounds_stepx"][name][3]
        p["bounds_stepx"][name][3] = d[name][nx-2]
      end
    end
  elseif  p["dims"][2] == 2
    ny = p["grid_size"][2]
    ##################     Left boundary of x dimension, x0
    if p["bounds_conds"] == "MurABS"
      for name in names
        d[name][1,:] = p["bounds_stepx"][name][:,2]
        p["bounds_stepx"][name][:,2] = p["bounds_stepx"][name][:,1]
        p["bounds_stepx"][name][:,1] = d[name][2,:]
      end
      ##################   right boundary of x dim
      for name in names
        d[name][nx-1,:] = p["bounds_stepx"][name][:,4]
        p["bounds_stepx"][name][:,4] = p["bounds_stepx"][name][:,3]
        p["bounds_stepx"][name][:,3] = d[name][nx-2,:]
      end
      #################    Left boundary of y dimension, y0
      for name in names
        d[name][:,1] = p["bounds_stepy"][name][:,2]
        p["bounds_stepy"][name][:,2] = p["bounds_stepy"][name][:,1]
        p["bounds_stepy"][name][:,1] = d[name][:,2]
      end
      #################   right boundary of y dimension
      for name in names
        d[name][:,ny-1] = p["bounds_stepy"][name][:,4]
        p["bounds_stepy"][name][:,4] = p["bounds_stepy"][name][:,3]
        p["bounds_stepy"][name][:,3] = d[name][:,ny-2]
      end
    end
  else
  end
end

function generate_boundary_Bfields!(d, p)
  nx = p["grid_size"][1]
  names = ("Bx","By","Bz")
  if p["dims"][2] == 1
    ######                Left boundary of x dimension
    if p["bounds_conds"] == "MurABS"
     for name in names
       d[name][1] = p["bounds_stepx"][name][2]
       p["bounds_stepx"][name][2] = p["bounds_stepx"][name][1]
       p["bounds_stepx"][name][1] = d[name][2]
       #####################   right boundary of x dimension
       d[name][nx-2] = p["bounds_stepx"][name][4]
       p["bounds_stepx"][name][4] = p["bounds_stepx"][name][3]
       p["bounds_stepx"][name][3] = d[name][nx-3]
     end
    end
  elseif  p["dims"][2] == 2
      ny = p["grid_size"][2]
      ###                        Left boundary of x dimension, x0
      if p["bounds_conds"] == "MurABS"
        for name in names
          d[name][1,:] = p["bounds_stepx"][name][:,2]
          p["bounds_stepx"][name][:,2] = p["bounds_stepx"][name][:,1]
          p["bounds_stepx"][name][:,1] = d[name][2,:]
        end
      ########################   right boundary of x dimension
        for name in names
          d[name][nx-2,:] = p["bounds_stepx"][name][:,4]
          p["bounds_stepx"][name][:,4] = p["bounds_stepx"][name][:,3]
          p["bounds_stepx"][name][:,3] = d[name][nx-3,:]
        end
      ###                        Left boundary of y dimension, y0
        for name in names
          d[name][:,1] = p["bounds_stepy"][name][:,2]
          p["bounds_stepy"][name][:,2] = p["bounds_stepy"][name][:,1]
          p["bounds_stepy"][name][:,1] = d[name][:,2]
        end
      ########################   right boundary of y dimension
        for name in names
          d[name][:,ny-2] = p["bounds_stepy"][name][:,4]
          p["bounds_stepy"][name][:,4] = p["bounds_stepy"][name][:,3]
          p["bounds_stepy"][name][:,3] = d[name][:,ny-3]
        end
      end
  else
  end
end
