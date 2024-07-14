"""
   Uniform meshgrids in physics space
    Ωₓ = [0,L1] × [0,L2] with steps dx and dy

   Input
     dim: Dimension = 2
     xspan = [x0,y0,x9,y9] , region of physics space
     dx: step of x dimension and  y dimension
     nx: grids number of x dimension and  y dimension

   Output
     mesh = xydxdy(xspan,nx,dims=dims)

"""

function xydxdy(xspan::Vector,nx::Vector;dims=dims)

    x0 = xspan[1:dims]       # [x0,y0,z0]
    x9 = xspan[dims+1:end]   # [x9,y9,z9]
    mesh = Array{Float64,dims}
    if dims == 1
        return collect(range(x0[1],stop=x9[1],length=nx[1]))
    elseif dims == 2
        i = 1
        xvec = collect(range(x0[i],stop=x9[i],length=nx[i]))
        i = 2
        yvec = collect(range(x0[i],stop=x9[i],length=nx[i]))
        return xvec .* yvec'
    else dims == 3
        i = 1
        xvec = collect(range(x0[i],stop=x9[i],length=nx[i]))
        i = 2
        yvec = collect(range(x0[i],stop=x9[i],length=nx[i]))
        xy = xvec .* yvec'
        i = 3
        zvec = collect(range(x0[i],stop=x9[i],length=nx[i]))
        for k = 1:nx[3]
            mesh[k] = zvec[k] * xy
        end
        return mesh
    end
end
