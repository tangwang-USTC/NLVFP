
"""
  dy/dv: Numerical first order derivative for vector y(v) with Central Difference method

   outputs:
    dy/dv         when orders = 1
    dyv = derivationCDS(yy,v)
"""

## for non-uniform grid with second order approximation
function derivationCDS(yy::AbstractVector,v::AbstractVector)
    nv = length(v)

    dv = diff(v,dims = 1)
    ##
    dyv = zeros(1:nv)
    i = 2:nv-1
    dyv[i] = (dv[i] ./ dv[i .- 1] .* (yy[i] - yy[i .- 1]) +
             dv[i .- 1] ./ dv[i] .* (yy[i .+ 1] - yy[i])) ./ (dv[i] + dv[i .- 1])
    # j = 1  # left boundary
    itpDL = QuadraticInterpolation(dyv[2:5],v[2:5])
    dyv[1] = itpDL.(v[1])
    # j = nf # right boundary
    itpDL = Spline1D(v[nv-5:nv-1],dyv[nv-5:nv-1])
    dyv[nv] = itpDL.(v[nv])
    return dyv
end

function ratio2!(ratio::AbstractVector,x0::AbstractVector;method::Int=0)

    nx = length(x0)
    ratio[1] = (x0[2] - x0[1]) / (x0[2] + x0[1])
    rat1 = ratio[1]
    rat2 = 0.0
    for i in 2:nx-1
        rat2 = (x0[i+1] - x0[i]) / (x0[i+1] + x0[i])
        ratio[i] = rat1 + rat2
        rat1 = 1rat2
    end
    ratio[nx] = rat2
end

## for uniform grid with first and second order approximation
function dvdtfvLCDS(yy::AbstractVector{T},nv::Int64,dv::T;order::Int64=1) where{T}
    
    dvy = zeros(1:nv)
    if order == 1       # ForwardDiff
        for i in 2:nv-1
            dvy[i] = (yy[i] - yy[i-1]) / dv
        end
    elseif order == - 1 # BackwardDiff
        for i in 2:nv-1
            dvy[i] = (yy[i+1] - yy[i]) / dv
        end
    elseif order == 2   # CentralDiff
        dv2 = 2dv
        for i in 2:nv-1
            dvy[i] = (yy[i+1] - yy[i-1]) / dv2
        end
    else
        eherh
    end
    return dvy
end

function dvdtfvLCDS(yy::AbstractVector{T},nv::Int64,dv::T,
    fLn::AbstractVector{T},ℓ::Int64;order::Int64=1) where{T}
    
    dvy = zeros(1:nv)
    if order == 1       # ForwardDiff
        for i in 2:nv-1
            dvy[i] = (yy[i] - yy[i-1]) / dv
        end
    elseif order == - 1 # BackwardDiff
        for i in 2:nv-1
            dvy[i] = (yy[i+1] - yy[i]) / dv
        end
    elseif order == 2   # CentralDiff
        dv2 = 2dv
        for i in 2:nv-1
            dvy[i] = (yy[i+1] - yy[i-1]) / dv2
        end
    else
        eherh
    end
    return dvy
end
