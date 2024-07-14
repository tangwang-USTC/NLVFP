
"""
  dy/dv: Numerical first order derivative for vector y(v) with Newton method or Central Difference method

   Inputs:
     orders:: # (=2, default), `orders ∈ [-1, 1, 2]` which denotes `[BackwardDiff, ForwardDiff, CentralDiff]`
     is_boundv0::Bool = false, default. It is `true` when `v[1] == 0.0` and the value `Rdiy[1] = 0.0`

   outputs:
     Rd1y = RdpdtfvL0CDS(Rd1y,yy,nv;orders=orders,is_boundv0=is_boundv0)
     Rd2y, Rd1y = RdpdtfvL0CDS(Rd2y, Rd1y,yy,nv;orders=orders,is_boundv0=is_boundv0)
     Rd3y, Rd2y, Rd1y = RdpdtfvL0CDS(Rd3y, Rd2y, Rd1y,yy,nv;orders=orders,is_boundv0=is_boundv0)
"""
## for uniform grid with first and second order approximation

# 1D, Rd1y = diff(RdtfvL) / (RdtfvL)
function RdpdtfvL0CDS(dvy::AbstractVector{T},yy::AbstractVector{T},
    nv::Int64;orders::Int64=1,is_boundv0::Bool=true) where{T}
    
    if orders == 1       # ForwardDiff
        for i in 2:nv-1
            dvy[i] = (1 - yy[i-1] / yy[i])
        end
        if is_boundv0 == false
            i = 1
            dvy[i] = 2dvy[i+1] - dvy[i+2]
        end
        i = nv
        if yy[end] == 0.0
            dvy[i] = 2dvy[i-1] - dvy[i-2]
        else
            dvy[i] = (1 - yy[i-1] / yy[i])
        end
    elseif orders == - 1 # BackwardDiff
        for i in 2:nv-1
            dvy[i] = (yy[i+1] / yy[i] - 1)
        end
        if is_boundv0 == false
            if yy[1] == 0.0
               dvy[1] = 2dvy[2] - dvy[3]
            else
                i = 1
                dvy[i] = (yy[i+1] / yy[i] - 1)
            end
        end
        i = nv
        dvy[i] = 2dvy[i-1] - dvy[i-2]
    elseif orders == 2   # CentralDiff
        for i in 2:nv-1
            dvy[i] = (yy[i+1] - yy[i-1]) / yy[i] / 2
        end
        if is_boundv0 == false
            # i = 1
            dvy[1] = 2dvy[2] - dvy[3]
        end
        i = nv
        dvy[i] = 2dvy[i-1] - dvy[i-2]
    else
        eherh
    end
    return dvy
end

# 1D,   Rd2y = diff(Rd1y) / Rd1y
function RdpdtfvL0CDS(Rd2y::AbstractVector{T},Rd1y::AbstractVector{T},
    yy::AbstractVector{T},nv::Int64;orders::Int64=1,is_boundv0::Vector{Bool}=[true,false]) where{T}
    
    Rd1y = RdpdtfvL0CDS(Rd1y,yy,nv;orders=orders,is_boundv0=is_boundv0[1])
    Rd2y = RdpdtfvL0CDS(Rd2y,Rd1y,nv;orders=orders,is_boundv0=is_boundv0[2])
    return Rd2y, Rd1y
end

# 1D,   Rd3y = diff(Rd2y) / Rd2y
function RdpdtfvL0CDS(Rd3y::AbstractVector{T},Rd2y::AbstractVector{T},Rd1y::AbstractVector{T},
    yy::AbstractVector{T},nv::Int64;orders::Int64=1,is_boundv0::Vector{Bool}=[true,false,false]) where{T}
    
    Rd1y = RdpdtfvL0CDS(Rd1y,yy,nv;orders=orders,is_boundv0=is_boundv0[1])
    Rd2y = RdpdtfvL0CDS(Rd2y,Rd1y,nv;orders=orders,is_boundv0=is_boundv0[2])
    Rd3y = RdpdtfvL0CDS(Rd3y,Rd2y,nv;orders=orders,is_boundv0=is_boundv0[3])
    return Rd3y, Rd2y, Rd1y
end

"""
   Inputs:
     orders:: # (=2, default), `orders ∈ [-1, 1, 2]` which denotes `[BackwardDiff, ForwardDiff, CentralDiff]`
     is_boundv0::Bool = false, default. It is `true` when `v[1] == 0.0` and the value `Rdiy[1] = 0.0`

   outputs:
     Rd1y = RdpdtfvL0CDS(Rd1y,yy,nv,dv,fvL,LM;orders=orders,is_boundv0=is_boundv0)
     Rd2y = RdpdtfvL0CDS(Rd2y,yy,nv,fvL,LM;orders=orders,is_boundv0=is_boundv0)
     Rd3y = RdpdtfvL0CDS(Rd3y,yy,nv,fvL,LM;orders=orders,is_boundv0=is_boundv0)
"""

# 2D, Rd1y = (diff(RdtfvL) / (RdtfvL))
function RdpdtfvL0CDS(Rd1y::AbstractArray{T,N},yy::AbstractArray{T,N},nv::Int64,
    fvL::AbstractArray{T,N},LM::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T,N}
    
    Ryy = deepcopy(yy[:,L1])
    for L1 in 1:LM+1
        Ryy = yy[:,L1] ./ fvL[:,L1]
        if fvL[1,L1] == 0.0
            Ryy[1] = 2Ryy[2] - Ryy[3]
        end
        Rd1y[:,L1] = RdpdtfvL0CDS(Rd1y[:,L1],Ryy,nv;orders=orders,is_boundv0=is_boundv0)
    end
    return Rd1y
end

# 2D,  Rd2y = diff(Rd1y) / Rd1y
function RdpdtfvL0CDS(Rd2y::AbstractArray{T,N},Rd1y::AbstractArray{T,N},yy::AbstractArray{T,N},nv::Int64,
    fvL::AbstractArray{T,N},LM::Int64;orders::Int64=1,is_boundv0::Vector{Bool}=[true,false]) where{T,N}
    
    Ryy = deepcopy(yy[:,L1])
    for L1 in 1:LM+1
        Ryy = yy[:,L1] ./ fvL[:,L1]
        if fvL[1,L1] == 0.0
            Ryy[1] = 2Ryy[2] - Ryy[3]
        end
        Rd2y[:,L1], Rd1y[:,L1] = RdpdtfvL0CDS(Rd2y[:,L1],Rd1y[:,L1],Ryy,nv;orders=orders,is_boundv0=is_boundv0)
    end
    return Rd2y, Rd1y
end

# 2D,  Rd3y = diff(Rd2y) / Rd2y
function RdpdtfvL0CDS(Rd3y::AbstractArray{T,N},Rd2y::AbstractArray{T,N},Rd1y::AbstractArray{T,N},yy::AbstractArray{T,N},
    nv::Int64,fvL::AbstractArray{T,N},LM::Int64;orders::Int64=1,is_boundv0::Vector{Bool}=[true,false,false]) where{T,N}
    
    Ryy = deepcopy(yy[:,1])
    for L1 in 1:LM+1
        Ryy = yy[:,L1] ./ fvL[:,L1]
        if fvL[1,L1] == 0.0
            Ryy[1] = 2Ryy[2] - Ryy[3]
        end
        Rd3y[:,L1],Rd2y[:,L1],Rd1y[:,L1] = RdpdtfvL0CDS(Rd3y[:,L1],Rd2y[:,L1],Rd1y[:,L1],Ryy,nv;orders=orders,is_boundv0=is_boundv0)
    end
    return Rd3y, Rd2y, Rd1y
end
