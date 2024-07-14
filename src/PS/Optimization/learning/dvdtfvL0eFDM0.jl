
"""
  dy/dv: Numerical first order derivative for vector y(v) with Newton method or Central Difference method

   Inputs:
     orders::Int =  1 or 2
     is_boundv0::Bool = false, default. When is true, `v[1] == 0.0`

   outputs:
     Rdvy = RdvdtfvL0CDS(Rdvy,yy,nv,dv;orders=orders,is_boundv0=is_boundv0)
     Rdy = RddtfvL0CDS(Rdy,yy,nv;orders=orders,is_boundv0=is_boundv0)
     Rddvy = RddvdtfvL0CDS(Rddvy,yy,nv,dv;orders=orders,is_boundv0=is_boundv0)
     Rddy = RdddtfvL0CDS(Rddy,yy,nv;orders=orders,is_boundv0=is_boundv0)
     RdRdy = RdRddtfvL0CDS(RdRdy,yy,nv;orders=orders,is_boundv0=is_boundv0)
     RdRdRdy = RdRdRddtfvL0CDS(RdRdRdy,yy,nv;orders=orders,is_boundv0=is_boundv0)
     dvy = dvdtfvL0CDS(dvy,yy,nv,dv;orders=orders,is_boundv0=is_boundv0)
     dy = ddtfvL0CDS(dy,yy,nv;orders=orders,is_boundv0=is_boundv0)
"""
## for uniform grid with first and second order approximation

# 1D, Rdvy = (diff(RdtfvL) / (RdtfvL)) / diff(vGe)
function RdvdtfvL0CDS(dvy::AbstractVector{T},yy::AbstractVector{T},
    nv::Int64,dv::T;orders::Int64=1,is_boundv0::Bool=false) where{T}
    
    if orders == 1       # ForwardDiff
        for i in 2:nv-1
            dvy[i] = (1 - yy[i-1] / yy[i]) / dv
        end
        if is_boundv0 == false
            i = 1
            dvy[i] = 2dvy[i+1] - dvy[i+2]
        end
        i = nv
        if yy[end] == 0.0
            dvy[i] = 2dvy[i-1] - dvy[i-2]
        else
            dvy[i] = (1 - yy[i-1] / yy[i]) / dv
        end
    elseif orders == - 1 # BackwardDiff
        for i in 2:nv-1
            dvy[i] = (yy[i+1] / yy[i] - 1) / dv
        end
        if is_boundv0 == false
            if yy[1] == 0.0
               dvy[1] = 2dvy[2] - dvy[3]
            else
                i = 1
                dvy[i] = (yy[i+1] / yy[i] - 1) / dv
            end
        end
        i = nv
        dvy[i] = 2dvy[i-1] - dvy[i-2]
    elseif orders == 2   # CentralDiff
        dv2 = 2dv
        for i in 2:nv-1
            dvy[i] = (yy[i+1] - yy[i-1]) / yy[i] / dv2
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

# 1D, Rdy = diff(RdtfvL) / (RdtfvL)
function RddtfvL0CDS(dvy::AbstractVector{T},yy::AbstractVector{T},
    nv::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T}
    
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

# 1D,  Rddvy = diff(Rdvy) / Rdvy
function RddvdtfvL0CDS(Rddvy::AbstractVector{T},yy::AbstractVector{T},
    nv::Int64,dv::T;orders::Int64=1,is_boundv0::Bool=false) where{T}
    
    dvy = zero.(yy)
    dvy = dvdtfvL0CDS(dvy,yy,nv,dv;orders=orders,is_boundv0=is_boundv0)
    Rddvy = RddtfvL0CDS(Rddvy,dvy,nv;orders=orders,is_boundv0=false)
    return Rddvy
end

# 1D,   RdRdy = diff(Rdy) / Rdy
function RdRddtfvL0CDS(RdRdy::AbstractVector{T},yy::AbstractVector{T},
    nv::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T}
    
    Rdy = zero.(yy)
    Rdy = RddtfvL0CDS(Rdy,yy,nv;orders=orders,is_boundv0=is_boundv0)
    RdRdy = RddtfvL0CDS(RdRdy,Rdy,nv;orders=orders,is_boundv0=false)
    return RdRdy
end

# 1D,   RdRdRdy = diff(RdRdy) / RdRdy
function RdRdRddtfvL0CDS(RdRdRdy::AbstractVector{T},yy::AbstractVector{T},
    nv::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T}
    
    Rdy = zero.(yy)
    Rdy = RddtfvL0CDS(Rdy,yy,nv;orders=orders,is_boundv0=is_boundv0)
    RdRdy = zero.(yy)
    RdRdy = RddtfvL0CDS(RdRdy,Rdy,nv;orders=orders,is_boundv0=false)
    RdRdRdy = RddtfvL0CDS(RdRdRdy,RdRdy,nv;orders=orders,is_boundv0=false)
    return RdRdRdy
end

# 1D, Rddy = diff(dy) / dy
function RdddtfvL0CDS(Rddy::AbstractVector{T},yy::AbstractVector{T}, 
    nv::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T}
    
    dy = zero.(yy)
    dy = ddtfvL0CDS(dy,yy,nv;orders=orders,is_boundv0=is_boundv0)
    Rddy = RddtfvL0CDS(Rddy,dy,nv;orders=orders,is_boundv0=false)
    return Rddy
end


# 1D, dvy = diff(RdtfvL) / diff(vGe)
function dvdtfvL0CDS(dvy::AbstractVector{T},yy::AbstractVector{T},
    nv::Int64,dv::T;orders::Int64=1,is_boundv0::Bool=false) where{T}
    
    if orders == 1       # ForwardDiff
        for i in 2:nv
            dvy[i] = (yy[i] - yy[i-1]) / dv
        end
        if is_boundv0 == false
            i = 1
            dvy[i] = 2dvy[i+1] - dvy[i+2]
        end
    elseif orders == - 1 # BackwardDiff
        for i in 2:nv-1
            dvy[i] = (yy[i+1] - yy[i]) / dv
        end
        if is_boundv0 == false
            i = 1
            dvy[i] = (yy[i+1] - yy[i]) / dv
        end
        i = nv
        dvy[i] = 2dvy[i-1] - dvy[i-2]
    elseif orders == 2   # CentralDiff
        dv2 = 2dv
        for i in 2:nv-1
            dvy[i] = (yy[i+1] - yy[i-1]) / dv2
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

# 1D, dy = diff(RdtfvL)
function ddtfvL0CDS(dvy::AbstractVector{T},yy::AbstractVector{T},
    nv::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T}
    
    if orders == 1       # ForwardDiff
        for i in 2:nv
            dvy[i] = (yy[i] - yy[i-1])
        end
        if is_boundv0 == false
            i = 1
            dvy[i] = 2dvy[i+1] - dvy[i+2]
        end
    elseif orders == - 1 # BackwardDiff
        for i in 2:nv-1
            dvy[i] = (yy[i+1] - yy[i])
        end
        if is_boundv0 == false
            i = 1
            dvy[i] = (yy[i+1] - yy[i])
        end
        i = nv
        dvy[i] = 2dvy[i-1] - dvy[i-2]
    elseif orders == 2   # CentralDiff
        for i in 2:nv-1
            dvy[i] = (yy[i+1] - yy[i-1]) / 2
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

"""
   Inputs:
   outputs:
     Rdvy = RdvdtfvL0CDS(Rdvy,yy,nv,dv,fvL,LM;orders=orders,is_boundv0=is_boundv0)
     Rdy = RddtfvL0CDS(Rdy,yy,nv,dv,fvL,LM;orders=orders,is_boundv0=is_boundv0)
     Rddvy = RddvdtfvL0CDS(Rddvy,yy,nv,dv,fvL,LM;orders=orders,is_boundv0=is_boundv0)
     Rddy = RdddtfvL0CDS(Rddy,yy,nv,fvL,LM;orders=orders,is_boundv0=is_boundv0)
     RdRdy = RdRddtfvL0CDS(RdRdy,yy,nv,fvL,LM;orders=orders,is_boundv0=is_boundv0)
     RdRdRdy = RdRdRddtfvL0CDS(RdRdRdy,yy,nv,fvL,LM;orders=orders,is_boundv0=is_boundv0)
     dvy = dvdtfvL0CDS(dvy,yy,nv,dv,fvL,LM;orders=orders,is_boundv0=is_boundv0)
     dy = ddtfvL0CDS(dy,yy,nv,fvL,LM;orders=orders,is_boundv0=is_boundv0)
"""

# 2D, Rdvy = (diff(RdtfvL) / (RdtfvL)) / diff(vGe)
function RdvdtfvL0CDS(Rdvy::AbstractArray{T,N},yy::AbstractArray{T,N},nv::Int64,dv::T,
    fvL::AbstractArray{T,N},LM::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T,N}
    
    Ryy = copy(yy[:,L1])
    for L1 in 1:LM+1
        Ryy = yy[:,L1] ./ fvL[:,L1]
        if fvL[1,L1] == 0.0
            Ryy[1] = 2Ryy[2] - Ryy[3]
        end
        Rdvy[:,L1] = RdvdtfvL0CDS(Rdvy[:,L1],Ryy,nv,dv;orders=orders,is_boundv0=is_boundv0)
    end
    return Rdvy
end

# 2D, Rdy = (diff(RdtfvL) / (RdtfvL))
function RddtfvL0CDS(Rdy::AbstractArray{T,N},yy::AbstractArray{T,N},nv::Int64,
    fvL::AbstractArray{T,N},LM::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T,N}
    
    Ryy = copy(yy[:,L1])
    for L1 in 1:LM+1
        Ryy = yy[:,L1] ./ fvL[:,L1]
        if fvL[1,L1] == 0.0
            Ryy[1] = 2Ryy[2] - Ryy[3]
        end
        Rdy[:,L1] = RddtfvL0CDS(Rdy[:,L1],Ryy,nv;orders=orders,is_boundv0=is_boundv0)
    end
    return Rdy
end

# 2D,  Rddvy = diff(Rdvy) / Rdvy
function RddvdtfvL0CDS(Rddvy::AbstractArray{T,N},yy::AbstractArray{T,N},nv::Int64,dv::T,
    fvL::AbstractArray{T,N},LM::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T,N}
    
    Ryy = copy(yy[:,L1])
    for L1 in 1:LM+1
        Ryy = yy[:,L1] ./ fvL[:,L1]
        if fvL[1,L1] == 0.0
            Ryy[1] = 2Ryy[2] - Ryy[3]
        end
        Rddvy[:,L1] = RddvdtfvL0CDS(Rddvy[:,L1],Ryy,nv,dv;orders=orders,is_boundv0=is_boundv0)
    end
    return Rddvy
end

# 2D,  Rddy = diff(dy) / dy
function RdddtfvL0CDS(Rddy::AbstractArray{T,N},yy::AbstractArray{T,N},nv::Int64,
    fvL::AbstractArray{T,N},LM::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T,N}
    
    Ryy = copy(yy[:,L1])
    for L1 in 1:LM+1
        Ryy = yy[:,L1] ./ fvL[:,L1]
        if fvL[1,L1] == 0.0
            Ryy[1] = 2Ryy[2] - Ryy[3]
        end
        Rddy[:,L1] = RdddtfvL0CDS(Rddy[:,L1],Ryy,nv;orders=orders,is_boundv0=is_boundv0)
    end
    return Rddy
end

# 2D,  RdRdy = diff(Rdy) / Rdy
function RdRddtfvL0CDS(RdRdy::AbstractArray{T,N},yy::AbstractArray{T,N},nv::Int64,
    fvL::AbstractArray{T,N},LM::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T,N}
    
    Ryy = copy(yy[:,L1])
    for L1 in 1:LM+1
        Ryy = yy[:,L1] ./ fvL[:,L1]
        if fvL[1,L1] == 0.0
            Ryy[1] = 2Ryy[2] - Ryy[3]
        end
        RdRdy[:,L1] = RdRddtfvL0CDS(RdRdy[:,L1],Ryy,nv;orders=orders,is_boundv0=is_boundv0)
    end
    return RdRdy
end

# 2D,  RdRdRdy = diff(RdRdy) / RdRdy
function RdRdRddtfvL0CDS(RdRdRdy::AbstractArray{T,N},yy::AbstractArray{T,N},nv::Int64,
    fvL::AbstractArray{T,N},LM::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T,N}
    
    Ryy = copy(yy[:,L1])
    for L1 in 1:LM+1
        Ryy = yy[:,L1] ./ fvL[:,L1]
        if fvL[1,L1] == 0.0
            Ryy[1] = 2Ryy[2] - Ryy[3]
        end
        RdRdRdy[:,L1] = RdRdRddtfvL0CDS(RdRdRdy[:,L1],Ryy,nv;orders=orders,is_boundv0=is_boundv0)
    end
    return RdRdRdy
end

# 2D, dvy = diff(RdtfvL) / diff(vGe)
function dvdtfvL0CDS(dvy::AbstractArray{T,N},yy::AbstractArray{T,N},nv::Int64,dv::T,
    fvL::AbstractArray{T,N},LM::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T,N}
    
    Ryy = copy(yy[:,L1])
    for L1 in 1:LM+1
        Ryy = yy[:,L1] ./ fvL[:,L1]
        if fvL[1,L1] == 0.0
            Ryy[1] = 2Ryy[2] - Ryy[3]
        end
        dvy[:,L1] = dvdtfvL0CDS(dvy[:,L1],Ryy,nv,dv;orders=orders,is_boundv0=is_boundv0)
    end
    return dvy
end

# 2D, dy = diff(RdtfvL)
function ddtfvL0CDS(dy::AbstractArray{T,N},yy::AbstractArray{T,N},nv::Int64,
    fvL::AbstractArray{T,N},LM::Int64;orders::Int64=1,is_boundv0::Bool=false) where{T,N}
    
    Ryy = copy(yy[:,L1])
    for L1 in 1:LM+1
        Ryy = yy[:,L1] ./ fvL[:,L1]
        if fvL[1,L1] == 0.0
            Ryy[1] = 2Ryy[2] - Ryy[3]
        end
        dy[:,L1] = ddtfvL0CDS(dy[:,L1],Ryy,nv;orders=orders,is_boundv0=is_boundv0)
    end
    return dy
end
