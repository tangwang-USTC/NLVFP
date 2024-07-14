
"""
   Inputs:
     k::Int ∈ N⁺, 

   outputs:
     yy = RdtfvL0interp(yy,vGe,nv,nvc0;k=k,Nitp=Nitp)
"""

# 1D,
function RdtfvL0interp(yy::AbstractVector{T},vGe::AbstractVector{T},
    nv::Int64,nvc0::Int64;k::Int64=2,Nitp::Int64=10) where{T}
    
    if Nitp == 0
        nvecitp = nvc0:nv
    else
        nvecitp = nvc0:min(nvc0+Nitp,nv)
    end
    # nvec9 = min(nvc0+Nitp,nv)
    itpDL = Dierckx.Spline1D(vGe[nvecitp],yy[nvecitp];k=k,bc="extrapolate")
    yy[1:nvc0-1] = itpDL.(vGe[1:nvc0-1])
    return yy
end

"""
   Inputs:
     vG = vGe[1:nvc0+Nitp]
     yy[1:nvc0-1] = yy[nvc0:nvc0+Nitp]

   outputs:
     yyitp = RdtfvL0interp(yy,vGe,vGitp;k=k)
"""

# 1D, Nitp == 0
function RdtfvL0interp(yy::AbstractVector{T},vGe::AbstractVector{T},vGitp::AbstractVector{T};k::Int64=2) where{T}
    
    itpDL = Dierckx.Spline1D(vGe,yy;k=k,bc="extrapolate")
    return itpDL.(vGitp)
end

"""
   Inputs:
     nvc: = [nvcy0, nvcd1, nvcd2, nvcd3]
     order_nvc_itp: (=2, default). # (1,2,3) → (nvcd1, nvcd2, nvcd3)

   outputs:
     yy = RdtfvL0interp(yy,vGe,nv,nvc,LM;k=k,Nitp=Nitp,order_nvc_itp=order_nvc_itp)
"""

# 2D, 
function RdtfvL0interp(yy::AbstractArray{T,N},vGe::AbstractVector{T},nv::Int64,
    nvc::Array{Int64},LM::Int64;k::Int64=2,Nitp::Int64=10,order_nvc_itp::Int64=2) where{T,N}
         
    for L1 in 1:LM+1
        nvc0 = nvc[1+order_nvc_itp,L1]
        if nvc0 ≥ 3
            if nvc[1,L1] ≥ nvc0 + Nitp
                if nv - nvc0 > Nitp
                    # yy[:,L1] = RdtfvL0interp(yy[:,L1],vGe,nv,nvc0;k=k,Nitp=Nitp)
                    # vGitp = vGe[nvc0:nvc0+Nitp]
                    yy[1:nvc0-1,L1] = RdtfvL0interp(yy[nvc0:nvc0+Nitp,L1],vGe[nvc0:nvc0+Nitp],vGe[1:nvc0-1];k=k)
                else
                    error("`yy` should be zeros according to the smoothness of the function. Checking it carefully!!!")
                end
            else
              @warn("`nvc0 + Nitp ≥ nvcy0` may cause the interpolations to be unstable, or even to be falured. L = ", L1-1)
              @show (L1,nv), nvc[:,L1]
              fghjfgh
            end
        end
    end
    return yy
end
