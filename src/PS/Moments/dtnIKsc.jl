
"""

  Inputs:
    dtfvLc: = dtfvL * (na / vth^3 / π^(3/2))

  Outputs:
    dtnIKsc!(dtnIKc,dtfvLce,vhe,ma,vth;atol_nIK=atol_nIK)
    dtnIKsc!(dtnIKc,dtfvLce,vhe,ma,vth,ns;atol_nIK=atol_nIK)
    dtnIKsc!(dtnIKc,dtfvLce,ve,ma;atol_nIK=atol_nIK)
    dtnIKsc!(dtnIKc,dtfvLce,ve,ma,ns;atol_nIK=atol_nIK)

    ## Testting
    dtnIKsc = zeros(3,ns)
    dtnIKsc!(dtnIKsc,dtfvLc0e,vhe,ma,vth,ns;atol_nIK=atol_nIK)

"""

# 2D, [dtnIKc,ns], vhe
function dtnIKsc!(dtnIKc::AbstractArray{T,N2},dtfvLc::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
    ma::AbstractVector{T},vth::AbstractVector{T},ns::Int64;atol_nIK::T=epsT100,is_out_err::Bool=true) where{T,N2}
  
    # nn = length(dtnIKc[:,1])
    if is_out_err
        errdtn, errdtI, errdtK = zeros(T,ns),zeros(T,ns),zeros(T,ns)
        for isp in 1:ns
            a = dtnIKc[:,isp]
            errdtn[isp], errdtI[isp], errdtK[isp] = dtnIKsc!(a,dtfvLc[isp],vhe[isp],
                                  ma[isp],vth[isp];atol_nIK=atol_nIK,is_out_err=is_out_err)
            dtnIKc[:,isp] = a
        end
        norm(errdtn) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜn̂a = 0`",errdtn)
        norm(errdtI) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜIa = 0`",errdtI)
        norm(errdtK) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜKa = 0`",errdtK)
    else
        for isp in 1:ns
            a = dtnIKc[:,isp]
            dtnIKsc!(a,dtfvLc[isp],vhe[isp],ma[isp],vth[isp];atol_nIK=atol_nIK,is_out_err=is_out_err)
            dtnIKc[:,isp] = a
        end
    end
end

# 1D, [dtnIKc], vhe
function dtnIKsc!(dtnIKc::AbstractVector{T},dtfvLc::AbstractArray{T,N},
    vhe::StepRangeLen,ma::T,vth::T;atol_nIK::T=epsT100,is_out_err::Bool=false) where{T,N}
    
    # n, I, K, T = dtnIKc[1:4]
    ve = vhe * vth
    v2 = ve .^2
    dtnIKc[1], errdtn = romberg(ve, (v2 .* dtfvLc[:,1]))
    dtnIKc[2], errdtI = romberg(ve, (v2 .* ve  .* dtfvLc[:,2]))
    dtnIKc[3], errdtK = romberg(ve, (v2 .^2  .* dtfvLc[:,1]))
    cc = 4 / sqrtpi
    dtnIKc[1] *= cc
    dtnIKc[2] *= ma / 3 * cc
    dtnIKc[3] *= ma / 2 * cc
    
    if is_out_err
        return errdtn, errdtI, errdtK
    else
        errdtn < atol_nIK || @warn("Convergence of integration `δₜna` if failure", errdtn)
        errdtI < atol_nIK || @warn("Convergence of integration `δₜIa` if failure", errdtI)
        errdtK < atol_nIK || @warn("Convergence of integration `δₜKa` if failure", errdtK)
    end
end

"""
  Outputs:
    nIKsc = zeros(3,ns)
    enIKc = zeros(3,ns)
    dtnIKsc!(nIKsc,enIKc,fvLc0k1,ve,ma,na,vth,ns;
            atol_nIK=atol_nIK,is_norm_error=is_norm_error)
"""

# 2D, [dtnIKc,ns], ve
function dtnIKsc!(dtnIKc::AbstractArray{T,N2},enIKc::AbstractArray{T,N2},dtfvLc::AbstractVector{Matrix{T}},
    ve::AbstractVector{StepRangeLen},ma::AbstractVector{T},na::AbstractVector{T},vath::AbstractVector{T},
    ns::Int64;atol_nIK::T=epsT100,is_norm_error::Bool=truee) where{T,N2}
  
    for isp in 1:ns
        a = dtnIKc[:,isp]
        enIKc[1,isp], enIKc[2,isp], enIKc[3,isp] = dtnIKsc!(a,dtfvLc[isp],
                     ve[isp],ma[isp];atol_nIK=atol_nIK,is_out_err=true)
        dtnIKc[:,isp] = a

        if is_norm_error
            isp = 1
            www = (ma[isp] * na[isp] * vath[isp])
            enIKc[1,isp] /= na[isp]
            enIKc[2,isp] /= www
            enIKc[3,isp] /= (www * vath[isp] / 2)
        end
    end
    @show dtnIKc
    @show sum(dtnIKc[3,:])
    @show enIKc
end

function dtnIKsc!(dtnIKc::AbstractArray{T,N2},dtfvLc::AbstractVector{Matrix{T}},ve::AbstractVector{StepRangeLen},
    ma::AbstractVector{T},ns::Int64;atol_nIK::T=epsT100,is_out_err::Bool=true) where{T,N2}
  
    # nn = length(dtnIKc[:,1])
    if is_out_err
        errdtn, errdtI, errdtK = zeros(T,ns),zeros(T,ns),zeros(T,ns)
        for isp in 1:ns
            a = dtnIKc[:,isp]
            errdtn[isp], errdtI[isp], errdtK[isp] = dtnIKsc!(a,dtfvLc[isp],
                         ve[isp],ma[isp];atol_nIK=atol_nIK,is_out_err=is_out_err)
            dtnIKc[:,isp] = a
        end
        norm(errdtn) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜn̂a = 0`",errdtn)
        norm(errdtI) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜIa = 0`",errdtI)
        norm(errdtK) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜKa = 0`",errdtK)
    else
        for isp in 1:ns
            a = dtnIKc[:,isp]
            dtnIKsc!(a,dtfvLc[isp],ve[isp],ma[isp];atol_nIK=atol_nIK,is_out_err=is_out_err)
            dtnIKc[:,isp] = a
        end
    end
end

# 1D, [dtnIKc], ve
function dtnIKsc!(dtnIKc::AbstractVector{T},dtfvLc::AbstractArray{T,N},ve::StepRangeLen,
    ma::T;atol_nIK::T=epsT100,is_out_err::Bool=false) where{T,N}
    
    # n, I, K, T = dtnIKc[1:4]
    v2 = ve .^2
    dtnIKc[1], errdtn = romberg(ve, (v2 .* dtfvLc[:,1]))
    dtnIKc[2], errdtI = romberg(ve, (v2 .* ve  .* dtfvLc[:,2]))
    dtnIKc[3], errdtK = romberg(ve, (v2 .^2  .* dtfvLc[:,1]))
    cc = 4 / sqrtpi
    dtnIKc[1] *= cc
    dtnIKc[2] *= ma / 3 * cc
    dtnIKc[3] *= ma / 2 * cc
    
    if is_out_err
        return errdtn, errdtI, errdtK
    else
        errdtn < atol_nIK || @warn("Convergence of integration `δₜna` if failure", errdtn)
        errdtI < atol_nIK || @warn("Convergence of integration `δₜIa` if failure", errdtI)
        errdtK < atol_nIK || @warn("Convergence of integration `δₜKa` if failure", errdtK)
    end
end
