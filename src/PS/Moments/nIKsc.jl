
"""

  Inputs:
    fvLc: = fvL * (na / vth^3 / π^(3/2))

  Outputs:
    nIKsc!(nIKc,fvLce,ve,ma;atol_nIK=atol_nIK)
    nIKsc!(nIKc,fvLce,ve,ma,ns;atol_nIK=atol_nIK)
    nIKsc!(nIKc,fvLce,vhe,ma,vth;atol_nIK=atol_nIK)
    nIKsc!(nIKc,fvLce,vhe,ma,vth,ns;atol_nIK=atol_nIK)

    ## Testting
    nIKsc = zeros(3,ns)
    nIKsc!(nIKsc,fvLc0e[:,1:2,:],vhe,ma,vth,ns;atol_nIK=atol_nIK)

"""

# 2D, [nIKc,ns], ve
function nIKsc!(nIKc::AbstractArray{T,N2},fvLc::AbstractVector{Matrix{T}},ve::AbstractVector{StepRangeLen},
  ma::AbstractVector{T},ns::Int64;atol_nIK::T=epsT100,is_out_err::Bool=true) where{T,N2}
 
  nn = length(nIKc[:,1])
  if is_out_err
      errn, errI, errK = zeros(T,ns),zeros(T,ns),zeros(T,ns)
      for isp in 1:ns
          a = zeros(T,nn)
          errn[isp], errI[isp], errK[isp] = nIKsc!(a,fvLc[isp],
              ve[isp],ma[isp];atol_nIK=atol_nIK,is_out_err=is_out_err)
          nIKc[:,isp] = a
      end
      norm(errn) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `n̂a = 0`",errn)
      norm(errI) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `Ia = 0`",errI)
      norm(errK) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `Ka = 0`",errK)
  else
    for isp in 1:ns
        a = zeros(T,nn)
        nIKsc!(a,fvLc[isp],ve[isp],ma[isp];atol_nIK=atol_nIK,is_out_err=is_out_err)
        nIKc[:,isp] = a
    end
  end
end

# 1D, [nIKc], ve
function nIKsc!(nIKc::AbstractVector{T},fvLc::AbstractArray{T,N},ve::StepRangeLen,
    ma::T;atol_nIK::T=epsT100,is_out_err::Bool=true) where{T,N}
    
  # n, I, K, T = nIKc[1:4]
  v2 = ve .^2
  nIKc[1], errn = romberg(ve, (v2 .* fvLc[:,1]))
  nIKc[2], errI = romberg(ve, (v2 .* ve  .* fvLc[:,2]))
  nIKc[3], errK = romberg(ve, (v2 .^2  .* fvLc[:,1]))
  cc = 4pi
  nIKc[1] *= cc
  nIKc[2] *= ma / 3 * cc
  nIKc[3] *= ma / 2 * cc
  
  if is_out_err
    return errn, errI, errK
  else
    errn < atol_nIK || @warn("Convergence of integration `na` if failure", errn)
    errI < atol_nIK || @warn("Convergence of integration `Ia` if failure", errI)
    errK < atol_nIK || @warn("Convergence of integration `Ka` if failure", errK)
  end
end

# 2D, [nIKc,ns], vhe
function nIKsc!(nIKc::AbstractArray{T,N2},fvLc::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
  ma::AbstractVector{T},vth::AbstractVector{T},ns::Int64;atol_nIK::T=epsT100,is_out_err::Bool=true) where{T,N2}
  
  nn = length(nIKc[:,1])
  if is_out_err
      errn, errI, errK = zeros(T,ns),zeros(T,ns),zeros(T,ns)
      for isp in 1:ns
          a = zeros(T,nn)
          errn[isp], errI[isp], errK[isp] = nIKsc!(a,fvLc[isp],
              vhe[isp],ma[isp],vth[isp];atol_nIK=atol_nIK,is_out_err=is_out_err)
          nIKc[:,isp] = a
      end
      norm(errn) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `n̂a = 0`",errn)
      norm(errI) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `Ia = 0`",errI)
      norm(errK) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `Ka = 0`",errK)
  else
    for isp in 1:ns
        a = zeros(T,nn)
        nIKsc!(a,fvLc[isp],vhe[isp],ma[isp],vth[isp];atol_nIK=atol_nIK,is_out_err=is_out_err)
        nIKc[:,isp] = a
    end
  end
end

# 1D, [nIKc], vhe
function nIKsc!(nIKc::AbstractVector{T},fvLc::AbstractArray{T,N},vhe::StepRangeLen,
    ma::T,vth::T;atol_nIK::T=epsT100,is_out_err::Bool=true) where{T,N}
    
  # n, I, K, T = nIKc[1:4]
  ve = vhe * vth
  v2 = ve .^2
  nIKc[1], errn = romberg(ve, (v2 .* fvLc[:,1]))
  nIKc[2], errI = romberg(ve, (v2 .* ve .* fvLc[:,2]))
  nIKc[3], errK = romberg(ve, (v2 .^2 .* fvLc[:,1]))
  cc = 4pi
  nIKc[1] *= cc
  nIKc[2] *= ma / 3 * cc
  nIKc[3] *= ma / 2 * cc
  
  if is_out_err
      return errn, errI, errK
  else
    errn < atol_nIK || @warn("Convergence of integration `na` if failure", errn)
    errI < atol_nIK || @warn("Convergence of integration `Ia` if failure", errI)
    errK < atol_nIK || @warn("Convergence of integration `Ka` if failure", errK)
  end
end
