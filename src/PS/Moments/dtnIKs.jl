
"""

  Inputs:
    dtfvL: = dtfvLc / (na / vth^3 / π^(3/2))

  Outputs:
    dtnIKs!(dtnIK,dtfvLe,vhe,ma,vth;errnIKc=errnIKc)
    dtnIKs!(dtnIK,dtfvLe,vhe,ma,vth,ns;errnIKc=errnIKc)
    dtnIKs!(dtnIK,dtfvLe,ve,ma;errnIKc=errnIKc)
    dtnIKs!(dtnIK,dtfvLe,ve,ma,ns;errnIKc=errnIKc)

    ## Testting
    dtnIKs = zeros(3,ns)
    dtnIKs!(dtnIKs,dtfvL0e[:,1:2,:],vhe,ma,vth,ns;errnIKc=errnIKc)

"""

# 2D, [dtnIK,ns], vhe
function dtnIKs!(nIK::AbstractArray{T,N2},dtfvL::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
    ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;errnIKc::T=epsT100) where{T,N2}
  
    nn = length(nIK[:,1])
    for isp in 1:ns
        a = zeros(T,nn)
        dtnIKs!(a,dtfvL[isp],vhe[isp],ma[isp],na[isp],vth[isp];errnIKc=errnIKc)
        nIK[:,isp] = a
    end
end

# 1D, [dtnIK], vhe
function dtnIKs!(nIK::AbstractVector{T},dtfvL::AbstractArray{T,N},
    vhe::StepRangeLen,ma::T,na::T,vth::T;errnIKc::T=epsT100) where{T,N}
    
    # n, I, K, T = nIK[1:4]
    v2 = vhe .^2
    nIK[1], errdtn = romberg(vhe, (v2 .* dtfvL[:,1]))
    nIK[2], errdtI = romberg(vhe, (v2 .* vhe .* dtfvL[:,2]))
    nIK[3], errdtK = romberg(vhe, (v2 .^2 .* dtfvL[:,1]))

    cc = 4 / sqrtpi * na
    nIK[1] *= cc
    cc *= (ma * vth)
    nIK[2] *= cc / 3
    cc *= vth
    nIK[3] *= cc / 2
    
    errdtn < errnIKc || @warn("Convergence of integration `δₜna` if failure", errdtn)
    errdtI < errnIKc || @warn("Convergence of integration `δₜIa` if failure", errdtI)
    errdtK < errnIKc || @warn("Convergence of integration `δₜKa` if failure", errdtK)
end

# 2D, [dtnIK,ns], ve
function dtnIKs!(nIK::AbstractArray{T,N2},ve::AbstractVector{StepRangeLen},dtfvL::AbstractVector{Matrix{T}},
    ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;errnIKc::T=epsT100) where{T,N2}
  
    nn = length(nIK[:,1])
    for isp in 1:ns
        a = zeros(T,nn)
        dtnIKs!(a,ve[isp],dtfvL[isp],ma[isp],na[isp],vth[isp];errnIKc=errnIKc)
        nIK[:,isp] = a
    end
end

# 1D, [dtnIK], ve
function dtnIKs!(nIK::AbstractVector{T},ve::StepRangeLen,dtfvL::AbstractArray{T,N},
    ma::T,na::T,vth::T;errnIKc::T=epsT100) where{T,N}
    
    # n, I, K, T = nIK[1:4]
    v2 = ve .^2
    nIK[1], errdtn = romberg(ve, (v2 .* dtfvL[:,1]))
    nIK[2], errdtI = romberg(ve, (v2 .* ve  .* dtfvL[:,2]))
    nIK[3], errdtK = romberg(ve, (v2 .^2  .* dtfvL[:,1]))

    cc = 4 / sqrtpi * (na / vth^3)
    nIK[1] *= cc
    cc *= ma
    nIK[2] *= cc / 3
    nIK[3] *= cc / 2
    
    errdtn < errnIKc || @warn("Convergence of integration `δₜna` if failure", errdtn)
    errdtI < errnIKc || @warn("Convergence of integration `δₜIa` if failure", errdtI)
    errdtK < errnIKc || @warn("Convergence of integration `δₜKa` if failure", errdtK)
end
