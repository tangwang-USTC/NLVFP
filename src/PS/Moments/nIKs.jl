
"""
  Moments such as n, u, K, T and vth

    na: = vₜₕ³ * ∫₀⁹(4πv̂²f₀(v))dv̂
    Iₐ: = ma / 3 vₜₕ⁴ ∫₀⁹(4πv̂³f₁(v))dv̂
    Ka: = 1/2 mₐ vₜₕ⁵ ∫₀⁹(4πv̂⁴f₀(v))dv̂
    Ta: = 2/3 (Ka/na - 0.5 mₐ uₐ²)

  Inputs:
    ma:
    nvgauss:
    v: GaussQuadrature collections
    fvLe: = f̂ₗₘ(v̂), harmonics of distribution functions on grids `vhe` without coefficients `cf` as:

         cf = 1/π^(3/2) * na /vth^3.

    which the effect of `cf` is included in the integral process.

  Outputs:

"""

# 2D, [nIK,ns], vhe
function nIKs!(nIK::AbstractArray{T,N2},fvL::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
    ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;errnIKc::T=epsT100) where{T,N2}

    nn = length(nIK[:,1])
    for isp in 1:ns
        a = zeros(T,nn)
        nIKs!(a,fvL[isp],vhe[isp],ma[isp],na[isp],vth[isp];errnIKc=errnIKc)
        nIK[:,isp] = a
    end
end

# 1D, [nIK], vhe
function nIKs!(nIK::AbstractVector{T},fvL::AbstractArray{T,N},
    vhe::StepRangeLen,ma::T,na::T,vth::T;errnIKc::T=epsT100) where{T,N}
  
    # n, I, K, T = nIK[1:4]
    v2 = vhe .^2
    nIK[1], errn = romberg(vhe, (v2 .* fvL[:,1]))
    nIK[2], errI = romberg(vhe, (v2 .* vhe .* fvL[:,2]))
    nIK[3], errK = romberg(vhe, (v2 .^2 .* fvL[:,1]))

    cc = 4 / sqrtpi * na
    nIK[1] *= cc
    cc *= (ma * vth)
    nIK[2] *= cc / 3
    cc *= vth
    nIK[3] *= cc / 2
    
    errn < errnIKc || @warn("Convergence of integration `na` if failure", errn)
    errI < errnIKc || @warn("Convergence of integration `Ia` if failure", errI)
    errK < errnIKc || @warn("Convergence of integration `Ka` if failure", errK)
end

# 2D, [nIK,ns], ve
function nIKs!(nIK::AbstractArray{T,N2},ve::AbstractVector{StepRangeLen},fvL::AbstractVector{Matrix{T}},
    ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;errnIKc::T=epsT100) where{T,N2}

    nn = length(nIK[:,1])
    for isp in 1:ns
        a = zeros(T,nn)
        nIKs!(a,ve[isp],fvL[isp],ma[isp],na[isp],vth[isp];errnIKc=errnIKc)
        nIK[:,isp] = a
    end
end

# 1D, [nIK], ve
function nIKs!(nIK::AbstractVector{T},ve::StepRangeLen,fvL::AbstractArray{T,N},
    ma::T,na::T,vth::T;errnIKc::T=epsT100) where{T,N}

    # n, I, K, T = nIK[1:4]
    v2 = ve .^2
    nIK[1], errn = romberg(ve, (v2 .* fvL[:,1]))
    nIK[2], errI = romberg(ve, (v2 .* ve  .* fvL[:,2]))
    nIK[3], errK = romberg(ve, (v2 .^2  .* fvL[:,1]))

    cc = 4 / sqrtpi * (na / vth^3)
    nIK[1] *= cc
    cc *= ma
    nIK[2] *= cc / 3
    nIK[3] *= cc / 2
    
    errn < errnIKc || @warn("Convergence of integration `na` if failure", errn)
    errI < errnIKc || @warn("Convergence of integration `Ia` if failure", errI)
    errK < errnIKc || @warn("Convergence of integration `Ka` if failure", errK)
end


"""
  Outputs:
    n̂a, Ia, Ka = nIKs(fvLe,vhe,ma,na,vth,ns,Rvth)
    n̂a, Ia, Ka = nIKs(fvLe,vhe,ma,na,vth,ns)

"""

function nIKs(fvLe::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},ma::AbstractVector{T},
    na::AbstractVector{T},vth::AbstractVector{T},ns::Int64) where{T}

    n̂a = zeros(ns)
    Ka = zeros(ns)
    Ia = zeros(ns)
    Ixvi = 4.0 / sqrtpi
    for isp = 1:ns
        v2 = vhe[isp] .^2
        ρa = ma[isp] .* na[isp]
        f0 = v2 .* fvLe[isp][:,1]
        f1 = v2 .* fvLe[isp][:,2]
        n̂a[isp], errn = romberg(vhe[isp], f0)
        Ia[isp], errI = romberg(vhe[isp], (vhe[isp] .* f1))
        Ka[isp], errK = romberg(vhe[isp], (v2 .* f0))
        ρa = ma[isp] * na[isp]
        n̂a[isp] *= Ixvi
        Ia[isp] *= Ixvi * (ρa * vth[isp]) / 3
        Ka[isp] *= (2ρa / sqrtpi * vth[isp]^2)
    end
    return n̂a, Ia, Ka
end

function nIKs!(nh::AbstractArray{T,N},Ia::AbstractArray{T,N},Ka::AbstractArray{T,N},
    fvLe::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},ma::AbstractVector{T},
    na::AbstractVector{T},vth::AbstractVector{T},ns::Int64,Rvth::AbstractVector{T}) where{T,N}

    Ixvi = 4.0 / sqrtpi
    for isp = 1:ns
        v2 = vhe[isp] .^2
        ρa = ma[isp] .* na[isp]
        f0 = v2 .* fvLe[isp][:,1]
        f1 = v2 .* fvLe[isp][:,2]
        nh[1,isp], nh[2,isp] = romberg(vhe[isp], f0)
        Ia[1,isp], Ia[2,isp] = romberg(vhe[isp], (vhe[isp] .* f1))
        Ka[1,isp], Ka[2,isp] = romberg(vhe[isp], (v2 .* f0))
        ρa = ma[isp] * na[isp]
        nh[1,isp] *= Ixvi
        Ia[1,isp] *= Ixvi * (ρa * vth[isp]) / 3
        Ka[1,isp] *= (2ρa / sqrtpi * vth[isp]^2)
    end
    return nh, Ia, Ka
end

"""
  Outputs:
    vthnIK!(vth,ma,na,Ia,Ka)
    vth = vthnIK(ρa,Ia,Ka)
"""

function vthnIK!(vth::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},
    Ia::AbstractVector{T},Ka::AbstractVector{T},ns::Int64) where{T}

    for isp in 1:ns
        vth[isp] = vthnIK(ma[isp]*na[isp],Ia[isp],Ka[isp])
    end
end

function vthnIK(ρa::T,Ia::T,Ka::T) where{T}

    if Ia ≤ epsT100
        return (4/3 * (Ka / ρa))^0.5
    else
        return (2/3 * (2Ka / ρa) - (Ia / ρa)^2)^0.5
    end
end
