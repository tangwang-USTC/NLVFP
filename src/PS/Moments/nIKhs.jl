
"""
  Normarmolized moments such as nh, Ih, Kh

    nh: = M̂000 = 1
    Ih: = M̂100 / 3
    Kh: = M̂200 = 3/2 * Th + Ih^2
    
  `[nh, Ih, Kh] = nIKh[1:3]` for `fvL` 
    
  or 
    
  `[Rδₜn, RδₜI, RδₜK] = [R̂000, R̂100, R̂200] = nIKh[1:3]` for `δₜfvL`

  Inputs:
    fvL: = fvLc / (na / vth^3 / π^(3/2))

  Outputs:
    nIKhs!(nIKh,fvLe,vhe)
    nIKhs!(nIKh,fvLe,vhe,ns)

    ## Testting
    nIKhs = zeros(3,ns)
    nIKhs!(nIKhs,fvL0e[:,1:2,:],vhe,ns)

"""

# [nIKh,ns]
function nIKhs!(nIKh::AbstractArray{T,N2},fvL::AbstractVector{Matrix{T}},
    vhe::AbstractVector{StepRangeLen},ns::Int64) where{T,N2}
    
    nn = length(nIKh[:,1])
    for isp in 1:ns
        a = zeros(T,nn)
        nIKhs!(a,fvL[isp],vhe[isp])
        nIKh[:,isp] = a
    end
end

# [nIKh]
function nIKhs!(nIKh::AbstractVector{T},fvL::AbstractArray{T,N},vhe::StepRangeLen;atol_IKTh::T=epsT100) where{T,N}

    v2 = vhe .^2
    nIKh[1], errnh = romberg(vhe, (v2 .* fvL[:,1]))
    nIKh[2], errIh = romberg(vhe, (v2 .* vhe  .* fvL[:,2]))
    nIKh[3], errKh = romberg(vhe, (v2 .^2  .* fvL[:,1]))
    
    cc = 4 / sqrtpi
    nIKh[1] *= cc
    nIKh[2] *= cc / 3
    nIKh[3] *= cc

    errnh < atol_IKTh || @warn("Convergence of integration `n̂a` if failure", errnh)
    errIh < atol_IKTh || @warn("Convergence of integration `Îa` if failure", errIh)
    errKh < atol_IKTh || @warn("Convergence of integration `K̂a` if failure", errKh)
end

"""
  where `Th ≡ 1` in theory but `Th = Rvth^2` in discrete.

  Inputs:

  Outputs:
    nIKThs!(nIKTh,fvLe,vhe;atol_IKTh=atol_IKTh,rtol_IKTh=rtol_IKTh)
    nIKThs!(nIKTh,fvLe,vhe,ns;atol_IKTh=atol_IKTh,rtol_IKTh=rtol_IKTh)

    nIKThs!(nIKTh,fvLe,vhe,Rvth,isp;atol_IKTh=atol_IKTh,rtol_IKTh=rtol_IKTh)
    nIKThs!(nIKTh,fvLe,vhe,Rvth,ns;atol_IKTh=atol_IKTh,rtol_IKTh=rtol_IKTh)

    ## Testting
    nIKThs!(nIKTh,fvL0e[:,1:2,isp3],vhe,1.0,1)
    nIKThs!(nIKThs,fvL0e[:,1:2,:],vhe,ones(ns),ns)
    nIKThs!(nIKTh,fvL0e[:,1:2,isp3],vhe)
    nIKThs!(nIKThs,fvL0e[:,1:2,:],vhe,ns)

"""

# 2D, [nIKTh,ns]
function nIKThs!(nIKTh::AbstractArray{T,N2},fvL::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
    ns::Int64;atol_IKTh::T=epsT100,rtol_IKTh::T=1e-1) where{T,N2}
    
    nn = length(nIKTh[:,1])
    for isp in 1:ns
        a = zeros(T,nn)
        nIKThs!(a,fvL[isp],vhe[isp];atol_IKTh=atol_IKTh,rtol_IKTh=rtol_IKTh)
        nIKTh[:,isp] = a
    end
end

function nIKThs!(nIKTh::AbstractArray{T,N2},fvL::AbstractVector{Matrix{T}},vhe::Vector{AbstractVector{T}},
    nvG::Vector{Int64},ns::Int64;atol_IKTh::T=epsT100,rtol_IKTh::T=1e-1) where{T,N2}
    
    nn = length(nIKTh[:,1])
    for isp in 1:ns
        a = zeros(T,nn)
        nIKThs!(a,fvL[isp],vhe[isp],nvG[isp];atol_IKTh=atol_IKTh,rtol_IKTh=rtol_IKTh)
        nIKTh[:,isp] = a
    end
end

function nIKThs!(nIKTh::AbstractArray{T,N2},fvL::AbstractVector{Matrix{T}},vhe::AbstractVector{T},
    nvG::Int64, ns::Int64;atol_IKTh::T=epsT100,rtol_IKTh::T=1e-1) where{T,N2}
    
    nn = length(nIKTh[:,1])
    for isp in 1:ns
        a = zeros(T,nn)
        nIKThs!(a,fvL[isp],vhe, nvG;atol_IKTh=atol_IKTh,rtol_IKTh=rtol_IKTh)
        nIKTh[:,isp] = a
    end
end

function nIKThs!(nIKTh::AbstractArray{T,N2},fvL::AbstractVector{Matrix{T}},vhe::StepRangeLen,
    ns::Int64;atol_IKTh::T=epsT100,rtol_IKTh::T=1e-1) where{T,N2}
    
    nn = length(nIKTh[:,1])
    for isp in 1:ns
        a = zeros(T,nn)
        nIKThs!(a,fvL[isp],vhe;atol_IKTh=atol_IKTh,rtol_IKTh=rtol_IKTh)
        nIKTh[:,isp] = a
    end
end

# 1D, [nIKTh] 
function nIKThs!(nIKTh::AbstractVector{T},fvL::AbstractArray{T,N},vhe::AbstractVector{T}, 
    nvG::Int64;atol_IKTh::T=epsT100,rtol_IKTh::T=1e-1) where{T,N}
    
    # nh, Ih, Kh, errTh = nIKTh[1:3]
    v2 = vhe .^2
    Ixv = - 2 / sqrtpi * (vhe[1] - vhe[end])
    wcck = clenshawcurtisweights(chebyshevmoments1(T, nvG))
    nIKTh[1] = Ixv * dot(wcck, (v2 .* fvL[:,1]))
    nIKTh[2] = Ixv * dot(wcck, (v2 .* vhe  .* fvL[:,2])) / 3
    nIKTh[3] = Ixv * dot(wcck, (v2 .^2  .* fvL[:,1]))
    
    # Checking the constraint: `n̂a = 1` 
    Dnh = nIKTh[1] .- 1
    if Dnh > atol_IKTh
        @warn("Number of meshgrids may be not enough to satisfy the convergence of `n̂a = 1`", Dnh)
        if Dnh > rtol_IKTh 
            printstyled("`norm(Dnh) ≥ rtol_IKTh` which means the algorithm is not convergent!!!",color=:red,"\n")
        end
    end

    # Checking the constraint: `K̂a = 3/2 * T̂a + ûa²` 
    nIKTh[4] = abs(nIKTh[3] - nIKTh[2] .^ 2 - 1.5)
end

function nIKThs!(nIKTh::AbstractVector{T},fvL::AbstractArray{T,N},vhe::StepRangeLen;atol_IKTh::T=epsT100,rtol_IKTh::T=1e-1) where{T,N}
    
    # nh, Ih, Kh, errTh = nIKTh[1:3]
    v2 = vhe .^2
    nIKTh[1], errnh = romberg(vhe, (v2 .* fvL[:,1]))
    nIKTh[2], errIh = romberg(vhe, (v2 .* vhe  .* fvL[:,2]))
    nIKTh[3], errKh = romberg(vhe, (v2 .^2  .* fvL[:,1]))

    cc = 4 / sqrtpi
    nIKTh[1] *= cc
    nIKTh[2] *= cc / 3
    nIKTh[3] *= cc

    errnh < atol_IKTh || @warn("Convergence of integration `n̂a` if failure", errnh)
    errIh < atol_IKTh || @warn("Convergence of integration `Îa` if failure", errIh)
    errKh < atol_IKTh || @warn("Convergence of integration `K̂a` if failure", errKh)
    
    # Checking the constraint: `n̂a = 1` 
    Dnh = nIKTh[1] .- 1
    if Dnh > atol_IKTh
        @warn("Number of meshgrids may be not enough to satisfy the convergence of `n̂a = 1`", Dnh)
        if Dnh > rtol_IKTh 
            printstyled("`norm(Dnh) ≥ rtol_IKTh` which means the algorithm is not convergent!!!",color=:red,"\n")
        end
    end

    # Checking the constraint: `K̂a = 3/2 * T̂a + ûa²` 
    nIKTh[4] = abs(nIKTh[3] - nIKTh[2] .^ 2 - 1.5)
end

# # 2D, [nIKTh,ns], Rvth
# function nIKThs!(nIKTh::AbstractArray{T,N2},fvL::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
#     Rvth::AbstractVector{T},ns::Int64;atol_IKTh::T=epsT100,rtol_IKTh::T=1e-1) where{T,N2}
    
#     nn = length(nIKTh[:,1])
#     for isp in 1:ns
#         a = zeros(T,nn)
#         nIKThs!(a,fvL[isp],vhe[isp],Rvth[isp],isp;atol_IKTh=atol_IKTh,rtol_IKTh=rtol_IKTh)
#         nIKTh[:,isp] = a
#     end
# end

# # 1D, [nIKTh]
# function nIKThs!(nIKTh::AbstractVector{T},fvL::AbstractArray{T,N},vhe::StepRangeLen,
#     Rvth::T,isp::Int64;atol_IKTh::T=epsT100,rtol_IKTh::T=1e-1) where{T,N}
    
#     # nh, Ih, Kh, errTh = nIKTh[1:4]
#     erfghnm
#     v2 = vhe .^2
#     nIKTh[1], errnh = romberg(vhe, (v2 .* fvL[:,1]))
#     nIKTh[2], errIh = romberg(vhe, (v2 .* vhe  .* fvL[:,2]))
#     nIKTh[3], errKh = romberg(vhe, (v2 .^2  .* fvL[:,1]))
#     cc = 4 / sqrtpi
#     nIKTh[1] *= cc
#     nIKTh[2] *= cc / 3
#     nIKTh[3] *= cc
#     if Rvth == 1.0
#         nIKTh[4] = abs(nIKTh[3] - nIKTh[2] .^ 2 - 1.5)
#     else
#         nIKTh[4] = abs(nIKTh[3] - nIKTh[2] .^ 2 - 1.5 * Rvth.^2)
#     end
    
#     errnh < atol_IKTh || @warn("Convergence of integration `n̂a` if failure", errnh)
#     errIh < atol_IKTh || @warn("Convergence of integration `Îa` if failure", errIh)
#     errKh < atol_IKTh || @warn("Convergence of integration `K̂a` if failure", errKh)
    
#     # Checking the constraint: `n̂a = 1` 
#     Dnh = nIKTh[1] .- 1
#     if Dnh > atol_IKTh
#         @warn("Number of meshgrids may be not enough to satisfy the convergence of `n̂a = 1`", Dnh)
#         if Dnh > rtol_IKTh 
#             printstyled("`norm(Dnh) ≥ rtol_IKTh` which means the algorithm is not convergent!!!",color=:red,"\n")
#         end
#     end

#     # Checking the constraint: `K̂a = 3/2 * T̂a + ûa²` 
#     if nIKTh[4] > atol_IKTh
#         @warn("Number of meshgrids may be not enough to satisfy the convergence of `K̂a = 3/2 * T̂a + ûa²`", (isp,nIKTh[4]))
#         if nIKTh[4] > rtol_IKTh 
#             printstyled("`errTh < 1e-1` which means the convergence of the algorithm is falure!",color=:red,"\n")
#         end
#     end
# end
