
"""
  Normalized kinetic dissipative forces when `is_normδtf = false` such as 

    Rdtn: = R̂000 = 0
    RdtI: = R̂100 / 3
    RdtK: = R̂200
    Rdtvth := 𝒲 / 3 = (R̂200 - 2 * Ih .* R̂100) / 3

  The relations between `RdtnIKTcs!` and `dtnIKsc!` are
    
  dtnIKsc[1,:] = RdtM[1,:] .* (vth.^3 * pi^1.5)
  dtnIKsc[2,:] = RdtM[2,:] .* (ma .* vth.^4 * pi^1.5)
  dtnIKsc[3,:] = RdtM[3,:] .* (ma .* vth.^5 * pi^1.5 / 2)

   and constraint according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`

  Inputs:
    Ih: = ûa
    dtfvLc: = dtfvL * (na / vth^3 / π^(3/2))

  Outputs:
    RdtnIKTcs!(RdtnIKT,dtfvLe,vhe,Ih,ns;atol_nIK=atol_nIK)
    RdtnIKTcs!(RdtnIKT,dtfvLe,vhe,Ih;atol_nIK=atol_nIK)

    ## Testting
    RdtnIKT = zeros(4)
    RdtnIKTcs!(RdtnIKT,δtfvL0[:,1:2,isp3],vhe,uh[isp3],1)
    RdtnIKTs = zeros(4,ns)
    RdtnIKTcs!(RdtnIKTs,δtfvL0[:,1:2,:],vhe,uh,ns)

"""

# 2D, [dtnIKTc,ns], vhe
function RdtnIKTcs!(RdtM::AbstractArray{T,N2},dtfvLc::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
  Ih::AbstractVector{T},ma::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
  atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N2}
  
  nn = length(RdtM[:,1])
  if is_out_errdt
      errdtn, errdtI, errdtK, errRdtT = zeros(T,ns),zeros(T,ns),zeros(T,ns),zeros(T,ns)
      for isp in 1:ns
          a = zeros(T,nn)
          errdtn[isp], errdtI[isp], errdtK[isp], errRdtT[isp] = RdtnIKTcs!(a,dtfvLc[isp],
                              vhe[isp],Ih[isp],ma[isp],vth[isp];
                              atol_nIK=atol_nIK,is_out_errdt=is_out_errdt)
          RdtM[:,isp] = a
      end
      norm(errdtn) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜn̂a = 0`",errdtn)
      norm(errdtI) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜIa = 0`",errdtI)
      norm(errdtK) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜKa = 0`",errdtK)
      norm(errRdtT) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜTa = 0`",errRdtT)
      
  else
    for isp in 1:ns
        a = zeros(T,nn)
        RdtnIKTcs!(a,dtfvLc[isp],vhe[isp],Ih[isp],ma[isp],vth[isp];atol_nIK=atol_nIK)
        RdtM[:,isp] = a
    end
  end
end

# 1D, [dtnIKTc], vhe
function RdtnIKTcs!(RdtM::AbstractVector{T},dtfvLc::AbstractArray{T,N},
    vhe::StepRangeLen,Ih::T,ma::T,vth::T;
    atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N}
  
    # Rdtn, RdtI, RdtK, Rdtvth = RdtM[1:4]
    v2 = vhe .^2
    RdtM[1], errdtn = romberg(vhe, (v2 .* dtfvLc[:,1]))
    RdtM[2], errdtI = romberg(vhe, (v2 .* vhe  .* dtfvLc[:,2]))
    RdtM[3], errdtK = romberg(vhe, (v2 .^2  .* dtfvLc[:,1]))
    
    cc = 4 / sqrtpi
    RdtM[1] *= cc
    RdtM[2] *= cc / 3
    RdtM[3] *= cc
    
    # w = RdtM[3] - 2 * Ih .* RdtI
    RdtM[4] = (RdtM[3] - 2 * Ih .* RdtM[2]) / 3    # `Rdtvth = w / 3`
    dtIh = RdtM[2] - Ih .* RdtM[4]                 # 

    # Checking the constraint according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
    errRdtKI = RdtM[3] - (2 * Ih .* dtIh + (3 .+ 2 * Ih.^2) .* RdtM[4])
    
    if is_out_errdt
      # giving the values of `dtn`, `dtI` and `dtK`
      RdtM[1,:] .*= (vth.^5 * sqrtpi3)              # dtn
      RdtM[2,:] .*= (ma .* vth.^4 * sqrtpi3)        # dtI
      RdtM[3,:] .*= (ma .* vth.^5 * sqrtpi3 / 2)    # dtK
      return errdtn, errdtI, errdtK, errRdtKI
    else
      # Checking the constraint according to: `δₜn̂a = 0`
      errdtn ≤ atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜn̂a = 0`",errdtn)
      errdtI < atol_nIK || @warn("Convergence of integration `RdtI` if failure", errdtI)
      errdtK < atol_nIK || @warn("Convergence of integration `RdtK` if failure", errdtK)
      
      errRdtKI ≤ atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`",errRdtKI)
    
      # giving the values of `dtn`, `dtI` and `dtK`
      RdtM[1,:] .*= (vth.^5 * sqrtpi3)              # dtn
      RdtM[2,:] .*= (ma .* vth.^4 * sqrtpi3)        # dtI
      RdtM[3,:] .*= (ma .* vth.^5 * sqrtpi3 / 2)    # dtK
    end
end

# 2D, [RdtnIKTc,ns]
function RdtnIKTcs!(RdtM::AbstractArray{T,N2},vhe::AbstractVector{StepRangeLen},dtfvLc::AbstractVector{Matrix{T}},
    Ih::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N2}
    
    nn = length(RdtM[:,1])
    if is_out_errdt
        errdtn, errdtI, errdtK, errRdtT = zeros(T,ns),zeros(T,ns),zeros(T,ns),zeros(T,ns)
        for isp in 1:ns
            a = zeros(T,nn)
            errdtn[isp], errdtI[isp], errdtK[isp], errRdtT[isp] = RdtnIKTcs!(a,vhe[isp],dtfvLc[isp],
                                  Ih[isp],na[isp],vth[isp];atol_nIK=atol_nIK,is_out_errdt=is_out_errdt)
            RdtM[:,isp] = a
        end
        norm(errdtn) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜn̂a = 0`",errdtn)
        norm(errdtI) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜIa = 0`",errdtI)
        norm(errdtK) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜKa = 0`",errdtK)
        norm(errRdtT) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜTa = 0`",errRdtT)
    else
      for isp in 1:ns
          a = zeros(T,nn)
          RdtnIKTcs!(a,vhe[isp],dtfvLc[isp],Ih[isp],na[isp],vth[isp];atol_nIK=atol_nIK)
          RdtM[:,isp] = a
      end
    end
end

# 1D, [RdtnIKTc]
function RdtnIKTcs!(RdtM::AbstractVector{T},vhe::StepRangeLen,dtfvLc::AbstractArray{T,N},
    Ih::T,na::T,vth::T;atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N}
    
    # Rdtn, RdtI, RdtK, Rdtvth = RdtM[1:4]
    v2 = vhe .^2
    RdtM[1], errdtn = romberg(vhe, (v2 .* dtfvLc[:,1]))
    RdtM[2], errdtI = romberg(vhe, (v2 .* vhe  .* dtfvLc[:,2]))
    RdtM[3], errdtK = romberg(vhe, (v2 .^2  .* dtfvLc[:,1]))

    cc = 4pi * (vth^3 / na)
    RdtM[1] *= cc
    RdtM[2] *= cc / 3
    RdtM[3] *= cc
    
    # w = RdtM[3] - 2 * Ih .* RdtI
    RdtM[4] = (RdtM[3] - 2 * Ih .* RdtM[2]) / 3    # `Rdtvth = w / 3`
    dtIh = RdtM[2] - Ih .* RdtM[4]                 # 
    
    # Checking the constraint according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
    errRdtKI = RdtM[3] - (2 * Ih .* dtIh + (3 .+ 2 * Ih.^2) .* RdtM[4])
    
    if is_out_errdt
        return errdtn, errdtI, errdtK, errRdtKI
    else
      # Checking the constraint according to: `δₜn̂a = 0`
      errdtn ≤ atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜn̂a = 0`",errdtn)
      errdtI < atol_nIK || @warn("Convergence of integration `RdtI` if failure", errdtI)
      errdtK < atol_nIK || @warn("Convergence of integration `RdtK` if failure", errdtK)
      errRdtKI ≤ atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`",errRdtKI)
    end
end