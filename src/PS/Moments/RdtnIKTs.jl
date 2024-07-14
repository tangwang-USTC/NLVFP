
"""
  Normalized kinetic dissipative forces when `is_normδtf = false` such as 

    Rdtn: = R̂000 = 0
    RdtI: = R̂100 / 3
    RdtK: = R̂200
    Rdtvth := 𝒲 / 3 = (R̂200 - 2 * Ih .* R̂100) / 3

  The relations between `RdtnIKTs!` and `dtnIKsc!` are
    
  dtnIKsc[1,:] = RdtM[1,:] .* (vth.^3 * pi^1.5)
  dtnIKsc[2,:] = RdtM[2,:] .* (ma .* vth.^4 * pi^1.5)
  dtnIKsc[3,:] = RdtM[3,:] .* (ma .* vth.^5 * pi^1.5 / 2)

   and constraint according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`

  Inputs:
    Ih: = ûa
    dtfvLc: = dtfvL * (na / vth^3 / π^(3/2))

  Outputs:
    RdtnIKTs!(RdtnIKT,dtfvLe,vhe,nvG,Ih,ns;atol_nIK=atol_nIK)
    RdtnIKTs!(RdtnIKT,dtfvLe,vhe,Ih,isp;atol_nIK=atol_nIK)

"""

# 2D, [ns], 
function RdtnIKTs!(RdtM::AbstractArray{T,N2},dtfvL::AbstractVector{Matrix{T}},vhe::AbstractVector{T},nvG::Int64,
    Ih::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N2}
  
    nn = length(RdtM[:,1])
    if is_out_errdt
        errRdtT = zeros(T,ns)
        for isp in 1:ns
            a = zeros(T,nn)
            errRdtT[isp] = RdtnIKTs!(a,dtfvL[isp],vhe,nvG,Ih[isp],ma[isp],na[isp],vth[isp];
                                atol_nIK=atol_nIK,is_out_errdt=is_out_errdt)
            RdtM[:,isp] = a
        end
        # abs(errRdtT) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜTa = 0`",errRdtT)
    else
        dfghggg
        for isp in 1:ns
            a = zeros(T,nn)
            RdtnIKTs!(a,dtfvL[isp],vhe,nvG,Ih[isp],ma[isp],na[isp],vth[isp];atol_nIK=atol_nIK)
            RdtM[:,isp] = a
        end
    end
end

function RdtnIKTs!(RdtM::AbstractArray{T,N2},dtfvL::AbstractVector{Matrix{T}},vhe::AbstractVector{T},
    Ih::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N2}
  
    nn = length(RdtM[:,1])
    if is_out_errdt
        errdtn, errdtI, errdtK, errRdtT = zeros(T,ns),zeros(T,ns),zeros(T,ns),zeros(T,ns)
        for isp in 1:ns
            a = zeros(T,nn)
            errdtn[isp], errdtI[isp], errdtK[isp], errRdtT[isp] = RdtnIKTs!(a,dtfvL[isp],
                                vhe,Ih[isp],ma[isp],na[isp],vth[isp];
                                atol_nIK=atol_nIK,is_out_errdt=is_out_errdt)
            RdtM[:,isp] = a
        end
        # norm(errdtn) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜn̂a = 0`",errdtn)
        # norm(errdtI) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜIa = 0`",errdtI)
        # norm(errdtK) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜKa = 0`",errdtK)
        # norm(errRdtT) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜTa = 0`",errRdtT)
    else
        for isp in 1:ns
            a = zeros(T,nn)
            RdtnIKTs!(a,dtfvL[isp],vhe,Ih[isp],ma[isp],na[isp],vth[isp];atol_nIK=atol_nIK)
            RdtM[:,isp] = a
        end
    end
end

function RdtnIKTs!(RdtM::AbstractArray{T,N2},dtfvL::AbstractVector{Matrix{T}},
    vhe::AbstractVector{T},nvG::Int64,Ih::AbstractVector{T},
    ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N2}
  
    nn = length(RdtM[:,1])
    if is_out_errdt
        errdtn, errdtI, errdtK, errRdtT = zeros(T,ns),zeros(T,ns),zeros(T,ns),zeros(T,ns)
        for isp in 1:ns
            a = zeros(T,nn)
            errdtn[isp], errdtI[isp], errdtK[isp], errRdtT[isp] = RdtnIKTs!(a,dtfvL[isp],
                                vhe,Ih[isp],ma[isp],na[isp],vth[isp];
                                atol_nIK=atol_nIK,is_out_errdt=is_out_errdt)
            RdtM[:,isp] = a
        end
    else
        for isp in 1:ns
            a = zeros(T,nn)
            RdtnIKTs!(a,dtfvL[isp],vhe,Ih[isp],ma[isp],na[isp],vth[isp];atol_nIK=atol_nIK)
            RdtM[:,isp] = a
        end
    end
end

function RdtnIKTs!(RdtM::AbstractArray{T,N2},dtfvL::AbstractVector{Matrix{T}},vhe::StepRangeLen,
    Ih::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N2}
  
    nn = length(RdtM[:,1])
    if is_out_errdt
        errdtn, errdtI, errdtK, errRdtT = zeros(T,ns),zeros(T,ns),zeros(T,ns),zeros(T,ns)
        for isp in 1:ns
            a = zeros(T,nn)
            errdtn[isp], errdtI[isp], errdtK[isp], errRdtT[isp] = RdtnIKTs!(a,dtfvL[isp],
                                vhe,Ih[isp],ma[isp],na[isp],vth[isp];
                                atol_nIK=atol_nIK,is_out_errdt=is_out_errdt)
            RdtM[:,isp] = a
        end
    else
        for isp in 1:ns
            a = zeros(T,nn)
            RdtnIKTs!(a,dtfvL[isp],vhe,Ih[isp],ma[isp],na[isp],vth[isp];atol_nIK=atol_nIK)
            RdtM[:,isp] = a
        end
    end
end

function RdtnIKTs!(RdtM::AbstractArray{T,N2},eRdtM::AbstractArray{T,N2},
    dtfvL::AbstractVector{Matrix{T}},vhe::StepRangeLen,Ih::AbstractVector{T},
    ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N2}
  
    nn = length(RdtM[:,1])
    if is_out_errdt
        for isp in 1:ns
            a = zeros(T,nn)
            eRdtM[1,isp], eRdtM[2,isp], eRdtM[3,isp], eRdtM[4,isp] = RdtnIKTs!(a,dtfvL[isp],
                                vhe,Ih[isp],ma[isp],na[isp],vth[isp];
                                atol_nIK=atol_nIK,is_out_errdt=is_out_errdt)
            RdtM[:,isp] = a
        end
    else
        for isp in 1:ns
            a = zeros(T,nn)
            RdtnIKTs!(a,dtfvL[isp],vhe,Ih[isp],ma[isp],na[isp],vth[isp];atol_nIK=atol_nIK)
            RdtM[:,isp] = a
        end
    end
end

function RdtnIKTs!(RdtM::AbstractArray{T,N2},eRdtM::AbstractArray{T,N2},
    dtfvL::AbstractVector{Matrix{T}},vhe::AbstractVector{T},nvG::Int64,
    Ih::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N2}
  
    nn = length(RdtM[:,1])
    if is_out_errdt
        for isp in 1:ns
            a = zeros(T,nn)
            ~, eRdtM[:,isp] = RdtnIKTs!(a,eRdtM[:,isp],dtfvL[isp],
                                vhe,nvG,Ih[isp],ma[isp],na[isp],vth[isp];
                                atol_nIK=atol_nIK,is_out_errdt=is_out_errdt)
            RdtM[:,isp] = a
        end
    else
        for isp in 1:ns
            a = zeros(T,nn)
            RdtnIKTs!(a,dtfvL[isp],vhe,Ih[isp],ma[isp],na[isp],vth[isp];atol_nIK=atol_nIK)
            RdtM[:,isp] = a
        end
    end
end

# 2D, [ns], vhe[2]
function RdtnIKTs!(RdtM::AbstractArray{T,N2},dtfvL::AbstractVector{Matrix{T}},vhe::AbstractVector{StepRangeLen},
    Ih::AbstractVector{T},ma::AbstractVector{T},na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N2}
  
    nn = length(RdtM[:,1])
    if is_out_errdt
        errdtn, errdtI, errdtK, errRdtT = zeros(T,ns),zeros(T,ns),zeros(T,ns),zeros(T,ns)
        for isp in 1:ns
            a = zeros(T,nn)
            errdtn[isp], errdtI[isp], errdtK[isp], errRdtT[isp] = RdtnIKTs!(a,dtfvL[isp],
                                vhe[isp],Ih[isp],ma[isp],na[isp],vth[isp];
                                atol_nIK=atol_nIK,is_out_errdt=is_out_errdt)
            RdtM[:,isp] = a
        end
        norm(errdtn) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜn̂a = 0`",errdtn)
        norm(errdtI) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜIa = 0`",errdtI)
        norm(errdtK) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜKa = 0`",errdtK)
        norm(errRdtT) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜTa = 0`",errRdtT)
        uytrewedrfgthj
    else
        for isp in 1:ns
            a = zeros(T,nn)
            RdtnIKTs!(a,dtfvL[isp],vhe[isp],Ih[isp],ma[isp],na[isp],vth[isp];atol_nIK=atol_nIK)
            RdtM[:,isp] = a
        end
    end
end

# 1D, [dtnIKT]
function RdtnIKTs!(RdtM::AbstractVector{T},eRdtM::AbstractVector{T},dtfvL::AbstractArray{T,N},
    vhe::AbstractVector{T},nvG::Int64,Ih::T,ma::T,na::T,vth::T;
    atol_nIK::T=epsT1,is_out_errdt::Bool=false) where{T,N}
  
    # Rdtn, RdtI, RdtK, Rdtvth = RdtM[1:4]
    v2 = vhe .^2
    Ixv = - 2 / sqrtpi * (vhe[1] - vhe[end])
    wcck = clenshawcurtisweights(chebyshevmoments1(T, nvG))
    RdtM[1] = Ixv * dot(wcck, (v2 .* dtfvL[:,1]))
    RdtM[2] = Ixv * dot(wcck, (v2 .* vhe  .* dtfvL[:,2])) / 3
    RdtM[3] = Ixv * dot(wcck, (v2 .^2  .* dtfvL[:,1]))

    # Computing the error of Gaussian quadratures
    vec = 1:2:nvG
    wcck = clenshawcurtisweights(chebyshevmoments1(T, Int((nvG+1)/2)))
    eRdtM[1] = Ixv * dot(wcck, (v2[vec] .* dtfvL[vec,1])) - RdtM[1]
    eRdtM[2] = Ixv * dot(wcck, (v2[vec] .* vhe[vec]  .* dtfvL[vec,2])) / 3 - RdtM[2]
    eRdtM[3] = Ixv * dot(wcck, (v2[vec] .^2  .* dtfvL[vec,1])) - RdtM[3]
    
    # w = RdtM[3] - 2 * Ih .* RdtI
    RdtM[4] = (RdtM[3] - 2 * Ih .* RdtM[2]) / 3    # `Rdtvth = w / 3`
    dtIh = RdtM[2] - Ih .* RdtM[4]                 # 

    # Checking the constraint according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`
    errRdtKI = RdtM[3] - (2 * Ih .* dtIh + (3 .+ 2 * Ih.^2) .* RdtM[4])
    
    if is_out_errdt
        # giving the values of `dtn`, `dtI` and `dtK`
        cc = na
        RdtM[1,:] .*= cc              # dtn
        cc *= ma .* vth
        RdtM[2,:] .*= cc           # dtI
        cc *= vth
        RdtM[3,:] .*= cc / 2    # dtK
        return errRdtKI, eRdtM
    else
        # Checking the constraint according to: 
        errRdtKI ≤ atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`",errRdtKI)
      
        # giving the values of `dtn`, `dtI` and `dtK`
        cc = na
        RdtM[1,:] .*= cc              # dtn
        cc *= ma .* vth
        RdtM[2,:] .*= cc           # dtI
        cc *= vth
        RdtM[3,:] .*= cc / 2    # dtK
        return eRdtM
    end 
end

function RdtnIKTs!(RdtM::AbstractVector{T},dtfvL::AbstractArray{T,N},
    vhe::StepRangeLen,Ih::T,ma::T,na::T,vth::T;
    atol_nIK::T=epsT1,is_out_errdt::Bool=false) where{T,N}
  
    # Rdtn, RdtI, RdtK, Rdtvth = RdtM[1:4]
    v2 = vhe .^2
    RdtM[1], errdtn = romberg(vhe, (v2 .* dtfvL[:,1]))
    RdtM[2], errdtI = romberg(vhe, (v2 .* vhe  .* dtfvL[:,2]))
    RdtM[3], errdtK = romberg(vhe, (v2 .^2  .* dtfvL[:,1]))
    
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
        cc = na
        RdtM[1,:] .*= cc              # dtn
        cc *= ma .* vth
        RdtM[2,:] .*= cc           # dtI
        cc *= vth
        RdtM[3,:] .*= cc / 2    # dtK
        return errdtn, errdtI, errdtK, errRdtKI
    else
        # Checking the constraint according to: `δₜn̂a = 0`
        errdtn < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜn̂a = 0`",errdtn)
        errdtI < atol_nIK || @warn("Convergence of integration `RdtI` if failure", errdtI)
        errdtK < atol_nIK || @warn("Convergence of integration `RdtK` if failure", errdtK)
    
        errRdtKI ≤ atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`",errRdtKI)
      
        # giving the values of `dtn`, `dtI` and `dtK`
        cc = na
        RdtM[1,:] .*= cc              # dtn
        cc *= ma .* vth
        RdtM[2,:] .*= cc           # dtI
        cc *= vth
        RdtM[3,:] .*= cc / 2    # dtK
    end 
end

# 2D, [RdtnIKT,ns], vhe
function RdtnIKTs!(RdtM::AbstractArray{T,N2},dtfvL::AbstractVector{Matrix{T}},
    vhe::AbstractVector{StepRangeLen},Ih::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    atol_nIK::T=epsT100,is_out_errdt::Bool=true) where{T,N2}
  
    nn = length(RdtM[:,1])
    if is_out_errdt
        errdtn, errdtI, errdtK, errRdtT = zeros(T,ns),zeros(T,ns),zeros(T,ns),zeros(T,ns)
        for isp in 1:ns
            a = zeros(T,nn)
            errdtn[isp], errdtI[isp], errdtK[isp], errRdtT[isp] = RdtnIKTs!(a,dtfvL[isp],
                                vhe,Ih[isp];atol_nIK=atol_nIK,is_out_errdt=is_out_errdt)
            RdtM[:,isp] = a
        end
        norm(errdtn) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜn̂a = 0`",errdtn)
        norm(errdtI) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜIa = 0`",errdtI)
        norm(errdtK) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜKa = 0`",errdtK)
        norm(errRdtT) < atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜTa = 0`",errRdtT)
    else
      for isp in 1:ns
          a = zeros(T,nn)
          RdtnIKTs!(a,dtfvL[isp],vhe[isp],Ih[isp];atol_nIK=atol_nIK,is_out_errdt=is_out_errdt)
          RdtM[:,isp] = a
      end
    end
end

# 1D, [RdtnIKT]
function RdtnIKTs!(RdtM::AbstractVector{T},dtfvL::AbstractArray{T,N},
    vhe::StepRangeLen,Ih::T;atol_nIK::T=epsT1,is_out_errdt::Bool=false) where{T,N}
  
    # Rdtn, RdtI, RdtK, Rdtvth = RdtM[1:4]
    v2 = vhe .^2
    RdtM[1], errdtn = romberg(vhe, (v2 .* dtfvL[:,1]))
    RdtM[2], errdtI = romberg(vhe, (v2 .* vhe  .* dtfvL[:,2]))
    RdtM[3], errdtK = romberg(vhe, (v2 .^2  .* dtfvL[:,1]))
    
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
        return errdtn, errdtI, errdtK, errRdtKI
    else
      # Checking the constraint according to: `δₜn̂a = 0`
      errdtn ≤ atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜn̂a = 0`",errdtn)
      errdtI < atol_nIK || @warn("Convergence of integration `RdtI` if failure", errdtI)
      errdtK < atol_nIK || @warn("Convergence of integration `RdtK` if failure", errdtK)
      errRdtKI ≤ atol_nIK || @warn("Number of meshgrids may be not enough to satisfy the convergence of `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)`",errRdtKI)
    end
end
