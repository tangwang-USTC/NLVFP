
"""
  Outputs:
    RdtvthRc!(Rdtvth,Ih,RdtI,RdtK,ns)
"""

function RdtvthRc!(Rdtvth::AbstractVector{T},Ih::AbstractVector{T},RdtI::AbstractVector{T},RdtK::AbstractVector{T},ns::Int64) where{T}

    # `Rdtvth = w / 3`
    for isp in 1:ns
        Rdtvth[isp] = (RdtK[isp] - 2 * Ih[isp] .* RdtI[isp]) / 3    
    end
    return Rdtvth
end

"""
  Checking the constraint according to: `δₜK̂ₐ = 2(ûₐ∂ₜûₐ + (3/2 + ûₐ²) * vₐₜₕ⁻¹∂ₜvₐₜₕ)` #### which can give `Rdtvth = vₐₜₕ⁻¹∂ₜvₐₜₕ` according to this constraint.
  
    dtIh = RdtM[2] - Ih .* RdtM[4]                 # 
  
    eRdtKI = RdtM[3] - (2 * Ih .* dtIh + (3 .+ 2 * Ih.^2) .* RdtM[4])
          
           = RdtM[3] - (2 * Ih .* dtIh + Kh .* RdtM[4])

  Outputs:
    eRdtKIRc!(eRdtKI,Ih,RdtI,RdtK,Rdtvth,ns)
"""

function eRdtKIRc!(eRdtKI::AbstractVector{T},Ih::AbstractVector{T},RdtI::AbstractVector{T},
    RdtK::AbstractVector{T},Rdtvth::AbstractVector{T},ns::Int64) where{T}

    for isp in 1:ns
        eRdtKI[isp] = RdtI[isp] - 2 * Ih[isp] .* Rdtvth[isp]
        eRdtKI[isp] = RdtK[isp] - (2 * Ih[isp] .* eRdtKI[isp] + (3 .+ 2 * Ih[isp].^2) .* Rdtvth[isp])
    end
    return eRdtKI
end

