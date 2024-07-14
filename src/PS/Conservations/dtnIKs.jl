
"""
  A priori anasys and posterior conservations for mass, momentum and 
    total energy during Fokker-Planck collision processes.

  The posterior conservations are based on the conservation laws in theory and 
    the super-convergence of the deviations of the conservative vaules in discrete.

  Inputs:
    Rc: The kinetic dissipative forces owing to Fokker-Planck collisions, which are normalized by factor `CjLL2(j,L)`
    err_Rc: The error of kinetic dissipative forces `Rc` given by Romberg method or Gaussian quadratures.
    1

  Outputs:
    dtnIKposteriorC!(Rc,err_Rc,nMjMs)
    dtnIKposteriorC!(dtnIK,err_dtnIK)

"""

# [ns * nMod = 2]
function dtnIKposteriorC!(Rc::AbstractArray{T,N},err_Rc::AbstractArray{T,N},nMjMs::Vector{Int64}) where{T,N}

    isp, iFv = 1, 2

    # Checking the convergences of quadratures

    # Posterior conservation for number density     # dtn
    Rc[1,1,:] .= 0.0                         
    
    # Posterior conservation for momentum           # dtI
    if abs(err_Rc[1,2,isp]) > abs(err_Rc[1,2,iFv])
        Rc[1,2,isp] = - Rc[1,2,iFv]            
    else
        Rc[1,2,iFv] = - Rc[1,2,isp]
    end
    
    # Posterior conservation for total energy       # dtK
    if abs(err_Rc[2,1,isp]) > abs(err_Rc[2,1,iFv])
        Rc[2,1,isp] = - Rc[2,1,iFv]         
    else
        Rc[2,1,iFv] = - Rc[2,1,isp]
    end
end

function dtnIKposteriorC!(dtnIK::AbstractArray{T,N},err_dtnIK::AbstractArray{T,N}) where{T,N}

    isp, iFv = 1, 2

    # Checking the convergences of quadratures

    # Posterior conservation for number density     # dtn
    dtnIK[1,:] .= 0.0                         
    
    # Posterior conservation for momentum           # dtI
    if abs(err_dtnIK[2,isp]) > abs(err_dtnIK[2,iFv])
        dtnIK[2,isp] = - dtnIK[2,iFv]            
    else
        dtnIK[2,iFv] = - dtnIK[2,isp]
    end
    
    # Posterior conservation for total energy       # dtK
    if abs(err_dtnIK[3,isp]) > abs(err_dtnIK[3,iFv])
        dtnIK[3,isp] = - dtnIK[3,iFv]         
    else
        dtnIK[3,iFv] = - dtnIK[3,isp]
    end
end


