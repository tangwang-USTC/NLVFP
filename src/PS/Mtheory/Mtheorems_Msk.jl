
"""
  M-theorem of isolated system:

  `Mhck1 ≤ Mhck` 
  
  for `∀(ℓ, j)`

  Inputs:
  Outputs:
    Mtheorems_RMcs!(RMcs,Mc,ρa,ns)
"""

function Mtheorems_RMcs!(RMcs::AbstractArray{T,N2},Mc::AbstractArray{T,N},
    ρa::AbstractVector{T},ns::Int64) where{T,N,N2}
    
    Ik = zeros(T,ns)
    Kk = zeros(T,ns)
    for isp in 1:ns
        Ik[isp] = Mc[2,1,isp]
        # Kk[isp] = Mc[1,2,isp] * (3/4)
        Kk[isp] = Mc[1,2,isp]
    end
    ρs = sum_kbn(ρa)
    # nas = sum_kbn(na)
    # nhS = na / nas
    # ms = ρs / nas
    ### mhS = ma / ms
    # ρhS = ρa / ρs
    # Is = sum_kbn(Ik)
    # us = sum_kbn(Ik) ./ ρs
    # Ks = sum_kbn(Kk)
    # vSth = (2 / 3 * (2Ks ./ ρs - us^2))
    # vSth = (2 / 3 * (2(sum_kbn(Kk)) ./ ρs - (sum_kbn(Ik) ./ ρs)^2))
    vSth = (sum_kbn(Kk) ./ ρs - 2 / 3 * (sum_kbn(Ik) ./ ρs)^2)
    RMcs[:,:] = sum(Mc;dims=3)[:,:,1]  # Mcs
    for nj in 1:njMs
        j = 2(nj - 1)
        for LL1 in 1:LM1
            LL = LL1 - 1
            RMcs[nj,LL1] /=  ρs * vSth^((j + LL))
        end
    end
end
