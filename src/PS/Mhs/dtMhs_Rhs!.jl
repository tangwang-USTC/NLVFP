
"""

  Inputs:
  Outputs:
    dtMhck!(dtMhc,Rhc,Mhc,njMs,LM,Rdtvthk,ns;L_Mh_limit=L_Mh_limit)
    dtMhck!(dtMhc,Rhc,Mhc,njMs,LM,Rdtvthk;L_Mh_limit=L_Mh_limit)
"""

# [j,L,ns]
function dtMhck!(dtMhc::Vector{Matrix{T}}, Rhc::Vector{Matrix{T}}, 
    Mhc::Vector{Matrix{T}}, njMs::Vector{Int64}, LM::Vector{Int64},
    Rdtvthk::AbstractVector{T}, ns::Int64;L_Mh_limit::Int64=0) where {T}

    # Computing the change rate of the normalized kinetic dissipations
    for isp in 1:ns
        if L_Mh_limit == 0
            for L in 0:LM[isp]
                L1 = L + 1
                for nj in 1:njMs[isp]
                    j = L + 2(nj - 1)
                    dtMhc[isp][nj,L1] = Rhc[isp][nj,L1] - Mhc[isp][nj,L1] * (j * Rdtvthk[isp])
                end
            end
        else
            for L in 0:min(LM[isp],L_Mh_limit)
                L1 = L + 1
                for nj in 1:njMs[isp]
                    j = L + 2(nj - 1)
                    dtMhc[isp][nj,L1] = Rhc[isp][nj,L1] - Mhc[isp][nj,L1] * (j * Rdtvthk[isp])
                end
            end
        end
    end
end

# [j,L]
function dtMhck!(dtMhc::AbstractArray{T,N}, Rhc::AbstractArray{T,N}, njMs::Int64, 
    LM::Int64, Rdtvthk::T;L_Mh_limit::Int64=0) where {T,N}

    # Computing the kinetic moments
    if L_Mh_limit == 0
        for L in 0:LM
            L1 = L + 1
            for nj in 1:njMs
                j = L + 2(nj - 1)
                dtMhc[nj,L1] = Rhc[nj,L1] - Mhc[nj,L1] * (j * Rdtvthk)
            end
        end
    else
        for L in 0:min(LM,L_Mh_limit)
            L1 = L + 1
            for nj in 1:njMs
                j = L + 2(nj - 1)
                dtMhc[nj,L1] = Rhc[nj,L1] - Mhc[nj,L1] * (j * Rdtvthk)
            end
        end
    end
end
