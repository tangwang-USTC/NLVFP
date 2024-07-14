
"""

  Inputs:
  Outputs:
    MckMhck!(Mc,Mhc,njMs,LM,ρa,vthk,ns;L_Mh_limit=L_Mh_limit)
    MckMhck!(Mc,Mhc,njMs,LM,ρa,vthk;L_Mh_limit=L_Mh_limit)
"""

# [j,L,ns]
function MckMhck!(Mc::AbstractArray{T,N}, Mhc::Vector{Matrix{T}}, njMs::Vector{Int64}, LM::Vector{Int64},
    ρk::AbstractVector{T}, vthk::AbstractVector{T}, ns::Int64;L_Mh_limit::Int64=0) where {T,N}

    # Computing the kinetic moments
    for isp in 1:ns
        if L_Mh_limit == 0
            for L in 0:LM[isp]
                L1 = L + 1
                for nj in 1:njMs[isp]
                    j = L + 2(nj - 1)
                    Mc[nj,L1,isp] = Mhc[isp][nj,L1] * (ρk[isp] * vthk[isp]^j)
                    # if L == 1 && j == 1
                    #     @show isp,L,j,Mc[nj,L1,isp]
                    # end
                end
            end
        else
            for L in 0:min(LM[isp],L_Mh_limit)
                L1 = L + 1
                for nj in 1:njMs[isp]
                    j = L + 2(nj - 1)
                    Mc[nj,L1,isp] = Mhc[isp][nj,L1] * (ρk[isp] * vthk[isp]^j)
                end
            end
        end
    end
end

# [j,L]
function MckMhck!(Mc::AbstractArray{T,N}, Mhc::AbstractArray{T,N}, njMs::Int64, 
    LM::Int64, ρk::T, vthk::T;L_Mh_limit::Int64=0) where {T,N}

    # Computing the kinetic moments
    if L_Mh_limit == 0
        for L in 0:LM
            L1 = L + 1
            for nj in 1:njMs
                j = L + 2(nj - 1)
                Mc[nj,L1] = Mhc[nj,L1] * (ρk * vthk^j)
            end
        end
    else
        for L in 0:min(LM,L_Mh_limit)
            L1 = L + 1
            for nj in 1:njMs
                j = L + 2(nj - 1)
                Mc[nj,L1] = Mhc[nj,L1] * (ρk * vthk^j)
            end
        end
    end
end

"""

  Inputs:
  Outputs:
    MhckMck!(Mhc,Mc,njMs,LM,ρa,ns)
"""

# [j,L,ns]
function MhckMck!(Mhc::Vector{Matrix{T}},Mc::AbstractArray{T,N},njMs::Vector{Int64},
    LM::Vector{Int64},ρk::AbstractVector{T},ns::Int64;L_Mh_limit::Int64=0) where{T,N}

    # Computing the re-normalized moments
    for isp in 1:ns
        vthk = (Mc[2,1,isp] / ρk[isp] - 2/3 * (Mc[1,2,isp] / ρk[isp])^2)^0.5
        if L_Mh_limit == 0
            for L in 0:LM[isp]
                L1 = L + 1
                for nj in 1:njMs[isp]
                    j = L + 2(nj - 1)
                    Mhc[isp][nj,L1] = Mc[nj,L1,isp] / (ρk[isp] * vthk^j)
                end
            end
        else
            for L in 0:min(LM[isp],L_Mh_limit)
                L1 = L + 1
                for nj in 1:njMs[isp]
                    j = L + 2(nj - 1)
                    Mhc[isp][nj,L1] = Mc[nj,L1,isp] / (ρk[isp] * vthk^j)
                end
            end
        end
    end
end

# [j,L,ns]
function MhckMck!(Mhc::Vector{Matrix{T}},Mc::AbstractArray{T,N},njMs::Vector{Int64},LM::Vector{Int64},
    ρk::AbstractVector{T},vthk::AbstractVector{T},ns::Int64;L_Mh_limit::Int64=0) where{T,N}

    # Computing the re-normalized moments
    for isp in 1:ns
        if L_Mh_limit == 0
            for L in 0:LM[isp]
                L1 = L + 1
                for nj in 1:njMs[isp]
                    j = L + 2(nj - 1)
                    Mhc[isp][nj,L1] = Mc[nj,L1,isp] / (ρk[isp] * vthk[isp]^j)
                end
            end
        else
            for L in 0:min(LM[isp],L_Mh_limit)
                L1 = L + 1
                for nj in 1:njMs[isp]
                    j = L + 2(nj - 1)
                    Mhc[isp][nj,L1] = Mc[nj,L1,isp] / (ρk[isp] * vthk[isp]^j)
                end
            end
        end
    end
end

# [t]
function MhckMck(Mc::AbstractVector{T},ρk::T,vthk::AbstractVector{T},j::Int64) where{T}

    efgb
    # Computing the re-normalized moments: Mhc
    return Mc ./ (ρk * vthk.^j)
end
