
"""
  Outputs:
    MhcknuT!(Mhck1,Mck1,ρa,vth,LM,ns,nMjMs,nModk1;is_renorm=is_renorm)
    MhcknuT!(Mhck1,LM,ns,nMjMs,nModk1;is_renorm=is_renorm)
"""

# is_Ms_nuT == true, Updating `Mhck1` and `Mck1`
function MhcknuT!(Mhck1::Vector{Matrix{T}}, Mck1::AbstractArray{T,N}, 
    ρak1::AbstractVector{T}, vthk1::AbstractVector{T}, LM::Vector{Int64}, 
    ns::Int64, nMjMs::Vector{Int64},nModk1::Vector{Int64}; is_renorm::Bool=true) where {T,N}

    for isp in 1:ns
        if nModk1[isp] == 1
            uh = Mck1[1,2,isp] / (ρak1[isp] * vthk1[isp])
            Mhck1[isp] = MsnntL2fL0(Mhck1[isp], nMjMs[isp], LM[isp], uh; is_renorm=is_renorm)
            MckMhck!(Mck1[:,:,isp],Mhck1[isp],ρak1[isp],vthk1[isp],LM[isp],nMjMs[isp])
        else
            for L in 0:LM[isp]
                L1 = L + 1
                for nj in 1:nMjMs[isp]
                    j = L + 2(nj - 1)
                    Mhck1[isp][nj,L1] = Mck1[nj,L1,isp] / (ρak1[isp] * vthk1[isp]^j)
                end
            end
        end
    end
end

# is_Ms_nuT == true
function MhcknuT!(Mhck1::Vector{Matrix{T}}, LM::Vector{Int64}, ns::Int64, 
    nMjMs::Vector{Int64}, nModk1::Vector{Int64}; is_renorm::Bool=true) where{T}
    
    sdfghjk
    for isp in 1:ns
        if nModk1[isp] == 1
            Mhck100 = 1Mhck1[isp]
            Mhck1[isp] = MsnntL2fL0(Mhck1[isp], nMjMs[isp], LM[isp], Mhck1[isp][1, 2]; is_renorm=is_renorm)
            @show Mhck100 - Mhck1[isp]
        else
        end
    end
end
