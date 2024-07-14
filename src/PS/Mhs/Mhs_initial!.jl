
"""
  Inputs:

  Outputs:
    Mhinitial_fDM!(nMjMs,ns,nMod,uai;is_MjMs_max=is_MjMs_max)
    Mhinitial_fDM!(nMjMs,ns,nMod,uai,njMs_fix;is_MjMs_max=is_MjMs_max)
    Mhinitial_fDM!(Mh,nMjMs,LM,LM1,ns;is_LM1_full=is_LM1_full,L_Mh_limit=L_Mh_limit)
"""
# fDM, fM
function Mhinitial_fDM!(nMjMs::Vector{Int64},ns::Int64,nMod::Vector{Int64},
    uai::Vector{AbstractVector{T}};is_MjMs_max::Bool=false) where{T}
    
    is_fM = true
    for isp in 1:ns
        if norm(uai[isp]) ≥ atol_uai
            is_fM = false
        end
    end
    if is_fM
        if is_MjMs_max
            nModmax1 = maximum(nMod) + 1
            for isp in 1:ns
                nMjMs[isp] = nModmax1
            end
        else
            for isp in 1:ns
                nMjMs[isp] = nMod[isp] + 1
            end
        end
    else
        if is_MjMs_max
            nMjMs .= 2 * maximum(nMod)
        else
            for isp in 1:ns
                nMjMs[isp] = 2 * nMod[isp]
            end
        end
    end
end

function Mhinitial_fDM!(nMjMs::Vector{Int64},ns::Int64,nMod::Vector{Int64},
    uai::Vector{AbstractVector{T}},njMs_fix::Int64;is_MjMs_max::Bool=false) where{T}
    
    is_fM = true
    for isp in 1:ns
        if norm(uai[isp]) ≥ atol_uai
            is_fM = false
        end
    end
    if is_fM
        if is_MjMs_max
            nModmax1 = max(njMs_fix, maximum(nMod) + 1)
            for isp in 1:ns
                nMjMs[isp] = nModmax1
            end
        else
            for isp in 1:ns
                nMjMs[isp] = nMod[isp] + 1
            end
        end
    else
        if is_MjMs_max
            iseven(njMs_fix) || (njMs_fix += 1)
            nMjMs .= max(njMs_fix, 2 * maximum(nMod))
        else
            for isp in 1:ns
                nMjMs[isp] = max(njMs_fix, 2 * nMod[isp])
            end
        end
    end
end

function Mhinitial_fDM!(nMjMs::Vector{Int64},ns::Int64,nMod::Vector{Int64},
    uai::Vector{AbstractVector{T}};is_MjMs_max::Bool=false) where{T}
    
    is_fM = true
    for isp in 1:ns
        if norm(uai[isp]) ≥ atol_uai
            is_fM = false
        end
    end
    if is_fM
        if is_MjMs_max
            nModmax1 = maximum(nMod) + 1
            for isp in 1:ns
                nMjMs[isp] = nModmax1
            end
        else
            for isp in 1:ns
                nMjMs[isp] = nMod[isp] + 1
            end
        end
    else
        if is_MjMs_max
            nMjMs .= 2 * maximum(nMod)
        else
            for isp in 1:ns
                nMjMs[isp] = 2 * nMod[isp]
            end
        end
    end
end

function Mhinitial_fDM!(nMjMs::Vector{Int64},ns::Int64,nMod::Vector{Int64},
    uai::Vector{AbstractVector{T}},njMs_fix::Int64;is_MjMs_max::Bool=false) where{T}
    
    is_fM = true
    for isp in 1:ns
        if norm(uai[isp]) ≥ atol_uai
            is_fM = false
        end
    end
    if is_fM
        if is_MjMs_max
            nModmax1 = max(njMs_fix, maximum(nMod) + 1)
            for isp in 1:ns
                nMjMs[isp] = nModmax1
            end
        else
            for isp in 1:ns
                nMjMs[isp] = nMod[isp] + 1
            end
        end
    else
        if is_MjMs_max
            iseven(njMs_fix) || (njMs_fix += 1)
            nMjMs .= max(njMs_fix, 2 * maximum(nMod))
        else
            for isp in 1:ns
                nMjMs[isp] = max(njMs_fix, 2 * nMod[isp])
            end
        end
    end
end

function Mhinitial_fDM!(Mh::Vector{Matrix{T}},nMjMs::Vector{Int64},LM::Vector{Int64},
    LM1::Int64,ns::Int64;is_LM1_full::Bool=true,L_Mh_limit::Int64=0) where{T}
    
    # no limit of `L` for `Mh`
    if L_Mh_limit == 0
        if is_LM1_full
            for isp in 1:ns
                Mh[isp] = zeros(nMjMs[isp],LM1)
            end
        else
            for isp in 1:ns
                Mh[isp] = zeros(nMjMs[isp],LM[isp]+1)
            end
        end
    else
        if is_LM1_full
            for isp in 1:ns
                Mh[isp] = zeros(nMjMs[isp],L_Mh_limit)
            end
        else
            for isp in 1:ns
                Mh[isp] = zeros(nMjMs[isp],min(LM[isp]+1,L_Mh_limit))
            end
        end
    end
end


# function Mhinitial_fDM!(Mh::Vector{Matrix{T}},nMjMs::Vector{Int64},
#     LM::Vector{Int64},LM1::Int64,ns::Int64,nMod::Vector{Int64};
#     njMs_fix::Int64=0,is_MjMs_max::Bool=false,is_LM1_full::Bool=true,L_Mh_limit::Int64=0) where{T}
    
#     if njMs_fix == 0
#         if is_MjMs_max
#             nModmax1 = maximum(nMod) + 1
#         end
#         # no limit of `L` for `Mh`
#         if L_Mh_limit == 0
#             if is_LM1_full
#                 for isp in 1:ns
#                     is_MjMs_max ? nMjMs[isp] = nModmax1 : nMjMs[isp] = nMod[isp] + 1
#                     Mh[isp] = zeros(nMjMs[isp],LM1)
#                 end
#             else
#                 for isp in 1:ns
#                     is_MjMs_max ? nMjMs[isp] = nModmax1 : nMjMs[isp] = nMod[isp] + 1
#                     Mh[isp] = zeros(nMjMs[isp],LM[isp]+1)
#                 end
#             end
#         else
#             if is_LM1_full
#                 for isp in 1:ns
#                     is_MjMs_max ? nMjMs[isp] = nModmax1 : nMjMs[isp] = nMod[isp] + 1
#                     Mh[isp] = zeros(nMjMs[isp],L_Mh_limit)
#                 end
#             else
#                 for isp in 1:ns
#                     is_MjMs_max ? nMjMs[isp] = nModmax1 : nMjMs[isp] = nMod[isp] + 1
#                     Mh[isp] = zeros(nMjMs[isp],min(LM[isp]+1,L_Mh_limit))
#                 end
#             end
#         end
#     else
#         for isp in 1:ns
#             Mh[isp] = zeros(nMjMs[isp],LM1)
#         end
#     end
# end
