"""
  The general `jᵗʰ`-order moment of `fM` in theory according to the following relations:
  
  ``

  Inputs:
  Outputs:
"""
# [nj,ns]

# # [nj], j ∈ 2:2:N⁺
# function Mcj_fM!(Mcj::AbstractVector{T},ρa::T,vth::T,nh::AbstractVector{T},vhth::AbstractVector{T},njMs::Int64) where{T}
    
#     nj = 1
#     # j = 0
#     Mcj[nj] = deepcopy(ρa) 
#     vhth2 = vhth ^2
#     for nj in 2:njMs
#         # j = 2(nj - 1)
#         if nj == 2
#             Mcj[nj] = ρa .* vth .^ j .* sum(nh .* vhth .^ j)
#         else
#             Mcj[nj] = ρa .* vth .^ j .* sum(nh .* vhth .^ j)
#         end
#     end
# end

"""
  Outputs:
    Mcj_fM!(Mcj,ρa,vth,nh,vhth,j)
    Mcj = Mcj_fM(ρa,vth,nh,vhth,j)
"""

# [ns]
function Mcj_fM!(Mcj::AbstractVector{T},ρa::AbstractVector{T},
    vth::AbstractVector{T},nh::Vector{TA},vhth::Vector{TA},j::Int64) where{T,TA}

    for isp in 1:ns
        Mcj[isp] = Mcj_fM(ρa[isp],vth[isp],nh[isp],vhth[isp],j)
    end
end

# []
function Mcj_fM(ρa::T,vth::T,nh::AbstractVector{T},vhth::AbstractVector{T},j::Int64) where{T}

    if j == 0
        return deepcopy(ρa) 
    else
        return ρa * vth ^ j * sum(nh .* vhth .^ j)
    end
end
