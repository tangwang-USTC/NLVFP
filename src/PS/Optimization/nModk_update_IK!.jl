


"""
  Reduceing the number of `nMod` according to `Rdtsaa`

  Inputs:

  Outputs:
    nMod_update!(is_nMod_renew, Rdtsaak, naik, uaik, vthik, nMod, ns;rtol_dtsa=rtol_dtsa)
"""

# [nMod, ns]
function nMod_update!(is_nMod_renew::Vector{Bool}, Rdtsaak::Vector{TA2},
    naik::Vector{TA}, uaik::Vector{TA},vthik::Vector{TA},nMod::Vector{Int64},
    ns::Int64; rtol_dtsa::T=1e-8) where{TA,TA2,T}
    
    for isp in 1:ns
        nModel = nMod[isp]
        if  nModel ≥ 2
            if norm(uaik[isp]) ≤ epsT10
                # is_update_nMod2(vthik[isp])
                is_nMod_renew[isp], naik[isp], vthik[isp], nModel = nMod_update!( 
                            Rdtsaak[isp], naik[isp], vthik[isp], nModel;rtol_dtsa=rtol_dtsa)
                if is_nMod_renew[isp]
                    nMod[isp] = nModel
                    uaik[isp] = zero.(vthik[isp])
                end
            else
                is_nMod_renew[isp], naik[isp], uaik[isp], vthik[isp], nModel = nMod_update!( 
                            Rdtsaak[isp], naik[isp], uaik[isp], vthik[isp], nModel;rtol_dtsa=rtol_dtsa)
                if is_nMod_renew[isp]
                    nMod[isp] = nModel
                end
            end
        else
            is_nMod_renew[isp] = false
        end
    end
end

################################################################

"""
  Outputs:
    is_nMod_renew, naik, uaik, vthik, nModel = nMod_update(Rdtsaak, naik, uaik, vthik, nModel;rtol_dtsa=rtol_dtsa)
    is_nMod_renew, naik, vthik, nModel = nMod_update!(Rdtsaak, naik, vthik, nModel;rtol_dtsa=rtol_dtsa)
"""

# [nMod] uaik .≠ 0
function nMod_update!(Rdtsaak::AbstractVector{T}, naik::AbstractVector{T}, 
    uaik::AbstractVector{T}, vthik::AbstractVector{T}, nModel::Int64; rtol_dtsa::T=1e-8) where{T}

    if nModel == 2
        is_update = is_update_nMod2(uaik, vthik, Rdtsaak[1];rtol_dtsa=rtol_dtsa)
        if is_update
            nModel -= 1
            return is_update, [1.0], [sum(naik .* uaik)], [1.0], nModel
        else
            return is_update, naik, uaik, vthik, nModel
        end
    elseif nModel == 3
        is_update = zeros(Bool,3)               # C(nMod,2)
        i = 1 
        i1 = 2
        is_update[1] = is_update_nMod2(uaik[[i,i1]], vthik[[i,i1]], Rdtsaak[1];rtol_dtsa=rtol_dtsa)

        i1 = 3
        is_update[2] = is_update_nMod2(uaik[[i,i1]], vthik[[i,i1]], Rdtsaak[2];rtol_dtsa=rtol_dtsa)

        i = 2
        i1 = 3
        is_update[3] = is_update_nMod2(uaik[[i,i1]], vthik[[i,i1]], Rdtsaak[3];rtol_dtsa=rtol_dtsa)

        # Dropping the identical spices
        N1 = sum(is_update)
        if N1 == 0
            return false, naik, uaik, vthik, nModel
        elseif N1 == 3 || N1 == 2
            nModel = 1
            return true, [1.0], [sum(naik .* uaik)], [1.0], nModel
        elseif N1 == 1
            nModel = 2
            if is_update[1]
                i = 1 
                i1 = 2
                s  = 3
                nhN = 1 - naik[s]
                uhN = sum(naik[[i,i1]] .* uaik[[i,i1]]) / nhN
                vhthN = (2 / 3 * (sum(naik[[i,i1]] / nhN .* (1.5 * vthik[[i,i1]].^2 + uaik[[i,i1]].^2)) - uhN^2)) ^0.5
                # @show vhthN + vthik[s]
                return true, [nhN,naik[s]], [uhN,uaik[s]], [vhthN,vthik[s]], nModel
            elseif is_update[2]
                i = 1
                i1 = 3
                s  = 2
                nhN = 1 - naik[s]
                uhN = sum(naik[[i,i1]] .* uaik[[i,i1]]) / nhN
                vhthN = (2 / 3 * (sum(naik[[i,i1]] / nhN .* (1.5 * vthik[[i,i1]].^2 + uaik[[i,i1]].^2)) - uhN^2)) ^0.5
                # @show vhthN + vthik[s]
                return true, [nhN,naik[s]], [uhN,uaik[s]], [vhthN,vthik[s]], nModel
            else
                i = 2
                i1 = 3
                s  = 1
                nhN = 1 - naik[s]
                uhN = sum(naik[[i,i1]] .* uaik[[i,i1]]) / nhN
                vhthN = (2 / 3 * (sum(naik[[i,i1]] / nhN .* (1.5 * vthik[[i,i1]].^2 + uaik[[i,i1]].^2)) - uhN^2)) ^0.5
                # @show vhthN + vthik[s]
                return true, [naik[s],nhN], [uaik[s],uhN], [vthik[s],vhthN], nModel
            end
        end
    else
        dfg
        for i in 1:nModel-1
        end
    end
end

function is_update_nMod2(uaik::AbstractVector{T}, vthik::AbstractVector{T},Rdtsaak::T; rtol_dtsa::T=1e-8) where {T<:Real}
    
    if abs(Rdtsaak) ≤ rtol_dtsa
        RDvthi = abs(vthik[1] / vthik[2] - 1.0)
        if RDvthi ≤ rtol_DnuTi_warn
            @warn("RDvthi_u: The relative differenc of `vthik` which will decide the parameter `nMod`.",RDvthi)
        else
            @show RDvthi
        end
    
        Duai = abs(uaik[1] - uaik[2])
        um = maximum(abs.(uaik))
        if um ≥ atol_uai
            RDuai = Duai / um
            @show RDuai
        else
            @show Duai
        end
        return true
    else
        return false
    end
end

# [nMod] uaik == 0
function nMod_update!(Rdtsaak::AbstractVector{T}, naik::AbstractVector{T},  vthik::AbstractVector{T}, nModel::Int64; rtol_dtsa::T=1e-8) where{T}

    if nModel == 2
        is_update = is_update_nMod2(vthik, Rdtsaak[1];rtol_dtsa=rtol_dtsa)
        if is_update
            nModel -= 1
            return is_update, [1.0], [1.0], nModel
            # return is_update, [1.0], [sum(naik .* vthik.^2)], nModel
        else
            return is_update, naik, vthik, nModel
        end
    elseif nModel == 3
        is_update = zeros(Bool,3)               # C(nMod,2)
        i = 1 
        i1 = 2
        is_update[1] = is_update_nMod2(vthik[[i,i1]],Rdtsaak[1];rtol_dtsa=rtol_dtsa)

        i1 = 3
        is_update[2] = is_update_nMod2(vthik[[i,i1]],Rdtsaak[2];rtol_dtsa=rtol_dtsa)

        i = 2
        i1 = 3
        is_update[3] = is_update_nMod2(vthik[[i,i1]],Rdtsaak[3];rtol_dtsa=rtol_dtsa)
        
        # Dropping the identical spices
        N1 = sum(is_update)
        if N1 == 0
            return false, naik, vthik, nModel
        elseif N1 == 3 || N1 == 2
            nModel = 1
            return true, [1.0], [1.0], nModel
        elseif N1 == 1
            nModel = 2
            if is_update[1]
                i = 1 
                i1 = 2
                s = 3
                return true, [1 - naik[s],naik[s]], [(sum(naik[[i,i1]] .* vthik[[i,i1]].^2) / (1 - naik[s]))^0.5,vthik[s]], nModel
            elseif is_update[2]
                i = 1
                i1 = 3
                s = 2
                return true, [1 - naik[s],naik[s]], [(sum(naik[[i,i1]] .* vthik[[i,i1]].^2) / (1 - naik[s]))^0.5,vthik[s]], nModel
            else
                i = 2
                i1 = 3
                s = 1
                return true, [naik[s], 1 - naik[s]], [vthik[s], (sum(naik[[i,i1]] .* vthik[[i,i1]].^2) / (1 - naik[s]))^0.5], nModel
            end
        end
    else
        dfg
        for i in 1:nModel-1
        end
    end
end

function is_update_nMod2(vthik::AbstractVector{T}, Rdtsaak::T; rtol_dtsa::T=1e-8) where {T<:Real}
    
    if abs(Rdtsaak) ≤ rtol_dtsa
        RDvthi = abs(vthik[1] / vthik[2] - 1.0)
        if RDvthi ≤ rtol_DnuTi_warn
            @warn("RDvthi: The relative differenc of `vthik` which will decide the parameter `nMod`.",RDvthi)
        else
            @show RDvthi
        end
        return true
    else
        return false
    end
end