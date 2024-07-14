"""
  The thermodynamic time scale `τab` for `fM-fM`

  # τ₀_ab = C * nᵦ * (ZₐZᵦ)²√(mₐmᵦ) / (mₐTᵦ + mᵦTₐ)^1.5 * lnAab
          ∝ nᵦmₐmᵦ(∑vₜₕ²)^(3/2)

  Inputs:
    tau_scale: ∈ [:min, :max, :minmax]
    Coeff_tau_mZ = Coeff_tau * 2^1.5 * Zqab^2 / mamb * lnA_const(spices)

  Outputs:
    tau_fM!(tau, ma, Zq, spices, na, vth, Coeff_tau, nai, vthi, nMod)
    tau_fM!(tau,mamb,Zqab spices,, nas, vaths, nbs, vbths, Coeff_tau, nMod)
    tau_fM!(tau,mamb,Zqab spices,, na, vth, Coeff_tau)
"""

#[nMod, ns≥3], Cab + Caa
function tau_fM!(tau::AbstractVector{T}, ma::AbstractVector{T}, Zq::Vector{Int64}, spices::Vector{Symbol},
    na::AbstractVector{T}, vth::AbstractVector{T}, Coeff_tau::T, 
    nai2::Vector{AbstractVector{T}}, vthi2::Vector{AbstractVector{T}},
    nai::Vector{AbstractVector{T}}, vthi::Vector{AbstractVector{T}}, nMod::Vector{Int64},ns::Int64) where {T<:Real}
    
    if ns == 2
        tau_fM!(tau, ma, Zq, spices, na, vth, Coeff_tau, nai, vthi, nMod)
    else
        isp = 1
        iFv = isp+1
        vec = [isp,iFv]
        nai2[1], vthi2[1] = nai[isp], vthi[isp]
        nai2[2], vthi2[2] = nai[iFv], vthi[iFv]
        tau_fM!(tau, ma[vec], Zq[vec], spices[vec], na[vec], vth[vec], Coeff_tau, nai2, vthi2, nMod[vec])

        tau2 = zeros(T,2)
        for iFv in isp+2:ns
            vec[2] = iFv
            nai2[2], vthi2[2] = nai[iFv], vthi[iFv]
            tau_fM!(tau2, ma[vec], Zq[vec], spices[vec], na[vec], vth[vec], Coeff_tau, nai2, vthi2, nMod[vec])
            tau[1] = min(tau[1], tau2[1])
            tau[2] = max(tau[2], tau2[2])
        end

        for isp in 2:ns-1
            vec[1] = isp
            nai2[1], vthi2[1] = nai[isp], vthi[isp]
            for iFv in isp+1:ns
                vec[2] = iFv
                nai2[2], vthi2[2] = nai[iFv], vthi[iFv]
                tau_fM!(tau2, ma[vec], Zq[vec], spices[vec], na[vec], vth[vec], Coeff_tau, nai2, vthi2, nMod[vec])
                tau[1] = min(tau[1], tau2[1])
                tau[2] = max(tau[2], tau2[2])
            end
        end
    end
end

#[nMod, ns=2], Cab + Caa
function tau_fM!(tau::AbstractVector{T}, ma::AbstractVector{T}, Zq::Vector{Int64}, spices::Vector{Symbol},
    na::AbstractVector{T}, vth::AbstractVector{T}, Coeff_tau::T, 
    nai::Vector{AbstractVector{T}}, vthi::Vector{AbstractVector{T}}, nMod::Vector{Int64}) where {T<:Real}
    
    # Computing the sub-collision time scales between `a` and `b`
    isp = 1
    iFv = 2
    mambs = ma[isp] * ma[iFv]
    Zqabs = Zq[isp] * Zq[iFv]
    if prod(nMod) == 1
        if min(na[isp], na[iFv]) ≥ n0_c
            Coeff_tau_mZ = Coeff_tau * 2^1.5 * Zqab^2 / mamb * lnA_const(spices)
            cach_fM!(tau, na, vth, Coeff_tau_mZ)
        end
    else
        if nMod[isp] == 1
            tau_fM!(tau,mambs,Zqabs, spices, na[isp], vth[isp], na[iFv] * nai[iFv], vth[iFv] * vthi[iFv], Coeff_tau, nMod[iFv])
        elseif nMod[iFv] == 1
            tau_fM!(tau,mambs,Zqabs, spices, na[iFv], vth[iFv], na[isp] * nai[isp], vth[isp] * vthi[isp], Coeff_tau, nMod[isp])
        else
            tau_fM!(tau,mambs,Zqabs, spices, na[isp] * nai[isp], vth[isp] * vthi[isp], na[iFv] * nai[iFv], vth[iFv] * vthi[iFv], Coeff_tau, nMod)
        end

        # Computing the self-collision time scales
        tau_copy = deepcopy(tau)
        isp = 1
        if nMod[isp] ≥ 2
            tau_fM!(tau,ma[isp],Zq[isp], spices[isp], na[isp], vth[isp], Coeff_tau, nai[isp], vthi[isp], nMod[isp])
            tau[1] = min(tau[1], tau_copy[1])
            tau[2] = max(tau[2], tau_copy[2])

            isp = 2                   # iFv
            if nMod[isp] ≥ 2
                tau_copy = deepcopy(tau)
                tau_fM!(tau,ma[isp],Zq[isp], spices[isp], na[isp], vth[isp], Coeff_tau, nai[isp], vthi[isp], nMod[isp])
                tau[1] = min(tau[1], tau_copy[1])
                tau[2] = max(tau[2], tau_copy[2])
            end
        else
            isp = 2               # iFv
            if nMod[isp] ≥ 2
                tau_fM!(tau,ma[isp],Zq[isp], spices[isp], na[isp], vth[isp], Coeff_tau, nai[isp], vthi[isp], nMod[isp])
                tau[1] = min(tau[1], tau_copy[1])
                tau[2] = max(tau[2], tau_copy[2])
            end
        end

        # Compute the total collision time scale 
        Coeff_tau_mZ = Coeff_tau * 2^1.5 * Zqab^2 / mamb * lnA_const(spices)
        tau_copy = deepcopy(tau)
        cach_fM!(tau, na, vth, Coeff_tau_mZ)
        tau[1] = min(tau[1], tau_copy[1])
        tau[2] = max(tau[2], tau_copy[2])
    end
end

# [ns=2], nMod .≥ 2, Cab
function tau_fM!(tau::AbstractVector{T},mamb::T, Zqab::Int64, spices::Vector{Symbol},
    nas::AbstractVector{T}, vaths::AbstractVector{T}, nbs::AbstractVector{T}, 
    vbths::AbstractVector{T}, Coeff_tau::T, nMod::Vector{Int64}) where {T<:Real}
    
    Coeff_tau_mZ = Coeff_tau * 2^1.5 * Zqab^2 / mamb * lnA_const(spices)
    tau_copy = deepcopy(tau)
    ia = 1
    ib = 1
    if min(nas[ia], nbs[ib]) ≥ n0_c
        cach_fM!(tau, [nas[ia], nbs[ib]], [vaths[ia], vbths[ib]], Coeff_tau_mZ)
    end
    for ib in 2:nMod[2]
        if min(nas[ia], nbs[ib]) ≥ n0_c
            tau_copy = deepcopy(tau)
            cach_fM!(tau, [nas[ia], nbs[ib]], [vaths[ia], vbths[ib]], Coeff_tau_mZ)
            tau[1] = min(tau[1], tau_copy[1])
            tau[2] = max(tau[2], tau_copy[2])
        end
    end

    for ia in 2:nMod[1]
        ib = 1
        if min(nas[ia], nbs[ib]) ≥ n0_c
            tau_copy = deepcopy(tau)
            cach_fM!(tau, [nas[ia], nbs[ib]], [vaths[ia], vbths[ib]], Coeff_tau_mZ)
            tau[1] = min(tau[1], tau_copy[1])
            tau[2] = max(tau[2], tau_copy[2])
        end

        for ib in 2:nMod[2]
            if min(nas[ia], nbs[ib]) ≥ n0_c
                tau_copy = deepcopy(tau)
                cach_fM!(tau, [nas[ia], nbs[ib]], [vaths[ia], vbths[ib]], Coeff_tau_mZ)
                tau[1] = min(tau[1], tau_copy[1])
                tau[2] = max(tau[2], tau_copy[2])
            end
        end
    end
end

# [ns=2], nMod=[1, nModb],  Cab
function tau_fM!(tau::AbstractVector{T},mamb::T, Zqab::Int64,spices::Vector{Symbol}, 
    nas::T, vaths::T, nbs::AbstractVector{T}, vbths::AbstractVector{T}, 
    Coeff_tau::T, nModb::Int64) where {T<:Real}

    Coeff_tau_mZ = Coeff_tau * 2^1.5 * Zqab^2 / mamb * lnA_const(spices)
    ib = 1
    if min(nas, nbs[ib]) ≥ n0_c
        cach_fM!(tau, [nas, nbs[ib]], [vaths, vbths[ib]], Coeff_tau_mZ)
    end
    for ib in 2:nModb
        if min(nas, nbs[ib]) ≥ n0_c
            tau_copy = deepcopy(tau)
            cach_fM!(tau, [nas, nbs[ib]], [vaths, vbths[ib]], Coeff_tau_mZ)
            tau[1] = min(tau[1], tau_copy[1])
            tau[2] = max(tau[2], tau_copy[2])
        end
    end
end

# [ns=2], nMod .=1, Cab
function tau_fM!(tau::AbstractVector{T}, mamb::T, Zqab::Int64, spices::Vector{Symbol},
    na::AbstractVector{T}, vth::AbstractVector{T}, Coeff_tau::T) where {T<:Real}

    if min(na[1], na[2]) ≥ n0_c
        tTn = (sum(vth.^2))^1.5 / (Coeff_tau * 2^1.5 * Zqab^2 / mamb * lnA_const(spices))
        tau[1] = tTn / maximum(na)
        tau[2] = tTn / minimum(na)
    end
end

# [ns=2], Cab 
function cach_fM!(tau::AbstractVector{T}, na::AbstractVector{T}, vth::AbstractVector{T},Coeff_tau_mZ::T) where {T<:Real}

    if min(na[1], na[2]) ≥ n0_c
        tTn = (sum(vth.^2))^1.5 / Coeff_tau_mZ
        tau[1] = tTn / maximum(na)
        tau[2] = tTn / minimum(na)
    end
end

""" 
  Inputs:
  Outputs:
    τab = tau_fM(ma,Zq, na, vth, Coeff_tau, nai, vthi, nMod;tau_scale=:min)
    τab_min, tab_max = tau_fM(ma,Zq, spices, na, vth, Coeff_tau, nai, vthi, nMod;tau_scale=:minmax)
    tau_fM!(tab,ma,Zq, spices, na, vth, Coeff_tau, nai, vthi, nMod)
"""

# [nMod], ns = 1, `nMod ≥ 2`, Caa
function tau_fM!(tau::AbstractVector{T}, ma::T, Zq::Int64, spices::Symbol, 
    na::T, vth::T, Coeff_tau::T, 
    nai::AbstractVector{T}, vthi::AbstractVector{T}, nMod::Int) where {T<:Real}
    
    cvT = Coeff_tau * 2^1.5 * (Zq^2 / ma)^2 * lnA_const(spices)
    if nMod == 2
        if min(nai[1], nai[2]) ≥ n0_c
            tTn = (sum((vth * vthi).^2))^1.5 / cvT
            tau[1] = tTn / (maximum(nai) * na)
            tau[2] = tTn / (minimum(nai) * na)
        end
    elseif nMod == 3
        vthis = vth * vthi
        nais = na * nai

        i, i1 = 1, 2
        if min(nais[i], nais[i1]) ≥ n0_c
            tTn = (sum((vthis[[i,i1]]).^2))^1.5 / cvT
            tau[1] = tTn / (maximum(nais[[i,i1]]))
            tau[2] = tTn / (minimum(nais[[i,i1]]))
        end

        i, i1 = 1, 3
        if min(nais[i], nais[i1]) ≥ n0_c
            tTn = (sum((vthis[[i,i1]]).^2))^1.5 / cvT
            tau[1] = min(tau[1], tTn / (maximum(nais[[i,i1]])))
            tau[2] = max(tau[2], tTn / (minimum(nais[[i,i1]])))
        end

        i, i1 = 2, 3
        if min(nais[i], nais[i1]) ≥ n0_c
            tTn = (sum((vthis[[i,i1]]).^2))^1.5 / cvT
            tau[1] = min(tau[1], tTn / (maximum(nais[[i,i1]])))
            tau[2] = max(tau[2], tTn / (minimum(nais[[i,i1]])))
        end
    else
        djfjff
    end
end

# [nMod], ns = 1,
function tau_fM(ma::T, Zq::Int64, spices::Symbol, na::T, vth::T, Coeff_tau::T, 
    nai::AbstractVector{T}, vthi::AbstractVector{T}, nMod::Int;tau_scale::Symbol=:min) where {T<:Real}
    
    cvT = Coeff_tau * 2^1.5 * (Zq^2 / ma)^2 * lnA_const(spices)
    if nMod == 2
        tTn = (sum((vth * vthi).^2))^1.5 / cvT
        if min(nai[1], nai[2]) ≥ n0_c
            if tau_scale == :min
                return tTn / (maximum(nai) * na)
            elseif tau_scale == :max
                return tTn / (minimum(nai) * na)
            else
                return tTn / (maximum(nai) * na), tTn / (minimum(nai) * na)
                    #    νT_min, νT_max
            end
        else
            if tau_scale == :min
                return tTn / (maximum(nai) * na)
            elseif tau_scale == :max
                return tTn / (minimum(nai) * na)
            else
                return tTn / (maximum(nai) * na), tTn / (minimum(nai) * na)
                    #    νT_min, νT_max
            end
        end
    elseif nMod == 3
        vthis = vth * vthi
        nais = na * nai

        i, i1 = 1, 2
        if min(nais[i], nais[i1]) ≥ n0_c
            tTn = (sum((vthis[[i,i1]]).^2))^1.5 / cvT
            tau_min = tTn / (maximum(nais[[i,i1]]))
            tau_max = tTn / (minimum(nais[[i,i1]]))
        end

        i, i1 = 1, 3
        if min(nais[i], nais[i1]) ≥ n0_c
            tTn = (sum((vthis[[i,i1]]).^2))^1.5 / cvT
            tau_min = min(tau_min, tTn / (maximum(nais[[i,i1]])))
            tau_max = max(tau_max, tTn / (minimum(nais[[i,i1]])))
        end

        i, i1 = 2, 3
        if min(nais[i], nais[i1]) ≥ n0_c
            tTn = (sum((vthis[[i,i1]]).^2))^1.5 / cvT
            tau_min = min(tau_min, tTn / (maximum(nais[[i,i1]])))
            tau_max = max(tau_max, tTn / (minimum(nais[[i,i1]])))
        end
        if tau_scale == :min
            return tau_min
        elseif tau_scale == :max
            return tau_max
        else
            return tau_min, tau_max
        end
    else
        djfjff
    end
end

"""
  Outputs:
    τab = tau_fM(ma,Zq, spices, na, Ta, Coeff_tau;tau_scale=:min)
    τab = tau_fM(mamb,Zqab, spices, na, vth, Coeff_tau;tau_scale=:min)
"""

# [ns=2], nMod .=1,
function tau_fM(ma::AbstractVector{T}, Zq::Vector{Int64}, spices::Vector{Symbol},
    na::AbstractVector{T}, Ta::AbstractVector{T}, Coeff_tau::T;tau_scale::Symbol=:min) where {T<:Real}

    ispt = 1
    iFvt = 2
    # cccd = (Dₐ^0.5 * Tk^1.5 / n20) * (nd / (md^0.5 * Td^1.5))
    # cccd = ((Dₐ / md) ^0.5 * (Tk / Td) ^1.5) * (nd / n20)
    # cccd = m_unit ^0.5 * T_unit ^1.5 / n_unit
    νT = Coeff_tau * (ma[ispt] * ma[iFvt])^0.5 * (Zq[ispt] * Zq[iFvt])^2 / (ma[ispt] * Ta[iFvt] + ma[iFvt] * Ta[ispt])^1.5 * lnA_const(spices)
    
    if tau_scale == :min
        return 1 / (maximum(na) * νT)
        # τmin = 1 / (maximum(n0) * νT)
    elseif tau_scale == :max
        return 1 / (minimum(na) * νT)
    else
        return 1 / (maximum(n0) * νT), 1 / (minimum(n0) * νT)
            #    νT_min, νT_max
    end
end

function tau_fM(mamb::T, Zqab::Int64, spices::Vector{Symbol}, 
    na::AbstractVector{T}, vth::AbstractVector{T}, Coeff_tau::T) where {T<:Real}

    νT = Coeff_tau * 2^1.5 * Zqab^2 / mamb / (sum(vth.^2))^1.5 * lnA_const(spices)
    if tau_scale == :min
        return 1 / (maximum(na) * νT)
    elseif tau_scale == :max
        return 1 / (minimum(na) * νT)
    else
        return 1 / (maximum(na) * νT), 1 / (minimum(na) * νT)
            #    νT_min, νT_max
    end
end
