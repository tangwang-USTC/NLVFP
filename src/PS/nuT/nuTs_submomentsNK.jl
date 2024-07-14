
"""
  For general 'nMod': The effective thermal temperature can be expressed as:

    Ta = 2/3 * (Ka - Eka) / na

  where

    nha = n1 + n2 + ⋯.
"""

"""
  Inputs:
    `nha`: is the normalized total number density of `spices[i]` which is normalized by `n20`.

  Outputs:
    nai = naiNormNK(nai,nMod,nha,is_uniform_nai,ns;is_eliminate=is_eliminate)
"""

function naiNormNK(nai::Vector{AbstractVector{T}},nMod::Vector{Int64},
    nha::AbstractVector{T},is_uniform_nai::Vector{Bool},ns::Int64;is_eliminate::Bool=false) where{T}

    for isp in 1:ns
        nModa = nMod[isp]
        if nModa > 1
            rat_nai = zeros(nModa)      # ∈ (0,1)
            if is_uniform_nai[isp]
                if nModa == 2
                    rat_nai .= 0.5
                else
                    rat_nai .= 1 / nModa
                end
            else
                if nModa == 2
                    if NK_initial == 1
                        rat_nai[1] = 1.0
                        rat_nai[2] = 1 - rat_nai[1]
                    else
                        rat_nai[1] = 0.45
                        rat_nai[2] = 1 - rat_nai[1]
                    end
                elseif nModa == 3
                    if NK_initial == 1
                        rat_nai[1:nModa-1] .= [1.0,0.0]
                        rat_nai[nModa] = 1 - sum(rat_nai[1:nModa-1])
                    else
                        rat_nai[1:nModa-1] .= [0.35,0.45]
                        rat_nai[nModa] = 1 - sum(rat_nai[1:nModa-1])
                    end
                elseif nModa == 4
                    if NK_initial == 1
                        rat_nai[1:nModa-1] .= [1.0,0.0,0.0]
                        rat_nai[nModa] = 1 - sum(rat_nai[1:nModa-1])
                    else
                        # rat_nai[1:nModa-1] .= [0.25,0.3,0.1]
                        rat_nai[1:nModa-1] .= [0.35,0.0,0.45]
                        rat_nai[nModa] = 1 - sum(rat_nai[1:nModa-1])
                    end
                elseif nModa == 4
                    if NK_initial == 1
                        rat_nai[1:nModa-1] .= [1.0,0.0,0.0,0.0]
                        rat_nai[nModa] = 1 - sum(rat_nai[1:nModa-1])
                    else
                        # rat_nai[1:nModa-1] .= [0.25,0.3,0.1]
                        rat_nai[1:nModa-1] .= [0.35,0.0,0.45,0.0]
                        rat_nai[nModa] = 1 - sum(rat_nai[1:nModa-1])
                    end
                else
                    sdfhmjh
                end
            end
            if prod(rat_nai) == 0
                @warn("Overfitting model, please choosing `is_eliminate` to decide whether or not to reduce the model!")
                # is_zero = (rat_nai .== 0)
            end
            abs(sum(rat_nai) - 1) < rtol_n || throw(ArgumentError("`∑(n̂ai)` must be unit!!!"))
        end
        if nModa == 1
            nai[isp][1] = nha[isp]
        else
            if is_uniform_nai[isp]
                nai[isp][1:nModa] .= 1.0 / nModa
            else
                nai[isp][1:nModa] = nha[isp] * rat_nai
            end
        end
        if is_eliminate && nMod[isp] ≥ 2
            vec = nai[isp] .> n0_c
            nMod[isp] = sum(vec)
            nai[isp][1:nMod[isp]] = nai[isp][vec]
            nai[isp][nMod[isp]+1:end] .= 0.0
        end
    end
    return nai,nMod
end

"""
  # Giving the normalized average velocity by `vth` as `uai[k] = ua[k] / vth, k = 1:nMod`

  Normalized effective Momentum:

    Îₐ = Iₐ / (mₐnₐvₜₕ) = ûₐ = ∑ₖ(n̂ₐₖ ûₐₖ)

  With definition `rat_uai` as `Rûₐₖ = ûₐₖ / ûₐ`, we obtain:

    ∑ₖ(n̂ₐₖ Rûₐₖ) = 1


  Inputs:
    uai: ûₐₖ = uai[k] / vth, the `kᵗʰ` normalized sub-component of the average velocity
    uha: ûₐ = ua / vth, the normalized effective average velocity

  Outputs:
    uai = uaiNormNK(uai,nai,nMod,uha,is_uniform_uai,ns)
"""

function uaiNormNK(uai::Vector{AbstractVector{T}},nai::Vector{AbstractVector{T}},nMod::Vector{Int64},
    uha::AbstractVector{T},is_uniform_uai::Vector{Bool},ns::Int64) where{T}

    for isp in 1:ns
        nModa = nMod[isp]
        if nModa == 1
            uai[isp][1] = uha[isp]
        else
            if is_uniform_uai[isp]
                uai[isp] .= uha[isp]
            else
                if nModa == 2
                    if NK_initial == 1
                        uai[isp][1] = 1 * uha[isp]
                    else
                        uai[isp][1] = 0.55 * uha[isp]
                    end
                else
                    if nModa == 3
                        if NK_initial == 1
                            uai[isp][1:nModa-1] = [1,0] * uha[isp]
                        else
                            uai[isp][1:nModa-1] = [0.85,1.5] * uha[isp]
                        end
                    elseif nModa == 4
                        if NK_initial == 1
                            uai[isp][1:nModa-1] = [1,0.0,0.0] * uha[isp]
                        else
                            uai[isp][1:nModa-1] = [0.25,0.0,0.0] * uha[isp]
                        end
                        # uai[isp][1:nModa-1] = [0.85,0.7,0.9] * uha[isp]
                    elseif nModa == 5
                        if NK_initial == 1
                            uai[isp][1:nModa-1] = [1,0.0,0.0,0.0] * uha[isp]
                        else
                            uai[isp][1:nModa-1] = [0.25,0.0,0.0,0.0] * uha[isp]
                        end
                        # uai[isp][1:nModa-1] = [0.85,0.7,0.9] * uha[isp]
                    else
                        sdfhmjh
                    end
                end
                if nModa == 2
                    uai[isp][nModa] = uha[isp] - (nai[isp][1] * uai[isp][1])
                    uai[isp][nModa] /= nai[isp][nModa]
                else
                    uai[isp][nModa] = uha[isp] - sum(nai[isp][1:nModa-1] .* uai[isp][1:nModa-1])
                    uai[isp][nModa] /= nai[isp][nModa]
                end
            end
            if abs(sum(nai[isp][1:nModa] .* uai[isp][1:nModa]) - uha[isp]) ≥ atol_IKTh
                @show abs(sum(nai[isp][1:nModa] .* uai[isp][1:nModa]) - uha[isp])
                throw(ArgumentError("`Îa - ∑(Îai)` must be zero!!!"))
            end
        end
    end
    return uai
end

"""
  # Giving the normalized thermal velocity by `vth` as `vthi[k] = vth[k] / vath, k = 1:nMod`

  Normalized total energy:

    K̂ₐ = Kₐ / (nₐTₐ) = ∑ₖ(3 / 2 * n̂ₐₖT̂ₐₖ + n̂ₐₖ ûₐₖ²)
    T̂ₐₖ = 2/3 (K̂ₐ - Êₖₐ)
        = 2/3 (K̂ₐ - ûₐ²)
        = ∑ₖ(n̂ₐₖT̂ₐₖ + 2/3 * n̂ₐₖ ûₐₖ²) - 2/3 * ûₐ²

  where

    T̂ₐₖ = Tₐₖ / Tₐ
    ûₐₖ = uₐₖ / vₜₕ
    ûₐ = uₐ / vₜₕ = Îₐ = ∑ₖ(n̂ₐₖ ûₐₖ)
    Êₖₐ = ûₐ²


  Inputs:
    vthi: v̂ₜₕₖ = vath[k] / vth[isp], the `kᵗʰ` normalized sub-component of the thermal velocity
    K̂a0: K̂ₐ = Ka / (nₐTₐ), the normalized effective total energy

  Outputs:
    vthi = vthiNormNK(vthi,nai,uai,nMod,uha,is_uniform_Tai,ns)
"""
 
function vthiNormNK(vthi::Vector{AbstractVector{T}},nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},
    nMod::Vector{Int64},uha::AbstractVector{T},is_uniform_Tai::Vector{Bool},ns::Int64) where{T}

    # ,spices::Vector{Symbol}
    for isp in 1:ns
        nModa = nMod[isp]
        if nModa == 1
            vthi[isp][1] = 1.0
            # vthi[isp] .= 1.0 + (1 / 3 * (uha[isp].^2 - sum(nai[isp] .* uai[isp].^2)))^0.5
        else
            if is_uniform_Tai[isp]
                vthi[isp] .= (1 + 2 / 3 * (uha[isp].^2 - sum(nai[isp] .* uai[isp].^2)))^0.5
            else
                if nModa == 2
                    if NK_initial == 1
                        vthi[isp][1] = 1.0
                    else
                        vthi[isp][1] = 0.8544533
                    end
                    # vthi[isp][1] = 0.999
                else
                    if nModa == 3
                        if NK_initial == 1
                            vthi[isp][1:nModa-1] = [1.0,vhthInitial]
                        else
                            vthi[isp][1:nModa-1] = [0.785,0.95]
                        end
                        # vthi[isp][1:nModa-1] = [0.965554,1.02]
                    elseif nModa == 4
                        if NK_initial == 1
                            vthi[isp][1:nModa-1] = [1.0,vhthInitial,vhthInitial]
                        else
                            vthi[isp][1:nModa-1] = [0.925,0.95,0.88]
                        end
                    elseif nModa == 5
                        if NK_initial == 1
                            vthi[isp][1:nModa-1] = [1.0,vhthInitial,vhthInitial,vhthInitial]
                        else
                            vthi[isp][1:nModa-1] = [0.925,0.95,0.88,0.78]
                        end
                    else
                        sdfhmjh
                    end
                end
                if nModa == 2
                    if nai[isp][nModa] ≤ n0_c
                        vthi[isp][nModa] = vhthInitial
                        @error("Initail vaule of `nai` is not good!")
                    else
                        # vthi^2
                        vthi[isp][2] = 1.0 / nai[isp][2] - nai[isp][1] / nai[isp][2] * (vthi[isp][1])^2
                        vthi[isp][2] += 2/3 * (uha[isp].^2 / nai[isp][2] - (nai[isp][1] / nai[isp][2] .* uai[isp][1].^2 + uai[isp][2].^2))
                        vthi[isp][2] = (vthi[isp][2])^0.5
                    end
                else
                    if nai[isp][nModa] ≤ n0_c
                        vthi[isp][nModa] = vhthInitial
                        @error("Initail vaule of `nai` is not good!")
                    else
                        vthi[isp][nModa] = 1 - sum(nai[isp][1:nModa-1] .* (vthi[isp][1:nModa-1]).^2)
                        vthi[isp][nModa] += 2/3 * (uha[isp].^2 - sum(nai[isp] .* uai[isp].^2))
                        vthi[isp][nModa] = (vthi[isp][nModa] / nai[isp][nModa])^0.5
                    end
                end
            end
            # abs(sum()) < atol_IKTh || throw(ArgumentError("`K̂a - ∑(Îai)` must be zero!!!"))
        end
    end
    return vthi
end

"""
  nai,uai,vthi = nuTsNormNK(nai,uai,vthi,nMod,nha,uha,is_uniform_nai,is_uniform_uai,is_uniform_Tai,ns;is_eliminate=is_eliminate)
"""

function nuTsNormNK(nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},
    nMod::Vector{Int64},nha::AbstractVector{T},uha::AbstractVector{T},
    is_uniform_nai::Vector{Bool},is_uniform_uai::Vector{Bool},
    is_uniform_Tai::Vector{Bool},ns::Int64;is_eliminate::Bool=false) where{T}

    nai,nMod = naiNormNK(nai,nMod,nha,is_uniform_nai,ns;is_eliminate=is_eliminate)
    uai = uaiNormNK(uai,nai,nMod,uha,is_uniform_uai,ns)
    vthi = vthiNormNK(vthi,nai,uai,nMod,uha,is_uniform_Tai,ns)
    if is_eliminate
        for isp in 1:ns
            isnh = nai[isp] .≥ atol_n
            nMod[isp] = sum(isnh)
            if nMod[isp] === 0
                ArgumentError("sum(nai0) must be unit!!!")
            else
                nai[isp] = nai[isp][isnh]
                uai[isp] = uai[isp][isnh]
                vthi[isp] = vthi[isp][isnh]
            end
        end
    else
        for isp in 1:ns
            for k in 1:nMod[isp]
                if nai[isp][k] ≤ atol_n
                    uai[isp][k] = 0.0
                end
            end
        end
    end
    return nai,uai,vthi,nMod
end

"""
  Outputs:
    uha, Tha = nuTsNormNK(uha,Tha,nai,uai,vthi)
    nuTsNormNK!(uha,Tha,nai,uai,vthi)
"""

# [nMod,ns]
function nuTsNormNK!(uha::AbstractVector{T},Tha::AbstractVector{T},
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},
    vthi::Vector{AbstractVector{T}},nMod::Vector{Int64}) where{T}

    uha[:] = [sum(nai[k][1:nMod[k]] .* uai[k][1:nMod[k]]) for k in 1:ns]
    # T̂a
    Tha[:] = [sum(nai[k][1:nMod[k]] .* vthi[k][1:nMod[k]].^2) + 2/3 * (sum(nai[k][1:nMod[k]] .* uai[k][1:nMod[k]].^2) - uha[k].^2) for k in 1:ns]
    return uha, Tha
end