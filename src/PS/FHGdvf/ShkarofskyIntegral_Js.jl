"""
  Numerical stability method for the Clenshaw-Curtis algorithm on multi-domain:

      1, Absolute stability: increasing the order of shape function: `ocp`

      2, Relative stability: For the section when `vk[i] → 0`, applying the skill to avoid small values with transformations

        `vk1 = (vk / cs) * cs`  where `cs = vk[ocp]`
"""


"""
  Defining the Shkarofsky integrals to compute the Rosenbluth potentials `Hₗᵐ` and `Gₗᵐ`.

    The integral `Jⱼ(Fₗᵐ)`:

      Jₗ₊₁(Fₗᵐ) = 𝓋^(L+1) * ∫𝓋⁹ dvᵦ(vᵦ² / vᵦ^(L+1) * Fₗᵐ(vᵦ)

      Jₗ₋₁(Fₗᵐ) = 𝓋^(L-1) * ∫𝓋⁹ dvᵦ(vᵦ² / vᵦ^(L-1) * Fₗᵐ(vᵦ)
              = 𝓋^(L) * ∫𝓋⁹ dvᵦ(vᵦ² / vᵦ^(L-1) * Fₗᵐ(vᵦ), L ≥ 1

      Jₗ(Fₗᵐ) =  ∫𝓋⁹ dvᵦ(vᵦ³ * Fₗᵐ(vᵦ), L = 0
      Jₗ(Fₗ₋₁ᵐ) = ∫𝓋⁹ dvᵦ(vᵦ * Fₗᵐ(vᵦ), L = 0

    Applying the Clenshaw-Curtis quadrature, the integrals will be:

      Jₗ₊₁(Fₗᵐ(𝓋ₖ[end])) = ∫_𝓋ₖ[end]^⁹ dvᵦ(vᵦ² * (𝓋ₖ[end] / vᵦ)^(L+1) * Fₗᵐ(vᵦ))
                   = - (𝓋ₖ[end] - 𝓋ₖ[end]) / 2 * dot(wcck, ((𝓋ₖ[end] / 𝓋ₖ)^(L+1) * 𝓋ₖ² * Fₗᵐ(𝓋ₖ)))

      Jₗ₋₁(Fₗᵐ(𝓋ₖ[end])) = ∫_𝓋ₖ[end]^⁹ dvᵦ(vᵦ² * (𝓋ₖ[end] / vᵦ)^(L-1) * Fₗᵐ(vᵦ))
                   = - (𝓋ₖ[end] - 𝓋ₖ[end]) / 2 * dot(wcck, ((𝓋ₖ[end] / 𝓋ₖ)^(L-1) * 𝓋ₖ² * Fₗᵐ(𝓋ₖ))), L ≥ 1.
    and

      Jₗ(Fₗᵐ(𝓋ₖ[end]))δₗ⁰ = ∫_𝓋ₖ[end]^⁹ dvᵦ(vᵦ³ * Fₗᵐ(vᵦ))
                   = - (𝓋ₖ[end] - 𝓋ₖ[end]) / 2 * dot(wcck, (𝓋ₖ³ * Fₗᵐ(𝓋ₖ))), L = 0.

  Inputs:
    FL0: The `Lᵗʰ`-order coefficients of normalized distribution functions,
         = F̂ₗ(𝓋̂) = FLn(𝓋̂,ℓ)
    va: denotes `𝓋̂ = vG * vabth`

  Outputs:
      JLFL0n1,JL0FL0 = shkarofskyJL0(JLFL0n1,JL0FL0,FLn,va,nc0,nck,ocp)
      JL1n2FL0,JLn1FL0 = shkarofskyJL1(JL1n2FL0,JLn1FL0,FLn,va,nc0,nck,ocp)
      JLFL0,JLn2FL0 = shkarofskyJL2(JLFL0,JLn2FL0,FLn,va,nvlevel0,nc0,nck,ocp)
      JL1FLm,JLn1FLm = shkarofskyJ(JL1FLm,JLn1FLm,FLn,va,nvlevel0,nc0,nck,ocp,L1)

"""

function shkarofskyJL0(JLFL0n1::AbstractVector{T},JL0FL0::AbstractVector{T},
    FLn::AbstractVector{T},va::AbstractVector{Tb},
    nc0::Int64,nck::Int64,ocp::Int64) where{T<:Real,Tb}

    # μk = chebyshevmoments1(T, ocp)
    wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
    # JLFL0n1 and JL0FL0
    k = nc0
    JLFL0n1[k] = 0.0
    JL0FL0[k] = 0.0
    nk9 = deepcopy(nck)
    nk1 = nk9 - ocp + 1
    vk = va[nk1:nk9]
    Ixvi = - (vk[1] - vk[end]) / 2
    Fvk = vk .* FLn[nk1:nk9]
    JLFL0n1[k-1] = Ixvi * dot(wcck, Fvk)
    JL0FL0[k-1] = Ixvi * dot(wcck, (vk.^2 .* Fvk))
    for k in nc0-1:-1:2
        nk9 = nk1
        nk1 = nk9 - ocp + 1
        vk = va[nk1:nk9]
        Ixvi = - (vk[1] - vk[end]) / 2
        Fvk = vk .* FLn[nk1:nk9]
        JLFL0n1[k-1] = JLFL0n1[k] + Ixvi * dot(wcck, Fvk)
        JL0FL0[k-1] = JL0FL0[k] + Ixvi * dot(wcck, (vk.^2 .* Fvk))
    end
    # # Checking the values of `vha[end]^(2+j) * F(vha[end])` which should be smaller than `epsT`
    # if is_checking_v2jF
    #     va[end].^2 .* Fvk[end] ≤ epsT01 || @warn("JLFLm: The values of `vha[end]^(2+j) * F(vha[end]) > epsT01`, checking the best `vGmax`!")
    # end
    return JLFL0n1, JL0FL0
end

function shkarofskyJL1(JL1n2FL0::AbstractVector{T},JLn1FL0::AbstractVector{T},
    FLn::AbstractVector{T},va::AbstractVector{Tb},
    nc0::Int64,nck::Int64,ocp::Int64) where{T<:Real,Tb}

    # μk = chebyshevmoments1(T, ocp)
    wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
    k = nc0
    JLn1FL0[k] = 0.0
    JL1n2FL0[k] = 0.0
    nk9 = deepcopy(nck)
    nk1 = nk9 - ocp + 1
    vk = va[nk1:nk9]
    Ixvi = - (vk[1] - vk[end]) / 2
    JL1n2FL0[k-1] = Ixvi * dot(wcck, FLn[nk1:nk9])
    JLn1FL0[k-1] = Ixvi * dot(wcck, (vk.^2 .* FLn[nk1:nk9]))
    for k in nc0-1:-1:2
        nk9 = nk1
        nk1 = nk9 - ocp + 1
        vk = va[nk1:nk9]
        Ixvi = - (vk[1] - vk[end]) / 2
        JL1n2FL0[k-1] = JL1n2FL0[k] + Ixvi * dot(wcck, FLn[nk1:nk9])
        JLn1FL0[k-1] = JLn1FL0[k] + Ixvi * dot(wcck, (vk.^2 .* FLn[nk1:nk9]))
    end
    return JL1n2FL0, JLn1FL0
end

function shkarofskyJL2(JLFL0::AbstractVector{T},JLn2FL0::AbstractVector{T},
    FLn::AbstractVector{T},va::AbstractVector{Tb},
    nvlevel0::AbstractVector{Int64},nc0::Int64,nck::Int64,ocp::Int64) where{T<:Real,Tb}

    # μk = chebyshevmoments1(T, ocp)
    wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
    va0 = va[nvlevel0]
    k = nc0
    JLn2FL0[k] = 0.0
    nk9 = deepcopy(nck)
    nk1 = nk9 - ocp + 1
    vk = va[nk1:nk9]
    Ixvi = - (vk[1] - vk[end]) / 2
    JLn2FL0[k-1] = Ixvi * dot(wcck, (vk .* FLn[nk1:nk9]))
    for k in nc0-1:-1:2
        nk9 = nk1
        nk1 = nk9 - ocp + 1
        vk = va[nk1:nk9]
        Ixvi = - (vk[1] - vk[end]) / 2
        JLn2FL0[k-1] = JLn2FL0[k] + Ixvi * dot(wcck, (vk .* FLn[nk1:nk9]))
    end
    # JLFL0
    k = nc0
    JLFL0[k] = 0.0
    # nk = nvlevel[k-1]
    nk9 = deepcopy(nck)
    nk1 = nk9 - ocp + 1
    vk = va[nk1:nk9]
    Ixvi = - (vk[1] - vk[end]) / 2
    vend = va0[k-1]
    JLFL0[k-1] = Ixvi * dot(wcck, ((vend ./ vk) .* FLn[nk1:nk9]))
    for k in nc0-1:-1:2
        i = nc0
        # nk = nvlevel[k-1]
        nk9 = nck
        nk1 = nk9 - ocp + 1
        vk = va[nk1:nk9]
        Ixvi = - (vk[1] - vk[end]) / 2
        vend = va0[k-1]
        JLFL0[k-1] = Ixvi * dot(wcck, ((vend ./ vk) .* FLn[nk1:nk9]))
        for i in nc0-1:-1:k
            # nk = nvlevel[k-1]
            nk9 = nk1
            nk1 = nk9 - ocp + 1
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            JLFL0[k-1] += Ixvi * dot(wcck, ((vend ./ vk) .* FLn[nk1:nk9]))
        end
    end
    if va0[1] == 0.0
        JLFL0[1] = 0.0
    end
    JLFL0 .*= va0
    return JLFL0, JLn2FL0
end

function shkarofskyJ(JL1FLm::AbstractVector{T},JLn1FLm::AbstractVector{T},
    FLn::AbstractVector{T},va::AbstractVector{Tb},nvlevel0::AbstractVector{Int64},
    nc0::Int64,nck::Int64,ocp::Int64,L1::Int64) where{T<:Real,Tb}

    if L1 ≤ 3
        @warn("Please use procedure `shkarofskyJLn.jl, n ∈ [0,1,2]`.")
        # μk = chebyshevmoments1(T, ocp)
        wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
        if L1 == 1
            # JLn1FLm and JL1FLm
            k = nc0
            JLn1FLm[k] = 0.0
            JL1FLm[k] = 0.0
            nk9 = deepcopy(nck)
            nk1 = nk9 - ocp + 1
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            Fvk = vk .* FLn[nk1:nk9]
            JL1FLm[k-1] = Ixvi * dot(wcck, Fvk)
            JLn1FLm[k-1] = Ixvi * dot(wcck, (vk.^2 .* Fvk))
            for k in nc0-1:-1:2
                nk9 = nk1
                nk1 = nk9 - ocp + 1
                vk = va[nk1:nk9]
                Ixvi = - (vk[1] - vk[end]) / 2
                Fvk = vk .* FLn[nk1:nk9]
                JL1FLm[k-1] = JL1FLm[k] + Ixvi * dot(wcck, Fvk)
                JLn1FLm[k-1] = JLn1FLm[k] + Ixvi * dot(wcck, (vk.^2 .* Fvk))
            end
            JL1FLm .*= va[nvlevel0]
            return JL1FLm, JLn1FLm
        elseif L1 == 2
            k = nc0
            JLn1FLm[k] = 0.0
            JL1FLm[k] = 0.0
            nk9 = deepcopy(nck)
            nk1 = nk9 - ocp + 1
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            JL1FLm[k-1] = Ixvi * dot(wcck, FLn[nk1:nk9])
            JLn1FLm[k-1] = Ixvi * dot(wcck, (vk.^2 .* FLn[nk1:nk9]))
            for k in nc0-1:-1:2
                nk9 = nk1
                nk1 = nk9 - ocp + 1
                vk = va[nk1:nk9]
                Ixvi = - (vk[1] - vk[end]) / 2
                JL1FLm[k-1] = JL1FLm[k] + Ixvi * dot(wcck, FLn[nk1:nk9])
                JLn1FLm[k-1] = JLn1FLm[k] + Ixvi * dot(wcck, (vk.^2 .* FLn[nk1:nk9]))
            end
            JL1FLm .*= va[nvlevel0].^L1
            return JL1FLm, JLn1FLm
        elseif L1 == 3
            k = nc0
            JLn1FLm[k] = 0.0
            JL1FLm[k] = 0.0
            nk9 = deepcopy(nck)
            nk1 = nk9 - ocp + 1
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            JL1FLm[k-1] = Ixvi * dot(wcck, FLn[nk1:nk9] ./ vk)
            JLn1FLm[k-1] = Ixvi * dot(wcck, (vk .* FLn[nk1:nk9]))
            for k in nc0-1:-1:2
                nk9 = nk1
                nk1 = nk9 - ocp + 1
                vk = va[nk1:nk9]
                Ixvi = - (vk[1] - vk[end]) / 2
                JL1FLm[k-1] = JL1FLm[k] + Ixvi * dot(wcck, FLn[nk1:nk9] ./ vk)
                JLn1FLm[k-1] = JLn1FLm[k] + Ixvi * dot(wcck, (vk .* FLn[nk1:nk9]))
            end
            va0 = va[nvlevel0]
            JL1FLm .*= va0.^L1
            va[1] == 0.0 ? JL1FLm[1] = 0.0 : nothing
            JLn1FLm .*= va0
            return JL1FLm, JLn1FLm
        end
    else
        # μk = chebyshevmoments1(T, ocp)
        wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
        va0 = va[nvlevel0]
        if L1 == 4
            k = nc0
            JLn1FLm[k] = 0.0
            JL1FLm[k] = 0.0
            nk9 = deepcopy(nck)
            nk1 = nk9 - ocp + 1
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            JL1FLm[k-1] = Ixvi * dot(wcck, FLn[nk1:nk9] ./ vk.^2)
            JLn1FLm[k-1] = Ixvi * dot(wcck, (FLn[nk1:nk9]))
            for k in nc0-1:-1:2
                nk9 = nk1
                nk1 = nk9 - ocp + 1
                vk = va[nk1:nk9]
                Ixvi = - (vk[1] - vk[end]) / 2
                JL1FLm[k-1] = JL1FLm[k] + Ixvi * dot(wcck, FLn[nk1:nk9] ./ vk.^2)
                JLn1FLm[k-1] = JLn1FLm[k] + Ixvi * dot(wcck, (FLn[nk1:nk9]))
            end
            if va[1] == 0.0
                JL1FLm[1], JLn1FLm[1] = 0.0, 0.0
            end
            JL1FLm .*= va0.^L1
            JLn1FLm .*= va0.^(L1-2)
            return JL1FLm, JLn1FLm
        elseif L1 == 5
            k = nc0
            JLn1FLm[k] = 0.0
            JL1FLm[k] = 0.0
            nk9 = deepcopy(nck)
            nk1 = nk9 - ocp + 1
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            JL1FLm[k-1] = Ixvi * dot(wcck, FLn[nk1:nk9] ./ vk.^3)
            JLn1FLm[k-1] = Ixvi * dot(wcck, (FLn[nk1:nk9] ./ vk))
            for k in nc0-1:-1:2
                nk9 = nk1
                nk1 = nk9 - ocp + 1
                vk = va[nk1:nk9]
                Ixvi = - (vk[1] - vk[end]) / 2
                JL1FLm[k-1] = JL1FLm[k] + Ixvi * dot(wcck, FLn[nk1:nk9] ./ vk.^3)
                JLn1FLm[k-1] = JLn1FLm[k] + Ixvi * dot(wcck, (FLn[nk1:nk9] ./ vk))
            end
            if va[1] == 0.0
                JL1FLm[1], JLn1FLm[1] = 0.0, 0.0
            end
            JL1FLm .*= va0.^L1
            JLn1FLm .*= va0.^(L1-2)
            return JL1FLm, JLn1FLm
        else
            # The right boundary point: `H₀⁰(∞) = Hₗᵐ(va[nck]) = 0`
            k = nc0
            JL1FLm[k] = 0.0
            JLn1FLm[k] = 0.0
            nk9 = deepcopy(nck)
            nk1 = nk9 - ocp + 1
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            vend = va0[k-1]
            FLnkL2 = (vend ./ vk).^(L1-4) .* FLn[nk1:nk9]
            JLn1FLm[k-1] = Ixvi * dot(wcck, FLnkL2)
            JL1FLm[k-1] = Ixvi * dot(wcck, ((vend ./ vk).^2 .* FLnkL2))
            for k in nc0-1:-1:2
                i = nc0
                nk9 = nck
                nk1 = nk9 - ocp + 1
                vk = va[nk1:nk9]
                Ixvi = - (vk[1] - vk[end]) / 2
                vend = va0[k-1]
                FLnkL2 = (vend ./ vk).^(L1-4) .* FLn[nk1:nk9]
                JLn1FLm[k-1] = Ixvi * dot(wcck, FLnkL2)
                JL1FLm[k-1] = Ixvi * dot(wcck, ((vend ./ vk).^2 .* FLnkL2))
                for i in nc0-1:-1:k
                    nk9 = nk1
                    nk1 = nk9 - ocp + 1
                    vk = va[nk1:nk9]
                    Ixvi = - (vk[1] - vk[end]) / 2
                    FLnkL2 = (vend ./ vk).^(L1-4) .* FLn[nk1:nk9]
                    JLn1FLm[k-1] += Ixvi * dot(wcck, FLnkL2)
                    JL1FLm[k-1] += Ixvi * dot(wcck, ((vend ./ vk).^2 .* FLnkL2))
                end
            end
            if va0[1] == 0.0
                JL1FLm[1], JLn1FLm[1] = 0.0, 0.0
            end
            JLn1FLm .*= va0.^2
            JL1FLm .*= va0.^2
            return JL1FLm, JLn1FLm
        end
    end
end

# The theoretical values
function shkarofskyIJt(ILFL0::AbstractVector{T},IL2FL0::AbstractVector{T},JL1FL0::AbstractVector{T},
    JLn1FL0::AbstractVector{T},v::AbstractVector{Tb},u::Float64,L1::Int64) where{T<:Real,Tb}

    v2 = v.^2
    if L1 == 1
        if u == 0.0
            fM = exp.(-v2)
            ILFL0 = - 0.5v .* fM + sqrtpi / 4 * erf.(v) |> Vector{T}
            IL2FL0 = - (3/4 ./ v + 0.5v) .* fM + 3/8 * sqrtpi ./ v2 .* erf.(v) |> Vector{T}
            JL1FL0 = v/2 .* fM |> Vector{T}
            JLn1FL0 = 0.5(v.^2 .+ 1.0) .* fM  |> Vector{T}
            # `v → 0`
            IL2FL0v0(v) = v.^3 / 5 - v.^5 / 7 + v.^7 / 18 - v.^9 / 66 + v.^11 / 312 -
                          v.^13 / 1800 + (v.^15 /12240) - v.^17 / 95760 + v.^19/846720
            count0 = 0
            for vv in v
                vv < 0.03 ? count0 += 1 : break
            end
            if count0 > 0
                IL2FL0[1:count0] = IL2FL0v0(v[1:count0])
            end
            return ILFL0, IL2FL0, JL1FL0, JLn1FL0
        else
            expuvp = exp.(-(u .+ v).^2)
            expuvn = exp.(-(u .- v).^2)
            expp = exp.((u .+ v).^2)
            erfn = sqrtpi * erf.(u .- v)
            erfp = sqrtpi * erf.(u .+ v)
            u2 = u^2
            ILFL0 = (-erfn + erfp) / 8.0 + (expuvp - expuvn) / 8u
            IL2FL0 = (3.0+2.0u2) / 16 .* (- erfn + erfp)
            IL2FL0 += ((1.0+u2 .- u*v+v2) .* expuvp - (1.0+u2 .+ u*v+v2) .* expuvn) / 8u
            IL2FL0 ./= v2
            JL1FL0 = v .* (erfn + erfp) / 8u
            JLn1FL0 =  (1 + 2u2) / 16u .* (erfn + erfp) + ((u.-v).* expuvp + (u.+v) .* expuvn) / 8u

            return ILFL0, IL2FL0, JL1FL0, JLn1FL0
        end
    else
        expuvp = exp.(-(u .+ v).^2)
        expuvn = exp.(-(u .- v).^2)
        expp = exp.((u .+ v).^2)
        erfn = sqrtpi * erf.(u .- v)
        erfp = sqrtpi * erf.(u .+ v)
        u2 = u^2
        if L1 == 2
            ILFL0 = (-1.0 /u .+ (1.0 - 1.0 / 2.0u2) ./ v) .*expuvp
            ILFL0 += (-1.0 /u .- (1.0 - 1.0 / 2.0u2) ./ v).*expuvn
            ILFL0 += (-u ./ v) .* erfn
            ILFL0 += (u ./ v) .* erfp
            ILFL0 *= (3 / 8)

            IL2FL0 = (-2.0 + 1.0 /(2u2) - u2 .+ (1/u + u)* v + (-1.0 + 1.0/(2u2)) * v2 + v.^3/u) .*expuvp
            IL2FL0 +=(2.0 - 1.0 /(2u2) + u2 .+ (1/u + u)* v - (-1.0 + 1.0/(2u2)) * v2 + v.^3/u).*expuvn
            IL2FL0 += (5/2 * u + u^3) .* erfn
            IL2FL0 += - (5/2 * u + u^3) .* erfp
            IL2FL0 .*= (- 3/8 * v.^-3)

            JL1FL0 = (expuvp - expuvn) .* v + v2 .* (erfn + erfp)
            JL1FL0 .*= (3 / 8 / u2)

            JLn1FL0 = 3/8 / u * (expuvp + expuvn)
            JLn1FL0 += 3/8 * (1.0 - 1.0 / 2u2) * (erfn + erfp)

            return ILFL0, IL2FL0, JL1FL0, JLn1FL0
        elseif L1 == 3
            u3 = u2 * u
            ILFL0 = (1/u .+ (3/4/u3 - 1/2/u + u)./v2 + (-1.0 + 3/2/u2)./v) .*expuvp
            ILFL0 -= (1/u .+ (3/4/u3 .- 1/2/u + u)./v2 - (-1 + 3/2/u2)./v).*expuvn
            ILFL0 += (u2 ./ v2) .* (erfp - erfn)
            ILFL0 *= (5 / 8)

            IL2FL0 = (-(3.0/4.0/u3+0.5/u+u).+(-3/4/u3+1.0/u-3u-u3)./v2+(2.0-3/2/u2+u2)./v+(1.0-3/2/u2)*v-v2/u).*expuvp
            IL2FL0 += ((3.0/4.0/u3+0.5/u+u).-(-3/4/u3+1.0/u-3u-u3)./v2+(2.0-3/2/u2+u2)./v+(1.0-3/2/u2)*v+v2/u).*expuvn
            IL2FL0 += ((0.5u2 * (7 + 2u2)) .* erfn - (0.5u2 * (7 + 2u2)) .* erfp) ./ v2
            IL2FL0 .*= (- 5/8 ./ v2)

            JL1FL0 = ((- 1.0 .+ 2.0(-u * v + v2)) .* expuvp + (1.0 .- 2.0(u * v + v2)) .* expuvn)
            JL1FL0 += (erfn + erfp) .* 2.0v.^3
            JL1FL0 .*= (5 / 16 / u3)

            JLn1FL0 = 15/16 / u3 * ( expuvn - expuvp)
            JLn1FL0 += 5/16 * (2.0 /u - 3.0 / u3) * v .* (erfn + erfp)

            return ILFL0, IL2FL0, JL1FL0, JLn1FL0
        else
        end
    end
end
