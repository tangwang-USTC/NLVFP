"""
  Numerical stability method for the Clenshaw-Curtis algorithm on multi-domain:

      1, Absolute stability: increasing the order of shape function: `ocp`

      2, Relative stability: For the section when `vk[i] → 0`, applying the skill to avoid small values with transformations

        `vk1 = (vk / cs) * cs`  where `cs = vk[ocp]`
"""


"""
  Defining the Shkarofsky integrals to compute the Rosenbluth potentials `Hₗᵐ` and `Gₗᵐ`.

    The integral `Iⱼ(Fₗᵐ)`:

      Iₗ(Fₗᵐ)   = 1 / 𝓋̂ ^(L+0) * ∫₀^𝓋̂ dvᵦ(vᵦ^(L+0) * vᵦ² * Fₗᵐ(vᵦ))
      Iₗ₊₂(Fₗᵐ) = 1 / 𝓋̂ ^(L+2) * ∫₀^𝓋̂ dvᵦ(vᵦ^(L+2) * vᵦ² * Fₗᵐ(vᵦ))

    where

      𝓋̂ = v / vᵦₜₕ / vₜₕ,

    `vᵦₜₕ` is the effective thermal velocity of `F(𝐯)`

    and the angular coefficient of spherical coodinate, `4π`, is not included in the shkarofsky integrals.

    Applying the Clenshaw-Curtis quadrature, the integrals will be:

      Iₗ₊₂(Fₗᵐ(𝓋̂ₖ[end])) = ∫₀^𝓋̂ₖ[end] dvᵦ((vᵦ / 𝓋̂ₖ[end])^(L+2) * vᵦ² * Fₗᵐ(vᵦ))
                   = - (0 - 𝓋̂ₖ[end]) / 2 * dot(wcck, ((𝓋̂ₖ / 𝓋̂ₖ[end])^(L+2) * 𝓋̂ₖ² * Fₗᵐ(𝓋̂ₖ)))

  Inputs:
    FL0: The `Lᵗʰ`-order coefficients of normalized distribution functions,
         = F̂ₗ(𝓋̂) = FLn(𝓋̂,ℓ)
    va: denotes `𝓋̂ = vG * vabth`
    Ixva = Ixv / vabth
          = 2 / (𝓋̂[1] - 𝓋̂[end])

  Outputs:
    ILn1FL0,IL1FL0 = shkarofskyIL0(ILn1FL0,IL1FL0,FLn,va,nvlevel0,nc0,ocp)
    ILFL0,IL2FL0 = va0 .* ILn1FL0, va0 .* IL1FL0
    ILn2FL0,IL2FL0 = shkarofskyIL1(ILn2FL0,IL2FL0,FLn,va,nvlevel0,nc0,ocp)
    ILn1FL0,IL1FL0 = shkarofskyIL2(ILn1FL0,IL1FL0,FLn,va,nvlevel0,nc0,ocp)
    ILFLm,IL2FLm = shkarofskyI(ILFLm,IL2FLm,FLn,va,nvlevel0,nc0,ocp,L1)

"""

function shkarofskyIL0(ILn1FL0::AbstractVector{T},IL1FL0::AbstractVector{T},
    FLn::AbstractVector{T},va::AbstractVector{Tb},
    nvlevel0::AbstractVector{Int64},nc0::Int64,ocp::Int64) where{T<:Real,Tb}

    # μk = chebyshevmoments1(T, ocp)
    wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))    # ILFLm
    va0 = va[nvlevel0]
    # IL1FL0 and ILn1FL0
    k = 1
    IL1FL0[k] = 0.0
    ILn1FL0[k] = 0.0
    vend = va0[k+1]
    nk1 = 1
    nk9 = ocp
    vk = va[nk1:nk9]
    Ixvi = - (vk[1] - vk[end]) / 2
    FLnvk = (vk / vend) .* vk .* FLn[nk1:nk9]
    ILn1FL0[k+1] = Ixvi * dot(wcck, FLnvk)
    IL1FL0[k+1] = Ixvi * dot(wcck, ((vk / vend).^2 .* FLnvk))
    for k in 2:nc0-1
        vend = va0[k+1]
        i = 1
        nk1 = 1
        nk9 = ocp
        vk = va[nk1:nk9]
        Ixvi = - (vk[1] - vk[end]) / 2
        FLnvk = (vk / vend) .* vk .* FLn[nk1:nk9]
        ILn1FL0[k+1] = Ixvi * dot(wcck, FLnvk)
        IL1FL0[k+1] = Ixvi * dot(wcck, ((vk / vend).^2 .* FLnvk))
        for i in 2:k
            nk1 = nk9
            nk9 = nk1 + ocp - 1
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            FLnvk = (vk / vend) .* vk .* FLn[nk1:nk9]
            ILn1FL0[k+1] += Ixvi * dot(wcck, FLnvk)
            IL1FL0[k+1] += Ixvi * dot(wcck, ((vk / vend).^2 .* FLnvk))
        end
    end
    return ILn1FL0,IL1FL0
end

function shkarofskyIL1(ILn2FL0::AbstractVector{T},IL2FL0::AbstractVector{T},
    FLn::AbstractVector{T},va::AbstractVector{Tb},
    nvlevel0::AbstractVector{Int64},nc0::Int64,ocp::Int64) where{T<:Real,Tb}

    # μk = chebyshevmoments1(T, ocp)
    wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
    va0 = va[nvlevel0]
    k = 1
    ILn2FL0[k] = 0.0
    IL2FL0[k] = 0.0
    vend = va0[k+1]
    nk1 = 1
    nk9 = ocp
    vk = va[nk1:nk9]
    Ixvi = - (vk[1] - vk[end]) / 2
    FLnvk = (vk / vend).^3 .* FLn[nk1:nk9]
    ILn2FL0[k+1] = Ixvi * dot(wcck, FLnvk)
    IL2FL0[k+1] = Ixvi * dot(wcck, (vk.^2 .* FLnvk))
    for k in 2:nc0-1
        vend = va0[k+1]
        i = 1
        nk1 = 1
        nk9 = ocp
        vk = va[nk1:nk9]
        Ixvi = - (vk[1] - vk[end]) / 2
        FLnvk = (vk / vend).^3 .* FLn[nk1:nk9]
        ILn2FL0[k+1] = Ixvi * dot(wcck, FLnvk)
        IL2FL0[k+1] = Ixvi * dot(wcck, (vk.^2 .* FLnvk))
        for i in 2:k
            nk1 = nk9
            nk9 = nk1 + ocp - 1
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            FLnvk = (vk / vend).^3 .* FLn[nk1:nk9]
            ILn2FL0[k+1] += Ixvi * dot(wcck, FLnvk)
            IL2FL0[k+1] += Ixvi * dot(wcck, (vk.^2 .* FLnvk))
        end
    end
    return ILn2FL0,IL2FL0
end

function shkarofskyIL2(ILn1FL0::AbstractVector{T},IL1FL0::AbstractVector{T},FLn::AbstractVector{T},
    va::AbstractVector{Tb},nvlevel0::AbstractVector{Int64},nc0::Int64,ocp::Int64) where{T<:Real,Tb}

    # μk = chebyshevmoments1(T, ocp)
    wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
    va0 = va[nvlevel0]
    k = 1
    ILn1FL0[k] = 0.0
    IL1FL0[k] = 0.0
    vend = va0[k+1]
    nk1 = 1
    nk9 = ocp
    vk = va[nk1:nk9]
    Ixvi = - (vk[1] - vk[end]) / 2
    FLnvk = (vk / vend).^3 .* vk .* FLn[nk1:nk9]
    ILn1FL0[k+1] = Ixvi * dot(wcck, FLnvk)
    IL1FL0[k+1] = Ixvi * dot(wcck, ((vk / vend).^2 .* FLnvk))
    for k in 2:nc0-1
        vend = va0[k+1]
        i = 1
        nk1 = 1
        nk9 = ocp
        vk = va[nk1:nk9]
        Ixvi = - (vk[1] - vk[end]) / 2
        FLnvk = (vk / vend).^3 .* vk .* FLn[nk1:nk9]
        ILn1FL0[k+1] = Ixvi * dot(wcck, FLnvk)
        IL1FL0[k+1] = Ixvi * dot(wcck, ((vk / vend).^2 .* FLnvk))
        for i in 2:k
            nk1 = nk9
            nk9 = nk1 + ocp - 1
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            FLnvk = (vk / vend).^3 .* vk .* FLn[nk1:nk9]
            ILn1FL0[k+1] += Ixvi * dot(wcck, FLnvk)
            IL1FL0[k+1] += Ixvi * dot(wcck, ((vk / vend).^2 .* FLnvk))
        end
    end
    return ILn1FL0,IL1FL0
end

function shkarofskyI(ILFLm::AbstractVector{T},IL2FLm::AbstractVector{T},FLn::AbstractVector{T},
    va::AbstractVector{Tb},nvlevel0::AbstractVector{Int64},nc0::Int64,ocp::Int64,L1::Int64) where{T<:Real,Tb}

    va0 = va[nvlevel0]
    if L1 ≤ 3
        @warn("Please use procedure `shkarofskyILn.jl, n ∈ [0,1,2]`.")
        FLnv2 = va .^ 2 .* FLn
        # μk = chebyshevmoments1(T, ocp)
        wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
        if L1 == 1
            # ILFLm
            k = 1
            ILFLm[k] = 0.0
            nk1 = 1
            nk9 = ocp
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            ILFLm[k+1] = Ixvi * dot(wcck, FLnv2[nk1:nk9])  #  .* vk.^(L=0)
            for k in 2:nc0-1
                nk1 = nk9
                nk9 = nk1 + ocp - 1
                vk = va[nk1:nk9]
                Ixvi = - (vk[1] - vk[end]) / 2
                ILFLm[k+1] = ILFLm[k] + Ixvi * dot(wcck, FLnv2[nk1:nk9])
            end
            # IL2FLm
            k = 1
            IL2FLm[k] = 0.0
            vend = va0[k+1]
            nk1 = 1
            nk9 = ocp
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            IL2FLm[k+1] = Ixvi * dot(wcck, ((vk / vend).^2 .* FLnv2[nk1:nk9]))
            for k in 2:nc0-1
                vend = va0[k+1]
                i = 1
                nk1 = 1
                nk9 = ocp
                vk = va[nk1:nk9]
                Ixvi = - (vk[1] - vk[end]) / 2
                IL2FLm[k+1] = Ixvi * dot(wcck, ((vk / vend).^2 .* FLnv2[nk1:nk9]))
                for i in 2:k
                    nk1 = nk9
                    nk9 = nk1 + ocp - 1
                    vk = va[nk1:nk9]
                    Ixvi = - (vk[1] - vk[end]) / 2
                    IL2FLm[k+1] += Ixvi * dot(wcck, ((vk / vend).^2 .* FLnv2[nk1:nk9]))
                end
            end
            return ILFLm, IL2FLm
        elseif L1 == 2
            k = 1
            ILFLm[k] = 0.0
            IL2FLm[k] = 0.0
            vend = va0[k+1]
            nk1 = 1
            nk9 = ocp
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            ILFLm[k+1] = Ixvi * dot(wcck, ((vk / vend) .* FLnv2[nk1:nk9]))
            IL2FLm[k+1] = Ixvi * dot(wcck, ((vk / vend).^(L1 + 1) .* FLnv2[nk1:nk9]))
            for k in 2:nc0-1
                vend = va0[k+1]
                i = 1
                nk1 = 1
                nk9 = ocp
                vk = va[nk1:nk9]
                Ixvi = - (vk[1] - vk[end]) / 2
                ILFLm[k+1] = Ixvi * dot(wcck, ((vk / vend) .* FLnv2[nk1:nk9]))
                IL2FLm[k+1] = Ixvi * dot(wcck, ((vk / vend).^(L1 + 1) .* FLnv2[nk1:nk9]))
                for i in 2:k
                    nk1 = nk9
                    nk9 = nk1 + ocp - 1
                    vk = va[nk1:nk9]
                    Ixvi = - (vk[1] - vk[end]) / 2
                    ILFLm[k+1] += Ixvi * dot(wcck, ((vk / vend) .* FLnv2[nk1:nk9]))
                    IL2FLm[k+1] += Ixvi * dot(wcck, ((vk / vend).^(L1 + 1) .* FLnv2[nk1:nk9]))
                end
            end
            return ILFLm, IL2FLm
        elseif L1 == 3
            k = 1
            ILFLm[k] = 0.0
            IL2FLm[k] = 0.0
            vend = va0[k+1]
            nk1 = 1
            nk9 = ocp
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            ILFLm[k+1] = Ixvi * dot(wcck, ((vk / vend).^(L1 - 1) .* FLnv2[nk1:nk9]))
            IL2FLm[k+1] = Ixvi * dot(wcck, ((vk / vend).^(L1 + 1) .* FLnv2[nk1:nk9]))
            for k in 2:nc0-1
                vend = va0[k+1]
                i = 1
                nk1 = 1
                nk9 = ocp
                vk = va[nk1:nk9]
                Ixvi = - (vk[1] - vk[end]) / 2
                ILFLm[k+1] = Ixvi * dot(wcck, ((vk / vend).^(L1 - 1) .* FLnv2[nk1:nk9]))
                IL2FLm[k+1] = Ixvi * dot(wcck, ((vk / vend).^(L1 + 1) .* FLnv2[nk1:nk9]))
                for i in 2:k
                    nk1 = nk9
                    nk9 = nk1 + ocp - 1
                    vk = va[nk1:nk9]
                    Ixvi = - (vk[1] - vk[end]) / 2
                    ILFLm[k+1] += Ixvi * dot(wcck, ((vk / vend).^(L1 - 1) .* FLnv2[nk1:nk9]))
                    IL2FLm[k+1] += Ixvi * dot(wcck, ((vk / vend).^(L1 + 1) .* FLnv2[nk1:nk9]))
                end
            end
            return ILFLm, IL2FLm
        end
    else
        # μk = chebyshevmoments1(T, ocp)
        wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
        k = 1
        IL2FLm[k] = 0.0
        vend = va0[k+1]
        nk1 = 1
        nk9 = ocp
        vk = va[nk1:nk9]
        Ixvi = - (vk[1] - vk[end]) / 2
        FLnv2 = (vk / vend).^(L1 - 1) .* va[nk1:nk9] .^ 2 .* FLn[nk1:nk9]
        ILFLm[k+1] = Ixvi * dot(wcck, FLnv2)
        IL2FLm[k+1] = Ixvi * dot(wcck, ((vk / vend).^2 .* FLnv2))
        for k in 2:nc0-1
            vend = va0[k+1]
            i = 1
            nk1 = 1
            nk9 = ocp
            vk = va[nk1:nk9]
            Ixvi = - (vk[1] - vk[end]) / 2
            FLnv2 = (vk / vend).^(L1 - 1) .* va[nk1:nk9] .^ 2 .* FLn[nk1:nk9]
            ILFLm[k+1] = Ixvi * dot(wcck, FLnv2)
            IL2FLm[k+1] = Ixvi * dot(wcck, ((vk / vend).^2 .* FLnv2))
            for i in 2:k
                nk1 = nk9
                nk9 = nk1 + ocp - 1
                vk = va[nk1:nk9]
                Ixvi = - (vk[1] - vk[end]) / 2
                FLnv2 = (vk / vend).^(L1 - 1) .* va[nk1:nk9] .^ 2 .* FLn[nk1:nk9]
                ILFLm[k+1] += Ixvi * dot(wcck, FLnv2)
                IL2FLm[k+1] += Ixvi * dot(wcck, ((vk / vend).^2 .* FLnv2))
            end
        end
        return ILFLm, IL2FLm
    end
end

