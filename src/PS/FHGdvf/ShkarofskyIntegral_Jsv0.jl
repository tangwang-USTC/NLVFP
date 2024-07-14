

"""
  Boundaries of Shkarofsky integrals to compute the Rosenbluth potentials `Hₗᵐ` and `Gₗᵐ`.

  Inputs:
    FL0: The `Lᵗʰ`-order coefficients of normalized distribution functions,
         = F̂ₗ(𝓋̂) = FLn(𝓋̂,ℓ)
    va: denotes `𝓋̂ = vG * vabth`

  Outputs:
    JLFL0n1bc, JL0FL0bc = shkarofskyJL0(uh)
    JL1n2FL0bc,JLn1FL0bc = shkarofskyJL1(uh)
    JLFL0bc,JLn2FL0bc = shkarofskyJL2(uh)
"""
# ns = 1, nMod = 1
function shkarofskyJL0(uh::T) where{T<:Real}

    if uh ≥ 1e-5
        JLFL0n1 = sqrtpi / 4uh * erf(uh)
        JL0FL0 = sqrtpi / 4 * (1 / 2uh + uh) * erf(uh) + 0.25 * exp(-uh^2)
    else
        uh2 = uh^2
        JLFL0n1 = 0.5 - uh2 / 6 + uh2^2 / 20
        JL0FL0 = 0.5 + uh2 / 6 - uh2^2 / 60
    end
    return JLFL0n1, JL0FL0
end

function shkarofskyJL1(uh::T) where{T<:Real}

    if uh ≥ 1e-5
        JL1n2FL0 =  (1.5 / uh) * (sqrtpi / 2uh * erf(uh) - exp(-uh^2))
        JLn1FL0 =  (0.75 / uh) * (sqrtpi / 2 * (2uh - 1/uh) * erf(uh) + exp(-uh^2))
    else
        uh2 = uh^2
        JL1n2FL0 = uh * (1.0 - 0.6 * uh2 + 3/14 * uh2^2)
        JLn1FL0 = uh * (1.0 - 0.2 * uh2 + 3/70 * uh2^2)
    end
    return JL1n2FL0, JLn1FL0
end

function shkarofskyJL2(uh::T) where{T<:Real}

    if uh ≥ 1e-5
        JLn2FL0 = (1.25 / uh) * (sqrtpi / 2 * (2.0 - 3.0  / uh^2) * erf(uh) + 3 / uh * exp(-uh^2))
    else
        uh2 = uh^2
        JLn2FL0 = uh2 * (2/3 - 2/7 * uh2 + 5/63 * uh2^2)
    end
    # JLFL0 = 0.0
    return 0.0, JLn2FL0
end

"""
"""

# [nMod]
function shkarofskyJL0(nh::AbstractVector{T},uh::AbstractVector{T},vhth::AbstractVector{T},nMod::Int64) where{T<:Real}

    JLFL0n1, JL0FL0 = shkarofskyJL0(uh / vhth)
    return JLFL0n1, JL0FL0
end

function shkarofskyJL1(uh::T) where{T<:Real}

    if uh ≥ 1e-5
        JL1n2FL0 =  (1.5 / uh) * (sqrtpi / 2uh * erf(uh) - exp(-uh^2))
        JLn1FL0 =  (0.75 / uh) * (sqrtpi / 2 * (2uh - 1/uh) * erf(uh) + exp(-uh^2))
    else
        uh2 = uh^2
        JL1n2FL0 = uh * (1.0 - 0.6 * uh2 + 3/14 * uh2^2)
        JLn1FL0 = uh * (1.0 - 0.2 * uh2 + 3/70 * uh2^2)
    end
    return JL1n2FL0, JLn1FL0
end

function shkarofskyJL2(uh::T) where{T<:Real}

    if uh ≥ 1e-5
        JLn2FL0 = (1.25 / uh) * (sqrtpi / 2 * (2.0 - 3.0  / uh^2) * erf(uh) + 3 / uh * exp(-uh^2))
    else
        uh2 = uh^2
        JLn2FL0 = uh2 * (2/3 - 2/7 * uh2 + 5/63 * uh2^2)
    end
    # JLFL0 = 0.0
    return 0.0, JLn2FL0
end
