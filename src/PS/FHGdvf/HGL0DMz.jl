
"""

  Inputs:
    u: the normalized velocity `uai = ua[isp] / vₜₕ[isp]`

  Outputs:
    HLnDM, dHLnDM, ddHLnDM, GLnDM, dGLnDM, ddGLnDM = HGL0DMabz(v,uai;L1=L1)

"""

# 1D, [v] `z` direction when

function HGL0DMabz(v::AbstractVector{T},u::T;L1::Int64=1) where{T<:Real}

    if L1 == 1 && u == 0.0
      erfv = erf.(v)
      expv2 = exp.(- v.^2)
      HLnDM =  sqrtpi / 4 * erfv ./ v
      dHLnDM = expv2 ./ 2v -  sqrtpi / 4 ./ v.^2 .* erfv
      ddHLnDM = - (1 .+ 1 ./ v.^2) .* expv2  + sqrtpi / 2 ./ v.^3 .* erfv
      GLnDM = 0.25expv2 +  (sqrtpi / 4 * v + (sqrtpi / 8) ./ v) .* erfv
      dGLnDM = expv2 ./ 4v + ( sqrtpi / 4 .- sqrtpi / 8 ./ v.^2) .* erfv
      ddGLnDM = - 0.5 ./ v.^2 .* expv2 + sqrtpi / 4 * erfv ./ v.^3
      return HLnDM, dHLnDM, ddHLnDM, GLnDM, dGLnDM, ddGLnDM
    else
      xi = 2u * v
      expuvp = exp.(-(u .+ v).^2)
      expuvn = exp.(-(u .- v).^2)
      expp = exp.((u .+ v).^2)
      erfn = sqrtpi * erf.(u .- v)
      erfp = sqrtpi * erf.(u .+ v)
      if L1 == 1
        u2 = u^2
        v2 = v.^2
        v3 = v2 .* v
        H = (expuvp - expuvn) ./ (u * v)
        H += (1/u .- 1 ./v) .* erfn + (1/u .+ 1 ./v) .* erfp
        H /= 8
        G = (2 .+ (1/u + u)./v + v/u).*expuvp
        G += (2 .- (1/u + u)./v - v/u).*expuvn
        G += (3/(2u) + 3u .+ (-(3/2) - u^2)./v - 3v + v.^2/u) .* erfn
        G += (3/(2u) + 3u .+ ((3/2) + u^2)./v + 3v + v.^2/u) .* erfp
        G /= 24
        dH = ((-expuvp + expuvn) ./ u + (erfn - erfp)) ./ 8v.^2
        dG = (2/u .- (1/u + u)./v.^2 + 1 ./v).*expuvp
        dG += (-2/u .+ (1/u + u)./v.^2 + 1 ./v).*expuvn
        dG += (-3 .+ (3/2 + u^2)./v.^2 + 2/u * v) .* erfn
        dG += (3 .- (3/2 + u^2)./v.^2 + 2/u * v) .* erfp
        dG /= 24
        ddH = ((expuvp - expuvn) .* (1.0 .+ v.^2)/u  + (erfp - erfn)) ./ 4v.^3
        ddG = (-1 .+ (1/u + u)./v + v/u).*expuvp
        ddG += (-1 .- (1/u + u)./v - v/u).*expuvn
        ddG += (-(3/2 + u^2)./v + v.^2/u) .* erfn
        ddG += ((3/2 + u^2)./v + v.^2/u) .* erfp
        ddG ./= 12v.^2
        return H, dH, ddH, G, dG, ddG
      elseif L1 == 2
        u2 = u^2
        v2 = v.^2
        v3 = v2 .* v
        H = (1/u2 .+ (1 - 1/(2u2))./v2 - 1/u ./ v).*expuvp
        H -= (1/u2 .+ (1 - 1/(2u2))./v2 + 1/u ./ v).*expuvn
        H += (-u./v2 + v/u2) .* erfn
        H += (u./v2 + v/u2) .* erfp
        H /= 8
        G = -(4 - 2/u2 .+ (-2 + 1/(2u2) - u2)./v2 + (1/u + u)./v + v/u - v2/u2).*expuvp
        G += (4 - 2/u2 .+ (-2 + 1/(2u2) - u2)./v2 - (1/u + u)./v - v/u - v2/u2).*expuvn
        G += (5u .- (u * (5/2 + u2))./v2 + (-5 + 5/(2u2))* v + v3/u2) .* erfn
        G -= (5u .- (u * (5/2 + u2))./v2 + (5 - 5/(2u2))* v - v3/u2) .* erfp
        G /= 40
        dH = -((2 - 1/u2)./v3 - 2/u ./v2 - 1/u2 ./v).*expuvp
        dH += ((2 - 1/u2)./v3 + 2/u ./v2 - 1/u2 ./v).*expuvn
        dH += (1/u2 .+ (2u)./v3) .* erfn
        dH += (1/u2 .- (2u)./v3) .* erfp
        dH /= 8
        dG = (-(3/(2u)).+(-2+1/(2u2)-u2)./v3+(1/u+u)./v2+(-1+1/(2u2))./v+3/(2u2) * v).*expuvp
        dG += (-(3/(2u)) .-(-2+1/(2u2)-u2)./v3+(1/u+u)./v2-(-1+1/(2u2))./v-3/(2u2) * v).*expuvn
        dG += (-5/2+5/(4u2).+(u*(5/2+u2))./v3+3/(2u2) * v2) .* erfn
        dG += (-5/2+5/(4u2).-(u*(5/2+u2))./v3+3/(2u2) * v2) .* erfp
        dG /= 20
        ddH = -(1/(2u2) .+ (-1 + 1/(2u2))./v2 + 1/u ./v + v/u).*expuvp
        ddH += (1/(2u2) .+ (-1 + 1/(2u2))./v2 - 1/u ./v - v/u).*expuvn
        ddH -= (u ./ v2) .* erfn
        ddH += (u ./ v2) .* erfp
        ddH .*= (3/4 ./ v2)
        ddG = (1-1/(2u2).+(2-1/(2u2)+u2)./v2-(1/u+u)./v-v/u+v2/u2).*expuvp
        ddG -= (1-1/(2u2).+(2-1/(2u2)+u2)./v2+(1/u+u)./v+v/u+v2/u2).*expuvn
        ddG -= (((5u)/2+u^3)./v2 - v3/u2) .* erfn
        ddG += (((5u)/2+u^3)./v2 + v3/u2) .* erfp
        ddG .*= (3/20 ./ v2)
        return H, dH, ddH, G, dG, ddG
      elseif L1 == 3
        u2 = u^2
        v2 = v.^2
        v3 = v2 .* v
        H = (expuvp - expuvn) ./ (u * v)
        H += (1/u .- 1 ./v) .* erfn + (1/u .+ 1 ./v) .* erfp
        H /= 8
        G = ().*expuvp
        G += ().*expuvn
        G += () .* erfn
        G += () .* erfp
        G /= 1
        dH = ().*expuvp
        dH += ().*expuvn
        dH += () .* erfn
        dH += () .* erfp
        dH /= 1
        dG = ().*expuvp
        dG += ().*expuvn
        dG += () .* erfn
        dG += () .* erfp
        dG /= 1
        ddH = ().*expuvp
        ddH += ().*expuvn
        ddH += () .* erfn
        ddH += () .* erfp
        ddH .*= (1)
        ddG = ().*expuvp
        ddG += ().*expuvn
        ddG += () .* erfn
        ddG += () .* erfp
        ddG .*= (1)
        return H, dH, ddH, G, dG, ddG
      else
        u2 = u^2
        v2 = v.^2
        v3 = v2 .* v
        H = (expuvp - expuvn) ./ (u * v)
        H += (1/u .- 1 ./v) .* erfn + (1/u .+ 1 ./v) .* erfp
        H /= 8
        G = ().*expuvp
        G += ().*expuvn
        G += () .* erfn
        G += () .* erfp
        G /= 1
        dH = ().*expuvp
        dH += ().*expuvn
        dH += () .* erfn
        dH += () .* erfp
        dH /= 1
        dG = ().*expuvp
        dG += ().*expuvn
        dG += () .* erfn
        dG += () .* erfp
        dG /= 1
        ddH = ().*expuvp
        ddH += ().*expuvn
        ddH += () .* erfn
        ddH += () .* erfp
        ddH .*= (1)
        ddG = ().*expuvp
        ddG += ().*expuvn
        ddG += () .* erfn
        ddG += () .* erfp
        ddG .*= (1)
        return H, dH, ddH, G, dG, ddG
        @warn("The analysis result when `L ≥ 4` were not given now.")
        # sdgfdf
      end
    end
  end