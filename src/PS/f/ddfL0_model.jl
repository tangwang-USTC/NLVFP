"""
  Inputs:
    u: = ûa
    v: = v̂

  Outputs:
    CddfLnDMv0L = CddfLnDMv0L(L)
    fLnDM, dfLnDM, ddfLnDM,fLnDMv0, dfLnDMv0, ddfLnDMv0 = fL0DM(ûa[isp];L1=1)
"""

# vecL = 1:L
function CddfLnDMv0(vecL::Vector{Int})

  L = vecL[end]
  if L == 0
    return nothing
  elseif L == 1
    return [6]
  elseif L == 2
    return [6.0,2.0/3.0]
  elseif L == 3
    return [6.0,2.0/3.0,2.0/5.0]
  elseif L == 4
    vec = zeros(L)
    vec[1:3] = [6.0,2.0/3.0,2.0/5.0]
    vec[L] = vec[L-1] * L / ((L-2) * (2L-1))
    return vec
  else
    vec = zeros(L)
    vec[1:3] = [6.0,2.0/3.0,2.0/5.0]
    k = 4
    vec[k] = vec[k-1] * k / ((k-2) * (2k-1))
    for k in 5:L
      vec[k] = vec[k-1] * k / ((k-2) * (2k-1))
    end
    return vec
  end
end

function CddfLnDMv0(L::Int)
  if L == 0
    return nothing
  elseif L == 1
    return 6.0
  elseif L == 2
    return 2.0/3.0
  elseif L == 3
    return 2.0/5.0
  elseif L == 4
    return 2.0/5.0 * L / ((L-2) * (2L-1))
  else
    k = 4
    vec = 2.0/5.0 * k / ((k-2) * (2k-1))
    for k in 5:L
      vec *= (k / ((k-2) * (2k-1)))
    end
    return vec
  end
end

function fL0DM(u::T;L::Int=0) where{T<:Real}

  if L == 0 && u == 0.0
    fLnM(v) = exp.(- v.^2)

    dfLnM(v) = - 2v .* exp.(- v.^2)

    ddfLnM(v) = (4v.^2 .- 2) .* exp.(- v.^2)
    return fLnM, dfLnM, ddfLnM, nothing, nothing, nothing
  else
    L1 = L + 1
    xi(v) = 2u * v
    expvuL(v) = (1 + 2L) / 2 * sqrtpi / u^0.5 * exp.(-u^2 .- v.^2)
    fLnDM(v) = expvuL(v) .* besseli.(0.5 + L, xi(v)) ./ v.^0.5
    dfLnDM(v) = expvuL(v) .* ((L ./ v.^1.5-2v.^0.5).*besseli.(1/2+L,xi(v))+2u ./ v.^0.5 .* besseli.(3/2+L,xi(v)))
    ddfLnDM(v) = expvuL(v).*(-4u*(1 ./ v.^1.5 + 2v.^0.5) .* besseli.(-1/2+L,xi(v)) +
            (L1 * (L1+1) ./ v.^2.5 + 2(L1 + L + 2u^2) ./ v.^0.5 + 4v.^1.5) .* besseli.(1/2+L,xi(v)))

    # function fLnDMv0u(u::T,L::Int) where{T}
    #
    #   if L == 0
    #     fLnv0L0(v) = exp(- u^2) * (1.0 + (2.0 / 3.0 * u^2 - 1.0) * v.^2)
    #     return fLnv0L0
    #   elseif L == 1
    #     fLnv0L1(v) = 2u * exp(- u^2) * (1.0 + (2.0 / 5.0 * u^2 - 1.0) * v.^2) .* v
    #     return fLnv0L1
    #   elseif L == 2
    #     fLnv0L2(v) = (2u)^L * exp(- u^2) / prod(3:2:(2L-1)) * (1.0 + (2.0 / 7.0 * u^2 - 1.0) * v.^2) .* v^L
    #     return fLnv0L2
    #   else
    #     fLnv0(v) = (2u)^L * exp(- u^2) / prod(3:2:(2L-1)) * (1.0 + (2.0 / (2L+3) * u^2 - 1.0) * v.^2) .* v^L
    #     return fLnv0
    #   end
    # end
    #
    # function dfLnDMv0u(u::T,L::Int) where{T}
    #
    #   if L == 0
    #     dfLnv0L0(v) = exp(- u^2) * (4.0 / 3.0 * u^2 - 2.0) * v
    #     return dfLnv0L0
    #   elseif L == 1
    #     dfLnv0L1(v) = 2u * exp(- u^2) * (1.0 + (6 / 5 * u^2 - 3.0) * v.^2)
    #     return dfLnv0L1
    #   elseif L == 2
    #     dfLnv0L2(v) = (2u)^L * exp(- u^2) * (L+2) / prod(3:2:(2L-1)) * (1.0/2.0 + (2.0/7.0 * u^2 - 1.0) * v.^2) .* v^(L-1)
    #     return dfLnv0L2
    #   else
    #     dfLnv0L(v) = (2u)^L * exp(- u^2) * (L+2) / prod(3:2:(2L-1)) * ((L+1)/(L+3) + (2.0/(2L+3) * u^2 - 1.0) * v.^2) .* v^(L-1)
    #     return dfLnv0L
    #   end
    # end
    #
    # function ddfLnDMv0u(u::T,L::Int) where{T}
    #
    #   if L == 0
    #     ddfLnv0L0(v) = exp(- u^2) * 2.0(2.0 / 3.0 * u^2 - 1.0)
    #     return ddfLnv0L0
    #   elseif L == 1
    #     ddfLnv0L1(v) = 2u * exp(- u^2) * (6.0(2.0 / 5.0 * u^2 - 1.0) * v.^2) .* v
    #     return ddfLnv0L1
    #   elseif L == 2
    #     ddfLnv0L2(v) = (2u)^L * exp(- u^2) * (4.0 / 6.0) .* v.^(L-2)
    #     return ddfLnv0L2
    #   else
    #     ddfLnv0L(v) = (2u)^L * exp(- u^2) * CddfLnDMv0(L) .* v.^(L-2)
    #     return ddfLnv0L
    #   end
    # end
    return fLnDM, dfLnDM, ddfLnDM, fLnDMv0u(u,L), dfLnDMv0u(u,L), ddfLnDMv0u(u,L)
  end
end

"""
  f(v=0;u,L) = fLnDMv0u(u,L)
"""

function fLnDMv0u(u::T,L::Int) where{T}

  if L == 0
    fLnv0L0(v) = exp(- u^2) * (1.0 + (2.0 / 3.0 * u^2 - 1.0) * v.^2)
    return fLnv0L0
  elseif L == 1
    fLnv0L1(v) = 2u * exp(- u^2) * (1.0 + (2.0 / 5.0 * u^2 - 1.0) * v.^2) .* v
    return fLnv0L1
  elseif L == 2
    fLnv0L2(v) = (2u)^L * exp(- u^2) / prod(3:2:(2L-1)) * (1.0 + (2.0 / 7.0 * u^2 - 1.0) * v.^2) .* v^L
    return fLnv0L2
  else
    fLnv0(v) = (2u)^L * exp(- u^2) / prod(3:2:(2L-1)) * (1.0 + (2.0 / (2L+3) * u^2 - 1.0) * v.^2) .* v^L
    return fLnv0
  end
end

function dfLnDMv0u(u::T,L::Int) where{T}

  if L == 0
    dfLnv0L0(v) = exp(- u^2) * (4.0 / 3.0 * u^2 - 2.0) * v
    return dfLnv0L0
  elseif L == 1
    dfLnv0L1(v) = 2u * exp(- u^2) * (1.0 + (6 / 5 * u^2 - 3.0) * v.^2)
    return dfLnv0L1
  elseif L == 2
    dfLnv0L2(v) = (2u)^L * exp(- u^2) * (L+2) / prod(3:2:(2L-1)) * (1.0/2.0 + (2.0/7.0 * u^2 - 1.0) * v.^2) .* v^(L-1)
    return dfLnv0L2
  else
    dfLnv0L(v) = (2u)^L * exp(- u^2) * (L+2) / prod(3:2:(2L-1)) * ((L+1)/(L+3) + (2.0/(2L+3) * u^2 - 1.0) * v.^2) .* v^(L-1)
    return dfLnv0L
  end
end

function ddfLnDMv0u(u::T,L::Int) where{T}

  if L == 0
    ddfLnv0L0(v) = exp(- u^2) * 2.0(2.0 / 3.0 * u^2 - 1.0)
    return ddfLnv0L0
  elseif L == 1
    ddfLnv0L1(v) = 2u * exp(- u^2) * (6.0(2.0 / 5.0 * u^2 - 1.0) * v.^2) .* v
    return ddfLnv0L1
  elseif L == 2
    ddfLnv0L2(v) = (2u)^L * exp(- u^2) * (4.0 / 6.0) .* v.^(L-2)
    return ddfLnv0L2
  else
    ddfLnv0L(v) = (2u)^L * exp(- u^2) * CddfLnDMv0(L) .* v.^(L-2)
    return ddfLnv0L
  end
end

"""
  f(0,u,L) = fLnDMv0u(v,u,L)
"""

function fLnDMv0u(v::Float64,u::T,L::Int) where{T}

  if L == 0
    return exp(- u^2) * (1.0 + (2.0 / 3.0 * u^2 - 1.0) * v.^2)
    return fLnv0L0
  elseif L == 1
    return 2u * exp(- u^2) * (1.0 + (2.0 / 5.0 * u^2 - 1.0) * v.^2) .* v
    return fLnv0L1
  elseif L == 2
    return (2u)^L * exp(- u^2) / prod(3:2:(2L-1)) * (1.0 + (2.0 / 7.0 * u^2 - 1.0) * v.^2) .* v^L
    return fLnv0L2
  else
    return (2u)^L * exp(- u^2) / prod(3:2:(2L-1)) * (1.0 + (2.0 / (2L+3) * u^2 - 1.0) * v.^2) .* v^L
    return fLnv0
  end
end

function dfLnDMv0u(v::Float64,u::T,L::Int) where{T}

  if L == 0
    return exp(- u^2) * (4.0 / 3.0 * u^2 - 2.0) * v
    return dfLnv0L0
  elseif L == 1
    return 2u * exp(- u^2) * (1.0 + (6 / 5 * u^2 - 3.0) * v.^2)
    return dfLnv0L1
  elseif L == 2
    return (2u)^L * exp(- u^2) * (L+2) / prod(3:2:(2L-1)) * (1.0/2.0 + (2.0/7.0 * u^2 - 1.0) * v.^2) .* v^(L-1)
    return dfLnv0L2
  else
    return (2u)^L * exp(- u^2) * (L+2) / prod(3:2:(2L-1)) * ((L+1)/(L+3) + (2.0/(2L+3) * u^2 - 1.0) * v.^2) .* v^(L-1)
    return dfLnv0L
  end
end

function ddfLnDMv0u(v::Float64,u::T,L::Int) where{T}

  if L == 0
    return exp(- u^2) * 2.0(2.0 / 3.0 * u^2 - 1.0)
    return ddfLnv0L0
  elseif L == 1
    return 2u * exp(- u^2) * (6.0(2.0 / 5.0 * u^2 - 1.0) * v.^2) .* v
    return ddfLnv0L1
  elseif L == 2
    return (2u)^L * exp(- u^2) * (4.0 / 6.0) .* v.^(L-2)
    return ddfLnv0L2
  else
    return (2u)^L * exp(- u^2) * CddfLnDMv0(L) .* v.^(L-2)
    return ddfLnv0L
  end
end


"""
  Inputs:
    u: = ûb
    v: = 𝓋̂ = v̂ * vabth = v̂ * (vath / vbth)

  Outputs:
    FLnDM,FLnDMv0 = FL0DM(ûa[iFv];L1=1)
    H, dH, ddH, G, dG, ddG = HGL0ab(ûa[iFv];L1=1)

"""

function FL0DM(u::T;L1::Int=1) where{T<:Real}

  if L1 == 1 && u == 0.0
    FLnM(v) = exp.(-  v.^2)
    # nv = length(v)
    # if nv == 1
    #   if v < 1e-10
    #     return v -> 1.0 .- v.^2
    #   else
    #     if v < 1e-6
    #       return v -> 1.0 .- v.^2 + v.^4/2 - v.^6/6
    #     else
    #       if v < 1e-2
    #         return v -> 1.0 .- v.^2 + v.^4/2 - v.^6/6 + v.^8/24 - v.^10/120 + v.^12/720
    #       else
    #         if v < 0.03
    #           return v -> 1.0 .- v.^2 + v.^4/2 - v.^6/6 + v.^8/24 - v.^10/120 + v.^12/720 - v.^14/5040 + v.^16/40320
    #         else
    #           if v < 1e-1
    #             return v -> 1.0 .- v.^2 + v.^4/2 - v.^6/6 + v.^8/24 - v.^10/120 + v.^12/720 - v.^14/5040 + v.^16/40320 - v.^18/362880 + v.^20/3628800
    #           else
    #             if v < 0.2
    #               return v -> 1.0 .- v.^2 + v.^4/2 - v.^6/6 + v.^8/24 - v.^10/120 + v.^12/720 - v.^14/5040 +
    #                          v.^16/40320 - v.^18/362880 + v.^20/3628800 - v.^22/39916800
    #             else
    #               return v -> 1.0 .- v.^2 + v.^4/2 - v.^6/6 + v.^8/24 - v.^10/120 + v.^12/720 - v.^14/5040 +
    #                          v.^16/40320 - v.^18/362880 + v.^20/3628800 - v.^22/39916800 + v.^24/479001600 - v.^26/6227020800
    #               if v > 0.5
    #                 @warn("When `v > 0.5`, the error of fM(v→0) will be larger than `exp(-v^2)`")
    #               end
    #             end
    #           end
    #         end
    #       end
    #     end
    #   end
    # else
    #   22222222222222222
    # end
    FLnMv0(v) = 1.0 .- v.^2 + v.^4/2 - v.^6/6 + v.^8/24 - v.^10/120 + v.^12/720 -
               v.^14/5040 + v.^16/40320 - v.^18/362880 + v.^20/3628800 -
               v.^22/39916800 + v.^24/479001600 - v.^26/6227020800
    return  FLnM, FLnMv0
  else
    L = L1 - 1
    FLnDM(v) = (2L+1)/2*sqrtpi/u^0.5*exp.(-u^2 .- v.^2).*besseli.(0.5+L,2u*v)./v.^0.5
    function FLnDMv0u(u::T,L::Int) where{T}

      if L == 0
        FLnv0L0(v) = exp(- u^2) * (1.0 + (2.0 / 3.0 * u^2 - 1.0) * v.^2)
        return FLnv0L0
      elseif L == 1
        FLnv0L1(v) = 2u * exp(- u^2) * (1.0 + (2.0 / 5.0 * u^2 - 1.0) * v.^2) .* v
        return FLnv0L1
      elseif L == 2
        FLnv0L2(v) = (2u)^L * exp(- u^2) / prod(3:2:(2L-1)) * (1.0 + (2.0 / 7.0 * u^2 - 1.0) * v.^2) .* v^L
        return FLnv0L2
      else
        FLnv0L(v) = (2u)^L * exp(- u^2) / prod(3:2:(2L-1)) * (1.0 + (2.0 / (2L+3) * u^2 - 1.0) * v.^2) .* v^L
        return FLnv0L
      end
    end
    return  FLnDM, FLnDMv0u(u,L)
  end
end


function HGL0DMabz(v::AbstractVector{T},u::T;L1::Int=1) where{T<:Real}

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
