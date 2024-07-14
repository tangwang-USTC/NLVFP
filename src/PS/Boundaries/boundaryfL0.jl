
"""
  When `v̂ = 0`: boundary conditions

  The boundaries of the first `pᵗʰ` derivatives of the `ℓᵗʰ`-order coefficients
  of the normalized distribution function which is drifted Maxwellian in `fL0DMz.jl`.

  Warning: coefficient `1/π^(3/2)` of `f̂ₗ(v̂)` is not included in following codes.

  Inputs:
    na: n̂ = nai[isp] = na[k,isp] / na[isp]
    vth: v̂ₜₕ = vthi[isp] = vth[k,isp] / vth[isp]
    u: û = uai[isp] = ua[k,isp] / vth[isp]

  Outputs:
    fLnv0 = fLnDMv0(na,u,vth,L)
    dfLnv0 = dfLnDMv0(na,u,vth,L)
    ddfLnv0 = ddfLnDMv0(na,u,vth,L)
"""
 
# L = 0
function fLnDMv0(na::T,u::T,vth::T) where{T}

  if vth == 1.0
    return na * exp(- u^2)
  else
    return na / vth^3 * exp(- (u/vth)^2)
  end
end

#
function fLnDMv0(na::T,u::T,vth::T,L::Int64) where{T}

  if L == 0
    if vth == 1.0
      return na * exp(- u^2)
    else
      return na / vth^3 * exp(- (u/vth)^2)
    end
  else
    return 0.0
  end
end

function dfLnDMv0(na::T,u::T,vth::T,L::Int64) where{T}

  if L == 1
    if vth == 1.0
      return u * na * (2u)* exp(- u^2)
    else
      return na / vth^4 * (2u/vth)* exp(- (u/vth)^2)
    end
  else
    return 0.0
  end
end

function ddfLnDMv0(na::T,u::T,vth::T,L::Int64) where{T}

  if L == 0
    if vth == 1.0
      return 2na * exp(- u^2) * (- 1.0 + 2/3 * u^2)
    else
      return 2na / vth^5 * exp(- (u/vth)^2) * (- 1.0 + 2/3 * (u/vth)^2)
    end
  elseif L == 2
    if vth == 1.0
      return 2/3 * na * (2u)^2 * exp(- u^2)
    else
      return 2/3 * na / vth^3 * (2u/vth)^2 * exp(- (u/vth)^2)
    end
  else
    return 0.0
  end
end

"""
  Outputs:
    fLnv0 = fLnDMv0(na,u,L)
    dfLnv0 = dfLnDMv0(na,u,L)
    ddfLnv0 = ddfLnDMv0(na,u,L)
"""

# vth = 1
function fLnDMv0(na::T,u::T,L::Int64) where{T}

  if L == 0
    return na * exp(- u^2)
  else
    return 0.0
  end
end

function dfLnDMv0(na::T,u::T,L::Int64) where{T}

  if L == 1
    return u * na * (2u)* exp(- u^2)
  else
    return 0.0
  end
end

function ddfLnDMv0(na::T,u::T,L::Int64) where{T}

  if L == 0
    return 2na * exp(- u^2) * (- 1.0 + 2/3 * u^2)
  elseif L == 2
    return 2/3 * na * (2u)^2 * exp(- u^2)
  else
    return 0.0
  end
end

"""
  Outputs:
    fLnv0 = fLnDMv0(na,u,L)
    dfLnv0 = dfLnDMv0(na,u,L)
    ddfLnv0 = ddfLnDMv0(na,u,L)
"""

# vth = 1, na = 1
function fLnDMv0(u::T,L::Int64) where{T}

  if L == 0
    return exp(- u^2)
  else
    return 0.0
  end
end

function dfLnDMv0(u::T,L::Int64) where{T}

  if L == 1
    return u * (2u)* exp(- u^2)
  else
    return 0.0
  end
end

function ddfLnDMv0(u::T,L::Int64) where{T}

  if L == 0
    return 2exp(- u^2) * (- 1.0 + 2/3 * u^2)
  elseif L == 2
    return 2/3 * (2u)^2 * exp(- u^2)
  else
    return 0.0
  end
end


"""
  Outputs:
    fLnv0 = fLnDMv0(na,u,vth,L)
    dfLnv0 = dfLnDMv0(na,u,vth,L)
    ddfLnv0 = ddfLnDMv0(na,u,vth,L)
"""
# [nMod], L = 0
function fLnDMv0(na::AbstractVector{T},u::AbstractVector{T},vth::AbstractVector{T},nMod::Int64) where{T}

  k = 1
  if vth[k] == 1
    fp = na[k] * exp(- (u[k])^2)
  else
    fp = na[k] / vth[k]^3 * exp(- (u[k]/vth[k])^2)
  end
  for k in 2:nMod
    if na[k] > 0
      if vth[k] == 1
        fp += na[k] * exp(- (u[k])^2)
      else
        fp += na[k] / vth[k]^3 * exp(- (u[k]/vth[k])^2)
      end
    end
  end
  return fp
end

# [nMod], L = 1
function dfLnDMv0(na::AbstractVector{T},u::AbstractVector{T},vth::AbstractVector{T},nMod::Int64) where{T}

  k = 1
  if u[k] ≠ 0
    if vth[k] == 1
      fp = na[k] * (2u[k])* exp(- (u[k])^2)
    else
      fp = na[k] / vth[k]^4 * (2u[k]/vth[k])* exp(- (u[k]/vth[k])^2)
    end
  else
    fp = 0.0
  end
  for k in 2:nMod
    if na[k] > 0 && u[k] ≠ 0
      if vth[k] == 1
        fp += na[k] * (2u[k])* exp(- (u[k])^2)
      else
        fp += na[k] / vth[k]^4 * (2u[k]/vth[k])* exp(- (u[k]/vth[k])^2)
      end
    end
  end
  return fp
end

# [nMod],
function fLnDMv0(na::AbstractVector{T},u::AbstractVector{T},vth::AbstractVector{T},L::Int64,nMod::Int64) where{T}

  if L == 0
    k = 1
    if vth[k] == 1
      fp = na[k] * exp(- (u[k])^2)
    else
      fp = na[k] / vth[k]^3 * exp(- (u[k]/vth[k])^2)
    end
    for k in 2:nMod
      if na[k] > 0
        if vth[k] == 1
          fp += na[k] * exp(- (u[k])^2)
        else
          fp += na[k] / vth[k]^3 * exp(- (u[k]/vth[k])^2)
        end
      end
    end
    return fp
  else
    return 0.0
  end
end

# [nMod],
function dfLnDMv0(na::AbstractVector{T},u::AbstractVector{T},vth::AbstractVector{T},L::Int64,nMod::Int64) where{T}

  if L == 1
    k = 1
    if u[k] ≠ 0
      if vth[k] == 1
        fp = na[k] * (2u[k])* exp(- (u[k])^2)
      else
        fp = na[k] / vth[k]^4 * (2u[k]/vth[k])* exp(- (u[k]/vth[k])^2)
      end
    else
      fp = 0.0
    end
    for k in 2:nMod
      if na[k] > 0 && u[k] ≠ 0
        if vth[k] == 1
          fp += na[k] * (2u[k])* exp(- (u[k])^2)
        else
          fp += na[k] / vth[k]^4 * (2u[k]/vth[k])* exp(- (u[k]/vth[k])^2)
        end
      end
    end
    return fp
  else
    return 0.0
  end
end

# [nMod],
function ddfLnDMv0(na::AbstractVector{T},u::AbstractVector{T},vth::AbstractVector{T},L::Int64,nMod::Int64) where{T}

  if L == 0
    k = 1
    if vth[k] == 1
      fp = 2na[k] * exp(- (u[k])^2) * (- 1.0 + 2/3 * (u[k])^2)
    else
      fp = 2na[k] / vth[k]^5 * exp(- (u[k]/vth[k])^2) * (- 1.0 + 2/3 * (u[k]/vth[k])^2)
    end
    for k in 2:nMod
      if na[k] > 0
        if vth[k] == 1
          fp += 2na[k] * exp(- (u[k])^2) * (- 1.0 + 2/3 * (u[k])^2)
        else
          fp += 2na[k] / vth[k]^5 * exp(- (u[k]/vth[k])^2) * (- 1.0 + 2/3 * (u[k]/vth[k])^2)
        end
      end
    end
    return fp
  elseif L == 2
    k = 1
    if u[k] ≠ 0
      if vth[k] == 1
        fp = 2/3 * u[k] * na[k] * (2u[k])^2 * exp(- (u[k])^2)
      else
        fp = 2/3 * u[k] * na[k] / vth[k]^3 * (2u[k]/vth[k])^2 * exp(- (u[k]/vth[k])^2)
      end
    else
      fp = 0.0
    end
    for k in 2:nMod
      if na[k] > 0 && u[k] ≠ 0
        if vth[k] == 1
          fp += 2/3 * u[k] * na[k] * (2u[k])^2 * exp(- (u[k])^2)
        else
          fp += 2/3 * u[k] * na[k] / vth[k]^3 * (2u[k]/vth[k])^2 * exp(- (u[k]/vth[k])^2)
        end
      end
    end
    return fp
  else
    return 0.0
  end
end

"""
  Outputs:
    fLnv0 = fLnDMv0(na,u,L)
    dfLnv0 = dfLnDMv0(na,u,L)
    ddfLnv0 = ddfLnDMv0(na,u,L)
"""

# [nMod] vth = 1
function fLnDMv0(na::AbstractVector{T},u::AbstractVector{T},L::Int64,nMod::Int64) where{T}

  if L == 0
    k = 1
    fp = na[k] * exp(- (u[k])^2)
    for k in 2:nMod
      if na[k] > 0
        fp += na[k] * exp(- (u[k])^2)
      end
    end
    return fp
  else
    return 0.0
  end
end

function dfLnDMv0(na::AbstractVector{T},u::AbstractVector{T},L::Int64,nMod::Int64) where{T}

  if L == 1
    k = 1
    if u[k] ≠ 0
      fp = na[k] * (2u[k])* exp(- (u[k])^2)
    else
      fp = 0.0
    end
    for k in 2:nMod
      if na[k] > 0 && u[k] ≠ 0
        fp += na[k] * (2u[k])* exp(- (u[k])^2)
      end
    end
    return fp
  else
    return 0.0
  end
end

function ddfLnDMv0(na::AbstractVector{T},u::AbstractVector{T},L::Int64,nMod::Int64) where{T}

  if L == 0
    k = 1
    fp = 2na[k] * exp(- (u[k])^2) * (- 1.0 + 2/3 * (u[k])^2)
    for k in 2:nMod
      if na[k] > 0
        fp += 2na[k] * exp(- (u[k])^2) * (- 1.0 + 2/3 * (u[k])^2)
      end
    end
    return fp
  elseif L == 2
    k = 1
    if u[k] ≠ 0
      fp = 2/3 * u[k] * na[k] * (2u[k])^2 * exp(- (u[k])^2)
    else
      fp = 0.0
    end
    for k in 2:nMod
      if na[k] > 0 && u[k] ≠ 0
        fp += 2/3 * u[k] * na[k] * (2u[k])^2 * exp(- (u[k])^2)
      end
    end
    return fp
  else
    return 0.0
  end
end
