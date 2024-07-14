"""
  M-theorems are the constraints inequations of the VFP equation.
"""

"""
  When `length(uh) == 1`, the procedure gives the deviation of the M functions to unit at different time.

  Inputs:
    uh::Float64, = uh(t)
    j::Int64, the order of the M function.

  Outputs:
    DMh = DMhsj_fDM(uht,j;ℓ=0)
    DMhsj_fDM!(DMhcjt,uht,njMs;ℓ=0)
"""

# When `nMod_output = 1` and `j ∈ ℓ:2:∞` and `ℓ=0`
function DMhsj_fDM(uh::T,j::Int64;ℓ::Int=0) where{T}
  
  if ℓ == 0
    if j == 0
        return 0 * uh
    elseif j == 2
        return 2/3 * uh^2
    elseif j == 4
        return 4/3 * uh^2 + 4/15 * uh^4
    elseif j == 6
        return 2 * uh^2 + 4/5 * uh^4 + 8/105 * uh^6
    else
        N = j / 2 |> Int    # `ℓ == 0`
        uh2 = uh^2
        DMh = 0uh
        for k in 1:N
            ck = 2^k * binomial(N, k) / prod(3:2:2k+1)
            DMh += ck * uh2 ^ k
        end
        return DMh
    end
  elseif ℓ == 1
    iiiiii
  else
    9iolllll
  end
end

function DMhsj_fDM!(DMhcj::AbstractVector{T},uh::T,njMs::Int64;ℓ::Int=0) where{T}
  
  if ℓ == 0
    # j = 2(kj - 1) + ℓ
    for kj in 1:njMs
        DMhcj[kj] = DMhsj_fDM(uh,2(kj - 1);ℓ=ℓ)
    end
  else
    for kj in 1:njMs
        DMhcj[kj] = DMhsj_fDM(uh,2(kj - 1) + ℓ;ℓ=ℓ)
    end
  end
end

# uh = uht
function DMhsj_fDM(uh::AbstractVector{T},j::Int64;ℓ::Int=0) where{T}
  
  if ℓ == 0
    if j == 0
        return 0 * uh
    elseif j == 2
        return 2/3 * uh.^2
    elseif j == 4
        return 4/3 * uh.^2 + 4/15 * uh.^4
    elseif j == 6
        return 2 * uh.^2 + 4/5 * uh.^4 + 8/105 * uh.^6
    else
        N = j / 2 |> Int    # `ℓ == 0`
        uh2 = uh.^2
        DMh = 0uh
        for k in 1:N
            ck = 2^k * binomial(N, k) / prod(3:2:2k+1)
            DMh += ck .* uh2 .^ k
        end
        return DMh
    end
  elseif ℓ == 1
    iiiiii
  else
    9iolllll
  end
end

function DMhsj_fDM!(DMhcj::AbstractArray{T,N},uh::AbstractVector{T},njMs::Int64;ℓ::Int=0) where{T,N}

  if ℓ == 0
    # j = 2(kj - 1) + ℓ
    for kj in 1:njMs
      DMhcj[:,kj] = DMhsj_fDM(uh,2(kj - 1);ℓ=ℓ)
    end
  else
    for kj in 1:njMs
      DMhcj[:,kj] = DMhsj_fDM(uh,2(kj - 1) + ℓ;ℓ=ℓ)
    end
  end
end

"""
  Inputs:
    uh::Float64, = uh(t)
    j::Int64, the order of the M function.

  Outputs:
    DMh = DMhsj_fDM(uht,j;ℓ=0)
    DMhsj_fDM!(DMhcjt,uht,njMs;ℓ=0)
"""
# When `nMod_output ≥ 2` and `j ∈ ℓ:2:∞` 
function Mhsj_fDM(nah::AbstractVector{T},uah::AbstractVector{T},Tah::AbstractVector{T},j::Int64) where{T}

  if j == 0
      Mh = 1.0
  elseif j == 2
      Mh = nah .* Tah .* (1 .+ 2/3 * uah.^2 ./ Tah)
  elseif j == 4
      uTah = uah.^2 ./ Tah 
      Mh = nah .* Tah.^2 .* (1 .+ 4/3 * uTah .+ 4/15 * uTah.^2)
  elseif j == 6
      uTah = uah.^2 ./ Tah 
      Mh = nah .* Tah.^3 .* (1 .+ 2 * uTah .+ 4/5 * uTah.^2 .+ 8/105 * uTah.^3)
  else
      N = j / 2 |> Int    # `ℓ == 0`
      uTah = uah.^2 ./ Tah 
      Mh = ones(T,length(Tah))
      for k in 1:N
          ck = 2^k * binomial(N, k) / prod(3:2:2k+1)
          Mh += ck .* uTah .^ k
      end
      Mh .*= nah .* Tah .^N
  end
  return Mh
end

function Mhsj_fDM!(Mhcsj,nah,nbh,uah,ubh,Tah,Tbh,njMs::Int64)

  for kj in 1:njMs
      j = 2(kj - 1)
      Mhcsj[:,kj] = Mhsj_fDM(nah,nbh,uah,ubh,Tah,Tbh,j) 
  end
end

function Mhsj_fDM(nah,nbh,uah::AbstractVector{T},ubh::AbstractVector{T},
  Tah::AbstractVector{T},Tbh::AbstractVector{T},j::Int64) where{T}

  if j == 0
      Mh = ones(T,length(Tah))
  elseif j == 2
      Mh = nah .* Tah .* (1 .+ 2/3 * uah.^2 ./ Tah)
      Mh += nbh .* Tbh .* (1 .+ 2/3 * ubh.^2 ./ Tbh)
  elseif j == 4
      uTah = uah.^2 ./ Tah 
      Mh = nah .* Tah.^2 .* (1 .+ 4/3 * uTah .+ 4/15 * uTah.^2)
      uTah = ubh.^2 ./ Tbh 
      Mh += nbh .* Tbh.^2 .* (1 .+ 4/3 * uTah .+ 4/15 * uTah.^2)
  elseif j == 6
      uTah = uah.^2 ./ Tah 
      Mh = nah .* Tah.^3 .* (1 .+ 2 * uTah .+ 4/5 * uTah.^2 .+ 8/105 * uTah.^3)
      uTah = ubh.^2 ./ Tbh 
      Mh += nbh .* Tbh.^3 .* (1 .+ 2 * uTah .+ 4/5 * uTah.^2 .+ 8/105 * uTah.^3)
  else
      N = j / 2 |> Int    # `ℓ == 0`
      uTah = uah.^2 ./ Tah 
      Mh = ones(T,length(Tah))
      for k in 1:N
          ck = 2^k * binomial(N, k) / prod(3:2:2k+1)
          Mh += ck .* uTah .^ k
      end
      Mh .*= nah .* Tah .^N
      
      uTah = ubh.^2 ./ Tbh 
      Mbh = ones(T,length(Tah))
      for k in 1:N
          ck = 2^k * binomial(N, k) / prod(3:2:2k+1)
          Mbh += ck .* uTah .^ k
      end
      Mh += nbh .* Tbh .^N .* Mbh
  end
  return Mh
end

function Mhsj_fDM!(Mhcsj,nah,nbh,uah,ubh,Tah,Tbh,njMs::Int64)

  for kj in 1:njMs
      j = 2(kj - 1)
      Mhcsj[:,kj] = Mhsj_fDM(nah,nbh,uah,ubh,Tah,Tbh,j) 
  end
end

