"""
  PDF: Probability distribution function with axisymmetry
     Harmonics of drift-Maxwellian distribution when
     `uai` is the relative velocity in Lagrange coordination


     f(v,μ,φ) = ∑∑fₗᵐYₗᵐ where fₗᵐ(v) ∝ vᴸ

      The `ℓᵗʰ`-order coefficients of normalized distribution function will be:


       f̂ₗᵐ(v̂,ℓ,m=0;û) = (-1)^ℓ * (2ℓ+1)//2 * (n̂ / v̂ₜₕ³) * exp((-û²-v̂²)/v̂ₜₕ²) * superRBFv(ℓ,ξ)

       superRBFv(ℓ,ξ) = 1 / √ξ * besseli(1/2 + ℓ, ξ)

     where

       ξ = 2 * |û| * v̂ / v̂ₜₕ²
       n̂ = nₛ / n₀
       v̂ₜₕ = vₜₕₛ / vₜₕ₀
       û = uₛ / vₜₕ₀
       v̂ = v / vₜₕ₀

     and `n₀`, `vₜₕ₀` are the effective values (or named as experimental values) of the specified total distribution function.

     With definition
   
     King(v̂;û,v̂th,ℓ) = (sign(û))^ℓ / v̂ₜₕ³ * exp(-(û²+v̂²)/v̂ₜₕ²) / √ξ * besseli(1/2 + ℓ, ξ)

     gives:

     f̂ₗᵐ(v̂,ℓ,m=0;û,v̂ₜₕ) = √(2π) * n̂ * King(v̂;û,v̂ₜₕ,ℓ)

  #  Singularity when `ξ = 2|û|v̂` tend to zero which can applying Tayler expansion:

     TaylerN = N : (0,N⁺) mean tayler expansion with first `Nₜₕ` terms form is used
     and terminated when `R_dSn = dSn/Sn ≤ 1e-2` (default )

    # For Maxwellian distribution which `uai = 0` will result `ℓ = 0`.
      However we will keep the order `ℓ = 1` when `uai = 0` for verifying the model.

  inputs:
    v: the normalized velocity `v̂ = v / vₜₕ`
    uai: the normalized group velocity `ûₐ = uₐ / vₜₕ`
    nai: `n̂ = 1.0` default, the normalized number density `n̂ = na / na0`, for single group of `fLn`.
    vthi: `v̂ₐₜₕ = 1.0` default, which denoes `v̂ₐₜₕ = vₐₜₕ / vₜₕ`, for single group of `fLn`.
    ξ: 2|û|v̂ / v̂ₐₜₕ²
    L_limit: the maximum order of coefficients of the distribution function on angular velocity.
    rel_dfLM ,the maximum rate of change of f(v)= ∑fL(v) for ℓM
    abs_dfLM ,the absolute minimum change of f(v), fL(v), for ℓM

  outputs:
    LM, fvL = fvLDMz(vhk,LM,ns,nMod,nai,uai,vthi;
            L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
"""


"""

  Inputs:
    v: the normalized velocity `v̂ = v / vₜₕ`

  Outputs:
    fLn = fvLDMz(fLn,v,ℓ,nai,uai,vthi)

"""

# 1D, [v] `z` direction when
function fvLDMz(fLn::AbstractVector{T},v::AbstractVector{T},ℓ::Int64,nai::T,uai::T,vthi::T) where{T}

    if uai == 0.0
        fLn[:] = fM(fLn,v,nai,vthi)
        return fLn
    else
        if uai > 0.0
            if vthi == 1.0
                fLn[1] = (ℓ + 1/2) * sqrt2pi * nai
                fLn[:] = fLn[1] * exp.(-(v.^2 .+ uai^2)) .* superRBFv(fLn,2uai * v,ℓ)
            else
                fLn[1] = (ℓ + 0.5) * sqrt2pi * nai / vthi^3
                fLn[:] = fLn[1] * exp.(-(v.^2 .+ uai^2) / vthi^2) .* superRBFv(fLn,2uai / vthi^2 * v,ℓ)
            end
        else
            if isodd(ℓ)
                if vthi == 1.0
                    fLn[1] = - (ℓ + 1/2) * sqrt2pi * nai
                    fLn[:] = fLn[1] * exp.(-(v.^2 .+ uai^2)) .* superRBFv(fLn,- 2uai * v,ℓ)
                else
                    fLn[1] = - (ℓ + 0.5) * sqrt2pi * nai / vthi^3
                    fLn[:] = fLn[1] * exp.(-(v.^2 .+ uai^2) / vthi^2) .* superRBFv(fLn,- 2uai / vthi^2 * v,ℓ)
                end
            else
                if vthi == 1.0
                    fLn[1] = (ℓ + 1/2) * sqrt2pi * nai
                    fLn[:] = fLn[1] * exp.(-(v.^2 .+ uai^2)) .* superRBFv(fLn,- 2uai * v,ℓ)
                else
                    fLn[1] = (ℓ + 0.5) * sqrt2pi * nai / vthi^3
                    fLn[:] = fLn[1] * exp.(-(v.^2 .+ uai^2) / vthi^2) .* superRBFv(fLn,- 2uai / vthi^2 * v,ℓ)
                end
            end
        end
        if fLn[end] ≥ epsT / 10
            @warn("Error: `fLn[end] > epsT / 10` which may cause errors of the moments.",ℓ)
            @show fLn[end]
        end
        return fLn
    end
end

"""
  fLn = fM(fLn,v,nai,vthi)
"""
# uai = 0
function fM(fLn::AbstractVector{T},v::AbstractVector{T},nai::T,vthi::T) where{T}

    if vthi == 1.0
        if nai == 1.0
            fLn = exp.(-v.^2)
        else
            fLn = exp.(-v.^2) * nai
        end
    else
        if nai == 1.0
            fLn = exp.(-v.^2 / vthi^2) / vthi^3
        else
            fLn = exp.(-v.^2 / vthi^2) * (nai / vthi^3)
        end
    end
    if fLn[end] ≥ epsT / 10
        @warn("Error: `fM[end] > epsT / 10` which may cause errors of the moments. ℓ=0")
        @show fLn[end]
    end
    return fLn
end

"""

  Kenerl function:

    superRBFv(ℓ,ξ) = 1 / √ξ * besseli(1/2 + ℓ, ξ)

  Inputs:
    fLn:
    xi: ξ = 2|û|v̂ / v̂ₜₕᵢ²
    ℓ:

  Outputs:
    fLn = superRBFv(fLn,xi,ℓ)

"""

function superRBFv(fLn::AbstractVector{T},xi::AbstractVector{T},ℓ::Int64) where{T}

    if  xi[1] == 0.0
        if ℓ == 0
            fLn[2:end] = sqrt2invpi * sinh.(xi[2:end]) ./ xi[2:end]
            fLn[1] = sqrt2invpi
        else
            fLn[2:end] = besseli.(0.5 + ℓ, xi[2:end]) ./ sqrt.(xi[2:end])
            fLn[1] = 0.0
        end
    else
        fLn = besseli.(0.5 + ℓ, xi) ./ sqrt.(xi)
    end
    return fLn
end


"""
  outputs:
    fLn = fvLDMz(fLn,vhk,ℓ,nai,uai,vthi,nMod)
"""

# 1.5D, [nMod,nv],     fLn = zeros(T,nck)
function fvLDMz(fLn::AbstractVector{T},vhk::AbstractVector{T},ℓ::Int64,nai::AbstractVector{T},
    uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64) where{T}

    k = 1
    fLn = fvLDMz(fLn,vhk,ℓ,nai[k],uai[k],vthi[k])
    fvLs = zero.(fLn)
    for k in 2:nMod
        if nai[k] > 0
            fvLs = fvLDMz(fvLs,vhk,ℓ,nai[k],uai[k],vthi[k])
        end
        fLn += fvLs
    end
    return fLn
end

# 2D, [nv,LM1]
function fvLDMz(fvL::AbstractArray{T,N},v::AbstractVector{T},LM::Int64,
    nai::T,uai::T,vthi::T;L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0) where{T,N}

    if uai == 0.0
        fvL[:,1] = fM(fvL[:,1],v,nai,vthi)
        # LM = 1
        return 1, fvL[:,1:2]
    else
        LM *= 0
        L1 = 1
        fvL[:,L1] = fvLDMz(fvL[:,L1],v,0,nai,uai,vthi)
        is_f = fvL[:,1] .> eps0
        fs = fvL[is_f,1]    # for `df_M = 1 ./ fs`
        if uai > 0
            for L1 in 2:L_limit+1
                # L = L1 - 1
                fvL[:,L1] = fvLDMz(fvL[:,L1],v,L1-1,nai,uai,vthi)
                LM += 1
                fs .+= fvL[is_f,L1]
                if maximum(fvL[is_f,L1]) ≤ abs_dfLM
                    LM < 2 || (LM -= 1)
                    break
                else
                    df_M = maximum(fvL[is_f,L1] ./ fs)
                    if df_M ≤ rel_dfLM
                        # LM < 2 || (LM -= 1)
                        break
                    end
                end
            end
        else
            for L1 in 2:L_limit+1
                fvL[:,L1] = fvLDMz(fvL[:,L1],v,L1-1,nai,uai,vthi)
                LM += 1
                if iseven(L1)
                    fs .-= fvL[is_f,L1]
                    if maximum(-fvL[is_f,L1]) ≤ abs_dfLM
                        # LM < 2 || (LM -= 1)
                        break
                    else
                        df_M = maximum(-fvL[is_f,L1] ./ fs)
                        if df_M ≤ rel_dfLM
                            LM < 2 || (LM -= 1)
                            break
                        end
                    end
                else
                    fs .+= fvL[is_f,L1]
                    if maximum(fvL[is_f,L1]) ≤ abs_dfLM
                        # LM < 2 || (LM -= 1)
                        break
                    else
                        df_M = maximum(fvL[is_f,L1] ./ fs)
                        if df_M ≤ rel_dfLM
                            # LM < 2 || (LM -= 1)
                            break
                        end
                    end
                end
            end
        end
        if isodd(LM)
            fLnmax = maximum(abs.(fvL[:,LM+1]))
        else
            fLnmax = maximum(fvL[:,LM+1])
        end
        if fLnmax ≤ abs_dfLM
            fvL[:,LM+1] .= 0.0
            if LM == 1
                @warn("The last order harmonic coefficients is lower than `abs_dfLM` and Maxwellian is a better model for",LM)
            else
                LM -= 1
            end
        end
        # fvL = fvL[:,1:LM+1]
        return LM, fvL[:,1:LM+1]
    end
end

"""
  outputs:
    LM, fvL = fvLDMz(fvL,vhk,nck,LM,nai,uai,vthi,nMod;
            L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
"""

# 2.5D, [nMod,nv,LM1],     fvL = zeros(T,nck,L_limit+1)
function fvLDMz(fvL::AbstractArray{T,N},vhk::AbstractVector{T},nck::Int,LM::Int64,
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64;
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0) where{T,N}

    k = 1
    LM, fvL = fvLDMz(fvL,vhk,LM,nai[k],uai[k],vthi[k];
            L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
    for k in 2:nMod
        if nai[k] > 0
            fvLs = zeros(T,nck,L_limit+1)
            LMs = 0
            LMs, fvLs = fvLDMz(fvLs,vhk,LMs,nai[k],uai[k],vthi[k];
                    L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
            if LMs ≤ LM
                fvL[:,1:LMs+1] += fvLs
            else
                fvLs[:,1:LM+1] += fvL
                LM = deepcopy(LMs)
                fvL = deepcopy(fvLs)
            end
        end
    end
    return LM, fvL
end

"""
  Inputs:

  outputs:
    LM, fvL = fvLDMz(fvL,vhk,LM,ns,nai,uai,vthi;
            L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
"""

# 3D, [nv,LM1,ns],     fvL = zeros(T,nck,L_limit+1,ns)
function fvLDMz(fvL::Vector{Matrix{T}},vhk::Vector{AbstractVector{T}},LM::Vector{Int},ns::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false) where{T}
    
    for isp in 1:ns
        v = vhk[isp]
        if nai[isp][1] > 0
            if uai[isp][1] == 0.0
                LM[isp] = 1
                fvL[isp][:,1] = fM(fvL[isp][:,1] ,v,nai[isp][1],vthi[isp][1])
            else
                LM[isp] = 0
                L1 = 1    # ℓ = 0
                fvL[isp][:,L1] = fvLDMz(fvL[isp][:,L1],v,0,nai[isp][1],uai[isp][1],vthi[isp][1])
                is_f = fvL[isp][:,1] .> eps0
                fs = fvL[isp][is_f,1] .+ eps0    # `for df_M = 1./fs`
                if uai[isp][1] > 0
                    for L1 in 2:L_limit+1
                        fvL[isp][:,L1] = fvLDMz(fvL[isp][:,L1],v,L1-1,nai[isp][1],uai[isp][1],vthi[isp][1])
                        LM[isp] += 1
                        fs .+= fvL[isp][is_f,L1]
                        if maximum(fvL[isp][is_f,L1]) ≤ abs_dfLM
                            # LM[isp] < 2 || (LM[isp] -= 1)
                            break
                        else
                            df_M = maximum(fvL[isp][is_f,L1] ./ fs)
                            if df_M ≤ rel_dfLM
                                # LM[isp] < 2 || (LM[isp] -= 1)
                                break
                            end
                        end
                    end
                else
                    for L1 in 2:L_limit+1
                        fvL[isp][:,L1] = fvLDMz(fvL[isp][:,L1],v,L1-1,nai[isp][1],uai[isp][1],vthi[isp][1])
                        LM[isp] += 1
                        if iseven(L1)
                            fs .-= fvL[isp][is_f,L1]
                            if maximum(-fvL[isp][is_f,L1]) ≤ abs_dfLM
                                # LM[isp] < 2 || (LM[isp] -= 1)
                                break
                            else
                                df_M = maximum(-fvL[isp][is_f,L1] ./ fs)
                                if df_M ≤ rel_dfLM
                                    # LM[isp] < 2 || (LM[isp] -= 1)
                                    break
                                end
                            end
                        else
                            fs .+= fvL[isp][is_f,L1]
                            if maximum(fvL[isp][is_f,L1]) ≤ abs_dfLM
                                # LM < 2 || (LM -= 1)
                                break
                            else
                                df_M = maximum(fvL[isp][is_f,L1] ./ fs)
                                if df_M ≤ rel_dfLM
                                    # LM < 2 || (LM -= 1)
                                    break
                                end
                            end
                        end
                    end
                    if isodd(LM[isp])
                        fLnmax = maximum(abs.(fvL[isp][:,LM[isp]+1]))
                    else
                        fLnmax = maximum(fvL[isp][:,LM[isp]+1])
                    end
                    if fLnmax ≤ abs_dfLM
                        fvL[isp][:,LM[isp]+1] .= 0.0
                        if LM[isp] == 1
                            @warn("The last order harmonic coefficients is lower than `abs_dfLM` and Maxwellian is a better model for",(isp,LM[isp]))
                        else
                            LM[isp] -= 1
                        end
                    end
                end
            end
        end
    end
    LM1 = maximum(LM) + 1
    if LM1 ≥ 1
        for isp in 1:ns
            if is_LM1_full
                sdfghj
            else
                if LM[isp] + 1 < LM1
                    if LM[isp] ≥ 0
                        fvL[isp][:,LM[isp]+2:LM1] .= 0.0
                    end
                end
            end
        end
        if LM1 < L_limit + 1
            for isp in 1:ns
                fvL[isp] = fvL[isp][:,1:LM1]
            end
        end
        return LM, fvL
    else
        # All `nai == 0`
        return LM, []
    end
end

"""
  Inputs:

  outputs:
    LM, fvL = fvLDMz(fvL,vhk,nck,LM,ns,nai,uai,vthi,nMod;L_limit=L_limit,
                    rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM,is_LM1_full=is_LM1_full)
"""

# 3.5D,  [nMod,nv,LM1,ns]
function fvLDMz(fvL::AbstractVector{Matrix{T}},vhk::Vector{AbstractVector{T}},nck::Vector{Int},LM::Vector{Int},ns::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false) where{T}

    for isp in 1:ns
        LM[isp],fvL2 = fvLDMz(fvL[isp],vhk[isp],nck[isp],LM[isp],
                nai[isp],uai[isp],vthi[isp],nMod[isp];
                L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
        fvL[isp][:,1:LM[isp]+1] = fvL2
    end
    LM1 = maximum(LM) + 1
    if is_LM1_full
        if LM1 < L_limit + 1
            for isp in 1:ns
                fvL[isp] = fvL[isp][:,1:LM1]
            end
        end
        LM, fvL = fvLDMzfull(fvL,vhk,nck,LM,LM1,ns,nai,uai,vthi,nMod)
        return LM, fvL
    else
        if LM1 > 0
            for isp in 1:ns
                if LM[isp] + 1 < LM1
                    if LM[isp] ≥ 0
                        fvL[isp][:,LM[isp]+2:LM1] .= 0.0
                    end
                end
            end
            if LM1 < L_limit + 1
                for isp in 1:ns
                    fvL[isp] = fvL[isp][:,1:LM1]
                end
            end
            return LM, fvL
        else
            # All `nai == 0`
            return LM, []
        end
    end
end



"""
  Inputs:
    uai:: ≠ 0.0
    LM1:: LM1 > LM + 1

  outputs:
    LM, fvL = fvLDMz(fvL,vhk,nck,LM,LM1,ns,nai,uai,vthi,nMod)
"""
# is_LM1_full = true
# 3.5D,  [nMod,nv,LM1,ns]
function fvLDMzfull(fvL::AbstractVector{Matrix{T}},vhk::Vector{AbstractVector{T}},nck::Vector{Int},LM::Vector{Int},LM1::Int,ns::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int}) where{T}

    for isp in 1:ns
        fvL[isp] = fvLDMzfull(fvL[isp],vhk[isp],nck[isp],LM1,nai[isp],uai[isp],vthi[isp],nMod[isp])
    end
    LM .= LM1 - 1
    return LM, fvL
end

# 2.5D, [nMod,nv,LM1],     fvL = zeros(T,nck,L_limit+1)
function fvLDMzfull(fvL::AbstractArray{T,N},vhk::AbstractVector{T},nck::Int,LM1::Int64,
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64) where{T,N}

    k = 1
    fvL = fvLDMzfull(fvL,vhk,LM1,nai[k],uai[k],vthi[k])
    for k in 2:nMod
        if nai[k] > 0
            fvL += fvLDMzfull(zero.(fvL),vhk,LM1,nai[k],uai[k],vthi[k])
        end
    end
    return fvL
end

# 2D, [nv,LM1]
function fvLDMzfull(fvL::AbstractArray{T,N},v::AbstractVector{T},
    LM1::Int64,nai::T,uai::T,vthi::T) where{T,N}

    if uai == 0.0
        fvL[:,1] = fM(fvL[:,1],v,nai,vthi)
        return fvL
    else
        for L1 in 1:LM1
            fvL[:,L1] = fvLDMz(fvL[:,L1],v,L1-1,nai,uai,vthi)
        end
        return fvL
    end
end