"""
  PDF: Probability distribution function with axisymmetry
     Harmonics of drift-Maxwellian distribution when
     `uai` is the relative velocity in Lagrange coordination


     f(v,μ,φ) = ∑∑fₗᵐYₗᵐ where fₗᵐ(v) ∝ vᴸ

      The `ℓᵗʰ`-order coefficients of normalized distribution function will be:


       f̂ₗᵐ(v̂,ℓ,m=0;û,v̂ₜₕ) = (sign(û))^ℓ * (2ℓ+1)//2 * √(2π) * (n̂ / v̂ₜₕ³) * exp(-(û²+v̂²)/v̂ₜₕ²) * superRBFv(ℓ,ξ)

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
    LM, fvL = fvLDMz(vhe,LM,ns,nMod,nai,uai,vthi;
            L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
"""

"""
  fLn = fM(fLn,v,nai,vthi)
"""
# 1D, uai = 0
function fM(fLn::AbstractVector{T},v::StepRangeLen,nai::T,vthi::T) where{T}

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

# 1D, uai = 0, nai = vthi = 1
function fM(fLn::AbstractVector{T},v::StepRangeLen) where{T}

    fLn = exp.(-v.^2)
    if fLn[end] ≥ epsT / 10
        @warn("Error: `fM[end] > epsT / 10` which may cause errors of the moments. ℓ=0")
        @show fLn[end]
    end
    return fLn
end

"""
  outputs:
    fLn = fvLDMz(fLn,vhe,ℓ,nai,uai,vthi,nMod)
"""

# 1.5D, [nMod,nv],     fLn = zeros(T,nvG)
function fvLDMz(fLn::AbstractVector{T},vhe::StepRangeLen,ℓ::Int64,nai::AbstractVector{T},
    uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64) where{T}

    k = 1
    fLn = fvLDMz(fLn,vhe,ℓ,nai[k],uai[k],vthi[k])
    fvLs = zero.(fLn)
    for k in 2:nMod
        if nai[k] > 0
            fvLs = fvLDMz(fvLs,vhe,ℓ,nai[k],uai[k],vthi[k])
        end
        fLn += fvLs
    end
    return fLn
end

# 2D, [nv,LM1], nMod = 1
function fvLDMz(fvL::AbstractArray{T,N},v::StepRangeLen,LM::Int64,
    nai::T,uai::T,vthi::T;L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0) where{T,N}

    if uai == 0.0
        fvL[:,1] = fM(fvL[:,1],v,nai,vthi)
        # LM = 1
        return 1, fvL[:,1:2]
    else
        LM *= 0
        L1 = 1
        fvL[:,L1] = fvLDMz(fvL[:,L1],v,0,nai,uai,vthi) # L = 0
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

# 2D, [nv,LM1], nMod = 1, nai = vthi = 1
function fvLDMz(fvL::AbstractArray{T,N},v::StepRangeLen,LM::Int64,uai::T;
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0) where{T,N}

    if uai == 0.0
        fvL[:,1] = fM(fvL[:,1],v)
        # LM = 1
        return 1, fvL[:,1:2]
    else
        LM *= 0
        L1 = 1
        fvL[:,L1] = fvLDMz(fvL[:,L1],v,0,uai)
        is_f = fvL[:,1] .> eps0
        fs = fvL[is_f,1]    # for `df_M = 1 ./ fs`
        if uai > 0
            for L1 in 2:L_limit+1
                # L = L1 - 1
                fvL[:,L1] = fvLDMz(fvL[:,L1],v,L1-1,uai)
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
                fvL[:,L1] = fvLDMz(fvL[:,L1],v,L1-1,uai)
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
    LM, fvL = fvLDMz(fvL,vhe,nvG,LM,nai,uai,vthi,nMod;
            L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
"""
 
# 2.5D, [nMod,nv,LM1],     fvL = zeros(T,nvG,L_limit+1)
function fvLDMz(fvL::AbstractArray{T,N},vhe::StepRangeLen,nvG::Int64,LM::Int64,
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64;
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0) where{T,N}

    k = 1
    LM, fvL = fvLDMz(fvL,vhe,LM,nai[k],uai[k],vthi[k];
            L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
    for k in 2:nMod
        if nai[k] > 0
            fvLs = zeros(T,nvG,L_limit+1)
            LMs = 0
            LMs, fvLs = fvLDMz(fvLs,vhe,LMs,nai[k],uai[k],vthi[k];
                    L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
            if LMs ≤ LM
                fvL[:,1:LMs+1] += fvLs
            else
                fvLs[:,1:LM+1] += fvL
                LM = copy(LMs)
                fvL = deepcopy(fvLs)
            end
        end
    end
    return LM, fvL
end

"""
  Inputs:

  outputs:
    LM, fvL = fvLDMz(fvL,vhe,LM,ns,nai,uai,vthi;
            L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
"""

# 3D, [nv,LM1,ns], nMod=1    fvL = zeros(T,nvG,L_limit+1,ns)
function fvLDMz(fvL::Vector{Matrix{T}},vhe::Vector{StepRangeLen},LM::Vector{Int64},ns::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false) where{T}

    for isp in 1:ns
        v = vhe[isp]
        if nai[isp][1] > 0
            if uai[isp][1] == 0.0
                LM[isp] = 1
                fvL[isp][:,1] = fM(fvL[isp][:,1],v,nai[isp][1],vthi[isp][1])
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

function fvLDMz(fvL::Vector{Matrix{T}},v::StepRangeLen,LM::Vector{Int64},ns::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false) where{T}

    for isp in 1:ns
        if nai[isp][1] > 0
            if uai[isp][1] == 0.0
                LM[isp] = 1
                fvL[isp][:,1] = fM(fvL[isp][:,1],v,nai[isp][1],vthi[isp][1])
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
    LM, fvL = fvLDMz(fvL,vhe,nvG,LM,ns,nai,uai,vthi,nMod;L_limit=L_limit,
                    rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM,is_LM1_full=is_LM1_full)
"""

# 3.5D,  [nMod,nv,LM1,ns], [LM, fvL0]
function fvLDMz(fvL::Vector{Matrix{T}},vhe::Vector{StepRangeLen},nvG::Vector{Int64},
    LM::Vector{Int64},ns::Int64,nai::Vector{AbstractVector{T}},
    uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false) where{T}

    for isp in 1:ns
        LM[isp],fvL2 = fvLDMz(fvL[isp],vhe[isp],nvG[isp],LM[isp],nai[isp],uai[isp],vthi[isp],nMod[isp];
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
        if is_LM1_full
            LM[:], fvL = fvLDMzfull(fvL,vhe,LM,LM1,ns,nai,uai,vthi,nMod) 
        end
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

function fvLDMz(fvL::Vector{Matrix{T}},vhe::StepRangeLen,nvG::Int64,
    LM::Vector{Int64},ns::Int64,nai::Vector{AbstractVector{T}},
    uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false) where{T}

    for isp in 1:ns
        LM[isp],fvL2 = fvLDMz(fvL[isp],vhe,nvG,LM[isp],nai[isp],uai[isp],vthi[isp],nMod[isp];
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
        if is_LM1_full
            LM[:], fvL = fvLDMzfull(fvL,vhe,LM,LM1,ns,nai,uai,vthi,nMod) 
        end
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

  outputs:
    LM, fvL = fvLDMz!(fvL,vhe,nvG,LM,ns,nai,uai,vthi,nMod;L_limit=L_limit,
                    rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM,is_LM1_full=is_LM1_full)
"""

# 3.5D,  [nMod,nv,LM1,ns], [LM1, fvL0]
function fvLDMz!(fvL::Vector{Matrix{T}},vhe::Vector{StepRangeLen},nvG::Vector{Int64},
    LM::Vector{Int64},ns::Int64,nai::Vector{AbstractVector{T}},
    uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false) where{T}

    for isp in 1:ns
        LM[isp],fvL2 = fvLDMz(fvL[isp],vhe[isp],nvG[isp],LM[isp],nai[isp],uai[isp],vthi[isp],nMod[isp];
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
        if is_LM1_full
            LM[:], fvL = fvLDMzfull(fvL,vhe,LM,LM1,ns,nai,uai,vthi,nMod)
        end
        return LM1, fvL
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
            return LM1, fvL
        else
            # All `nai == 0`
            return LM1, []
        end
    end
end

function fvLDMz!(fvL::Vector{Matrix{T}},vhe::Vector{StepRangeLen},nvG::Vector{Int64},
    LM::Vector{Int64},ns::Int64,nai::Vector{AbstractVector{T}},
    uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false) where{T}

    for isp in 1:ns
        LM[isp],fvL2 = fvLDMz(fvL[isp],vhe[isp],nvG[isp],LM[isp],nai[isp],uai[isp],vthi[isp],nMod[isp];
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
        if is_LM1_full
            LM[:], fvL = fvLDMzfull(fvL,vhe,LM,LM1,ns,nai,uai,vthi,nMod)
        end
        return LM1, fvL
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
            return LM1, fvL
        else
            # All `nai == 0`
            return LM1, []
        end
    end
end

# 3.5D,  [nv,LM1,ns], [LM1, fvL0], nMod = 1
function fvLDMz!(fvL::Vector{Matrix{T}},vhe::Vector{StepRangeLen},
    LM::Vector{Int64},ns::Int64,uhkL::AbstractVector{T};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false) where{T}

    for isp in 1:ns
        LM[isp],fvL2 = fvLDMz(fvL[isp],vhe[isp],LM[isp],uhkL[isp];
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
        if is_LM1_full
            LM[:], fvL = fvLDMzfull(fvL,vhe,LM,LM1,ns,uhkL)
        end
        return LM1, fvL
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
            return LM1, fvL
        else
            # All `nai == 0`
            return LM1, []
        end
    end
end

function fvLDMz!(fvL::Vector{Matrix{T}},vhe::StepRangeLen,
    LM::Vector{Int64},ns::Int64,uhkL::AbstractVector{T};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false) where{T}

    for isp in 1:ns
        LM[isp],fvL2 = fvLDMz(fvL[isp],vhe,LM[isp],uhkL[isp];
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
        if is_LM1_full
            LM[:], fvL = fvLDMzfull(fvL,vhe,LM,LM1,ns,uhkL)
        end
        return LM1, fvL
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
            return LM1, fvL
        else
            # All `nai == 0`
            return LM1, []
        end
    end
end

# 3.5D,  [nv,LM1,ns=2], LM1, nMod = 1
function fvLDMz!(fvL::Vector{Matrix{T}},vhe::Vector{StepRangeLen},
    LM::Vector{Int64},uai::AbstractVector{T};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0) where{T}

    for isp in 1:2
        LM[isp], fvL2 = fvLDMz(fvL[isp],vhe[isp],LM[isp],uai[isp];
                            L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
        fvL[isp][:,1:LM[isp]+1] = fvL2
    end
    LM1 = maximum(LM) + 1
    if LM1 < L_limit + 1
        for isp in 1:2
            fvL[isp] = fvL[isp][:,1:LM1]
        end
    end
    return LM1
end

function fvLDMz!(fvL::Vector{Matrix{T}},vhe::StepRangeLen,
    LM::Vector{Int64},uai::AbstractVector{T};
    L_limit::Int64=25,rel_dfLM::Real=eps0,abs_dfLM::Real=eps0) where{T}

    for isp in 1:2
        LM[isp], fvL2 = fvLDMz(fvL[isp],vhe,LM[isp],1.0,uai[isp],1.0;
                            L_limit=L_limit,rel_dfLM=rel_dfLM,abs_dfLM=abs_dfLM)
        fvL[isp][:,1:LM[isp]+1] = fvL2
    end
    LM1 = maximum(LM) + 1
    if LM1 < L_limit + 1
        for isp in 1:2
            fvL[isp] = fvL[isp][:,1:LM1]
        end
    end
    return LM1
end

"""
  Inputs:
    uai:: ≠ 0.0
    LM1:: LM1 > LM + 1

  outputs:
    LM, fvL = fvLDMz(fvL,vhe,LM,LM1,ns,nai,uai,vthi,nMod)
"""
# is_LM1_full = true
# 3.5D,  [nMod,nv,LM1,ns]
function fvLDMzfull(fvL::Vector{Matrix{T}},vhe::Vector{StepRangeLen},
    LM::Vector{Int64},LM1::Int64,ns::Int64,nai::Vector{AbstractVector{T}},
    uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64}) where{T}

    for isp in 1:ns
        fvL[isp] = fvLDMzfull(fvL[isp],vhe[isp],LM1,nai[isp],uai[isp],vthi[isp],nMod[isp])
        LM[isp] = LM1 - 1
    end
    return LM, fvL
end

function fvLDMzfull(fvL::Vector{Matrix{T}},vhe::StepRangeLen,
    LM::Vector{Int64},LM1::Int64,ns::Int64,nai::Vector{AbstractVector{T}},
    uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64}) where{T}

    for isp in 1:ns
        fvL[isp] = fvLDMzfull(fvL[isp],vhe,LM1,nai[isp],uai[isp],vthi[isp],nMod[isp])
        LM[isp] = LM1 - 1
    end
    return LM, fvL
end

# 3.5D,  [nv,LM1,ns]
function fvLDMzfull(fvL::Vector{Matrix{T}},vhe::Vector{StepRangeLen},
    LM::Vector{Int64},LM1::Int,ns::Int64,uai::Vector{AbstractVector{T}}) where{T}

    for isp in 1:ns
        fvL[isp] = fvLDMzfull(fvL[isp],vhe[isp],LM1,uai[isp])
        LM[isp] = LM1 - 1
    end
    return LM, fvL
end

function fvLDMzfull(fvL::Vector{Matrix{T}},vhe::StepRangeLen,
    LM::Vector{Int64},LM1::Int,ns::Int64,uai::Vector{AbstractVector{T}}) where{T}

    for isp in 1:ns
        fvL[isp] = fvLDMzfull(fvL[isp],vhe,LM1,uai[isp])
        LM[isp] = LM1 - 1
    end
    return LM, fvL
end


# 2.5D, [nMod,nv,LM1],     fvL = zeros(T,nvG,L_limit+1)
function fvLDMzfull(fvL::AbstractArray{T,N},vhe::StepRangeLen,LM1::Int64,
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64) where{T,N}

    k = 1
    fvL = fvLDMzfull(fvL,vhe,LM1,nai[k],uai[k],vthi[k])
    for k in 2:nMod
        if nai[k] > 0
            fvL += fvLDMzfull(zero.(fvL),vhe,LM1,nai[k],uai[k],vthi[k])
        end
    end
    return fvL
end

# 2D, [nv,LM1], nMod = 1, 
function fvLDMzfull(fvL::AbstractArray{T,N},v::StepRangeLen,
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

# 2D, [nv,LM1], nMod = 1, nai = vthi = 1
function fvLDMzfull(fvL::AbstractArray{T,N},v::StepRangeLen,LM1::Int64,uai::T) where{T,N}

    if uai == 0.0
        fvL[:,1] = fM(fvL[:,1],v)
        return fvL
    else
        for L1 in 1:LM1
            fvL[:,L1] = fvLDMz(fvL[:,L1],v,L1-1,uai)
        end
        return fvL
    end
end
