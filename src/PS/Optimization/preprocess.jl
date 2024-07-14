
"""
  Normalization of the original datas `fLn` and filtering the normalized data.

  Inputs:
  Outputs: 
    yscale, ys = normalfLn(fLn,ℓ,uai)
    yscale, ys = normalfLn(fLn,ℓ,uai,nMod)
    ys, vs = filterfLn(fLn,vG;n10=n10,dnvs=dnvs)
"""

function normalfLn(fLn::AbstractVector{T},ℓ::Int64,uai::TA,nMod::Int64) where{T,TA}
    
    if nMod == 1
        if uai[1] ≥ 0.0
            yscale = maximum(fLn)
            yscale = 10^(round(log10(yscale)))
        else
            if isodd(ℓ)
                yscale = maximum(- fLn)
                yscale = - 10^(round(log10(yscale)))
            else
                yscale = maximum(fLn)
                yscale = 10^(round(log10(yscale)))
            end
        end
        if yscale ≠ 1.0
            return yscale, fLn / yscale
        else
            return yscale, deepcopy(fLn)
        end
    else
        if isodd(ℓ)
            yscale, id = findmax(abs.(fLn))
            yscale = sign(fLn[id]) * 10^(round(log10(yscale)))
        else
            yscale = maximum(fLn)
            yscale = 10^(round(log10(yscale)))
        end
        if yscale ≠ 1.0
            return yscale, fLn / yscale
        else
            return yscale, deepcopy(fLn)
        end
    end
end

# nMod = 1
function normalfLn(fLn::AbstractVector{T},ℓ::Int64,uai::T) where{T}
    
    if uai ≥ 0.0
        yscale = maximum(fLn)
        yscale = 10^(round(log10(yscale)))
    else
        if isodd(ℓ)
            yscale = maximum(- fLn)
            yscale = - 10^(round(log10(yscale)))
        else
            yscale = maximum(fLn)
            yscale = 10^(round(log10(yscale)))
        end
    end
    if yscale ≠ 1.0
        return yscale, fLn / yscale
    else
        return yscale, deepcopy(fLn)
    end
end
function filterfLn(fLn::AbstractVector{T},vG::AbstractVector{T};
    isfilter0::Bool=true,n10::Int64=1,dnvs::Int64=1) where{T}

    # filter
    if n10 < 0
        is_ys_0 = fLn .> epsT * 10.0^-(n10)
    else
        is_ys_0 = fLn .> epsT * 10.0^-(5 * n10)
    end
    if isfilter0 && vG[1] == 0.0
        is_ys_0[1] = false
    end
    if dnvs == 1
        return deepcopy(fLn[is_ys_0]), deepcopy(vG[is_ys_0])
    elseif dnvs ≥ 2
        return deepcopy(fLn[is_ys_0][1:dnvs:end]), deepcopy(vG[is_ys_0][1:dnvs:end])
    else
        @error("dnvs must be no larger than `0`!")
    end
end
