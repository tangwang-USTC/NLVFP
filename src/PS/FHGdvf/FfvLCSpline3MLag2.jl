
# IK

"""
    In Lagrange coordinate system where `nc0[isp] = nc0[iFv]` for `ns * nMod = 2`

    Type of boundary conditions:

    1, The Neumman (first) boundary condition: giving the first order derivatives:

        `∂ᵥf(v=0)` and `∂ᵥf(v=∞)`

    2, The second boundary condition: giving the second order derivatives:

        `∂²ᵥf(v=0)` and `∂²ᵥf(v=∞)`

         Especially, when `∂²ᵥf(v=0) = ∂²ᵥf(v=∞) = 0', this is the so-called

         Nature Boundary conditions (NBC).

    3, The first mixed boundary conditions:

        `∂ᵥf(v=0) = 0` and `∂²ᵥf(v=∞) = 0`

    4, he second mixed boundary conditions:

        `∂²ᵥf(v=0) = 0` and `∂ᵥf(v=∞) = 0`


"""

"""
  Warning: When `isrenormalization = true`, the left-end of domain of the velocity axis,
           `vGmax`, should be determined by `FvL[end,L1,isp]`, which will affect
           the accuracy of the Shkarosky integrals and then the accuracy of the finial results.

  Warning: The biggest relative error of `FvL` comes from the extrapolations

  Inputs:
    FvL = FvLc / cf3, the normalzied harmonic of the distribution function

  Outputs:
    FfvLCSLag!(ddfvL0,dfvL0,FvL,FvLa,vaa,nvlevel0a,ncF,
              fvL0,vhk,nc0,ocp,nvlevele,vth,uai,LM1;
              is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
              autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
              p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# When `uai[1] = - uai[2]`, `nai = vthi = 1` in the Lagrange coordinate system with relative velocity `uC`
 
# 3.5D, [nv,LM,ns=2]  (ddfvL, dfvL, FvL, FvLa, vaa, nvlevel0a, ncF), is_extrapolate_FLn = true
function FfvLCSLag!(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}},
    FvL::AbstractVector{Matrix{T}}, FvLa::Array{Any}, vaa::Vector{AbstractVector{T}}, 
    nvlevel0a::Vector{Vector{Int64}}, ncF::Vector{Int64}, fvL0::AbstractVector{Matrix{T}},
    vhk::Vector{AbstractVector{T}},nc0::Vector{Int64},ocp::Vector{Int64},
    nvlevele::Vector{Vector{Int64}},vth::AbstractVector{T},uai::Vector{T},LM1::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false) where {T}

    for isp in 1:2
        isp == 1 ? iFv = 2 : iFv = 1
        # vbath = vth[iFv] / vth[isp]
        vb = vhk[iFv] * (vth[iFv] / vth[isp])             # `= vᵦ / vath``, for the mapping process
        if abs(vabth-1) ≤ epsT10
            is_map_F = false                         # same meshgrids `vhk[isp] = vhk[iFv] * vbath`
        else
            is_map_F = true
        end
        L1 = 1
        ddfvL[isp][:, L1], dfvL[isp][:, L1], FvL[iFv][:, L1], 
                a, b, c, ncF[iFv] = FfvLCSLag(ddfvL[isp][:, L1], 
                dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], 
                vhk[isp], vb, nc0[isp], ocp[isp],nvlevele[isp],uai[isp][1],L1 - 1;
                ncF=ncF[iFv], is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
        if ncF[iFv] == 0
            for L1 in 2:LM1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCSLag(ddfvL[isp][:, L1],
                            dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], 
                            vhk[isp], vb, nc0[isp],nvlevele[isp], uai[isp][1],L1 - 1;
                            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                end
            end
            FvLa[2:LM1, iFv] .= [nothing]
        else
            FvLa[L1, iFv], vaa[iFv], nvlevel0a[iFv] = a, b, c
            for L1 in 2:LM1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], FvL[iFv][:, L1],
                            FvLa[L1, iFv] = FfvLCSLag(ddfvL[isp][:, L1], 
                            dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], 
                            vhk[isp], vb, vaa[iFv], nc0[isp],nvlevele[isp], uai[isp][1],L1 - 1;
                            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                else
                    FvLa[L1, iFv] = zero.(vaa[iFv])
                end
            end
        end
    end
    # return ddfvL, dfvL, FvL, FvLa, vaa, nvlevel0a, ncF
end

function FfvLCSLag!(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}},
    FvL::AbstractVector{Matrix{T}}, FvLa::Array{Any}, vaa::Vector{AbstractVector{T}}, 
    nvlevel0a::Vector{Vector{Int64}}, ncF::Vector{Int64}, fvL0::AbstractVector{Matrix{T}},
    vhk::AbstractVector{T},nc0::Int64,ocp::Int64,
    nvlevele::Vector{Int64},vth::AbstractVector{T},uai::Vector{T},LM1::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false) where {T}

    for isp in 1:2
        isp == 1 ? iFv = 2 : iFv = 1
        # vbath = vth[iFv] / vth[isp]
        vb = vhk * (vth[iFv] / vth[isp])             # `= vᵦ / vath``, for the mapping process
        if abs(vabth-1) ≤ epsT10
            is_map_F = false                         # same meshgrids `vhk[isp] = vhk[iFv] * vbath`
        else
            is_map_F = true
        end
        L1 = 1
        ddfvL[isp][:, L1], dfvL[isp][:, L1], FvL[iFv][:, L1], 
                a, b, c, ncF[iFv] = FfvLCSLag(ddfvL[isp][:, L1], 
                dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], 
                vhk, vb, nc0, ocp,nvlevele,uai[isp][1],L1 - 1;
                ncF=ncF[iFv], is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
        if ncF[iFv] == 0
            for L1 in 2:LM1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCSLag(ddfvL[isp][:, L1],
                            dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], 
                            vhk, vb, nc0,nvlevele, uai[isp][1],L1 - 1;
                            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                end
            end
            FvLa[2:LM1, iFv] .= [nothing]
        else
            FvLa[L1, iFv], vaa[iFv], nvlevel0a[iFv] = a, b, c
            for L1 in 2:LM1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], FvL[iFv][:, L1],
                            FvLa[L1, iFv] = FfvLCSLag(ddfvL[isp][:, L1], 
                            dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], 
                            vhk, vb, vaa[iFv], nc0,nvlevele, uai[isp][1],L1 - 1;
                            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                else
                    FvLa[L1, iFv] = zero.(vaa[iFv])
                end
            end
        end
    end
    # return ddfvL, dfvL, FvL, FvLa, vaa, nvlevel0a, ncF
end

"""
  Inputs:
    vha:
    va: 
    FLb = F̂(𝓋̂) `
    ncF::Int64, if `ncF ≥ 1`, the number of the extrapolation points where  `FLn[ncF] ≤ eps_FLn_limit`.
              will be given by the input parameter `ncF`.
              Or else, the parameter `ncF will be determined by the inner algorithm.

  Outputs:
    ddfLn0,dfLn0,fLn,FLn,FLnba,vaa,nvlevel0a,ncF = FfvLCSLag(ddfLn0,dfLn0,
            FLnb,fLn0,vha,va,nc0,ocp,nvlevele,uai,ℓ;ncF=ncF,
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    ddfLn0,dfLn0,fLn,FLn = FfvLCSLag(ddfLn0,dfLn0,FLnb,fLn0,vha,va,vaa,nc0,nvlevele,uai,ℓ;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 1D, (ddfLn0,dfLn0,FLnb, FLnba, vaa, nvlevel0a, ncF)
function FfvLCSLag(M2::AbstractVector{T}, M1::AbstractVector{T}, FLnb::AbstractVector{T}, 
    fLn0::AbstractVector{T}, vha::AbstractVector{T}, va::AbstractVector{T}, 
    nc0::Int64, ocp::Int64,nvlevele::Vector{Int64}, uai::T,ℓ::Int64; ncF::Int64=0,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false , is_map_F::Bool=true,
    nai::T=1.0, vthi::T=1.0) where {T}
    
    vG = vha[nvlevele]
    if is_normal
        yscale, ys = normalfLn(fLn0,ℓ,uai)
    else
        ys = deepcopy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vG; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end
    modelap(v) = modela(v, pa)
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vG[1] ≠ 0.0 || modelav0 === nothing
        M1[:] = modeldfLn0(vG)
        M2[:] = ForwardDiff.derivative.(modeldfLn0, vG)
    else
        M1[2:end] = modeldfLn0(vG[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vG[2:end])
        if ℓ == 0 || ℓ == 2
            M2[1] = ddfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
    end
    if ncF ≤ 0
        ncF = 0        # find the value `ncF` for `FLn` where `FLn[ncF] ≤ eps_FLn_limit`.
        # # Mapping
        if is_map_F
            va9 = deepcopy(va[end])
            va93 = va9^3
            FLnb9 = va93 * modelap(va9)
            if FLnb9 > eps_FLn_limit
                # nc0F = nc0 + ncF + 1   # The total number of the new `FLn = [FLnb;FLnba]`
                dva9 = va[end] - va[end-ocp+1]
                while FLnb9 > eps_FLn_limit
                    va9 += dva9
                    FLnb9 = va93 * modelap(va9)
                    ncF += 1
                end
                ncF += 1  # va9
                nvak = (ncF - 1) * ocp - ncF + 2
                vaa0 = range(va[end], step=dva9, length=ncF)
                vaa, nvlevel0a = zeros(T, nvak), zeros(Int, ncF)
                vaa, nvlevel0a = vCmapping(vaa, nvlevel0a, vaa0, ncF, ocp)
                if va[1] ≠ 0.0 || modelav0 === nothing
                    FLnb[:] = modelap(va)
                else
                    FLnb[2:end] = modelap(va[2:end])
                    if ℓ == 0
                        FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
                    else
                        FLnb[1] = 0.0
                    end
                end
                return M2, M1, FLnb, modelap(vaa), vaa, nvlevel0a, ncF
                FLnba = modelap(vaa)
                return M2, M1, FLnb, FLnba, vaa, nvlevel0a, ncF
            else
                if va[1] ≠ 0.0 || modelav0 === nothing
                    FLnb[:] = modelap(va)
                else
                    FLnb[2:end] = modelap(va[2:end])
                    if ℓ == 0
                        FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
                    else
                        FLnb[1] = 0.0
                    end
                end
                return M2, M1, FLnb, [nothing], [nothing], [nothing], ncF
            end
        else
            if vha[1] ≠ 0.0 || modelav0 === nothing
                FLnb[:] = modelap(vha)
            else
                FLnbM0[2:end] = modelap(vha[2:end])
                if ℓ == 0
                    FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
                else
                    FLnb[1] = 0.0
                end
            end
            return M2, M1, FLnb, [nothing], [nothing], [nothing], ncF
        end
    else 
        dva9 = va[end] - va[end-1]
        vaa0 = range(va[end], step=dva9, length=ncF)
        nvak = (ncF - 1) * ocp - ncF + 2
        vaa, nvlevel0a = zeros(T, nvak), zeros(Int, ncF)
        vaa, nvlevel0a = vCmapping(vaa, nvlevel0a, vaa0, ncF, ocp)
        if va[1] ≠ 0.0 || modelav0 === nothing
            FLnb[:] = modelap(va)
        else
            FLnb[2:end] = modelap(va[2:end])
            if ℓ == 0
                FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
            else
                FLnb[1] = 0.0
            end
        end
        # Check whether `FLnb[end] ≪ eps(T)`
        if FLnb[end] ≥ eps_FLn_limit
            @warn("Error: `FLn[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
            @show ncF, va[end], FLnb[end]
        end
        return M2, M1, FLnb, modelap(vaa), vaa, nvlevel0a, ncF
        FLnba = modelap(vaa)
        return M2, M1, FLnb, FLnba, vaa, nvlevel0a, ncF
    end
end

# 1D, (ddfLn0,dfLn0,FLnb), vaa
function FfvLCSLag(M2::AbstractVector{T}, M1::AbstractVector{T}, FLnb::AbstractVector{T}, 
    fLn0::AbstractVector{T}, vha::AbstractVector{T}, va::AbstractVector{T}, 
    vaa::AbstractVector{T}, nc0::Int64, nvlevele::Vector{Int64},uai::T,ℓ::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false , is_map_F::Bool=true,
    nai::T=1.0, vthi::T=1.0) where {T}
    
    vG = vha[nvlevele]
    if is_normal
        yscale, ys = normalfLn(fLn0,ℓ,uai)
    else
        ys = deepcopy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vG; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale                  
    end
    modelap(v) = modela(v, pa)
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vG[1] ≠ 0.0 || modelav0 === nothing
        M1[:] = modeldfLn0(vG)
        M2[:] = ForwardDiff.derivative.(modeldfLn0, vG)
    else
        M1[2:end] = modeldfLn0(vG[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vG[2:end])
        if ℓ == 0 || ℓ == 2
            # M2[1] = ddfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
    end
    if va[1] ≠ 0.0 || modelav0 === nothing
        FLnb[:] = modelap(va)
    else
        FLnb[2:end] = modelap(va[2:end])
        if ℓ == 0
            FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
        else
            FLnb[1] = 0.0
        end
    end
    FLnba = modelap(vaa)
    # Check whether `FLnba[end] ≪ eps(T)`
    if FLnba[end] ≥ eps_FLn_limit
        @warn("Error: `FLnba[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
        @show "vaa", va[end], FLnba[end]
    end
    return M2, M1, FLnb, FLnba
end

"""
  Inputs:
    vha:
    va: 
    FLnb = F̂(𝓋̂) `

  Outputs:
    FfvLCSLag!(ddfvL0,dfvL0,FvL,FvLa,vaa,nvlevel0a,ncF,
            fvL0,vhk,nc0,ocp,nvlevele,vth,uai,LM1;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    ddfLn0,dfLn0,fLn,FLn = FfvLCSLag(ddfLn0,dfLn0,FLnb,fLn0,vha,va,nc0,nvlevele,uai,ℓ;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 3.5D, [nv,LM,ns=2], (ddfvL, dfvL, FvL), is_extrapolate_FLn = false, where `uai[1] = - uai[2]`, 
function FfvLCSLag!(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}},
    FvL::AbstractVector{Matrix{T}}, fvL0::AbstractVector{Matrix{T}},
    vhk::Vector{AbstractVector{T}},nc0::Vector{Int64}, 
    nvlevele::Vector{Vector{Int64}},vth::AbstractVector{T},uai::Vector{T},LM1::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false) where {T}

    for isp in 1:2
        isp == 1 ? iFv = 2 : iFv = 1
        # vbath = vth[iFv] / vth[isp]
        vb = vhk[iFv] * (vth[iFv] / vth[isp])             # `= vᵦ / vath``, for the mapping process
        for L1 in 1:LM1
            if norm(fvL0[isp][:, L1]) ≥ epsT
                ddfvL[isp][:, L1], dfvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCSLag(ddfvL[isp][:, L1],
                        dfvL[isp][:, L1],  FvL[iFv][:, L1], fvL0[isp][:, L1], 
                        vhk[isp], vb, nc0[isp],nvlevele[isp], uai[isp][1],L1 - 1;
                        is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
            end
        end
    end
    # return ddfvL, dfvL, FvL
end
function FfvLCSLag!(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}},
    FvL::AbstractVector{Matrix{T}}, fvL0::AbstractVector{Matrix{T}},
    vhk::AbstractVector{T},nc0::Int64,
    nvlevele::Vector{Int64},vth::AbstractVector{T},uai::Vector{T},LM1::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false) where {T}

    for isp in 1:2
        isp == 1 ? iFv = 2 : iFv = 1
        # vbath = vth[iFv] / vth[isp]
        vb = vhk * (vth[iFv] / vth[isp])             # `= vᵦ / vath``, for the mapping process
        for L1 in 1:LM1
            if norm(fvL0[isp][:, L1]) ≥ epsT
                ddfvL[isp][:, L1], dfvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCSLag(ddfvL[isp][:, L1],
                        dfvL[isp][:, L1],  FvL[iFv][:, L1], fvL0[isp][:, L1], vhk, vb, nc0,nvlevele, uai[isp][1],L1 - 1;
                        is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
            end
        end
    end
    # return ddfvL, dfvL, FvL
end

"""
"""

# # 1D, (ddfLn0,dfLn0,FLnb)
function FfvLCSLag(M2::AbstractVector{T}, M1::AbstractVector{T}, FLnb::AbstractVector{T}, 
    fLn0::AbstractVector{T}, vha::AbstractVector{T}, va::AbstractVector{T}, 
    nc0::Int64,nvlevele::Vector{Int64}, uai::T,ℓ::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false , is_map_F::Bool=true,
    nai::T=1.0, vthi::T=1.0) where {T}
    
    vG = vha[nvlevele]

    if is_normal
        yscale, ys = normalfLn(fLn0,ℓ,uai)
    else
        ys = deepcopy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vG; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end
    modelap(v) = modela(v, pa)
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vG[1] ≠ 0.0 || modelav0 === nothing
        M1[:] = modeldfLn0(vG)
        M2[:] = ForwardDiff.derivative.(modeldfLn0, vG)
    else
        M1[2:end] = modeldfLn0(vG[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vG[2:end])
        if ℓ == 0 || ℓ == 2
            M2[1] = ddfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
    end
    if is_map_F
        # # Mapping
        if va[1] ≠ 0.0 || modelav0 === nothing
            FLnb[:] = modelap(va)
        else
            FLnb[2:end] = modelap(va[2:end])
            if ℓ == 0
                FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
            else
                FLnb[1] = 0.0
            end
        end
        # # Check whether `FLnb[end] ≪ eps(T)`
        # if FLnb[end] ≥ eps_FLn_limit
        #     @warn("Error: `FLn[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
        #     @show FLnb[end]
        # end
    else
        if vha[1] ≠ 0.0 || modelav0 === nothing
            FLnb[:] = modelap(vha)                        # M0
        else
            FLnb[2:end] = modelap(vha[2:end])
            if ℓ == 0
                FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
            else
                FLnb[1] = 0.0
            end
        end
    end
    return M2, M1, FLnb
end

