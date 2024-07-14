

"""
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

  Warning: When multi-modules (nMod ≥ 2) is applied and the disparities of
           `vthi[k], k=1:1:nMod` is so big that the `old problem`, a suitable lef-endpoint,
           will be important to the algorithm. The the basic meshgrids
           decided by the bigest `vabth[k]` is more safe but maybe less efficient.


  Warning: The biggest relative error of `FvL` comes from the extrapolations

            when `vabth > 1.0` which is calculated by (1~3)ᵗʰ order spline interpolations of the weighted function now.


  Inputs:
    nai:

  Outputs:
    ddfvL0,dfvL0,fvL,FvL = FfvLCS(ddfvL0,dfvL0,fvL,FvL,fvL0,vGk,
              nc0,ocp,nvlevele,vth,nai,uai,vthi,LM,LM1,ns,nMod;
              isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
              autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
              p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 3.5D, [nMod]  (ddfvL, dfvL, fvL, FvL, FvLa, vaa, nvlevel0a, ncF)
function FfvLCS(ddfvL::AbstractArray{T,N}, dfvL::AbstractArray{T,N}, fvL::AbstractArray{T,N},
    FvL::AbstractArray{T,N},fvL0::AbstractArray{T,N},vGk::AbstractVector{Tb},nc0::Int64,ocp::Int64,
    nvlevele::Vector{Int},vth::Vector{T},nai::AbstractVector{TA},uai::AbstractVector{TA},
    vthi::AbstractVector{TA},LM::Vector{Int}, LM1::Int64, ns::Int64, nMod::Vector{Int};
    isnormal::Bool=true, restartfit::Vector{Int}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1) where {T,TA,N,Tb}

    nsp_vec = 1:ns
    vGe = vGk[nvlevele]
    ncF, vaa = zeros(Int, ns), Vector((undef), ns)
    FvLa = Array{Any}(undef, LM1, ns)
    nvlevel0a = Vector((undef), ns)
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec.≠isp]
        iFv = nspF[1]
        vbath = vth[iFv] / vth[isp]
        vb = vGk * vbath
        if nMod[isp] == 1
            L1 = 1
            ddfvL[:, L1, isp], dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv], FvLa[L1, iFv],
                vaa[iFv], nvlevel0a[iFv], ncF[iFv] = FfvLCS(ddfvL[:, L1, isp],
                    dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv], fvL0[:, L1, isp], vGe, vb,
                    nc0, ocp, vbath,nai[isp][1],uai[isp][1],vthi[isp][1],L1 - 1;
                    ncF=ncF[iFv], isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
                    autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
            if ncF[iFv] == 0
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[:, L1, isp]) ≥ epsT
                        ddfvL[:, L1, isp], dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv] = FfvLCS(ddfvL[:, L1, isp],
                                dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv], fvL0[:, L1, isp], vGe, vb,
                                nc0, vbath, nai[isp][1],uai[isp][1],vthi[isp][1],L1 - 1;
                                isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
                                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
                    end
                end
                FvLa[2:LM1, iFv] .= [nothing]
            else
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[:, L1, isp]) ≥ epsT
                        ddfvL[:, L1, isp], dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv],
                            FvLa[L1, iFv] = FfvLCS(ddfvL[:, L1, isp], dfvL[:, L1, isp],
                                fvL[:, L1, isp], FvL[:, L1, iFv], fvL0[:, L1, isp], vGe, vb, vaa[iFv],
                                nc0, vbath,nai[isp][1],uai[isp][1],vthi[isp][1],L1 - 1;
                                isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
                                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
                    else
                        FvLa[L1, iFv] = zero.(vaa[iFv])
                    end
                end
                if LM[isp]+1 ≠ LM1
                    FvLa[LM[isp]+2:LM1, iFv] .= [nothing]
                end
            end
        else
            L1 = 1
            ddfvL[:, L1, isp], dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv], FvLa[L1, iFv],
                vaa[iFv], nvlevel0a[iFv], ncF[iFv] = FfvLCS(ddfvL[:, L1, isp],
                    dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv], fvL0[:, L1, isp], vGe, vb,
                    nc0, ocp, vbath, nai[isp],uai[isp],vthi[isp],L1 - 1, nMod[isp];
                    ncF=ncF[iFv], isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
                    autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
            if ncF[iFv] == 0
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[:, L1, isp]) ≥ epsT
                        ddfvL[:, L1, isp], dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv] = FfvLCS(ddfvL[:, L1, isp],
                                dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv], fvL0[:, L1, isp], vGe, vb, nc0,
                                vbath, nai[isp],uai[isp],vthi[isp],L1 - 1, nMod[isp];
                                isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
                                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
                    end
                end
                FvLa[2:LM1, iFv] .= [nothing]
            else
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[:, L1, isp]) ≥ epsT
                        ddfvL[:, L1, isp], dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv],
                          FvLa[L1, iFv] = FfvLCS(ddfvL[:, L1, isp], dfvL[:, L1, isp],
                            fvL[:, L1, isp], FvL[:, L1, iFv], fvL0[:, L1, isp], vGe, vb, vaa[iFv],
                            nc0, vbath, nai[isp],uai[isp],vthi[isp],L1 - 1, nMod[isp];
                            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
                    else
                        FvLa[L1, iFv] = zero.(vaa[iFv])
                    end
                end
                if LM[isp]+1 ≠ LM1
                    FvLa[LM[isp]+2:LM1, iFv] .= [nothing]
                end
            end
        end
    end
    return ddfvL, dfvL, fvL, FvL, FvLa, vaa, nvlevel0a, ncF
end

"""
  Inputs:
    vGe:
    va: `va = vGk * vabth = vGk * (vth[isp] / vth[iFv])` when `fLn0 = fvL0[:,L1,iFv]` gives `FvL[:,L1,isp]`
         which is equivalent to `vb = vGk / vabth` when `fLn0 = fvL0[:,L1,isp]` gives `FvL[:,L1,iFv]`
    fLn0: fLn0e = fLn[nvlevele]
    FLn:  FvL(va=vGk*vabth)
    ncF::Int64, if `ncF ≥ 1`, the number of the extrapolation points where  `FLn[ncF] ≤ epsT / 10`.
              will be given by the input parameter `ncF`.
              Or else, the parameter `ncF will be determined by the inner algorithm.

  Outputs:
    `FLnb = F̂(𝓋̂) `
    ddfLn0,dfLn0,fLn,FLn,FLnba,vaa,nvlevel0a,ncF = FfvLCS(ddfLn0,dfLn0,fLn,
            FLnb,fLn0,vGe,va,nc0,ocp,vabth,nai,uai,vthi,ℓ,nMod;
            ncF=ncF,isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 1.5D, [nMod], (ddfLn0,dfLn0,fLnnew, FLnb, FLnba, vaa, nvlevel0a, ncF)
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T}, M0::AbstractVector{T},
    FLnb::AbstractVector{T}, fLn0::AbstractVector{T}, vGe::AbstractVector{Tb},
    va::AbstractVector{Tb}, nc0::Int64, ocp::Int64,vabth::T,nai::AbstractVector{TA},
    uai::AbstractVector{TA},vthi::AbstractVector{TA},ℓ::Int64, nMod::Int64; ncF::Int64=0,
    isnormal::Bool=true, restartfit::Vector{Int}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1) where {T,Tb,TA}

    if isnormal
        yscale, ys = normalfLn(fLn0,ℓ,uai,nMod)
    else
        ys = copy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vGe; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end 
    modelap(v) = modela(v, pa)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vGe)
    else
        M0 = zero.(vGe)
        M0[2:end] = modelap(vGe[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai,uai,vthi, nMod)
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vGe)
        M2 = ForwardDiff.derivative.(modeldfLn0, vGe)
    else
        M1 = zeros(T, nc0)
        M1[2:end] = modeldfLn0(vGe[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai,uai,vthi, nMod)
        end
        M2 = zeros(T, nc0)
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vGe[2:end])
        if ℓ == 0 || ℓ == 2
            M2[1] = ddfLnDMv0(nai,uai,vthi, ℓ, nMod)
        end
    end
    if ncF ≤ 0
        ncF = 0        # find the value `ncF` for `FLn` where `FLn[ncF] ≤ epsT / 10`.
        if vabth == 1.0
            return M2, M1, M0, M0, [nothing], [nothing], [nothing], ncF
        else
            # # Mapping
            FLnb9 = modelap(va[end])
            if FLnb9 > epsT / 10
                # nc0F = nc0 + ncF + 1   # The total number of the new `FLn = [FLnb;FLnba]`
                va9 = copy(va[end])
                dva9 = va[end] - va[end-ocp+1]
                while FLnb9 > epsT / 10
                    va9 += dva9
                    FLnb9 = modelap(va9)
                    ncF += 1
                end
                ncF += 1
                nvak = (ncF - 1) * ocp - ncF + 2
                vaa0 = range(va[end], step=dva9, length=ncF)
                vaa, nvlevel0a = zeros(T, nvak), zeros(Int, ncF)
                vaa, nvlevel0a = vCmapping(vaa, nvlevel0a, vaa0, ncF, ocp)
                if va[1] ≠ 0.0 || modelav0 === nothing
                    FLnb = modelap(va)
                else
                    FLnb[2:end] = modelap(va[2:end])
                    if ℓ == 0
                        FLnb[1] = M0[1]
                        # FLnb[1] = fLnDMv0(nai,uai,vthi,nMod)
                    else
                        FLnb[1] = 0.0
                    end
                end
                return M2, M1, M0, FLnb, modelap(vaa), vaa,  nvlevel0a, ncF
                FLnba = modelap(vaa)
                return M2, M1, M0, FLnb, FLnba, vaa, nvlevel0a, ncF
            else
                if va[1] ≠ 0.0 || modelav0 === nothing
                    FLnb = modelap(va)
                else
                    FLnb[2:end] = modelap(va[2:end])
                    if ℓ == 0
                        FLnb[1] = M0[1]
                        # FLnb[1] = fLnDMv0(nai,uai,vthi,nMod)
                    else
                        FLnb[1] = 0.0
                    end
                end
                return M2, M1, M0, FLnb, [nothing], [nothing], [nothing], ncF
            end
        end
    else
        dva9 = va[end] - va[end-1]
        vaa0 = range(va[end], step=dva9, length=ncF)
        nvak = (ncF - 1) * ocp - ncF + 2
        vaa, nvlevel0a = zeros(T, nvak), zeros(Int, ncF)
        vaa, nvlevel0a = vCmapping(vaa, nvlevel0a, vaa0, ncF, ocp)
        if va[1] ≠ 0.0 || modelav0 === nothing
            FLnb = modelap(va)
        else
            FLnb[2:end] = modelap(va[2:end])
            if ℓ == 0
                FLnb[1] = M0[1]
                # FLnb[1] = fLnDMv0(nai,uai,vthi,nMod)
            else
                FLnb[1] = 0.0
            end
        end
        # Check whether `FLnb[end] ≪ eps(T)`
        if FLnb[end] ≥ epsT / 10
            @warn("Error: `FLn[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
            @show "nMod,ncF", va[end], FLnb[end]
        end
        return M2, M1, M0, FLnb, modelap(vaa), vaa, nvlevel0a, ncF
        FLnba = modelap(vaa)
        return M2, M1, M0, FLnb, FLnba, vaa, nvlevel0a, ncF
    end
end

# 1.5D, [nMod], (ddfLn0,dfLn0,fLnnew,FLnb), vaa
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T}, M0::AbstractVector{T},
    FLnb::AbstractVector{T}, fLn0::AbstractVector{T}, vGe::AbstractVector{Tb},
    va::AbstractVector{Tb}, vaa::AbstractVector{Tb}, nc0::Int64,vabth::T,nai::AbstractVector{TA},
    uai::AbstractVector{TA},vthi::AbstractVector{TA},ℓ::Int64,nMod::Int64;
    isnormal::Bool=true, restartfit::Vector{Int}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1) where {T,Tb,TA}

    if isnormal
        yscale, ys = normalfLn(fLn0,ℓ,uai,nMod)
    else
        ys = copy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vGe; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end
    modelap(v) = modela(v, pa)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vGe)
    else
        M0 = zero.(vGe)
        M0[2:end] = modelap(vGe[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai,uai,vthi, nMod)
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vGe)
        M2 = ForwardDiff.derivative.(modeldfLn0, vGe)
    else
        M1 = zeros(T, nc0)
        M1[2:end] = modeldfLn0(vGe[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai,uai,vthi, nMod)
        end
        M2 = zeros(T, nc0)
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vGe[2:end])
        if ℓ == 0 || ℓ == 2
            a = ddfLnDMv0(nai,uai,vthi, ℓ, nMod)
            M2[1] = a
        end
    end
    if va[1] ≠ 0.0 || modelav0 === nothing
        FLnb = modelap(va)
    else
        FLnb[2:end] = modelap(va[2:end])
        if ℓ == 0
            FLnb[1] = M0[1]
            # FLnb[1] = fLnDMv0(nai,uai,vthi,nMod)
        else
            FLnb[1] = 0.0
        end
    end
    FLnba = modelap(vaa)
    # Check whether `FLnba[end] ≪ eps(T)`
    if FLnba[end] ≥ epsT / 10
        @warn("Error: `FLnba[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
        @show "nMod,vaa", va[end], FLnba[end]
    end
    return M2, M1, M0, FLnb, modelap(vaa)
end

"""
  Inputs:
    vGe:
    va: `va = vGk * vabth = vGk * (vth[isp] / vth[iFv])` when `fLn0 = fvL0[:,L1,iFv]` gives `FvL[:,L1,isp]`
         which is equivalent to `vb = vGk / vabth` when `fLn0 = fvL0[:,L1,isp]` gives `FvL[:,L1,iFv]`
    fLn0:
    FLnb = F̂(𝓋̂)

  Outputs:
    ddfLn0,dfLn0,fLn,FLn,FLnba, vaa, nvlevel0a, ncF = FfvLCS(ddfLn0,dfLn0,
            fLn,FLnb,fLn0,vGe,va,nc0,ocp,vabth,nai,uai,vthi,ℓ,nMod;
            isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 1.5D, [nMod], (ddfLn0,dfLn0,fLnnew,FLnb)
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T}, M0::AbstractVector{T},
    FLnb::AbstractVector{T}, fLn0::AbstractVector{T}, vGe::AbstractVector{Tb},
    va::AbstractVector{Tb}, nc0::Int64, vabth::T,nai::AbstractVector{TA},
    uai::AbstractVector{TA},vthi::AbstractVector{TA},ℓ::Int64, nMod::Int64;
    isnormal::Bool=true, restartfit::Vector{Int}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1) where {T,Tb,TA}

    if isnormal
        yscale, ys = normalfLn(fLn0,ℓ,uai,nMod)
    else
        ys = copy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vGe; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end
    modelap(v) = modela(v, pa)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vGe)
    else
        M0 = zero.(vGe)
        M0[2:end] = modelap(vGe[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai,uai,vthi, nMod)
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vGe)
        M2 = ForwardDiff.derivative.(modeldfLn0, vGe)
    else
        M1 = zeros(T, nc0)
        M1[2:end] = modeldfLn0(vGe[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai,uai,vthi, nMod)
        end
        M2 = zeros(T, nc0)
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vGe[2:end])
        if ℓ == 0 || ℓ == 2
            M2[1] = ddfLnDMv0(nai,uai,vthi, ℓ, nMod)
        end
    end
    if vabth == 1.0
        return M2, M1, M0, M0
    else
        # # Mapping
        if va[1] ≠ 0.0 || modelav0 === nothing
            FLnb = modelap(va)
        else
            FLnb[2:end] = modelap(va[2:end])
            if ℓ == 0
                FLnb[1] = M0[1]
                # FLnb[1] = fLnDMv0(nai,uai,vthi,nMod)
            else
                FLnb[1] = 0.0
            end
        end
        # Check whether `FLnb[end] ≪ eps(T)`
        if FLnb[end] ≥ epsT / 10
            @warn("Error: `FLn[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
            @show "nMod", va[end], FLnb[end]
        end
        return M2, M1, M0, FLnb
    end
end

"""
  Warning: When `isrenormalization = true`, the left-end of domain of the velocity axis,
           `vGmax`, should be determined by `FvL[end,L1,isp]`, which will affect
           the accuracy of the Shkarosky integrals and then the accuracy of the finial results.

  Warning: When multi-modules (nMod ≥ 2) is applied and the disparities of
           `vthi[k], k=1:1:nMod` is so big that the `old problem`, a suitable lef-endpoint,
           will be important to the algorithm. The the basic meshgrids
           decided by the bigest `vabth[k]` is more safe but maybe less efficient.

  Warning: The biggest relative error of `FvL` comes from the extrapolations

            when `vabth > 1.0` which is calculated by (1~3)ᵗʰ order spline interpolations of the weighted function now.

  Inputs:
    nai:

  Outputs:
    ddfvL0,dfvL0,fvL,FvL,vaa,FvLa,nvlevel0a,ncF = FfvLCS(ddfvL0,dfvL0,
              fvL,FvL,fvL0,vGk,nc0,ocp,nvlevele,vth,nai,uai,vthi,LM,LM1,ns;
              isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
              autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
              p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 3D  (ddfvL0,dfvL0,fvL,FvL), nMod[:] .== 1
function FfvLCS(ddfvL::AbstractArray{T,N}, dfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},
    FvL::AbstractArray{T,N}, fvL0::AbstractArray{T,N},vGk::AbstractVector{Tb},
    nc0::Int64,ocp::Int64,nvlevele::Vector{Int}, vth::Vector{T},nai::AbstractVector{TA},
    uai::AbstractVector{TA},vthi::AbstractVector{TA},LM::Vector{Int}, LM1::Int64, ns::Int64;
    isnormal::Bool=true, restartfit::Vector{Int}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18, n10::Int64=1, dnvs::Int64=1) where {T,TA,N,Tb}

    nsp_vec = 1:ns
    vGe = vGk[nvlevele]
    ncF, vaa = zeros(Int, ns), Vector((undef), ns)
    FvLa = Array{Any}(undef, LM1, ns)
    nvlevel0a = Vector((undef), ns)
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec.≠isp]
        iFv = nspF[1]
        vbath = vth[iFv] / vth[isp]
        vb = vGk * vbath
        L1 = 1
        ddfvL[:, L1, isp], dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv], FvLa[L1, iFv],
              vaa[iFv], nvlevel0a[iFv], ncF[iFv] = FfvLCS(ddfvL[:, L1, isp],
                dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv], fvL0[:, L1, isp], vGe, vb,
                nc0, ocp, vbath,nai[isp][1],uai[isp][1],vthi[isp][1],L1 - 1; 
                ncF=ncF[iFv],isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
        if ncF[iFv] == 0
            for L1 in 2:LM[isp]+1
                if norm(fvL0[:, L1, isp]) ≥ epsT
                    ddfvL[:, L1, isp], dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv] = FfvLCS(ddfvL[:, L1, isp],
                            dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv], fvL0[:, L1, isp], vGe, vb,
                            nc0, vbath,nai[isp][1],uai[isp][1],vthi[isp][1],L1 - 1;
                            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
                end
            end
            FvLa[2:LM1, iFv] .= [nothing]
        else
            for L1 in 2:LM[isp]+1
                if norm(fvL0[:, L1, isp]) ≥ epsT
                    ddfvL[:, L1, isp], dfvL[:, L1, isp], fvL[:, L1, isp], FvL[:, L1, iFv],
                          FvLa[L1, iFv] = FfvLCS(ddfvL[:, L1, isp], dfvL[:, L1, isp],
                            fvL[:, L1, isp], FvL[:, L1, iFv], fvL0[:, L1, isp], vGe, vb, vaa[iFv],
                            nc0, vbath,nai[isp][1],uai[isp][1],vthi[isp][1],L1 - 1;
                            isnormal=isnormal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs)
                else
                    FvLa[L1, iFv] = zero.(vaa[iFv])
                end
            end
            if LM[isp]+1 ≠ LM1
                FvLa[LM[isp]+2:LM1, iFv] .= [nothing]
            end
        end
    end
    return ddfvL, dfvL, fvL, FvL, FvLa, vaa, nvlevel0a, ncF
end


"""
  Inputs:
    vGe:
    va: `va = vGk * vabth = vGk * (vth[isp] / vth[iFv])` when `fLn0 = fvL0[:,L1,iFv]` gives `FvL[:,L1,isp]`
         which is equivalent to `vb = vGk / vabth` when `fLn0 = fvL0[:,L1,isp]` gives `FvL[:,L1,iFv]`
    fLn0:
    FLb = F̂(𝓋̂) `
    ncF::Int64, if `ncF ≥ 1`, the number of the extrapolation points where  `FLn[ncF] ≤ epsT / 10`.
              will be given by the input parameter `ncF`.
              Or else, the parameter `ncF will be determined by the inner algorithm.

  Outputs:
    ddfLn0,dfLn0,fLn,FLn,FLnba,vaa,nvlevel0a,ncF = FfvLCS(ddfLn0,dfLn0,
            fLn,FLnb,fLn0,vGe,va,nc0,ocp,vabth,nai,uai,vthi,ℓ;ncF=ncF,
            isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    ddfLn0,dfLn0,fLn,FLn = FfvLCS(ddfLn0,dfLn0,fLn,
            FLnb,fLn0,vGe,va,vaa,nc0,vabth,nai,uai,vthi,ℓ,
            isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 1D, (ddfLn0,dfLn0,fLnnew,FLnb,ncF)
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T}, M0::AbstractVector{T},
    FLnb::AbstractVector{T}, fLn0::AbstractVector{T}, vGe::AbstractVector{Tb},
    va::AbstractVector{Tb}, nc0::Int64, ocp::Int64,vabth::T,nai::T,uai::T,vthi::T,ℓ::Int64; 
    ncF::Int64=0,isnormal::Bool=true, restartfit::Vector{Int}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1) where {T,Tb}

    if isnormal
        yscale, ys = normalfLn(fLn0,ℓ,uai)
    else
        ys = copy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vGe; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end
    modelap(v) = modela(v, pa)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vGe)
    else
        M0 = zero.(vGe)
        M0[2:end] = modelap(vGe[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai[1],uai[1],vthi[1])
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vGe)
        M2 = ForwardDiff.derivative.(modeldfLn0, vGe)
    else
        M1 = zeros(T, nc0)
        M1[2:end] = modeldfLn0(vGe[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
        M2 = zeros(T, nc0)
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vGe[2:end])
        if ℓ == 0 || ℓ == 2
            M2[1] = ddfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
    end
    if ncF ≤ 0
        ncF = 0        # find the value `ncF` for `FLn` where `FLn[ncF] ≤ epsT / 10`.
        if vabth == 1.0
            return M2, M1, M0, M0, [nothing], [nothing], [nothing], ncF
        else
            # # Mapping
            FLnb9 = modelap(va[end])
            if FLnb9 > epsT / 10
                # nc0F = nc0 + ncF + 1   # The total number of the new `FLn = [FLnb;FLnba]`
                va9 = copy(va[end])
                dva9 = va[end] - va[end-ocp+1]
                while FLnb9 > epsT / 10
                    va9 += dva9
                    FLnb9 = modelap(va9)
                    ncF += 1
                end
                ncF += 1  # va[end]
                nvak = (ncF - 1) * ocp - ncF + 2
                vaa0 = range(va[end], step=dva9, length=ncF)
                vaa, nvlevel0a = zeros(T, nvak), zeros(Int, ncF)
                vaa, nvlevel0a = vCmapping(vaa, nvlevel0a, vaa0, ncF, ocp)
                if va[1] ≠ 0.0 || modelav0 === nothing
                    FLnb = modelap(va)
                else
                    FLnb[2:end] = modelap(va[2:end])
                    if ℓ == 0
                        FLnb[1] = M0[1]
                        # FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
                    else
                        FLnb[1] = 0.0
                    end
                end
                return M2, M1, M0, FLnb, modelap(vaa), vaa, nvlevel0a, ncF
                FLnba = modelap(vaa)
                return M2, M1, M0, FLnb, FLnba, vaa, nvlevel0a, ncF
            else
                if va[1] ≠ 0.0 || modelav0 === nothing
                    FLnb = modelap(va)
                else
                    FLnb[2:end] = modelap(va[2:end])
                    if ℓ == 0
                        FLnb[1] = M0[1]
                        # FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
                    else
                        FLnb[1] = 0.0
                    end
                end
                return M2, M1, M0, FLnb, [nothing], [nothing], [nothing], ncF
            end
        end
    else
        dva9 = va[end] - va[end-1]
        vaa0 = range(va[end], step=dva9, length=ncF)
        nvak = (ncF - 1) * ocp - ncF + 2
        vaa, nvlevel0a = zeros(T, nvak), zeros(Int, ncF)
        vaa, nvlevel0a = vCmapping(vaa, nvlevel0a, vaa0, ncF, ocp)
        if va[1] ≠ 0.0 || modelav0 === nothing
            FLnb = modelap(va)
        else
            FLnb[2:end] = modelap(va[2:end])
            if ℓ == 0
                FLnb[1] = M0[1]
                # FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
            else
                FLnb[1] = 0.0
            end
        end
        # Check whether `FLnb[end] ≪ eps(T)`
        if FLnb[end] ≥ epsT / 10
            @warn("Error: `FLn[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
            @show "ncF", va[end], FLnb[end]
        end
        return M2, M1, M0, FLnb, modelap(vaa), vaa, nvlevel0a, ncF
        FLnba = modelap(vaa)
        return M2, M1, M0, FLnb, FLnba, vaa, nvlevel0a, ncF
    end
end

# 1D, (ddfLn0,dfLn0,fLnnew,FLnb), vaa
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T}, M0::AbstractVector{T},
    FLnb::AbstractVector{T}, fLn0::AbstractVector{T}, vGe::AbstractVector{Tb},
    va::AbstractVector{Tb}, vaa::AbstractVector{Tb}, nc0::Int64,
    vabth::T,nai::T,uai::T,vthi::T,ℓ::Int64;
    isnormal::Bool=true, restartfit::Vector{Int}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1) where {T,Tb,N}

    if isnormal
        yscale, ys = normalfLn(fLn0,ℓ,uai)
    else
        ys = copy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vGe; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale                  
    end
    modelap(v) = modela(v, pa)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vGe)
    else
        M0 = zero.(vGe)
        M0[2:end] = modelap(vGe[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai[1],uai[1],vthi[1])
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vGe)
        M2 = ForwardDiff.derivative.(modeldfLn0, vGe)
    else
        M1 = zeros(T, nc0)
        M1[2:end] = modeldfLn0(vGe[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
        M2 = zeros(T, nc0)
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vGe[2:end])
        if ℓ == 0 || ℓ == 2
            # M2[1] = ddfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
    end
    if va[1] ≠ 0.0 || modelav0 === nothing
        FLnb = modelap(va)
    else
        FLnb[2:end] = modelap(va[2:end])
        if ℓ == 0
            FLnb[1] = M0[1]
            # FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
        else
            FLnb[1] = 0.0
        end
    end
    FLnba = modelap(vaa)
    # Check whether `FLnba[end] ≪ eps(T)`
    if FLnba[end] ≥ epsT / 10
        @warn("Error: `FLnba[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
        @show "vaa", va[end], FLnba[end]
    end
    return M2, M1, M0, FLnb, FLnba
end

"""
  Inputs:
    vGe:
    va: `va = vGk * vabth = vGk * (vth[isp] / vth[iFv])` when `fLn0 = fvL0[:,L1,iFv]` gives `FvL[:,L1,isp]`
         which is equivalent to `vb = vGk / vabth` when `fLn0 = fvL0[:,L1,isp]` gives `FvL[:,L1,iFv]`
    fLn0:
    FLnb = F̂(𝓋̂) `

  Outputs:
    ddfLn0,dfLn0,fLn,FLn = FfvLCS(ddfLn0,dfLn0,fLn,FLnb,Ln0,vGe,va,nc0,vabth,nai,uai,vthi,ℓ;
            isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 1D, (ddfLn0,dfLn0,fLnnew,FLnb)
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T}, M0::AbstractVector{T},
    FLnb::AbstractVector{T}, fLn0::AbstractVector{T}, vGe::AbstractVector{Tb},
    va::AbstractVector{Tb}, nc0::Int64, vabth::T,nai::T,uai::T,vthi::T,ℓ::Int64;
    isnormal::Bool=true, restartfit::Vector{Int}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1) where {T,Tb,N}

    if isnormal
        yscale, ys = normalfLn(fLn0,ℓ,uai)
    else
        ys = copy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vGe; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ;
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end
    modelap(v) = modela(v, pa)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vGe)
    else
        M0 = zero.(vGe)
        M0[2:end] = modelap(vGe[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai[1],uai[1],vthi[1])
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vGe[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vGe)
        M2 = ForwardDiff.derivative.(modeldfLn0, vGe)
    else
        M1 = zeros(T, nc0)
        M1[2:end] = modeldfLn0(vGe[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
        M2 = zeros(T, nc0)
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vGe[2:end])
        if ℓ == 0 || ℓ == 2
            M2[1] = ddfLnDMv0(nai[1],uai[1],vthi[1], ℓ)
        end
    end
    if vabth == 1.0
        return M2, M1, M0, M0
    else
        # # Mapping
        if va[1] ≠ 0.0 || modelav0 === nothing
            FLnb = modelap(va)
        else
            FLnb[2:end] = modelap(va[2:end])
            if ℓ == 0
                FLnb[1] = M0[1]
                # FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
            else
                FLnb[1] = 0.0
            end
        end
        # Check whether `FLnb[end] ≪ eps(T)`
        if FLnb[end] ≥ epsT / 10
            @warn("Error: `FLn[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
            @show FLnb[end]
        end
        return M2, M1, M0, FLnb
    end
end
