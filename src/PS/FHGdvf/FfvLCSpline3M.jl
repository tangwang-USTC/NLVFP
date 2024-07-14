

"""
    In Euler coordinate system for general `nc0, nMod, ns`

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
           will be important to the algorithm. 


  Warning: The biggest relative error of `FvL` comes from the extrapolations

  Inputs:
    nai:

  Outputs:
    ddfvL0,dfvL0,fvL,FvL = FfvLCS(ddfvL0,dfvL0,fvL,FvL,fvL0,vhk,
              nc0,ocp,nvlevele,LM,LM1,
              nai,uai,vthi,nMod,vth,ns;
              is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
              autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
              p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 3.5D, [nMod,nv,LM,ns]  (ddfvL, dfvL, fvL, FvL, FvLa, vaa, nvlevel0a, ncF)
function FfvLCS(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}}, fvL::AbstractVector{Matrix{T}},
    FvL::AbstractVector{Matrix{T}},fvL0::AbstractVector{Matrix{T}},vhk::Vector{AbstractVector{T}},
    nc0::Vector{Int64},ocp::Vector{Int64},nvlevele::Vector{Vector{Int64}},LM::Vector{Int64}, LM1::Int64, 
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}}, nMod::Vector{Int64},
    vth::Vector{T},ns::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false) where {T}

    nsp_vec = 1:ns
    ncF, vaa = zeros(Int, ns), Vector((undef), ns)
    FvLa = Array{Any}(undef, LM1, ns)
    nvlevel0a = Vector{Vector{Int64}}(undef, ns)
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec.≠isp]
        iFv = nspF[1]
        vbath = vth[iFv] / vth[isp]
        vha = vhk[isp]                    # The grids of `fvL[isp]`
        vb = vhk[iFv] * vbath             # `= vᵦ / vath``, for the mapping process `fvL[va;vath,isp] → FvL[vb;vath,iFv]`
        if norm([vabth-1, nc0[isp]-nc0[iFv], ocp[isp]-ocp[iFv], vhk[isp][end]-vhk[iFv][end]]) ≤ epsT10
            is_map_F = false   # same meshgrids `vhk[isp] = vhk[iFv] * vbath`
            # is_map_F = true
        else
            is_map_F = true
        end
        if nMod[isp] == 1
            L1 = 1
            ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1], 
                    a, b, c, ncF[iFv] = FfvLCS(ddfvL[isp][:, L1],
                    dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vha, vb,
                    nc0[isp], ocp[isp],nvlevele[isp],L1 - 1,
                    nai[isp][1],uai[isp][1],vthi[isp][1];
                    ncF=ncF[iFv], is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                    autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
            if ncF[iFv] == 0
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[isp][:, L1]) ≥ epsT
                        ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                                dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vha, vb,
                                nc0[isp],nvlevele[isp],L1 - 1,
                                nai[isp][1],uai[isp][1],vthi[isp][1];
                                is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                    end
                end
                FvLa[2:LM1, iFv] .= [nothing]
            else
                FvLa[L1, iFv], vaa[iFv], nvlevel0a[iFv] = a, b, c
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[isp][:, L1]) ≥ epsT
                        ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1],
                            FvLa[L1, iFv] = FfvLCS(ddfvL[isp][:, L1], dfvL[isp][:, L1],
                                FvL[iFv][:, L1], fvL0[isp][:, L1], vha, vb, vaa[iFv],
                                nc0[isp],nvlevele[isp],L1 - 1,
                                nai[isp][1],uai[isp][1],vthi[isp][1];
                                is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
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
            ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1], 
                    a, b, c, ncF[iFv] = FfvLCS(ddfvL[isp][:, L1],
                    dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], 
                    vha, vb, nc0[isp], ocp[isp],nvlevele[isp],L1 - 1,
                    nai[isp],uai[isp],vthi[isp],nMod[isp];
                    ncF=ncF[iFv], is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                    autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                    is_fit_f=is_fit_f, is_map_F=is_map_F)
                    
            if ncF[iFv] == 0
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[isp][:, L1]) ≥ epsT
                        ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                                dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], 
                                vha, vb, nc0[isp],nvlevele[isp],L1 - 1,nai[isp],
                                uai[isp],vthi[isp],nMod[isp];
                                is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                    end
                end
                FvLa[2:LM1, iFv] .= [nothing]
            else
                FvLa[L1, iFv], vaa[iFv], nvlevel0a[iFv] = a, b, c
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[isp][:, L1]) ≥ epsT
                        ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1],
                          FvLa[L1, iFv] = FfvLCS(ddfvL[isp][:, L1], dfvL[isp][:, L1],
                            FvL[iFv][:, L1], fvL0[isp][:, L1], vha, vb, vaa[iFv],
                            nc0[isp], nvlevele[isp],L1 - 1,
                            nai[isp],uai[isp],vthi[isp],nMod[isp];
                            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
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
    # rtghjm
    return ddfvL, dfvL, fvL, FvL, FvLa, vaa, nvlevel0a, ncF
end

# 3.5D, [nMod,nv,LM,ns]  (ddfvL, dfvL, fvL, FvL)
function FfvLCS(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}}, fvL::AbstractVector{Matrix{T}},
    FvL::AbstractVector{Matrix{T}},fvL0::AbstractVector{Matrix{T}},
    vhk::Vector{AbstractVector{T}},nc0::Vector{Int64},nvlevele::Vector{Vector{Int64}},LM::Vector{Int64}, 
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64},
    vth::Vector{T},ns::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false) where {T}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec.≠isp]
        iFv = nspF[1]
        vbath = vth[iFv] / vth[isp]
        vha = vhk[isp]                    # The grids of `fvL[isp]`
        vb = vhk[iFv] * vbath             # `= vᵦ / vath``, for the mapping process `fvL[va;vath,isp] → FvL[vb;vath,iFv]`
        if nMod[isp] == 1
            for L1 in 1:LM[isp]+1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                            dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vha, vb,
                            nc0[isp],nvlevele[isp],L1 - 1,
                            nai[isp][1],uai[isp][1],vthi[isp][1];
                            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                end
            end
        else
            for L1 in 1:LM[isp]+1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                            dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], 
                            vha, vb, nc0[isp],nvlevele[isp],L1 - 1,
                            nai[isp],uai[isp],vthi[isp],nMod[isp];
                            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                end
            end
        end
    end
    # rtghjm
    return ddfvL, dfvL, fvL, FvL
end

# 3.5D, [nv,LM,ns]  (ddfvL, dfvL, fvL, FvL, FvLa, vaa, nvlevel0a, ncF)
function FfvLCS(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}}, fvL::AbstractVector{Matrix{T}},
    FvL::AbstractVector{Matrix{T}},fvL0::AbstractVector{Matrix{T}},vhk::Vector{AbstractVector{T}},
    nc0::Vector{Int64},ocp::Vector{Int64},nvlevele::Vector{Vector{Int64}},LM::Vector{Int64},LM1::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}}, 
    vth::Vector{T},ns::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false) where {T}

    nsp_vec = 1:ns
    ncF, vaa = zeros(Int, ns), Vector((undef), ns)
    FvLa = Array{Any}(undef, LM1, ns)
    nvlevel0a = Vector{Vector{Int64}}(undef, ns)
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec.≠isp]
        iFv = nspF[1]
        vbath = vth[iFv] / vth[isp]
        vha = vhk[isp]                    # The grids of `fvL[isp]`
        vb = vhk[iFv] * vbath             # `= vᵦ / vath``, for the mapping process `fvL[va;vath,isp] → FvL[vb;vath,iFv]`
        if norm([vabth-1, nc0[isp]-nc0[iFv], ocp[isp]-ocp[iFv], vhk[isp][end]-vhk[iFv][end]]) ≤ epsT10
            is_map_F = false   # same meshgrids `vhk[isp] = vhk[iFv] * vbath`
            # is_map_F = true
        else
            is_map_F = true
        end
        L1 = 1
        ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1], 
                a, b, c, ncF[iFv] = FfvLCS(ddfvL[isp][:, L1],
                dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vha, vb,
                nc0[isp], ocp[isp],nvlevele[isp],L1 - 1,
                nai[isp][1],uai[isp][1],vthi[isp][1];
                ncF=ncF[iFv], is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
        if ncF[iFv] == 0
            for L1 in 2:LM[isp]+1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                            dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vha, vb,
                            nc0[isp],nvlevele[isp],L1 - 1,
                            nai[isp][1],uai[isp][1],vthi[isp][1];
                            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                end
            end
            FvLa[2:LM1, iFv] .= [nothing]
        else
            FvLa[L1, iFv], vaa[iFv], nvlevel0a[iFv] = a, b, c
            for L1 in 2:LM[isp]+1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1],
                        FvLa[L1, iFv] = FfvLCS(ddfvL[isp][:, L1], dfvL[isp][:, L1],
                            FvL[iFv][:, L1], fvL0[isp][:, L1], vha, vb, vaa[iFv],
                            nc0[isp],nvlevele[isp],L1 - 1, 
                            nai[isp][1],uai[isp][1],vthi[isp][1];
                            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                else
                    FvLa[L1, iFv] = zero.(vaa[iFv])
                end
            end
            if LM[isp]+1 ≠ LM1
                FvLa[LM[isp]+2:LM1, iFv] .= [nothing]
            end
        end
    end
    # rtghjm
    return ddfvL, dfvL, fvL, FvL, FvLa, vaa, nvlevel0a, ncF
end

"""
  Inputs:
    vha:
    va: `va = vha[isp] * vabth = vha * (vth[isp] / vth[iFv])` for `f(va,isp) → F(vb/vath,iFv)`
    fLn0: fLn0e = fLn[nvlevele]
    FLn:  FvL(va)
    ncF::Int64, if `ncF ≥ 1`, the number of the extrapolation points where  `FLn[ncF] ≤ epsT01`.
              will be given by the input parameter `ncF`.
              Or else, the parameter `ncF will be determined by the inner algorithm.

  Outputs:
    `FLnb = F̂(𝓋̂) `
    ddfLn0,dfLn0,fLn,FLn,FLnba,vaa,nvlevel0a,ncF = FfvLCS(ddfLn0,dfLn0,
            FLnb,fLn0,vha,va,nc0,ocp,nvlevele,ℓ,
            nai,uai,vthi,nMod;
            ncF=ncF,is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,is_map_F=is_map_F)

"""

# 1.5D, [nMod], (ddfLn0,dfLn0,fLnnew, FLnb, FLnba, vaa, nvlevel0a, ncF)
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T}, FLnb::AbstractVector{T}, 
    fLn0::AbstractVector{T}, vha::AbstractVector{T},va::AbstractVector{T}, 
    nc0::Int64, ocp::Int64,nvlevele::Vector{Int64},ℓ::Int64,
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64; ncF::Int64=0,
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false, is_map_F::Bool=true) where {T}
    
    vG = vha[nvlevele]
    if is_normal
        yscale, ys = normalfLn(fLn0,ℓ,uai,nMod)
    else
        ys = deepcopy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vG; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end 
    modelap(v) = modela(v, pa)
    if vha[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vha)
    else
        M0 = zero.(vha)
        M0[2:end] = modelap(vha[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai,uai,vthi, nMod)
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vG[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vG)
        M2 = ForwardDiff.derivative.(modeldfLn0, vG)
    else
        M1[2:end] = modeldfLn0(vG[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai,uai,vthi, nMod)
        end
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vG[2:end])
        if ℓ == 0 || ℓ == 2
            M2[1] = ddfLnDMv0(nai,uai,vthi, ℓ, nMod)
        end
    end
    if ncF ≤ 0
        ncF = 0        # find the value `ncF` for `FLn` where `FLn[ncF] ≤ epsT01`.
        # # Mapping
        if is_map_F
            va9 = deepcopy(va[end])
            va93 = va9^3
            FLnb9 = va93 * modelap(va9)
            if FLnb9 > epsT01
                # nc0F = nc0 + ncF + 1   # The total number of the new `FLn = [FLnb;FLnba]`
                dva9 = va[end] - va[end-ocp+1]
                while FLnb9 > epsT01
                    va9 += dva9
                    FLnb9 = va93 * modelap(va9)
                    ncF += 1
                end
                ncF += 1
                @show 1, ncF
                nvak = (ncF - 1) * ocp - ncF + 2
                vaa0 = range(va[end], step=dva9, length=ncF)
                vaa, nvlevel0a = zeros(T, nvak), zeros(Int, ncF)
                vaa, nvlevel0a = vCmapping(vaa, nvlevel0a, vaa0, ncF, ocp)
                if va[1] ≠ 0.0 || modelav0 === nothing
                    FLnb[:] = modelap(va)
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
                    FLnb[:] = modelap(va)
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
        else
            return M2, M1, M0, M0, [nothing], [nothing], [nothing], ncF
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
                FLnb[1] = M0[1]
                # FLnb[1] = fLnDMv0(nai,uai,vthi,nMod)
            else
                FLnb[1] = 0.0
            end
        end
        # Check whether `FLnb[end] ≪ eps(T)`
        if is_warn_FLnb9 && FLnb[end] ≥ epsT01
            @warn("Error: `FLn[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
            @show "nMod,ncF", va[end], FLnb[end]
        end
        return M2, M1, M0, FLnb, modelap(vaa), vaa, nvlevel0a, ncF
        FLnba = modelap(vaa)
        return M2, M1, M0, FLnb, FLnba, vaa, nvlevel0a, ncF
    end
end

"""
  Inputs:
    vha:
    va: 

  Outputs:
    ddfLn0,dfLn0,fLn,FLn,FLnba, vaa, nvlevel0a, ncF = FfvLCS(ddfLn0,dfLn0,
            FLnb,fLn0,vha,va,nc0,nvlevele,ℓ,nai,uai,vthi,nMod,vabth;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 1.5D, [nMod], (ddfLn0,dfLn0,fLnnew,FLnb)
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T}, FLnb::AbstractVector{T}, 
    fLn0::AbstractVector{T}, vha::AbstractVector{T}, va::AbstractVector{T}, 
    nc0::Int64,nvlevele::Vector{Int64},ℓ::Int64, 
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false , is_map_F::Bool=true) where {T}
    
    vG = vha[nvlevele]
    if is_normal
        yscale, ys = normalfLn(fLn0,ℓ,uai,nMod)
    else
        ys = deepcopy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vG; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end
    modelap(v) = modela(v, pa)
    if vha[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vha)
    else
        M0 = zero.(vha)
        M0[2:end] = modelap(vha[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai,uai,vthi, nMod)
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vG[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vG)
        M2 = ForwardDiff.derivative.(modeldfLn0, vG)
    else
        M1[2:end] = modeldfLn0(vG[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai,uai,vthi, nMod)
        end
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vG[2:end])
        if ℓ == 0 || ℓ == 2
            M2[1] = ddfLnDMv0(nai,uai,vthi, ℓ, nMod)
        end
    end
    if is_map_F
        # # Mapping
        if va[1] ≠ 0.0 || modelav0 === nothing
            FLnb[:] = modelap(va)
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
        if is_warn_FLnb9 && FLnb[end] ≥ epsT01
            @warn("Error: `FLn[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
            @show nMod, va[end], FLnb[end]
        end
        return M2, M1, M0, FLnb
    else
        return M2, M1, M0, M0
    end
end

# 1.5D, [nMod], (ddfLn0,dfLn0,fLnnew,FLnb), vaa
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T}, FLnb::AbstractVector{T}, 
    fLn0::AbstractVector{T}, vha::AbstractVector{T}, va::AbstractVector{T}, 
    vaa::AbstractVector{T}, nc0::Int64, nvlevele::Vector{Int64},ℓ::Int64,
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false, is_map_F::Bool=true) where {T}
    
    vG = vha[nvlevele]
    if is_normal
        yscale, ys = normalfLn(fLn0,ℓ,uai,nMod)
    else
        ys = deepcopy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys, vG; n10=n10, dnvs=dnvs)
    if ℓ == 0
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    else
        counts, pa, modela, modelav0 = fvLmodel(vs, ys, nai,uai,vthi, ℓ; nMod=nMod,
            yscale=yscale, restartfit=restartfit, maxIterTR=maxIterTR,
            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, is_fit_f=is_fit_f)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end
    modelap(v) = modela(v, pa)
    if vha[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vha)
    else
        M0 = zero.(vha)
        M0[2:end] = modelap(vha[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai,uai,vthi, nMod)
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vG[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vG)
        M2 = ForwardDiff.derivative.(modeldfLn0, vG)
    else
        M1[2:end] = modeldfLn0(vG[2:end])
        if ℓ == 1
            M1[1] = dfLnDMv0(nai,uai,vthi, nMod)
        end
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0, vG[2:end])
        if ℓ == 0 || ℓ == 2
            a = ddfLnDMv0(nai,uai,vthi, ℓ, nMod)
            M2[1] = a
        end
    end
    if va[1] ≠ 0.0 || modelav0 === nothing
        FLnb[:] = modelap(va)
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
    if FLnba[end] ≥ epsT01
        @warn("Error: `FLnba[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
        @show "nMod,vaa", va[end], FLnba[end]
    end
    return M2, M1, M0, FLnb, modelap(vaa)
end

"""
  Inputs:
    vha:
    va: 
    FLb = F̂(𝓋̂) `
    ncF::Int64, if `ncF ≥ 1`, the number of the extrapolation points where  `FLn[ncF] ≤ epsT01`.
              will be given by the input parameter `ncF`.
              Or else, the parameter `ncF will be determined by the inner algorithm.

  Outputs:
    ddfLn0,dfLn0,fLn,FLn,FLnba,vaa,nvlevel0a,ncF = FfvLCS(ddfLn0,dfLn0,
            FLnb,fLn0,vha,va,nc0,ocp,nvlevele,ℓ,nai,uai,vthi;ncF=ncF,
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    ddfLn0,dfLn0,fLn,FLn = FfvLCS(ddfLn0,dfLn0,FLnb,fLn0,
            vha,va,vaa,nc0,nvlevele,ℓ,nai,uai,vthi;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 1D, (ddfLn0,dfLn0,fLnnew,FLnb, FLnba, vaa, nvlevel0a, ncF)
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T}, 
    FLnb::AbstractVector{T}, fLn0::AbstractVector{T}, vha::AbstractVector{T},
    va::AbstractVector{T}, nc0::Int64, ocp::Int64,nvlevele::Vector{Int64},ℓ::Int64,
    nai::T,uai::T,vthi::T; ncF::Int64=0,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false , is_map_F::Bool=true) where {T}
    
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
    if vha[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vha)
    else
        M0 = zero.(vha)
        M0[2:end] = modelap(vha[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai[1],uai[1],vthi[1])
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vG[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vG)
        M2 = ForwardDiff.derivative.(modeldfLn0, vG)
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
        ncF = 0        # find the value `ncF` for `FLn` where `FLn[ncF] ≤ epsT01`.
        # # Mapping
        if is_map_F
            va9 = deepcopy(va[end])
            va93 = va9^3
            FLnb9 = va93 * modelap(va9)
            if FLnb9 > epsT01
                # nc0F = nc0 + ncF + 1   # The total number of the new `FLn = [FLnb;FLnba]`
                dva9 = va[end] - va[end-ocp+1]
                while FLnb9 > epsT01
                    va9 += dva9
                    FLnb9 = va93 * modelap(va9)
                    ncF += 1
                end
                ncF += 1  # va9
                @show 2, ncF
                nvak = (ncF - 1) * ocp - ncF + 2
                vaa0 = range(va[end], step=dva9, length=ncF)
                vaa, nvlevel0a = zeros(T, nvak), zeros(Int, ncF)
                vaa, nvlevel0a = vCmapping(vaa, nvlevel0a, vaa0, ncF, ocp)
                if va[1] ≠ 0.0 || modelav0 === nothing
                    FLnb[:] = modelap(va)
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
                    FLnb[:] = modelap(va)
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
        else
            return M2, M1, M0, M0, [nothing], [nothing], [nothing], ncF
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
                FLnb[1] = M0[1]
                # FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
            else
                FLnb[1] = 0.0
            end
        end
        # Check whether `FLnb[end] ≪ eps(T)`
        if is_warn_FLnb9 && FLnb[end] ≥ epsT01
            @warn("Error: `FLn[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
            @show "ncF", va[end], FLnb[end]
        end
        return M2, M1, M0, FLnb, modelap(vaa), vaa, nvlevel0a, ncF
        FLnba = modelap(vaa)
        return M2, M1, M0, FLnb, FLnba, vaa, nvlevel0a, ncF
    end
end

"""
  Inputs:
    vha:
    va: 
    FLnb = F̂(𝓋̂) `

  Outputs:
    ddfLn0,dfLn0,fLn,FLn = FfvLCS(ddfLn0,dfLn0,FLnb,
            fLn0,vha,va,nc0,nvlevele,ℓ,nai,uai,vthi;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# # 1D, (ddfLn0,dfLn0,fLnnew,FLnb)
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T}, FLnb::AbstractVector{T}, 
    fLn0::AbstractVector{T}, vha::AbstractVector{T}, va::AbstractVector{T}, 
    nc0::Int64,nvlevele::Vector{Int64},ℓ::Int64, nai::T,uai::T,vthi::T;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false , is_map_F::Bool=true) where {T}
    
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
    if vha[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vha)
    else
        M0 = zero.(vha)
        M0[2:end] = modelap(vha[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai[1],uai[1],vthi[1])
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vG[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vG)
        M2 = ForwardDiff.derivative.(modeldfLn0, vG)
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
                FLnb[1] = M0[1]
                # FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
            else
                FLnb[1] = 0.0
            end
        end
        # Check whether `FLnb[end] ≪ eps(T)`
        if is_warn_FLnb9 && FLnb[end] ≥ epsT01
            @warn("Error: `FLn[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
            @show FLnb[end]
        end
        return M2, M1, M0, FLnb
    else
        return M2, M1, M0, M0
    end
end

# 1D, (ddfLn0,dfLn0,fLnnew,FLnb), vaa
function FfvLCS(M2::AbstractVector{T}, M1::AbstractVector{T},
    FLnb::AbstractVector{T}, fLn0::AbstractVector{T}, vha::AbstractVector{T},
    va::AbstractVector{T}, vaa::AbstractVector{T}, nc0::Int64,
    nvlevele::Vector{Int64},ℓ::Int64, nai::T,uai::T,vthi::T;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false , is_map_F::Bool=true) where {T}
    
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
    if vha[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vha)
    else
        M0 = zero.(vha)
        M0[2:end] = modelap(vha[2:end])
        if ℓ == 0
            M0[1] = fLnDMv0(nai[1],uai[1],vthi[1])
        else
            M0[1] = 0.0
        end
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap, v)
    if vG[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vG)
        M2 = ForwardDiff.derivative.(modeldfLn0, vG)
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
            FLnb[1] = M0[1]
            # FLnb[1] = fLnDMv0(nai[1],uai[1],vthi[1])
        else
            FLnb[1] = 0.0
        end
    end
    FLnba = modelap(vaa)
    # Check whether `FLnba[end] ≪ eps(T)`
    if FLnba[end] ≥ epsT01
        @warn("Error: `FLnba[end] > epsT/10` which may cause errors of the Shkrafsky integrals,ℓ=",ℓ)
        @show "vaa", va[end], FLnba[end]
    end
    return M2, M1, M0, FLnb, FLnba
end



"""
  Inputs:
    nai:

  Outputs:
    ddfvL0,dfvL0,fvL,FvL,vaa,FvLa,nvlevel0a,ncF = FfvLCS(ddfvL0,dfvL0,
              fvL,FvL,fvL0,vhk,nc0,ocp,nvlevele,LM,LM1,nai,uai,vthi,vth,ns;
              is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
              autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
              p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# # 3D  (ddfvL0,dfvL0,fvL,FvL), nMod[:] .== 1
# function FfvLCS(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}},fvL::AbstractVector{Matrix{T}},
#     FvL::AbstractVector{Matrix{T}}, fvL0::AbstractVector{Matrix{T}},vhk::Vector{AbstractVector{T}},
#     nc0::Vector{Int64},ocp::Vector{Int64},nvlevele::Vector{Vector{Int64}}, vth::Vector{T},
    #   nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},
#     LM::Vector{Int64}, LM1::Int64, ns::Int64;
#     is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
#     autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
#     p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18, 
#     n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false) where {T}
#     ddddd
#     nsp_vec = 1:ns
#     ncF, vaa = zeros(Int, ns), Vector((undef), ns)
#     FvLa = Array{Any}(undef, LM1, ns)
#     nvlevel0a = Vector((undef), ns)
#     for isp in nsp_vec
#         nspF = nsp_vec[nsp_vec.≠isp]
#         iFv = nspF[1]
#         vbath = vth[iFv] / vth[isp]
#         vha = vhk[isp]
#         vb = vhk[iFv] * vbath
#         L1 = 1
#         ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1], FvLa[L1, iFv],
#               vaa[iFv], nvlevel0a[iFv], ncF[iFv] = FfvLCS(ddfvL[isp][:, L1],
#                 dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vha, vb,
#                 nc0[isp], ocp[isp], nvlevele[isp],L1 - 1,nai[isp][1],uai[isp][1],vthi[isp][1]; 
#                 ncF=ncF[iFv],is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
#                 autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
#                 p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
#         if ncF[iFv] == 0
#             for L1 in 2:LM[isp]+1
#                 if norm(fvL0[isp][:, L1]) ≥ epsT
#                     ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
#                             dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vha, vb,
#                             nc0[isp], nvlevele[isp],L1 - 1,nai[isp][1],uai[isp][1],vthi[isp][1];
#                             is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
#                             autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
#                             p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
#                 end
#             end
#             FvLa[2:LM1, iFv] .= [nothing]
#         else
#             for L1 in 2:LM[isp]+1
#                 if norm(fvL0[isp][:, L1]) ≥ epsT
#                     ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1],
#                           FvLa[L1, iFv] = FfvLCS(ddfvL[isp][:, L1], dfvL[isp][:, L1],
#                             FvL[iFv][:, L1], fvL0[isp][:, L1], vha, vb, vaa[iFv],
#                             nc0[isp], nvlevele[isp],L1 - 1,nai[isp][1],uai[isp][1],vthi[isp][1];
#                             is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
#                             autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
#                             p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
#                 else
#                     FvLa[L1, iFv] = zero.(vaa[iFv])
#                 end
#             end
#             if LM[isp]+1 ≠ LM1
#                 FvLa[LM[isp]+2:LM1, iFv] .= [nothing]
#             end
#         end
#     end
#     return ddfvL, dfvL, fvL, FvL, FvLa, vaa, nvlevel0a, ncF
# end
