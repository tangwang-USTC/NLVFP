
# Ms

"""
    In Lagrange coordinate system where `nc0[isp] = nc0[iFv]`

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
    ddfvL0,dfvL0,fvL,FvL = FfvLCSLag(ddfvL0,dfvL0,fvL,FvL,fvL0,vhk,
              nc0,ocp,nvlevele,vth,nai,uai,vthi,LM,LM1,ns,nMod;
              is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
              autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
              p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 3.5D, [nMod≥2,nv,LM,ns]  (ddfvL, dfvL, fvL, FvL, FvLa, vaa, nvlevel0a, ncF), is_extrapolate_FLn = true
function FfvLCSLag!(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}}, 
    fvL::AbstractVector{Matrix{T}},FvL::AbstractVector{Matrix{T}},fvL0::AbstractVector{Matrix{T}},
    vhk::AbstractVector{T},nc0::Int64,ocp::Int64,nvlevele::Vector{Int64},LM::Vector{Int64}, LM1::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}}, nMod::Vector{Int64},
    vth::Vector{T}, ns::Int64;
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
        vb = vhk * vbath             # `= vᵦ / vath``, for the mapping process `fvL[va;vath,isp] → FvL[vb;vath,iFv]`
        if abs(vabth-1) ≤ epsT10
            is_map_F = false   # same meshgrids `vhk = vhk * vbath`
        else
            is_map_F = true
        end

        if nMod[isp] == 1
            L1 = 1
            ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1], FvLa[L1, iFv],
                    vaa[iFv], nvlevel0a[iFv], ncF[iFv] = FfvLCS(ddfvL[isp][:, L1],
                    dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vhk, vb,
                    nc0, ocp,nvlevele,L1 - 1, nai[isp][1],uai[isp][1],vthi[isp][1];
                    ncF=ncF[iFv], is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                    autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
            if ncF[iFv] == 0
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[isp][:, L1]) ≥ epsT
                        ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                                dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vhk, vb,
                                nc0,nvlevele,L1 - 1, nai[isp][1],uai[isp][1],vthi[isp][1];
                                is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                    end
                end
                FvLa[2:LM1, iFv] .= [nothing]
            else
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[isp][:, L1]) ≥ epsT
                        ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1],
                            FvLa[L1, iFv] = FfvLCS(ddfvL[isp][:, L1], dfvL[isp][:, L1],
                                FvL[iFv][:, L1], fvL0[isp][:, L1], vhk, vb, vaa[iFv],
                                nc0,nvlevele,L1 - 1, nai[isp][1],uai[isp][1],vthi[isp][1];
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
            ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1], FvLa[L1, iFv],
                vaa[iFv], nvlevel0a[iFv], ncF[iFv] = FfvLCS(ddfvL[isp][:, L1],
                    dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vhk, vb,
                    nc0, ocp,nvlevele,L1 - 1, nai[isp],uai[isp],vthi[isp], nMod[isp];
                    ncF=ncF[iFv], is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                    autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,
                    is_fit_f=is_fit_f, is_map_F=is_map_F)
                    
            if ncF[iFv] == 0
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[isp][:, L1]) ≥ epsT
                        ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                                dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vhk, vb, nc0,
                                nvlevele,L1 - 1, nai[isp],uai[isp],vthi[isp], nMod[isp];
                                is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                                autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                                p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                    end
                end
                FvLa[2:LM1, iFv] .= [nothing]
            else
                for L1 in 2:LM[isp]+1
                    if norm(fvL0[isp][:, L1]) ≥ epsT
                        ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1],
                          FvLa[L1, iFv] = FfvLCS(ddfvL[isp][:, L1], dfvL[isp][:, L1],
                            FvL[iFv][:, L1], fvL0[isp][:, L1], vhk, vb, vaa[iFv],
                            nc0, nvlevele,L1 - 1, nai[isp],uai[isp],vthi[isp], nMod[isp];
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

# 3.5D, [nMod≥2,nv,LM,ns]  (ddfvL, dfvL, fvL, FvL), is_extrapolate_FLn = false
function FfvLCSLag!(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}}, 
    fvL::AbstractVector{Matrix{T}},FvL::AbstractVector{Matrix{T}},fvL0::AbstractVector{Matrix{T}},
    vhk::AbstractVector{T},nc0::Int64,nvlevele::Vector{Int64},LM::Vector{Int64},
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}}, nMod::Vector{Int64},
    vth::Vector{T}, ns::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false) where {T}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec.≠isp]
        iFv = nspF[1]
        vbath = vth[iFv] / vth[isp]
        vb = vhk * vbath             # `= vᵦ / vath``, for the mapping process `fvL[va;vath,isp] → FvL[vb;vath,iFv]`
        if nMod[isp] == 1
            for L1 in 1:LM[isp]+1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                            dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vhk, vb,
                            nc0,nvlevele,L1 - 1, nai[isp][1],uai[isp][1],vthi[isp][1];
                            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                end
            end
        else
            for L1 in 1:LM[isp]+1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                            dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vhk, vb, nc0,
                            nvlevele,L1 - 1, nai[isp],uai[isp],vthi[isp], nMod[isp];
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

# 3.5D, [nv,LM,ns]  (ddfvL, dfvL, fvL, FvL), is_extrapolate_FLn = false
function FfvLCSLag!(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}}, 
    fvL::AbstractVector{Matrix{T}},FvL::AbstractVector{Matrix{T}},fvL0::AbstractVector{Matrix{T}},
    vhk::Vector{AbstractVector{T}},nc0::Vector{Int64},nvlevele::Vector{Vector{Int64}},LM::Vector{Int64}, 
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}}, nMod::Vector{Int64},
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
        vb = vhk[isp] * vbath             # `= vᵦ / vath``, for the mapping process `fvL[va;vath,isp] → FvL[vb;vath,iFv]`
        if nMod[isp] == 1
            for L1 in 1:LM[isp]+1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                            dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vhk[isp], vb,
                            nc0[isp],nvlevele[isp],L1 - 1, nai[isp][1],uai[isp][1],vthi[isp][1];
                            is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                            autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                            p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
                end
            end
        else
            for L1 in 1:LM[isp]+1
                if norm(fvL0[isp][:, L1]) ≥ epsT
                    ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                            dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vhk[isp], vb, nc0[isp],
                            nvlevele[isp],L1 - 1, nai[isp],uai[isp],vthi[isp], nMod[isp];
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
function FfvLCSLag!(ddfvL::AbstractVector{Matrix{T}}, dfvL::AbstractVector{Matrix{T}}, 
    fvL::AbstractVector{Matrix{T}},FvL::AbstractVector{Matrix{T}},fvL0::AbstractVector{Matrix{T}},
    vhk::AbstractVector{T},nc0::Int64,nvlevele::Vector{Int64},vth::Vector{T},
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},
    LM::Vector{Int64}, ns::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0, 0, 100], maxIterTR::Int64=1000,
    autodiff::Symbol=:forward, factorMethod::Symbol=:QR, show_trace::Bool=false,
    p_tol::Float64=1e-18, f_tol::Float64=1e-18, g_tol::Float64=1e-18,
    n10::Int64=1, dnvs::Int64=1, is_fit_f::Bool=false) where {T}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec.≠isp]
        iFv = nspF[1]
        vbath = vth[iFv] / vth[isp]
        vb = vhk * vbath             # `= vᵦ / vath``, for the mapping process `fvL[va;vath,isp] → FvL[vb;vath,iFv]`
        for L1 in 1:LM[isp]+1
            if norm(fvL0[isp][:, L1]) ≥ epsT
                ddfvL[isp][:, L1], dfvL[isp][:, L1], fvL[isp][:, L1], FvL[iFv][:, L1] = FfvLCS(ddfvL[isp][:, L1],
                        dfvL[isp][:, L1], FvL[iFv][:, L1], fvL0[isp][:, L1], vhk, vb,
                        nc0,nvlevele,L1 - 1, nai[isp][1],uai[isp][1],vthi[isp][1];
                        is_normal=is_normal, restartfit=restartfit, maxIterTR=maxIterTR,
                        autodiff=autodiff, factorMethod=factorMethod, show_trace=show_trace,
                        p_tol=p_tol, f_tol=f_tol, g_tol=g_tol, n10=n10, dnvs=dnvs,is_fit_f=is_fit_f)
            end
        end
    end
    # rtghjm
    return ddfvL, dfvL, fvL, FvL
end
