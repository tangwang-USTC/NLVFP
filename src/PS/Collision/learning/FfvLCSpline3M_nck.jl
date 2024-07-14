

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

# μu ??????????

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
              nvlevel0,vth,nai,vthi,uai,μu,LM,ns;nMod=nMod,
              isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
              autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
              p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 3.5D, [nMod]  (ddfvL0,dfvL0,fvL,FvL),
function FfvLCS(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},
    fvL::AbstractArray{T,N},FvL::AbstractArray{T,N},fvL0::AbstractArray{T,N},
    vGk::AbstractVector{Tb},nvlevel0::Vector{Int},vth::Vector{T},
    nai::AbstractArray{T,N2},vthi::AbstractArray{T,N2},uai::AbstractArray{T,N2},
    μu::AbstractArray{T,N2},LM::Vector{Int},ns::Int;nMod::Int=2,
    isnormal::Bool=true,restartfit::Vector{Int}=[0,0,100],maxIterTR::Int=1000,
    autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int=1,dnvs::Int=1) where{T,N,N2,Tb}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec .≠ isp]
        iFv = nspF[1]
        vbath = vth[iFv] / vth[isp]
        vb = vGk * vbath
        for L1 in 1:LM1
            if norm(fvL0[:,L1,isp]) > epsT5
                ddfvL[:,L1,isp],dfvL[:,L1,isp],fvL[:,L1,isp],FvL[:,L1,iFv] = FfvLCS(ddfvL[:,L1,isp],
                        dfvL[:,L1,isp],fvL[:,L1,isp],FvL[:,L1,iFv],fvL0[:,L1,isp],vGk,vb,
                        nvlevel0,vbath,nai[:,isp],vthi[:,isp],uai[:,isp],μu[:,isp],L1-1;nMod=nMod,
                        isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
                        autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
            end
        end
    end
    return ddfvL, dfvL,fvL, FvL
end

"""
  Inputs:
    vGk:
    va: `va = vGk * vabth = vGk * (vth[isp] / vth[iFv])` when `fLn0 = fvL0[:,L1,iFv]` gives `FvL[:,L1,isp]`
         which is equivalent to `vb = vGk / vabth` when `fLn0 = fvL0[:,L1,isp]` gives `FvL[:,L1,iFv]`
    fLn0:
    FLnb = F̂(𝓋̂)

  Outputs:
    ddfLn0,dfLn0,fLn,FLn, ncF, vaa, FLnba, nvlevel0a,nvlevela = FfvLCS(ddfLn0,dfLn0,
            fLn,FLnb,fLn0,vGk,va,nc0,ocp,nvlevel0,vabth,nai,vthi,uai,μu,ℓ;nMod=nMod,
            isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 1.5D, [nMod], (ddfLn0,dfLn0,fLnnew,FLnb)
function FfvLCS(M2::AbstractVector{T},M1::AbstractVector{T},M0::AbstractVector{T},
    FLnb::AbstractVector{T},fLn0::AbstractVector{T},vGk::AbstractVector{Tb},
    va::AbstractVector{Tb},nvlevel0::AbstractVector{Int},vabth::T,
    nai::AbstractVector{T},vthi::AbstractVector{T},uai::AbstractVector{T},
    μu::AbstractVector{T},ℓ::Int;nMod::Int=2,
    isnormal::Bool=true,restartfit::Vector{Int}=[0,0,100],maxIterTR::Int=1000,
    autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int=1,dnvs::Int=1,
    isboundaryf::Bool=true) where{T,Tb,N}

    vG0 = vGk[nvlevel0]
    if isnormal
        yscale, ys = normalfLn(fLn0)
    else
        ys = copy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys,vG0;n10=n10,dnvs=dnvs)
    if ℓ == 0
        counts,pa,modela,modelav0 = fvLmodel(vs,ys,nai,vthi,uai;nMod=nMod,
                    yscale=yscale,restartfit=restartfit,maxIterTR=maxIterTR,
                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    else
        counts,pa,modela,modelav0 = fvLmodel(vs,ys,nai,vthi,uai,μu,ℓ;nMod=nMod,
                    yscale=yscale,restartfit=restartfit,maxIterTR=maxIterTR,
                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end
    modelap(v) =  modela(v,pa)
    if vGk[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vGk)
    else
        modelav0p = v -> modelav0(v,pa)
        M0 = zero.(vGk)
        M0[2:end] = modelap(vGk[2:end])
        ℓ == 0 ? M0[1] = modelav0p(0.0) : M0[1] = 0.0
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap,v)
    if vG0[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vG0)
        M2 = ForwardDiff.derivative.(modeldfLn0,vG0)
    else
        modeldfLn0v0 = v -> ForwardDiff.derivative.(modelav0p,v)
        M1 = zero.(vG0)
        M1[2:end] = modeldfLn0(vG0[2:end])
        if ℓ == 1
            # if isboundaryf
            #     M1[1] = dfLnDMv0u(0.0,uai,ℓ)
            # else
            #     M1[1] = modeldfLn0v0(0.0)
            # end
            M1[1] = M1[2]
        end
        M2 = zero.(vG0)
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0,vG0[2:end])
        if ℓ == 0 ||  ℓ == 2
            # if isboundaryf
            #     M2[1] = ddfLnDMv0u(0.0,uai,ℓ)
            # else
            #     M2[1] = ForwardDiff.derivative.(modeldfLn0v0,0.0)
            # end
            M2[1] = M2[2]
        end
    end
    if vabth == 1.0
        return M2,M1,M0,M0
    else
        # # Mapping
        if va[1] ≠ 0.0 || modelav0 === nothing
            FLnb = modelap(va)
        else
            FLnb[2:end] = modelap(va[2:end])
            ℓ == 0 ? FLnb[1] = modelav0p(va[1]) : FLnb[1] = 0.0
        end
        # Check whether `FLnb[end] ≪ eps(T)`
        if FLnb[end] ≥ epsT
            @warn("Error: `FLn[end] > epsT` which may cause errors of the Shkrafsky integrals.")
            @show FLnb[end]
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
    ddfvL0,dfvL0,fvL,FvL = FfvLCS(ddfvL0,dfvL0,fvL,FvL,fvL0,vGk,
              nvlevel0,vth,nai,vthi,uai,μu,LM,ns;
              isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
              autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
              p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 3D  (ddfvL0,dfvL0,fvL,FvL), nMod = 1
function FfvLCS(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},
    fvL::AbstractArray{T,N},FvL::AbstractArray{T,N},fvL0::AbstractArray{T,N},
    vGk::AbstractVector{Tb},nvlevel0::Vector{Int},vth::Vector{T},
    nai::AbstractVector{T},vthi::AbstractVector{T},uai::AbstractVector{T},
    μu::AbstractVector{T},LM::Vector{Int},ns::Int;
    isnormal::Bool=true,restartfit::Vector{Int}=[0,0,100],maxIterTR::Int=1000,
    autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int=1,dnvs::Int=1) where{T,N,Tb}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec .≠ isp]
        iFv = nspF[1]
        vbath = vth[iFv] / vth[isp]
        vb = vGk * vbath
        for L1 in 1:LM1
            if norm(fvL0[:,L1,isp]) > epsT5
                ddfvL[:,L1,isp],dfvL[:,L1,isp],fvL[:,L1,isp],FvL[:,L1,iFv] = FfvLCS(ddfvL[:,L1,isp],
                        dfvL[:,L1,isp],fvL[:,L1,isp],FvL[:,L1,iFv],fvL0[:,L1,isp],vGk,vb,
                        nvlevel0,vbath,nai[isp],vthi[isp],uai[isp],μu[isp],L1-1;
                        isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
                        autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
            end
        end
    end
    return ddfvL, dfvL,fvL, FvL
end

"""
  Outputs:
    ddfvL0,dfvL0,fvL,FvL = FfvLCS(ddfvL0,dfvL0,fvL,FvL,fvL0,vGk,vb,
                  nvlevel0,vabth,nai,vthi,uai,μu,LM1;
                  isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
                  autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                  p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 2D (ddfvL0,dfvL0,fvL0,FvLb)
function FfvLCS(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},
    FvLb::AbstractArray{T,N},fvL0::AbstractArray{T,N},vGk::AbstractVector{Tb},vb::AbstractVector{Tb},
    nvlevel0::AbstractVector{Int},vabth::T,nai::T,vthi::T,uai::T,μu::T,LM1::Int;
    isnormal::Bool=true,restartfit::Vector{Int}=[0,0,100],maxIterTR::Int=1000,
    autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int=1,dnvs::Int=1) where{T,Tb,N}

    for L1 in 1:LM1
        if norm(fvL0[:,L1]) > epsT5
            ddfvL[:,L1],dfvL[:,L1],fvL[:,L1],FvLb[:,L1] = FfvLCS(ddfvL[:,L1],
                    dfvL[:,L1],fvL[:,L1],FvLb[:,L1],fvL0[:,L1],vGk,vb,
                    nvlevel0,vabth,nai,vthi,uai,μu,L1-1;
                    isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
        end
    end
    return ddfvL, dfvL, fvL, FvLb
end


"""
  Inputs:
    vGk:
    va: `va = vGk * vabth = vGk * (vth[isp] / vth[iFv])` when `fLn0 = fvL0[:,L1,iFv]` gives `FvL[:,L1,isp]`
         which is equivalent to `vb = vGk / vabth` when `fLn0 = fvL0[:,L1,isp]` gives `FvL[:,L1,iFv]`
    fLn0:
    FLnb = F̂(𝓋̂) `

  Outputs:
    ddfLn0,dfLn0,fLn,FLn = FfvLCS(ddfLn0,dfLn0,fLn,FLnb,
            fLn0,vGk,va,nvlevel0,vabth,nai,vthi,uai,μu,ℓ;
            isnormal=isnormal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 1D, (ddfLn0,dfLn0,fLnnew,FLnb)
function FfvLCS(M2::AbstractVector{T},M1::AbstractVector{T},M0::AbstractVector{T},
    FLnb::AbstractVector{T},fLn0::AbstractVector{T},vGk::AbstractVector{Tb},va::AbstractVector{Tb},
    nvlevel0::AbstractVector{Int},vabth::T,nai::T,vthi::T,uai::T,μu::T,ℓ::Int;
    isnormal::Bool=true,restartfit::Vector{Int}=[0,0,100],maxIterTR::Int=1000,
    autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int=1,dnvs::Int=1,
    isboundaryf::Bool=true) where{T,Tb,N}

    vG0 = vGk[nvlevel0]
    if isnormal
        yscale, ys = normalfLn(fLn0)
    else
        ys = copy(fLn0)
        yscale = 1.0
    end
    # filter
    ys, vs = filterfLn(ys,vG0;n10=n10,dnvs=dnvs)
    if ℓ == 0
        counts,pa,modela,modelav0 = fvLmodel(vs,ys,nai,vthi,uai;
                    yscale=yscale,restartfit=restartfit,maxIterTR=maxIterTR,
                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    else
        counts,pa,modela,modelav0 = fvLmodel(vs,ys,nai,vthi,uai,μu,ℓ;
                    yscale=yscale,restartfit=restartfit,maxIterTR=maxIterTR,
                    autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    end
    if yscale ≠ 1
        pa[1:3:end] *= yscale
    end
    modelap(v) =  modela(v,pa)
    if vGk[1] ≠ 0.0 || modelav0 === nothing
        M0 = modelap(vGk)
    else
        modelav0p = v -> modelav0(v,pa)
        M0 = zero.(vGk)
        M0[2:end] = modelap(vGk[2:end])
        ℓ == 0 ? M0[1] = modelav0p(0.0) : M0[1] = 0.0
    end
    # Computing the first two derivatives of model function: `∂ᵥmodelap(v)` and `∂ᵥ∂ᵥmodelap(v)`
    modeldfLn0 = v -> ForwardDiff.derivative.(modelap,v)
    if vG0[1] ≠ 0.0 || modelav0 === nothing
        M1 = modeldfLn0(vG0)
        M2 = ForwardDiff.derivative.(modeldfLn0,vG0)
    else
        modeldfLn0v0 = v -> ForwardDiff.derivative.(modelav0p,v)
        M1 = zero.(vG0)
        M1[2:end] = modeldfLn0(vG0[2:end])
        if ℓ == 1
            if isboundaryf
                M1[1] = dfLnDMv0u(0.0,uai,ℓ)
            else
                M1[1] = modeldfLn0v0(0.0)
            end
        end
        M2 = zero.(vG0)
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0,vG0[2:end])
        if ℓ == 0 ||  ℓ == 2
            if isboundaryf
                M2[1] = ddfLnDMv0u(0.0,uai,ℓ)
            else
                M2[1] = ForwardDiff.derivative.(modeldfLn0v0,0.0)
            end
        end
    end
    if vabth == 1.0
        return M2,M1,M0,M0
    else
        # # Mapping
        if va[1] ≠ 0.0 || modelav0 === nothing
            FLnb = modelap(va)
        else
            FLnb[2:end] = modelap(va[2:end])
            ℓ == 0 ? FLnb[1] = modelav0p(va[1]) : FLnb[1] = 0.0
        end
        # Check whether `FLnb[end] ≪ eps(T)`
        if FLnb[end] ≥ epsT
            @warn("Error: `FLn[end] > epsT` which may cause errors of the Shkrafsky integrals.")
            @show FLnb[end]
        end
        return M2, M1, M0, FLnb
    end
end
