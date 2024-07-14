# 1: (FvL)
# 3: (ddfvL0,dfvL0,fvL0)

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
  Warnning: The biggest relative error of `FvL` comes from the extrapolations

            when `vabth > 1.0` which is calculated by (1~3)ᵗʰ order spline interpolations

            of the weighted function now.

  Inputs:
    nc0: The number of the initial grid points.

  Outputs:
    FvL = FfvLCS(FvL,fvL,vGk,nc0,nvlevel0,uai,vthi,LM,ns;
             isnormal=isnormal,restartfit=restartfit,
             maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
             p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
    ddfvL,dfvL,fvLn = FfvLCS(ddfvL,dfvL,fvLn,fvL,vGk,nc0,nvlevel0,ua,vth,LM,ns;
             isnormal=isnormal,restartfit=restartfit,
             maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
             p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)


"""

# 3D  (FvL)
function FfvLCS(FvL::AbstractArray{T,N},fvL::AbstractArray{T,N},vGk::AbstractVector{Tb},
    nc0::Int,nvlevel0::Vector{Int},ua::AbstractVector{T},vth::Vector{T},LM::Vector{Int},ns::Int;
    isnormal::Bool=true,restartfit::Vector{Int}=[0,0,100],
    maxIterTR::Int=1000,autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int=1,dnvs::Int=1) where{T,Tb,N,N2}

    nsp_vec = 1:ns
    for isp in nsp_vec
        nspF = nsp_vec[nsp_vec .≠ isp]
        iFv = nspF[1]
        vabth = vth[isp] / vth[iFv]
        va = vGk * vabth
        for L1 in 1:LM1
            if norm(fvL[:,L1,isp]) > epsT5
                FvL[:,L1,iFv] = FfvLCS(FvL[:,L1,iFv],fvL[:,L1,isp],
                        vGk,va,nc0,nvlevel0,ua[isp],L1-1,vabth;
                        isnormal=isnormal,restartfit=restartfit,
                        maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
            end
        end
    end
    return FvL
end

# 3D  (ddfvL0,dfvL0,fvLnew)
function FfvLCS(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvLn::AbstractArray{T,N},
    fvL::AbstractArray{T,N},vGk::AbstractVector{Tb},nc0::Int,nvlevel0::Vector{Int},
    ua::AbstractVector{T},vth::Vector{T},LM::Vector{Int},ns::Int;
    isnormal::Bool=true,restartfit::Vector{Int}=[0,0,100],
    maxIterTR::Int=1000,autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int=1,dnvs::Int=1) where{T,Tb,N,N2}

    for isp in 1:ns
        for L1 in 1:LM1
            if norm(fvL[:,L1,isp]) > epsT5
                ddfvL[:,L1,isp],dfvL[:,L1,isp],fvLn[:,L1,isp] = FfvLCS(ddfvL[:,L1,isp],
                        dfvL[:,L1,isp],fvLn[:,L1,isp],fvL[:,L1,isp],vGk,nc0,nvlevel0,ua[isp],L1-1;
                        isnormal=isnormal,restartfit=restartfit,
                        maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
            end
        end
    end
    return ddfvL, dfvL,fvLn
end

"""
  Outputs:
    ddfvL0,dfvL0,fvLn,FvL = FfvLCS(ddfvL0,dfvL0,fvLn,FvL,fvL,vGk,va,nc0,nvlevel0,u,LM1,vabth;
                        isnormal=isnormal,restartfit=restartfit,
                        maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""

# 2D (FvL)
function FfvLCS(FvLb::AbstractArray{T,N},fvL::AbstractArray{T,N},
    vGk::AbstractVector{Tb},va::AbstractVector{Tb},nc0::Int,
    nvlevel0::AbstractVector{Int},u::Float64,LM1::Int,vabth::Float64;
    isnormal::Bool=true,restartfit::Vector{Int}=[0,0,100],
    maxIterTR::Int=1000,autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int=1,dnvs::Int=1) where{T,Tb,N}

    for L1 in 1:LM1
        if norm(fvL[:,L1]) > epsT5
            FvLb[:,L1] = FfvLCS(FvLb[:,L1],fvL[:,L1],vGk,va,nc0,nvlevel0,u,L1-1,vabth;
                    isnormal=isnormal,restartfit=restartfit,
                    maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
        end
    end
    return FvLb
end

# 2D (ddfvL0,dfvL0,fvL0), `vabth = 1.0`
function FfvLCS(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvLn::AbstractArray{T,N},
    fvL::AbstractArray{T,N},vGk::AbstractVector{Tb},nc0::Int,nvlevel0::AbstractVector{Int},u::Float64,LM1::Int;
    isnormal::Bool=true,restartfit::Vector{Int}=[0,0,100],
    maxIterTR::Int=1000,autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int=1,dnvs::Int=1) where{T,Tb,N}

    for L1 in 1:LM1
        if norm(fvL[:,L1]) > epsT5
            ddfvL[:,L1],dfvL[:,L1],fvLn[:,L1] = FfvLCS(ddfvL[:,L1],
                    dfvL[:,L1],fvLn[:,L1],fvL[:,L1],vGk,nc0,nvlevel0,u,L1-1;
                    isnormal=isnormal,restartfit=restartfit,
                    maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                    p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
        end
    end
    return ddfvL, dfvL, fvLn
end

"""
  Inputs:
    vGk:
    va: `= vGk * vabth`
    fLnk: `=  f̂(𝓋̂)`
    FLn: = fLnk

  Outputs:
    `FLnb = F̂(𝓋̂) `
    ddfLn0,dfLn0,fLnnew,FLn = FfvLCS(ddfLn0,dfLn0,fLnnew,FLnb,fLnk,vGk,va,nc0,nvlevel0,u,ℓ,vabth;
            isnormal=isnormal,restartfit=restartfit,
            maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)

"""
# 1D, (FLnb)
function FfvLCS(FLnb::AbstractVector{T},fLn0::AbstractVector{T},vGk::AbstractVector{Tb},
    va::AbstractVector{Tb},nc0::Int,nvlevel0::AbstractVector{Int},u::Float64,ℓ::Int,vabth::Float64;
    isnormal::Bool=true,restartfit::Vector{Int}=[0,0,100],
    maxIterTR::Int=1000,autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int=1,dnvs::Int=1,
    isboundaryf::Bool=true,μu::Float64=1.0) where{T,Tb,N}

    if vabth == 1.0
        return nothing
    else
        pa,modela,modelav0
        modelap(v) =  modela(v,pa)
        # # Mapping
        if va[1] ≠ 0.0 || modelav0 === nothing
            FLnb = modelap(va)
        else
            FLnb[2:end] = modelap(va[2:end])
            ℓ == 0 ? FLnb[1] = modelav0p(va[1]) : FLnb[1] = 0.0
        end
        return FLnb
    end
end

# 1D, (ddfLn0,dfLn0,fLnnew), `vabth = 1.0`
function FfvLCS(M2::AbstractVector{T},M1::AbstractVector{T},M0::AbstractVector{T},
    fLnk::AbstractVector{T},vGk::AbstractVector{Tb},nc0::Int,nvlevel0::AbstractVector{Int},u::Float64,ℓ::Int;
    isnormal::Bool=true,restartfit::Vector{Int}=[0,0,100],
    maxIterTR::Int=1000,autodiff::Symbol=:forward,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=1e-18,f_tol::Float64=1e-18,g_tol::Float64=1e-18,n10::Int=1,dnvs::Int=1,
    isboundaryf::Bool=true,μu::Float64=1.0) where{T,Tb,N}

    vG0 = vGk[nvlevel0]
    fLn0 = fLnk[nvlevel0]
    counts,pa,modela,modelav0,MnDMka = fvLmodel(vG0,fLn0,ℓ,u,μu;
                isnormal=isnormal,restartfit=restartfit,
                maxIterTR=maxIterTR,autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs)
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
                M1[1] = dfLnDMv0u(0.0,u,ℓ)
            else
                M1[1] = modeldfLn0v0(0.0)
            end
        end
        M2 = zero.(vG0)
        M2[2:end] = ForwardDiff.derivative.(modeldfLn0,vG0[2:end])
        if ℓ == 0 ||  ℓ == 2
            if isboundaryf
                M2[1] = ddfLnDMv0u(0.0,u,ℓ)
            else
                M2[1] = ForwardDiff.derivative.(modeldfLn0v0,0.0)
            end
        end
    end
    return M2,M1,M0
end
