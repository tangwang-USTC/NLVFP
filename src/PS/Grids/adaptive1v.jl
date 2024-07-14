
"""
  Discreting the velocity axis with background regular grids according to
  moment convergence from characteristics `nai, uai, vthi` and shape function `fLnshape`.

    I, `vGmax`: The left boundary will be decided by `L_shape` and `jMax` inner adaptively;

    II, `nvG`: When `is_nvG_adapt = true`. The number of the first level meshgrids will be decided by the convergence of the renormalized kinetic moments adaptively.
               This optimization is for the integrals of kinetic moments `Mc` and kinetic dissipations `Rc`

    III, `nc0`: The third level.

    IV, `nck`: The second level. Together with `nc0` for the integrals of Rosenbluth potentials `H(v)` and  `G(v)`

  Inputs:
    vhe = Vector{AbstractVector{T}}(undef,ns)
    vhk = Vector{AbstractVector{T}}(undef,ns)
    nvlevele0 = Vector{Vector{Int64}}(undef,ns)
    nvlevel0 = Vector{Vector{Int64}}(undef,ns)
    nvG = zeros(Int64,ns)
    nc0 = zeros(Int64,ns)        # The real number of meshgrids in level-0 (include the chebyshev grids but maybe not limited)
    nck = zeros(Int64,ns)        # `nck` is the total number of meshgrids in level-1
    #                            # `nvlevel is the number vector of subdivision chebyshev grids in level-1`
    
  Outputs:
    vHadapt1D!(vhe,vhk, vGdom, nvG, nc0, nck, ocp, 
          nvlevele0, nvlevel0, nai, uai, vthi, nMod, ns;
          eps_fup=eps_fup,eps_flow=eps_flow,
          maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
          abstol=abstol,reltol=reltol,
          vadaptlevels=vadaptlevels,gridv_type=gridv_type,
          is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
"""
# 2.5V, [nMod,nv,LM,ns]
function vHadapt1D!(vhe::Vector{StepRangeLen}, vhk::Vector{AbstractVector{T}}, vGdom::AbstractArray{T,N},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64}, 
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, 
    nai::Vector{AbstractVector{T}}, uai::Vector{AbstractVector{T}}, vthi::Vector{AbstractVector{T}}, 
    nMod::Vector{Int}, ns::Int64;
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T,N}
 
    for isp in 1:ns
        if norm(uai[isp]) ≤ epsT1000
            nvG[isp], nc0[isp], nck[isp], vhe[isp], vhk[isp], vGdom[:,isp], 
                nvlevele0[isp], nvlevel0[isp] = vHadapt1D(
                nvG[isp], ocp[isp], vGdom[:,isp], nai[isp], vthi[isp], nMod[isp];
                eps_fup=eps_fup,eps_flow=eps_flow,
                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                abstol=abstol,reltol=reltol,
                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
        else
            nvG[isp], nc0[isp], nck[isp], vhe[isp], vhk[isp], vGdom[:,isp], 
                nvlevele0[isp], nvlevel0[isp] = vHadapt1D(
                nvG[isp], ocp[isp], vGdom[:,isp], nai[isp], uai[isp], vthi[isp], nMod[isp];
                eps_fup=eps_fup,eps_flow=eps_flow,
                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                abstol=abstol,reltol=reltol,
                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
        end
    end
end
function vHadapt1D!(vhe::Vector{AbstractVector{T}}, vhk::Vector{AbstractVector{T}}, vGdom::AbstractArray{T,N},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64}, 
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, 
    nai::Vector{AbstractVector{T}}, uai::Vector{AbstractVector{T}}, vthi::Vector{AbstractVector{T}}, 
    nMod::Vector{Int}, ns::Int64;
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T,N}
 
    for isp in 1:ns
        if norm(uai[isp]) ≤ epsT1000
            nvG[isp], nc0[isp], nck[isp], vhe[isp], vhk[isp], vGdom[:,isp], 
                nvlevele0[isp], nvlevel0[isp] = vHadapt1D(
                nvG[isp], ocp[isp], vGdom[:,isp], nai[isp], vthi[isp], nMod[isp];
                eps_fup=eps_fup,eps_flow=eps_flow,
                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                abstol=abstol,reltol=reltol,
                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
        else
            nvG[isp], nc0[isp], nck[isp], vhe[isp], vhk[isp], vGdom[:,isp], 
                nvlevele0[isp], nvlevel0[isp] = vHadapt1D(
                nvG[isp], ocp[isp], vGdom[:,isp], nai[isp], uai[isp], vthi[isp], nMod[isp];
                eps_fup=eps_fup,eps_flow=eps_flow,
                maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
                abstol=abstol,reltol=reltol,
                vadaptlevels=vadaptlevels,gridv_type=gridv_type,
                is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
        end
    end
end

# 2.5V,               , uai .= 0
function vHadapt1D!(vhe::Vector{StepRangeLen}, vhk::Vector{AbstractVector{T}}, vGdom::AbstractArray{T,N},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64}, 
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, 
    nai::Vector{AbstractVector{T}}, vthi::Vector{AbstractVector{T}}, 
    nMod::Vector{Int}, ns::Int64;
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T,N}
 
    for isp in 1:ns
        nvG[isp], nc0[isp], nck[isp], vhe[isp], vhk[isp], vGdom[:,isp], 
            nvlevele0[isp], nvlevel0[isp] = vHadapt1D(
            nvG[isp], ocp[isp], vGdom[:,isp], nai[isp], vthi[isp], nMod[isp];
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
    end
end
function vHadapt1D!(vhe::Vector{AbstractVector{T}}, vhk::Vector{AbstractVector{T}}, vGdom::AbstractArray{T,N},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64}, 
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, 
    nai::Vector{AbstractVector{T}}, vthi::Vector{AbstractVector{T}}, 
    nMod::Vector{Int}, ns::Int64;
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T,N}
 
    for isp in 1:ns
        nvG[isp], nc0[isp], nck[isp], vhe[isp], vhk[isp], vGdom[:,isp], 
            nvlevele0[isp], nvlevel0[isp] = vHadapt1D(
            nvG[isp], ocp[isp], vGdom[:,isp], nai[isp], vthi[isp], nMod[isp];
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
    end
end

"""
  Inputs:
    L_shape: = ([0,1], default)
    Nshape: = (2, default)
    nvlevele0 = zeros(Int64,nvG)
    nvlevel0 = zeros(Int64,nc0)

  Outputs:
    vHadapt1D!(vhe, vhk, vGdom, nvG, nc0, nck, ocp, 
            nvlevele0, nvlevel0, uhk, L_shape, Nshape, ns;
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
"""

# 2V, [nv,LM,ns], nMod = 1, nai = vthi = 1
function vHadapt1D!(vhe::Vector{StepRangeLen},  vhk::Vector{AbstractVector{T}}, vGdom::AbstractArray{T,N},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64}, 
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, uhk::AbstractVector{T},  ns::Int64;
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T,N}
 
    for isp in 1:ns
        nvG[isp], nc0[isp], nck[isp], vhe[isp], vhk[isp], vGdom[:,isp], 
            nvlevele0[isp], nvlevel0[isp] = vHadapt1D(
            nvG[isp], ocp[isp], vGdom[:,isp], uhk[isp][1];
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
    end
end
function vHadapt1D!(vhe::Vector{AbstractVector{T}},  vhk::Vector{AbstractVector{T}}, vGdom::AbstractArray{T,N},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64}, 
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, uhk::AbstractVector{T}, ns::Int64;
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T,N}
 
    for isp in 1:ns
        nvG[isp], nc0[isp], nck[isp], vhe[isp], vhk[isp], vGdom[:,isp], 
            nvlevele0[isp], nvlevel0[isp] = vHadapt1D(
            nvG[isp], ocp[isp], vGdom[:,isp], uhk[isp][1];
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
    end
end

"""
  L_shape = 0, Nshape = 1

  Inputs:
    nvlevele0 = zeros(Int64,nvG)
    nvlevel0 = zeros(Int64,nc0)

  Outputs:
    vHadapt1D!(vhe, vhk, vGdom, nvG, nc0, nck, ocp, 
            nvlevele0, nvlevel0, ns;
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
"""

# 2V, [nv,ns], nMod = 1, nai = vthi = 1, uai = 0
function vHadapt1D!(vhe::Vector{StepRangeLen},  vhk::Vector{AbstractVector{T}}, vGdom::AbstractArray{T,N},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64}, 
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, ns::Int64;
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T,N}
  
    for isp in 1:ns
        nvG[isp], nc0[isp], nck[isp], vhe[isp], vhk[isp], vGdom[:,isp], 
            nvlevele0[isp], nvlevel0[isp] = vHadapt1D(
            nvG[isp], ocp[isp], vGdom[:,isp], uhk[isp];
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
    end
end
function vHadapt1D!(vhe::Vector{StepRangeLen},  vhk::Vector{AbstractVector{T}}, vGdom::AbstractArray{T,N},
    nvG::Vector{Int64}, nc0::Vector{Int64}, nck::Vector{Int64}, ocp::Vector{Int64}, 
    nvlevele0::Vector{Vector{Int64}}, nvlevel0::Vector{Vector{Int64}}, ns::Int64;
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100, vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5, reltol::Float64=1e-5, 
    vadaptlevels::Int=4, gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T,N}
 
    for isp in 1:ns
        nvG[isp], nc0[isp], nck[isp], vhe[isp], vhk[isp], vGdom[:,isp], 
            nvlevele0[isp], nvlevel0[isp] = vHadapt1D(
            nvG[isp], ocp[isp], vGdom[:,isp], uhk[isp];
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
    end
end

"""
  Inputs:
    nvlevele0 = zeros(Int64,nvG)
    nvlevel0 = zeros(Int64,nc0)

  Outputs:
    nc0, nck, vhe, vhk, vGdom, nvlevele0, 
            nvlevel0 = vHadapt1D(nvG, ocp, vGdom, nai, uai, vthi, nMod;
            eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
            abstol=abstol,reltol=reltol,
            vadaptlevels=vadaptlevels,gridv_type=gridv_type,
            is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
"""

# 1.5V, [is_nvG_adapt,vadaptlevels], ns = 1
# [nMod,nv]
function vHadapt1D(nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nai::AbstractVector{T}, uai::AbstractVector{T}, vthi::AbstractVector{T}, nMod::Int64;
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5,reltol::Float64=1e-5, 
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T}

    L_shape = [0, 1]
    Nshape = length(L_shape)
    # nj = 1 + nMod
    jVconst = Vector{Vector{Int64}}(undef,Nshape)
    jVconst[1] = [0,2nMod] |> Vector{Int64}      # L = 0
    jVconst[2] = jVconst[1] .+ 1                 # L = 1
    jMax = [jVconst[1][end],jVconst[2][end]]

    # `vGmax`: Estimating the right-boundary of the velocity axis and giving the shape functions;
    fLnshape = Vector{Function}(undef,Nshape)
    vGdom[2] = vGmaxMhs!(fLnshape,Nshape,nai,uai,vthi,nMod;
        L_shape=L_shape,jMax=jMax,eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit)

    # `nvG`: Giving the initial regular uniform meshgrids or the Chebyshev grids as the background grids
    if gridv_type == :uniform
        vhe = range(vGdom[1], vGdom[2], length=nvG) # |> Vector{Float64}
    elseif gridv_type == :chebyshev
        # vc0 = vccn(nvG;datatype=T)
        vhe = vCmapping(vccn(nvG;datatype=T),vGdom[1],vGdom[2];isinv=true)
    else
        ertyuyhgtf
    end
    vh0 = vhe |> Vector{T}

    # Evoluating the convergence of normolized kinetic moments `Mhc` to optimizing `nc0` for Romberg integral or GQ.
    if is_nvG_adapt
        if nvG ≥ nvG_limit
            if nvG > nvG_limit
                nvG = deepcopy(nvG_limit)
                @warn("Warning:`nvG` has been bigger than `nvG_limit1` and been replaced; Checking for efficiency and accuracy!")
            end
        else
            k = 1
            L = deepcopy(L_shape[k])
            nj = 2nMod
            fLn = fLnshape[k](vhe)

            for k in 2:Nshape
                L = deepcopy(L_shape[k])
                fLn = fLnshape[k](vhe)
            end
        end
    end

    nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0 = vHadapt1D(
        vh0,vhe,nvG,ocp,vGdom,fLnshape,Nshape,jVconst;
        abstol=abstol, reltol=reltol,vadaptlevels=vadaptlevels)

    return nvG, nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
end

# [nMod,nv],                         , uai .= 0
function vHadapt1D(nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    nai::AbstractVector{T}, vthi::AbstractVector{T}, nMod::Int64;
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5,reltol::Float64=1e-5, 
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T}

    L_shape = [0]
    Nshape = length(L_shape)
    # nj = 1 + nMod
    if Nshape == 1
        jVconst = [0,2nMod]
        jMax = [jVconst[end]]
    else
        jVconst = Vector{Vector{Int64}}(undef,Nshape)
        jVconst[1] = [0,2nMod] |> Vector{Int64}      # L = 0
        jMax = [jVconst[1][end]]
    end

    # `vGmax`: Estimating the right-boundary of the velocity axis and giving the shape functions;
    fLnshape = Vector{Function}(undef,Nshape)
    vGdom[2] = vGmaxMhs!(fLnshape,Nshape,nai,vthi,nMod;
        L_shape=L_shape,jMax=jMax,eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit)

    # `nvG`: Giving the initial regular uniform meshgrids or the Chebyshev grids as the background grids
    if gridv_type == :uniform
        vhe = range(vGdom[1], vGdom[2], length=nvG) # |> Vector{Float64}
    elseif gridv_type == :chebyshev
        # vc0 = vccn(nvG;datatype=T)
        vhe = vCmapping(vccn(nvG;datatype=T),vGdom[1],vGdom[2];isinv=true)
    else
        ertyuyhgtf
    end
    vh0 = vhe |> Vector{T}

    # Evoluating the convergence of normolized kinetic moments `Mhc` to optimizing `nc0` for Romberg integral or GQ.
    if is_nvG_adapt
        if nvG ≥ nvG_limit
            if nvG > nvG_limit
                nvG = deepcopy(nvG_limit)
                @warn("Warning:`nvG` has been bigger than `nvG_limit1` and been replaced; Checking for efficiency and accuracy!")
            end
        else
            k = 1
            L = deepcopy(L_shape[k])
            nj = 2nMod
            fLn = fLnshape[k](vhe)

            for k in 2:Nshape
                L = deepcopy(L_shape[k])
                fLn = fLnshape[k](vhe)
            end
        end
    end

    if Nshape == 1
        nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0 = vHadapt1D(
            vh0,vhe,nvG,ocp,vGdom,fLnshape[1],jVconst;
            abstol=abstol, reltol=reltol,vadaptlevels=vadaptlevels)
    else
        nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0 = vHadapt1D(
            vh0,vhe,nvG,ocp,vGdom,fLnshape,Nshape,jVconst;
            abstol=abstol, reltol=reltol,vadaptlevels=vadaptlevels)
    end

    return nvG, nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
end

#1V, [nv], nMod = 1, nai = vthi = 1
function vHadapt1D(nvG::Int64, ocp::Int64, vGdom::AbstractVector{T}, uai::T;
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5,reltol::Float64=1e-5, 
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T}

    L_shape = [0, 1]
    Nshape = length(L_shape)
    # nj = 1 + nMod
    jVconst = Vector{Vector{Int64}}(undef,Nshape)
    jVconst[1] = [0,2]      # j ∈ [0,2], L = 0
    jVconst[2] = [1,3]      # j ∈ [1,3], L = 1
    jMax = [2,3]

    # `vGmax`: Estimating the right-boundary of the velocity axis and giving the shape functions;
    fLnshape = Vector{Function}(undef,Nshape)
    vGdom[2] = vGmaxMhs!(fLnshape,Nshape,uai;
        L_shape=L_shape,jMax=jMax,eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit)

    # `nvG`: Giving the initial regular uniform meshgrids or the Chebyshev grids as the background grids
    if gridv_type == :uniform
        vhe = range(vGdom[1], vGdom[2], length=nvG) # |> Vector{Float64}
    elseif gridv_type == :chebyshev
        # vc0 = vccn(nvG;datatype=T)
        vhe = vCmapping(vccn(nvG;datatype=T),vGdom[1],vGdom[2];isinv=true)
    else
        ertyuyhgtf
    end
    vh0 = vhe |> Vector{T}

    # Evoluating the convergence of kinetic moments to optimizing `nc0` for Romberg integral or GQ.
    if is_nvG_adapt
        if nvG ≥ nvG_limit
            if nvG > nvG_limit
                nvG = deepcopy(nvG_limit)
                @warn("Warning:`nvG` has been bigger than `nvG_limit1` and been replaced; Checking for efficiency and accuracy!")
            end
        else
            k = 1
            L = deepcopy(L_shape[k])            # = 0
            fLn = fLnshape[k](vhe)

            k = 1
            L = deepcopy(L_shape[k])            # = 1
            fLn = fLnshape[k](vhe)

            for k in 3:Nshape
                L = deepcopy(L_shape[k])
                fLn = fLnshape[k](vhe)
            end
        end
    end

    nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0 = vHadapt1D(
        vh0,vhe,nvG,ocp,vGdom,fLnshape,Nshape,jVconst;
        abstol=abstol, reltol=reltol,vadaptlevels=vadaptlevels)

    return nvG, nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
end

# [nv],                         , uai = 0
function vHadapt1D(nvG::Int64, ocp::Int64, vGdom::AbstractVector{T};
    eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::Vector{T}=[5.0, 20],
    abstol::Float64=epsT5,reltol::Float64=1e-5, 
    vadaptlevels::Int=4,gridv_type::Symbol=:uniform,
    is_nvG_adapt::Bool=false,nvG_limit::Int64=9) where {T}

    # L_shape = 0             
    # nj = 1 + nMod           # 2, j ∈ [0,2]
    jVconst = [0,2]         
    jMax = 2

    # `vGmax`: Estimating the right-boundary of the velocity axis and giving the shape functions;
    vGdom[2], fLnshape = vGmaxMhs(;jMax=jMax,eps_fup=eps_fup,eps_flow=eps_flow,
        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit)

    # `nvG`: Giving the initial regular uniform meshgrids or the Chebyshev grids as the background grids
    if gridv_type == :uniform
        vhe = range(vGdom[1], vGdom[2], length=nvG) # |> Vector{Float64}
    elseif gridv_type == :chebyshev
        # vc0 = vccn(nvG;datatype=T)
        vhe = vCmapping(vccn(nvG;datatype=T),vGdom[1],vGdom[2];isinv=true)
    else
        ertyuyhgtf
    end
    vh0 = vhe |> Vector{T}

    # Evoluating the convergence of kinetic moments to optimizing `nc0` for Romberg integral or GQ.
    if is_nvG_adapt
        if nvG ≥ nvG_limit
            if nvG > nvG_limit
                nvG = deepcopy(nvG_limit)
                @warn("Warning:`nvG` has been bigger than `nvG_limit1` and been replaced; Checking for efficiency and accuracy!")
            end
        else
            fLn = fLnshape(vhe)
        end
    end

    nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0 = vHadapt1D(
        vh0,vhe,nvG,ocp,vGdom,fLnshape,jVconst;
        abstol=abstol, reltol=reltol,vadaptlevels=vadaptlevels)

    return nvG, nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
end

"""

  Outputs:
    nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0 = vHadapt1D(
        vh0,vhe,nvG,ocp,vGdom,fLnshape,Nshape,jVconst;
        abstol=abstol, reltol=reltol,vadaptlevels=vadaptlevels)
"""

# 1V, [jVconst,Nshape,vadaptlevels,nv]
function vHadapt1D(vh0::AbstractVector{T},vhe::StepRangeLen,
    nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    fLnshape::Vector{Function},Nshape::Int64,jVconst::Vector{Vector{Int64}};
    abstol::Float64=epsT5,reltol::Float64=1e-5,vadaptlevels::Int=4) where {T}

    if vadaptlevels == 0
        nc0, nck = copy(nvG), copy(nvG)
        vhk = deepcopy(vh0)
        nvlevele0 = 1:nvG |> Vector{Int64}
        nvlevel0 = 1:nc0 |> Vector{Int64}
        return nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
    else
        # `nc0`: [level,nv], Generating the refining meshgrids according to the dichotomy method
        if vadaptlevels ≥ 2
            # isconvergeds, 
            nc0 = length(vh0)
            if Nshape == 1
                RδdMs = zeros(T,2)
                num0, vh0 = vHadaptive!(RδdMs,fLnshape[1],vh0,nc0,ocp,jVconst[1];
                                abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels)
                num0 == 0 ? isconverged = true : isconverged = false
                Lvh0 = 1         # ∈ [1:Nshape]
            else
                isconverged, num0, RδdMs, Lvh0, vh0 = vHadaptive(fLnshape,Nshape,vh0,nc0,ocp,jVconst[1];
                                abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels)
            end
        else
            Lvh0 = 1
            @warn("`vadaptlevels = 1` will be not proposed!")
        end

        #`nck`: [j,level,nv], Refinement by chebyshev grids on subdivision of the initial meshgrids with number `nc0`.
        # # （2,1） [j_lower,j_higher], shape_lowest
        nc0 = length(vh0)
        nck, nvlevel, vlevel = vchebyHadaptive(fLnshape[Lvh0],vh0,jVconst[Lvh0];
                            nc0=nc0, ocp=ocp, abstol=abstol, reltol=1e-4, vadaptlevels=1)
            
        vhk = zeros(T,nck)
        vGvlevel1!(vhk, nvlevel, vlevel; isinv=false)
    
        nvlevele0 = zeros(Int64,nvG)
        nvlevel0invGk!(nvlevele0, vhe, vh0, nvG, nc0) # The position of the initial equally spaced points `vGe` among in `vh0`.
    
        nvlevel0 = zeros(Int64,nc0)
        nvlevel0invGk!(nvlevel0, vh0, vhk, nc0, nck)  # the position of initial chebyshev grids `vh0` among in `vGk`
        return nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
    end
end
function vHadapt1D(vh0::AbstractVector{T},vhe::AbstractVector{T},
    nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    fLnshape::Vector{Function},Nshape::Int64,jVconst::Vector{Vector{Int64}};
    abstol::Float64=epsT5,reltol::Float64=1e-5,vadaptlevels::Int=4) where {T}

    if vadaptlevels == 0
        nc0, nck = copy(nvG), copy(nvG)
        vhk = deepcopy(vh0)
        nvlevele0 = 1:nvG |> Vector{Int64}
        nvlevel0 = 1:nc0 |> Vector{Int64}
        return nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
    else
        # `nc0`: [level,nv], Generating the refining meshgrids according to the dichotomy method
        if vadaptlevels ≥ 2
            # isconvergeds, 
            nc0 = length(vh0)
            if Nshape == 1
                RδdMs = zeros(T,2)
                num0, vh0 = vHadaptive!(RδdMs,fLnshape[1],vh0,nc0,ocp,jVconst[1];
                                abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels)
                num0 == 0 ? isconverged = true : isconverged = false
                Lvh0 = 1         # ∈ [1:Nshape]
            else
                isconverged, num0, RδdMs, Lvh0, vh0 = vHadaptive(fLnshape,Nshape,vh0,nc0,ocp,jVconst[1];
                                abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels)
            end
        else
            Lvh0 = 1
            @warn("`vadaptlevels = 1` will be not proposed!")
        end

        #`nck`: [j,level,nv], Refinement by chebyshev grids on subdivision of the initial meshgrids with number `nc0`.
        # # （2,1） [j_lower,j_higher], shape_lowest
        nc0 = length(vh0)
        nck, nvlevel, vlevel = vchebyHadaptive(fLnshape[Lvh0],vh0,jVconst[Lvh0];
                            nc0=nc0, ocp=ocp, abstol=abstol, reltol=1e-4, vadaptlevels=1)
            
        vhk = zeros(T,nck)
        vGvlevel1!(vhk, nvlevel, vlevel; isinv=false)
    
        nvlevele0 = zeros(Int64,nvG)
        nvlevel0invGk!(nvlevele0, vhe, vh0, nvG, nc0) # The position of the initial equally spaced points `vGe` among in `vh0`.
    
        nvlevel0 = zeros(Int64,nc0)
        nvlevel0invGk!(nvlevel0, vh0, vhk, nc0, nck)  # the position of initial chebyshev grids `vh0` among in `vGk`
        return nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
    end
end

# 1V, Nshape = 1
function vHadapt1D(vh0::AbstractVector{T},vhe::StepRangeLen,
    nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    fLnshape::Function,jVconst::Vector{Int64};
    abstol::Float64=epsT5,reltol::Float64=1e-5,vadaptlevels::Int=4) where {T}

    if vadaptlevels == 0
        nc0, nck = copy(nvG), copy(nvG)
        vhk = deepcopy(vh0)
        nvlevele0 = 1:nvG |> Vector{Int64}
        nvlevel0 = 1:nc0 |> Vector{Int64}
        return nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
    else
        # `nc0`: [level,nv], Generating the refining meshgrids according to the dichotomy method
        if vadaptlevels ≥ 2
            # isconvergeds, 
            nc0 = length(vh0)
            RδdMs = zeros(T,2)
            num0, vh0 = vHadaptive!(RδdMs,fLnshape,vh0,nc0,ocp,jVconst[1];
                            abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels)
            num0 == 0 ? isconverged = true : isconverged = false
        end

        #`nck`: [j,level,nv], Refinement by chebyshev grids on subdivision of the initial meshgrids with number `nc0`.
        # # （2,1） [j_lower,j_higher], shape_lowest
        nc0 = length(vh0)
        nck, nvlevel, vlevel = vchebyHadaptive(fLnshape,vh0,jVconst;
                            nc0=nc0, ocp=ocp, abstol=abstol, reltol=1e-4, vadaptlevels=1)
            
        vhk = zeros(T,nck)
        vGvlevel1!(vhk, nvlevel, vlevel; isinv=false)
    
        nvlevele0 = zeros(Int64,nvG)
        nvlevel0invGk!(nvlevele0, vhe, vh0, nvG, nc0) # The position of the initial equally spaced points `vGe` among in `vh0`.
    
        nvlevel0 = zeros(Int64,nc0)
        nvlevel0invGk!(nvlevel0, vh0, vhk, nc0, nck)  # the position of initial chebyshev grids `vh0` among in `vGk`
        return nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
    end
end
function vHadapt1D(vh0::AbstractVector{T},vhe::AbstractVector{T},
    nvG::Int64, ocp::Int64, vGdom::AbstractVector{T},
    fLnshape::Function,jVconst::Vector{Int64};
    abstol::Float64=epsT5,reltol::Float64=1e-5,vadaptlevels::Int=4) where {T}

    if vadaptlevels == 0
        nc0, nck = copy(nvG), copy(nvG)
        vhk = deepcopy(vh0)
        nvlevele0 = 1:nvG |> Vector{Int64}
        nvlevel0 = 1:nc0 |> Vector{Int64}
        return nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
    else
        # `nc0`: [level,nv], Generating the refining meshgrids according to the dichotomy method
        if vadaptlevels ≥ 2
            # isconvergeds, 
            nc0 = length(vh0)
            RδdMs = zeros(T,2)
            num0, vh0 = vHadaptive!(RδdMs,fLnshape,vh0,nc0,ocp,jVconst[1];
                            abstol=abstol, reltol=reltol, vadaptlevels=vadaptlevels)
            num0 == 0 ? isconverged = true : isconverged = false
        end

        #`nck`: [j,level,nv], Refinement by chebyshev grids on subdivision of the initial meshgrids with number `nc0`.
        # # （2,1） [j_lower,j_higher], shape_lowest
        nc0 = length(vh0)
        nck, nvlevel, vlevel = vchebyHadaptive(fLnshape,vh0,jVconst;
                            nc0=nc0, ocp=ocp, abstol=abstol, reltol=1e-4, vadaptlevels=1)
            
        vhk = zeros(T,nck)
        vGvlevel1!(vhk, nvlevel, vlevel; isinv=false)
    
        nvlevele0 = zeros(Int64,nvG)
        nvlevel0invGk!(nvlevele0, vhe, vh0, nvG, nc0) # The position of the initial equally spaced points `vGe` among in `vh0`.
    
        nvlevel0 = zeros(Int64,nc0)
        nvlevel0invGk!(nvlevel0, vh0, vhk, nc0, nck)  # the position of initial chebyshev grids `vh0` among in `vGk`
        return nc0, nck, vhe, vhk, vGdom, nvlevele0, nvlevel0
    end
end
