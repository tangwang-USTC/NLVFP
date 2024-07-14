"""
  Findig out the best values of the right boundary of `vGe`: vGmax, for very spices.
  
  Inputs:
    L_shape: (=0, default) Shape function for meshgrids on the velocity axis
    jMax: (=[2,3], default) The maximum number of moments of `ℓᵗʰ`-order amplitude function to decide the values of `vGm`
    maxiter_vGm: (=10, default) The maxinum number of iteration to find out the vaule of `vGmax`
        
"""

"""
  Outputs:
    vGm = vGmaxMhs(fLnshape,Nshape,nai,uai,vthi,nMod;
                  L_shape=L_shape,jMax=jMax,eps_fup=eps_fup,eps_flow=eps_flow,
                  maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,is_out_shape=is_out_shape)
  
"""

# 1V, [nLshape,nMod]
function vGmaxMhs!(fLnshape::Vector{Function},Nshape::Int64,
    nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64;
    L_shape::Vector{Int64}=[0,1],jMax::Vector{Int64}=[2,3],eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::AbstractVector{T}=[5,20],) where{T}
    
    k = 1
    vGm, fLnshape[k] = vGmaxMhs(nai,uai,vthi,nMod;
              L_shape=L_shape[k],jMax=jMax[k],eps_fup=eps_fup,eps_flow=eps_flow,
              maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,is_out_shape=true)
    for k in 2:Nshape
        vGmk, fLnshape[k] = vGmaxMhs(nai,uai,vthi,nMod;
                  L_shape=L_shape[k],jMax=jMax[k],eps_fup=eps_fup,eps_flow=eps_flow,
                  maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,is_out_shape=true)
        vGm = max(vGm, vGmk)
    end
    return vGm
end

# 1V, uai .= 0
function vGmaxMhs!(fLnshape::Vector{Function},Nshape::Int64,
    nai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64;
    L_shape::Vector{Int64}=[0,2],jMax::Vector{Int64}=[2,4],eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::AbstractVector{T}=[5,20],) where{T}
    
    k = 1
    vGm, fLnshape[k] = vGmaxMhs(nai,vthi,nMod;
              L_shape=L_shape[k],jMax=jMax[k],eps_fup=eps_fup,eps_flow=eps_flow,
              maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,is_out_shape=true)
    for k in 2:Nshape
        vGmk, fLnshape[k] = vGmaxMhs(nai,vthi,nMod;
                  L_shape=L_shape[k],jMax=jMax[k],eps_fup=eps_fup,eps_flow=eps_flow,
                  maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,is_out_shape=true)
        vGm = max(vGm, vGmk)
    end
    return vGm
end


# 0.5V, [nMod], `ns = nLshape = 1`
function vGmaxMhs(nai::AbstractVector{T},uai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64;
    L_shape::Int64=0,jMax::Int64=2,eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::AbstractVector{T}=[5,20],
    is_out_shape::Bool=false) where{T}
    
    # Shape function for meshgrids on the velocity axis
    p = zeros(3nMod)
    pnuTi!(p, nai, uai, vthi,nMod)
    fLnshape(v) = fvLmodel(L_shape,nMod)(v,p)  # shape function for create the background meshgrids.

    # Finding the values of `vGmax`
    jM2 = max(jMax, 2nMod) + 2  # 2 + j

    dom = deepcopy(vGm_limit)
    vGms, vGmb = dom[1], dom[2]
    vGm = sum(dom) / 2

    fLn9 = fLnshape(vGm) * vGm^jM2
    if eps_flow ≤ fLn9 ≤ eps_fup
        if is_out_shape
            return vGm, fLnshape
        else
            return vGm
        end
    else
        for i in 1:maxiter_vGm
            fLn9 = fLnshape(vGm) * vGm^jM2
            if fLn9 > eps_fup
                vGms = vGm
                dom = [vGms, vGmb]
                vGm = sum(dom) / 2
            elseif fLn9 < eps_flow
                vGmb = vGm
                dom = [vGms, vGm]
                vGm = sum(dom) / 2
            else
                break
            end
        end
        if is_out_shape
            return vGm, fLnshape
        else
            return vGm
        end
    end
end

# 0.5V, [nMod],                    uai .= 0
function vGmaxMhs(nai::AbstractVector{T},vthi::AbstractVector{T},nMod::Int64;
    L_shape::Int64=0,jMax::Int64=2,eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::AbstractVector{T}=[5,20],
    is_out_shape::Bool=false) where{T}
    
    # Shape function for meshgrids on the velocity axis
    p = zeros(3nMod)
    pnuTi!(p, nai, 0nai, vthi,nMod)
    fLnshape(v) = fvLmodel(L_shape,nMod)(v,p)  # shape function for create the background meshgrids.

    # Finding the values of `vGmax`
    jM2 = max(jMax, 2nMod) + 2  # 2 + j

    dom = deepcopy(vGm_limit)
    vGms, vGmb = dom[1], dom[2]
    vGm = sum(dom) / 2

    fLn9 = fLnshape(vGm) * vGm^jM2
    if eps_flow ≤ fLn9 ≤ eps_fup
        if is_out_shape
            return vGm, fLnshape
        else
            return vGm
        end
    else
        for i in 1:maxiter_vGm
            fLn9 = fLnshape(vGm) * vGm^jM2
            if fLn9 > eps_fup
                vGms = vGm
                dom = [vGms, vGmb]
                vGm = sum(dom) / 2
            elseif fLn9 < eps_flow
                vGmb = vGm
                dom = [vGms, vGm]
                vGm = sum(dom) / 2
            else
                break
            end
        end
        if is_out_shape
            return vGm, fLnshape
        else
            return vGm
        end
    end
end

# 0.5V, [nLshape]
function vGmaxMhs!(fLnshape::Vector{Function},Nshape::Int64,uai::T;
    L_shape::Vector{Int64}=[0,1],jMax::Vector{Int64}=[2,3],eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::AbstractVector{T}=[5,20]) where{T}
    
    k = 1
    vGm, fLnshape[k] = vGmaxMhs(uai;
              L_shape=L_shape[k],jMax=jMax[k],eps_fup=eps_fup,eps_flow=eps_flow,
              maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,is_out_shape=true)
    for k in 2:Nshape
        vGmk, fLnshape[k] = vGmaxMhs(uai;
                  L_shape=L_shape[k],jMax=jMax[k],eps_fup=eps_fup,eps_flow=eps_flow,
                  maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,is_out_shape=true)
        vGm = max(vGm, vGmk)
    end
    return vGm
end

# 0V, [], `ns = nLshape = 1`, nMod = 1, vGmax = vGmax(uai;L_shape,jMax)
function vGmaxMhs(uai::T;
    L_shape::Int64=0,jMax::Int64=2,eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::AbstractVector{T}=[5,20],
    is_out_shape::Bool=true) where{T}
    
    # Shape function for meshgrids on the velocity axis
    fLnshape(v) = fvLmodel(L_shape,1)(v,[1.0, uai, 1.0])  # shape function for create the background meshgrids.

    # Finding the values of `vGmax` by dichotomy.
    j2 = max(jMax,2) + 2

    dom = deepcopy(vGm_limit)
    vGms, vGmb = dom[1], dom[2]       # Bounds for the left boundary, [5,20]
    vGm = sum(dom) / 2

    fLn9 = fLnshape(vGm) * vGm^j2
    if eps_flow ≤ fLn9 ≤ eps_fup
        # printstyled("vGmaxMhs:fLn9=",(fLnshape(vGm),fLn9),color=:red,"\n")
        if is_out_shape
            return vGm, fLnshape
        else
            return vGm
        end
    else
        for i in 1:maxiter_vGm
            fLn9 = fLnshape(vGm) * vGm^j2
            if fLn9 > eps_fup
                vGms = vGm
                dom = [vGms, vGmb]
                vGm = sum(dom) / 2
            elseif fLn9 < eps_flow
                vGmb = vGm
                dom = [vGms, vGm]
                vGm = sum(dom) / 2
            else
                break
            end
        end
        # printstyled("vGmaxMhs:fLn9=",(vGm,fLnshape(vGm),fLn9),color=:red,"\n")
        if is_out_shape
            return vGm, fLnshape
        else
            return vGm
        end
    end
end

# 0V, [],                                vGmax = vGmax(;jMax)
function vGmaxMhs(;jMax::Int64=2,eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::AbstractVector{T}=[5,20],
    is_out_shape::Bool=true) where{T}
    
    # Shape function for meshgrids on the velocity axis
    fLnshape(v) = fvLmodel(0,1)(v,[1.0, 0.0, 1.0])  # shape function for create the background meshgrids.

    # Finding the values of `vGmax` by dichotomy.
    j2 = max(jMax,2) + 2

    dom = deepcopy(vGm_limit)
    vGms, vGmb = dom[1], dom[2]       # Bounds for the left boundary, [5,20]
    vGm = sum(dom) / 2

    fLn9 = fLnshape(vGm) * vGm^j2
    if eps_flow ≤ fLn9 ≤ eps_fup
        # printstyled("vGmaxMhs:fLn9=",(fLnshape(vGm),fLn9),color=:red,"\n")
        if is_out_shape
            return vGm, fLnshape
        else
            return vGm
        end
    else
        for i in 1:maxiter_vGm
            fLn9 = fLnshape(vGm) * vGm^j2
            if fLn9 > eps_fup
                vGms = vGm
                dom = [vGms, vGmb]
                vGm = sum(dom) / 2
            elseif fLn9 < eps_flow
                vGmb = vGm
                dom = [vGms, vGm]
                vGm = sum(dom) / 2
            else
                break
            end
        end
        # printstyled("vGmaxMhs:fLn9=",(vGm,fLnshape(vGm),fLn9),color=:red,"\n")
        if is_out_shape
            return vGm, fLnshape
        else
            return vGm
        end
    end
end

"""
  Outputs:
    vGmaxMhs!(vGm,fLnshape,Nshape,nai,uai,vthi,nMod;
            L_shape=L_shape,jMax=jMax,eps_fup=eps_fup,eps_flow=eps_flow,
            maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit)
"""

# 1.5V, [nLshape,nMod,ns]
function vGmaxMhs!(vGm::AbstractVector{T},fLnshape::Vector{Vector{Function}},Nshape::Vector{Int64},
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},
    nMod::Vector{Int64},ns::Int64;
    L_shape::Vector{Int64}=[0,1],jMax::Int64=2,eps_fup::T=1e-17,eps_flow::T=1e-18,
    maxiter_vGm::Int64=100,vGm_limit::AbstractVector{T}=[5,20]) where{T}

    efrghnjm
    for isp in 1:ns
        a = fLnshape[isp]
        vGm[isp] = vGmaxMhs!(a,Nshape,nai[isp], uai[isp], vthi[isp],nMod[isp];
                        L_shape=L_shape,jMax=jMax,eps_fup=eps_fup,eps_flow=eps_flow,
                        maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit)
        fLnshape[isp] = a
    end
end
