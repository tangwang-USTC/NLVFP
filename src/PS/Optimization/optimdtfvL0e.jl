"""
  Optimization the function `δₜfₗᵐ(v̂)` when `v̂ → 0` according to the asymptotic behavior of the function:

    `ℓ = 0`: δₜfₗᵐ(v̂ → 0) → C₀, ∂ᵥδₜfₗᵐ(v̂ → 0) → 0;

    `ℓ ≥ 1`: δₜfₗᵐ(v̂ → 0) → 0;
  
  Optimization strategy for the function `δₜfₗᵐ(v̂` → 0)` when the step `Δv̂ᵢ` is const:
    y = δₜfₗᵐ / fₗᵐ                       # The relative function
    Rd1yᵢ = (yᵢ₊₁ - yᵢ₋₁) / 2yᵢ          # The first orders relative difference of `y` with `CDs`
    Rd2yᵢ = (Rd1yᵢ₊₁ - Rd1yᵢ₋₁) / 2Rd1yᵢ   # The second orders relative difference of `y` with `CDs`

    `y(v̂→0) → Cₗᵐ`
    `Rd1y(v̂→0) → 0`
  
  The critic number `nvc` :
    
    length(nvc) = 2 * (order_smooth + 1)
  
  When `order_smooth = 3" gives

    nvc = zeros(Int64,7,LM1,ns)  # where `nvc[:,L1,isp] = [nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3]`
    
    for most case, `nvcy00 ≤ nvcd1 ≤ nvcd2 ≤ nvcy1 ≤ nvcy0`
  
  where
    
    nvcd1: which is decided according to `Rd1y` when `abs(Rd1y[k])` is small enough; 
          such as `abs(Rd1y[k]) ≤ abstol_Rd1y`.
    nvcd2: which is decided according to `Rd2y` when is is smooth enough; 
          such as `abs(Rd2y[k]) ≤ abstol_Rd2y`.
    nvcd3: which is decided according to `Rd3y` when is is smooth enough; 
          such as `abs(Rd3y[k]) ≤ abstol_Rd3y`.
    nvcy00: (=1, default) if `nvcy00 ≥ 1`, gives `k < nvcy00`, `δₜfₗᵐ[k] = 0.0` when `v̂ → 0` and `nvcy00 ≤ nvcd1`
    nvcy0: `fₗᵐ[nvcy0] ~ 0.0` when `nvcy0 ≥ nvcd1`
    nvcy1: `y[nvcy1] ~ 0.0` when `nvcy1 ≥ nvcd2`
    nvcy2: `y[nvcy2] ~ 0.0` when `nvcy2 ≥ nvcd3`
          [1:nvcy00-1], [nvcy00:nvcd1-1], [nvcd1:nvcd2-1], [nvcd2:nvcd3]

  That is (`CDs`):
    Rd1yᵢ₋₁ = Rd1yᵢ₊₁ - 2Rd2yᵢ * Rd1yᵢ
    yᵢ₋₁ = yᵢ₊₁ - 2Rd1yᵢ * yᵢ        # `y(v̂` → 0)` can be optimized by smoothing ([nvcd1:nvcd2]) and interpolating ([nvcy00:nvcd1-1]) methods 
                                        to eliminate the higher-orders frequency oscillations;
                                     # `y[nvcd1:nvcd2]` could be smoothed to obtain the optimized performance when `abstol_Rdy` is small enough, it. et. `abstol_Rdy ≤ 0.3`.
                                     # `y[nvcy00:nvcd1-1]` or `y[nvcy00:nvcd2-1]` will be interpolated according to the values `yᵢ` when `k ≥ nc1`
    δₜfₗᵐᵢ = yᵢ * fₗᵐᵢ                 # 

  Inputs:
    orders:: # (=2, default), `orders ∈ [-1, 1, 2]` which denotes `[BackwardDiff, ForwardDiff, CentralDiff]`
    order_smooth: (=2, default), which is `∈ N⁺`, the highest order of smoothness to smooth the function `dtfvL`.
    order_smooth_itp:: (=0, default), (0,1) → (y=Rdtf,Rd1y), the order of function to be extrapolated to smooth the function `dtfvL`.
    order_nvc_itp:: (=2, default), Applying the critic number of grids `nvcdi` 
                according to the order of smoothness to interpolate the function `dtfvL(v̂→0)`.
                (1,2,3,23) → (nvcd1,nvcd2,nvcd3,max(nvcd2,nvcd3))
    is_boundv0::Vector{Bool}=zeros(Bool,order_smooth), (=[true,false], default). 
                When it is `true`, the value `Rdiy(v[1]=0.0)` will be `true`.
    Nsmooth: (=3, default), number of points to smooth the function `dtfvL`.
    nvc0_limit:: (=4, default), The lower bound of `nvc(order_nvc_itp)` to applying to the extrapolation for `dtfLn(v̂→0)`
    L1nvc_limit:: (=3, default), The lower bound of `nvc(order_nvc_itp)` to applying to the extrapolation for `dtfvL(v̂→0;L1)`
    k_δtf: (=2, default), the order of the Spline Interpolations for `dtfLn(v̂→0)`
    Nitp: (=10, default), the number of gridpoints to generate the interpolating function for `dtfLn(v̂→0)`

  Outputs:
    nvc = nvcfind(nvc,dtf,fvL,nvG,ns,LM;orders=order_dvdtf,is_boundv0=is_boundv0,
                  Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)
    optimdtfvL0e!(nvc,dtf,fvL,vGe,nvG,ns,LM;orders=order_dvdtf,is_boundv0=is_boundv0,
                  Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy,
                  k=k,Nitp=Nitp,order_smooth_itp=order_smooth_itp,order_nvc_itp=order_nvc_itp,
                  nvc0_limit=nvc0_limit,L1nvc_limit=L1nvc_limit)
"""

# 3D, [nvc]
function nvcfind(nvc::Array{Int64,N}, dtf::AbstractArray{T,N}, fvL::AbstractArray{T,N},
    nv::Int64,ns::Int64, LM::Vector{Int64}; orders::Int64=1, is_boundv0::Vector{Bool}=[true,false],  
    Nsmooth::Int=3,order_smooth::Int64=2, abstol_Rdy::AbstractVector{T}=[0.35,0.35]) where {T,N}

    for isp in 1:ns
        L1 = 1
        nvc[:,L1,isp] = nvcfind(nvc[:,L1,isp],dtf[:,L1,isp],fvL[:,L1,isp], nv, L1; 
            orders=orders, is_boundv0=is_boundv0, Nsmooth=Nsmooth, 
            order_smooth=order_smooth, abstol_Rdy=abstol_Rdy)
        if LM[isp] == 1
            L1 = 2
            if norm(fvL[:, L1,isp]) ≥ abs_dfLM
                nvc[:,L1,isp] = nvcfind(nvc[:,L1,isp],dtf[:,L1,isp],fvL[:,L1,isp], nv, L1; 
                    orders=orders, is_boundv0=is_boundv0, Nsmooth=Nsmooth, nvcup=nvc[:,L1-1,isp],
                    order_smooth=order_smooth, abstol_Rdy=abstol_Rdy)
            end
        else
            for L1 in 2:LM[isp]+1
                nvc[:,L1,isp] = nvcfind(nvc[:,L1,isp],dtf[:,L1,isp],fvL[:,L1,isp], nv, L1; 
                    orders=orders, is_boundv0=is_boundv0, Nsmooth=Nsmooth, nvcup=nvc[:,L1-1,isp],
                    order_smooth=order_smooth, abstol_Rdy=abstol_Rdy)
            end
        end
    end
    return nvc
end

# 3D, [nvc, δtf]
function optimdtfvL0e!(nvc::Array{Int64,N}, dtf::AbstractArray{T,N}, 
    fvL::AbstractArray{T,N},vGe::AbstractVector{T},nv::Int64,ns::Int64, LM::Vector{Int64}; 
    orders::Int64=1, is_boundv0::Vector{Bool}=[true,false], Nsmooth::Int=3, 
    order_smooth::Int64=2,abstol_Rdy::AbstractVector{T}=[0.35,0.35],k::Int64=2,Nitp::Int64=10,
    order_smooth_itp::Int64=1,order_nvc_itp::Int64=2,nvc0_limit::Int64=5,L1nvc_limit::Int64=3) where {T,N}

    for isp in 1:ns
        L1 = 1
        nvc[:,L1,isp] = nvcfind(nvc[:,L1,isp],dtf[:,L1,isp],fvL[:,L1,isp], nv, L1; 
            orders=orders, is_boundv0=is_boundv0, Nsmooth=Nsmooth, 
            order_smooth=order_smooth, abstol_Rdy=abstol_Rdy)
        if LM[isp] == 1
            L1 = 2
            if norm(fvL[:, L1,isp]) ≥ abs_dfLM
                nvc[:,L1,isp],dtf[:,L1,isp] = optimdtfvL0e!(nvc[:,L1,isp], dtf[:,L1,isp], 
                    fvL[:, L1,isp], vGe, nv, L1; orders=orders, is_boundv0=is_boundv0, nvcup=nvc[:,L1-1,isp],
                    Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy,k=k,Nitp=Nitp,
                    order_smooth_itp=order_smooth_itp,order_nvc_itp=order_nvc_itp,nvc0_limit=nvc0_limit)
            else
                dtf[:,L1,isp] .= 0.0
            end
        else
            for L1 in 2:LM[isp]+1
                nvc[:,L1,isp],dtf[:,L1,isp] = optimdtfvL0e!(nvc[:,L1,isp], dtf[:,L1,isp], 
                    fvL[:, L1,isp], vGe, nv, L1; orders=orders, is_boundv0=is_boundv0, nvcup=nvc[:,L1-1,isp],
                    Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy,k=k,Nitp=Nitp,
                    order_smooth_itp=order_smooth_itp,order_nvc_itp=order_nvc_itp,nvc0_limit=nvc0_limit)
            end
        end
    end
end

"""
  Inputs:

  Outputs:
    nvc = nvcfind(nvc,dtf,fvL,nvG,LM;orders=orders,is_boundv0=is_boundv0,
                  Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)
    optimdtfvL0e!(nvc,dtf,fvL,vGe,nvG,LM;orders=orders,is_boundv0=is_boundv0,
                  Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy,
                  k=k,Nitp=Nitp,order_smooth_itp=order_smooth_itp,order_nvc_itp=order_nvc_itp,
                  nvc0_limit=nvc0_limit,L1nvc_limit=L1nvc_limit)
"""

# 2D, [nvc]
function nvcfind(nvc::Array{Int64,N}, dtf::AbstractArray{T,N}, fvL::AbstractArray{T,N},
    nv::Int64, LM::Int64; orders::Int64=1, is_boundv0::Vector{Bool}=[true,false], 
    Nsmooth::Int=3, order_smooth::Int64=2, abstol_Rdy::AbstractVector{T}=[0.35,0.35]) where {T,N}

    L1 = 1
    nvc[:, L1] = nvcfind(nvc[:, L1], dtf[:, L1], fvL[:, L1], nv, L1; 
            orders=orders, is_boundv0=is_boundv0, Nsmooth=Nsmooth, 
            order_smooth=order_smooth, abstol_Rdy=abstol_Rdy)
    if LM == 1
        L1 = 2
        if norm(fvL[:, L1,isp]) ≥ abs_dfLM
            nvc[:, L1] = nvcfind(nvc[:, L1], dtf[:, L1], fvL[:, L1], nv, L1; 
                orders=orders, is_boundv0=is_boundv0, Nsmooth=Nsmooth, nvcup=nvc[:,L1-1],
                order_smooth=order_smooth, abstol_Rdy=abstol_Rdy)
        end
    else
        for L1 in 2:LM+1
            nvc[:, L1] = nvcfind(nvc[:, L1], dtf[:, L1], fvL[:, L1], nv, L1; 
                orders=orders, is_boundv0=is_boundv0, Nsmooth=Nsmooth, nvcup=nvc[:,L1-1],
                order_smooth=order_smooth, abstol_Rdy=abstol_Rdy)
        end
    end
    return nvc
end

# 2D, [nvc, δtf]
function optimdtfvL0e!(nvc::Array{Int64,N}, dtf::AbstractArray{T,N}, 
    fvL::AbstractArray{T,N},vGe::AbstractVector{T},nv::Int64, LM::Int64; 
    orders::Int64=1, is_boundv0::Vector{Bool}=[true,false],Nsmooth::Int=3, 
    order_smooth::Int64=2,abstol_Rdy::AbstractVector{T}=[0.35,0.35],k::Int64=2,Nitp::Int64=10,
    order_smooth_itp::Int64=1,order_nvc_itp::Int64=2,nvc0_limit::Int64=5,L1nvc_limit::Int64=3) where {T,N}

    L1 = 1
    if L1 ≥ L1nvc_limit
        nvc[:, L1], dtf[:, L1] = optimdtfvL0e!(nvc[:, L1], dtf[:, L1], 
            fvL[:, L1], vGe, nv, L1; orders=orders, is_boundv0=is_boundv0,
            Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy,k=k,Nitp=Nitp,
            order_smooth_itp=order_smooth_itp,order_nvc_itp=order_nvc_itp,nvc0_limit=nvc0_limit)
    else
        nvc[:, L1] = nvcfind(nvc[:, L1], dtf[:, L1], fvL[:, L1], nv, L1; 
            orders=orders, is_boundv0=is_boundv0,Nsmooth=Nsmooth,
            order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)
    end
    if LM == 1
        L1 = 2
        if norm(fvL[:, L1,isp]) ≥ abs_dfLM
            if L1 ≥ L1nvc_limit
                nvc[:, L1], dtf[:, L1] = optimdtfvL0e!(nvc[:, L1], dtf[:, L1], 
                    fvL[:, L1], vGe, nv, L1; orders=orders, is_boundv0=is_boundv0, nvcup=nvc[:,L1-1],
                    Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy,k=k,Nitp=Nitp,
                    order_smooth_itp=order_smooth_itp,order_nvc_itp=order_nvc_itp,nvc0_limit=nvc0_limit)
            else
                nvc[:, L1] = nvcfind(nvc[:, L1], dtf[:, L1], fvL[:, L1], nv, L1; 
                    orders=orders, is_boundv0=is_boundv0,Nsmooth=Nsmooth, nvcup=nvc[:,L1-1],
                    order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)
            end
        end
    else
        for L1 in 2:LM+1
            if L1 ≥ L1nvc_limit
                nvc[:, L1], dtf[:, L1] = optimdtfvL0e!(nvc[:, L1], dtf[:, L1], 
                    fvL[:, L1], vGe, nv, L1; orders=orders, is_boundv0=is_boundv0, nvcup=nvc[:,L1-1],
                    Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy,k=k,Nitp=Nitp,
                    order_smooth_itp=order_smooth_itp,order_nvc_itp=order_nvc_itp,nvc0_limit=nvc0_limit)
            else
                nvc[:, L1] = nvcfind(nvc[:, L1], dtf[:, L1], fvL[:, L1], nv, L1; 
                    orders=orders, is_boundv0=is_boundv0,Nsmooth=Nsmooth, nvcup=nvc[:,L1-1],
                    order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)
            end
        end
    end
end

"""
  Find the critic value of `nvc`.

  Inputs:

  Outputs:
    nvc = nvcfind(nvc,dtf,fLn,nvG,L1;nvcup=nvc,orders=orders,is_boundv0=is_boundv0,
                  Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)
    nvc, dtf = nvcfind!(nvc,dtf,fLn,vGe,nvG,L1;nvcup=nvc,orders=orders,is_boundv0=is_boundv0,
                  Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy,k=k_δtf,Nitp=Nitp,
                  order_smooth_itp=order_smooth_itp,order_nvc_itp=order_nvc_itp,nvc0_limit=nvc0_limit)
    
"""

# 1D, [nvc]
function nvcfind(nvc::Vector{Int64}, dtf::AbstractVector{T}, fLn::AbstractVector{T}, nv::Int64, 
    L1::Int64; nvcup::Vector{Int64}=nvc,orders::Int64=1, is_boundv0::Vector{Bool}=[true,false], 
    Nsmooth::Int=3, order_smooth::Int64=2,abstol_Rdy::AbstractVector{T}=[0.35,0.35]) where {T}

    # # Finding the critic value of `nvcy00`
    # k = 0
    # for k in 1:nv
    #     if abs(dtf[k]) > abs_dtfLM
    #         nvc[1] = k - 1
    #         break
    #     end
    # end
 
    # # Finding the critic value of `nvcy0`.
    nvcy0 = nvc0find(fLn, nv)
    nvcy0 == 0 ? nvc[2] = nv : nvc[2] = nvcy0

    # # Finding the critic value of `nvcd1`.            order = 1
    y = dtf ./ fLn
    if fLn[1] == 0.0
        y[1] = 2y[2] - y[3]
    end
    Rd1y = zero.(y)
    Rd1y = RdpdtfvL0CDS(Rd1y, y, nv; orders=orders, is_boundv0=is_boundv0[1])
    nvcd1 = nvcfind(Rd1y; Nsmooth=Nsmooth, abstol_Rdy=abstol_Rdy[1])
    nvc[3] = nvcd1

    # # Finding the critic value of `nvcy1`
    nvcy1 = deepcopy(nvcd1)
    nvcy1 = nvc0find(dtf[nvcy1:nv], nv - nvcy1 + 1)
    # nvcy1 = nvc0find(y[nvcy1:nv], nv - nvcy1 + 1)
    if nvcy1 == 0
        nvc[4] = nv
    else
        nvc[4] = nvcd1 + (nvcy1 - 1)
    end
    if L1 ≥ 9
        i = 2
        nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
        i = 3
        nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
        i = 4
        nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
    end

    if order_smooth ≥ 2
        # # Finding the critic value of `nvcd2`.            order = 2
        if nvcd1 ≤ 3
            Rd2y = zeros(T, nv)
            Rd2y = RdpdtfvL0CDS(Rd2y, Rd1y, nv; orders=orders, is_boundv0=is_boundv0[2])
            nvcd2 = nvcfind(Rd2y; Nsmooth=Nsmooth, abstol_Rdy=abstol_Rdy[2])
            nvc[5] = nvcd2
        elseif nv - nvcd1 ≤ 5
            nvc[5] = nv
        else
            Rd2y = zeros(T, nv)
            Rd2y = RdpdtfvL0CDS(Rd2y, Rd1y, nv; orders=orders, is_boundv0=is_boundv0[2])
            nvcd2 = nvcfind(Rd2y, nvcd1; Nsmooth=Nsmooth, abstol_Rdy=abstol_Rdy[2])
            nvc[5] = nvcd2
        end

        # # Finding the critic value of `nvcy2`
        nvcy2 = deepcopy(nvcd2)
        nvcy2 = nvc0find(Rd1y[nvcy2:nv], nv - nvcy2 + 1)
        if nvcy2 == 0
            nvc[6] = nv
        else
            nvc[6] = nvcd2 + (nvcy2 - 1)
        end
        if L1 ≥ 9
            i = 5
            nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
            i = 6
            nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
        end

        if order_smooth ≥ 3
            # # Finding the critic value of `nvcd3`.            order = 3
            if nvcd2 ≤ 3
                Rd3y = zeros(T, nv)
                Rd3y = RdpdtfvL0CDS(Rd3y, Rd2y, nv; orders=orders, is_boundv0=is_boundv0[3])
                nvcd3 = nvcfind(Rd3y; Nsmooth=Nsmooth, abstol_Rdy=abstol_Rdy[3])
                nvc[7] = nvcd3
            elseif nv - nvcd2 ≤ 5
                nvc[7] = nv
            else
                Rd3y = zeros(T, nv)
                Rd3y = RdpdtfvL0CDS(Rd3y, Rd2y, nv; orders=orders, is_boundv0=is_boundv0[3])
                nvcd3 = nvcfind(Rd3y, nvcd2; Nsmooth=Nsmooth, abstol_Rdy=abstol_Rdy[3])
                nvc[7] = nvcd3
            end

            # # Finding the critic value of `nvcy3`
            nvcy3 = deepcopy(nvcd3)
            nvcy3 = nvc0find(Rd2y[nvcy3:nv], nv - nvcy3 + 1)
            if nvcy3 == 0
                nvc[8] = nv
            else
                nvc[8] = nvcd3 + (nvcy3 - 1)
            end
            if L1 ≥ 9
                i = 7
                nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
                i = 8
                nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
            end
        end
    end

    return nvc
end

# 1D, [nvc, dtf]
function optimdtfvL0e!(nvc::Vector{Int64}, dtf::AbstractVector{T}, 
    fLn::AbstractVector{T}, vGe::AbstractVector{T}, nv::Int64, L1::Int64; nvcup::Vector{Int64}=nvc,
    orders::Int64=1, is_boundv0::Vector{Bool}=[true,false], Nsmooth::Int=3, 
    order_smooth::Int64=2,abstol_Rdy::AbstractVector{T}=[0.35,0.35],k::Int64=2,Nitp::Int64=10,
    order_smooth_itp::Int64=1,order_nvc_itp::Int64=2,nvc0_limit::Int64=5) where {T}
    
    nvc0_limit ≥ 3 || throw(ArgumentError("`nvc0_limit` is smaller than `3` which is not proposed."))
    # # Finding the critic value of `nvcy00`
    # k = 0
    # for k in 1:nv
    #     if abs(dtf[k]) > abs_dtfLM
    #         nvc[1] = k - 1
    #         break
    #     end
    # end
 
    # # Finding the critic value of `nvcy0`.
    nvcy0 = nvc0find(fLn, nv)
    nvcy0 == 0 ? nvc[2] = nv : nvc[2] = nvcy0

    # # Finding the critic value of `nvcd1`.            order = 1
    y = dtf ./ fLn
    if fLn[1] == 0.0
        y[1] = 2y[2] - y[3]
    end
    Rd1y = zero.(y)
    Rd1y = RdpdtfvL0CDS(Rd1y, y, nv; orders=orders, is_boundv0=is_boundv0[1])
    nvcd1 = nvcfind(Rd1y; Nsmooth=Nsmooth, abstol_Rdy=abstol_Rdy[1])
    nvc[3] = nvcd1

    # # Finding the critic value of `nvcy1`
    nvcy1 = deepcopy(nvcd1)
    nvcy1 = nvc0find(dtf[nvcy1:nv], nv - nvcy1 + 1)
    # nvcy1 = nvc0find(y[nvcy1:nv], nv - nvcy1 + 1)
    if nvcy1 == 0
        nvc[4] = nv
    else
        nvc[4] = nvcd1 + (nvcy1 - 1)
    end
    if L1 ≥ 9
        i = 2
        nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
        i = 3
        nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
        i = 4
        nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
    end

    if order_smooth ≥ 2
        # # Finding the critic value of `nvcd2`.            order = 2
        if nvcd1 ≤ 3
            Rd2y = zeros(T, nv)
            Rd2y = RdpdtfvL0CDS(Rd2y, Rd1y, nv; orders=orders, is_boundv0=is_boundv0[2])
            nvcd2 = nvcfind(Rd2y; Nsmooth=Nsmooth, abstol_Rdy=abstol_Rdy[2])
            nvc[5] = nvcd2
        elseif nv - nvcd1 ≤ 5
            nvc[5] = nv
        else
            Rd2y = zeros(T, nv)
            Rd2y = RdpdtfvL0CDS(Rd2y, Rd1y, nv; orders=orders, is_boundv0=is_boundv0[2])
            nvcd2 = nvcfind(Rd2y, nvcd1; Nsmooth=Nsmooth, abstol_Rdy=abstol_Rdy[2])
            nvc[5] = nvcd2
        end

        # # Finding the critic value of `nvcy2`
        nvcy2 = deepcopy(nvcd2)
        nvcy2 = nvc0find(Rd1y[nvcy2:nv], nv - nvcy2 + 1)
        if nvcy2 == 0
            nvc[6] = nv
        else
            nvc[6] = nvcd2 + (nvcy2 - 1)
        end
        if L1 ≥ 9
            i = 5
            nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
            i = 6
            nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
        end

        if order_smooth ≥ 3
            # # Finding the critic value of `nvcd3`.            order = 3
            if nvcd2 ≤ 3
                Rd3y = zeros(T, nv)
                Rd3y = RdpdtfvL0CDS(Rd3y, Rd2y, nv; orders=orders, is_boundv0=is_boundv0[3])
                nvcd3 = nvcfind(Rd3y; Nsmooth=Nsmooth, abstol_Rdy=abstol_Rdy[3])
                nvc[7] = nvcd3
            elseif nv - nvcd2 ≤ 5
                nvc[7] = nv
            else
                Rd3y = zeros(T, nv)
                Rd3y = RdpdtfvL0CDS(Rd3y, Rd2y, nv; orders=orders, is_boundv0=is_boundv0[3])
                nvcd3 = nvcfind(Rd3y, nvcd2; Nsmooth=Nsmooth, abstol_Rdy=abstol_Rdy[3])
                nvc[7] = nvcd3
            end

            # # Finding the critic value of `nvcy3`
            nvcy3 = copy(nvcd3)
            nvcy3 = nvc0find(Rd2y[nvcy3:nv], nv - nvcy3 + 1)
            if nvcy3 == 0
                nvc[8] = nv
            else
                nvc[8] = nvcd3 + (nvcy3 - 1)
            end
            if L1 ≥ 9
                i = 7
                nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
                i = 8
                nvc[i] < nvcup[i] ? nvc[i] = nvcup[i] : nothing
            end
        end
    end

    # Interpolations on grids `vG[1:nvci-1]`
    if order_smooth_itp == 0
        if order_nvc_itp > 3
            nvc0 = max(nvc[5],nvc[7])
        else
            nvc0 = nvc[1+2 * order_nvc_itp]
        end
        @show L1, nvc0, nvc0_limit
        if nvc0 ≥ nvc0_limit
            if nvc[2] ≥ nvc0 + Nitp
                if nv - nvc0 > Nitp
                    y[1:nvc0-1] = RdtfvL0interp(y[nvc0:nvc0+Nitp],vGe[nvc0:nvc0+Nitp],vGe[1:nvc0-1];k=k)
                else
                    error("`y` should be zeros according to the smoothness of the function. Checking it carefully!!!")
                end
            else
              @warn("`nvc0 + Nitp ≥ nvcy0` may cause the interpolations of `y` to be unstable, or even to be falured. L = ", L1-1)
              @show (L1,nv), nvc
              fghjfgh
            end
            if abs(y[1]) ≥ 1.0
                @warn("`|RdtfLn[1]| ≥ 1` may increase the relative error of `dtfLn` when",(L1-1,y[1]))
            end
        end
        dtf = y .* fLn
    else order_smooth_itp == 1
        if order_nvc_itp > 3
            nvc0 = max(nvc[5],nvc[7])
        else
            nvc0 = nvc[1+2 * order_nvc_itp]
        end
        if nvc0 ≥ nvc0_limit
            if nvc[4] ≥ nvc0 + Nitp
                if nv - nvc0 ≥ Nitp
                    Rd1y[2:nvc0-1] = RdtfvL0interp([0;Rd1y[nvc0:nvc0+Nitp]],[0;vGe[nvc0:nvc0+Nitp]],vGe[2:nvc0-1];k=k)
                else
                    @warn("`Rd1y` is not suitable when using the standard interpolations. Accept a compromised results!!!")
                    Nitp2 = nv - nvc0
                    if Nitp2 ≥ 5
                        Rd1y[2:nvc0-1] = RdtfvL0interp([0;Rd1y[nvc0:nvc0+Nitp2]],[0;vGe[nvc0:nvc0+Nitp2]],vGe[2:nvc0-1];k=k)
                    elseif order_nvc_itp > 3
                        nvc0 = round(Int64,(nvc[5]+nvc[7])/2)
                        Nitp2 = nv - nvc0
                        if Nitp2 ≥ 10
                            Rd1y[2:nvc0-1] = RdtfvL0interp([0;Rd1y[nvc0:nvc0+Nitp]],[0;vGe[nvc0:nvc0+Nitp]],vGe[2:nvc0-1];k=k)
                        else
                            if Nitp2 ≥ 5
                                Rd1y[2:nvc0-1] = RdtfvL0interp([0;Rd1y[nvc0:nvc0+Nitp2]],[0;vGe[nvc0:nvc0+Nitp2]],vGe[2:nvc0-1];k=k)
                            else
                                error("`Rd1y` should be zeros according to the smoothness of the function. Checking it carefully!!!")
                            end
                        end
                    else
                        gtfhjk
                        if order_nvc_itp == 1
                        elseif order_nvc_itp == 2
                        else
                        end
                        nvc0 = nvc[1+2 * order_nvc_itp]
                    end
                end
            else
              @warn("`nvc0 + Nitp ≥ nvcy0` may cause the interpolations of `Rd1y` to be unstable, or even to be falured. L = ", L1-1)
              @show (L1,nv),(nvc0,Nitp,nv), nvc
            #   fghjfghykujiik
            end
            if abs(Rd1y[2]) ≥ 0.5
                @warn("`|Rd1y[2]| ≥ 0.5` may increase the relative error of `y=RdtfLn` when",(L1-1,y[1]))
            end
        end
        for i in nvc0:-1:2
            y[i-1] = y[i+1] - 2Rd1y[i] * y[i]
        end
        dtf = y .* fLn
    end
    return nvc, dtf
end

"""
  Find the critic value of `nvcd1`, `nvcd2` and so on.

  Inputs:
    Nsmooth: (=3, default), number of points to smooth the function `dtfvL`.

  Outputs:
    nvcd1 = nvcfind(Rdy, nvcdi;Nsmooth=Nsmooth,abstol_Rdy=abstol_Rdy)
    nvcd1 = nvcfind(Rdy;Nsmooth=Nsmooth,abstol_Rdy=abstol_Rdy)
"""

# 1D `nvcdi`, which could be optimized to improve performance
function nvcfind(Rdy::AbstractVector{T}, nvcdi::Int64; Nsmooth::Int=3, abstol_Rdy::T=0.5) where {T}

    Vsign = Int.(sign.(Rdy[1:Nsmooth]))
    issame = isvecsame(Vsign, Nsmooth)             # Bool
    Vcons = abs.(Rdy[1:Nsmooth]) .≤ abstol_Rdy
    # sum(Vcons) / Nsmooth == 1 ? iscons = true : iscons = false
    if issame === true && sum(Vcons) == Nsmooth
        iscons = true
    else
        iscons = false
    end
    if iscons === true
        # nvcdi2 = nvcdi
        return nvcdi
    else
        nvcdi2 = 1
        if Nsmooth === 2
            for k in Nsmooth+1:1:length(Rdy)
                Vsign[1] = Vsign[2]
                Vsign[2] = Int(sign(Rdy[k]))
                issame = isvecsame(Vsign, Nsmooth)             # Bool
                Vcons[1] = Vcons[2]
                Vcons[2] = abs(Rdy[k]) ≤ abstol_Rdy
                if issame === true && sum(Vcons) == Nsmooth
                    iscons = true
                else
                    iscons = false
                end
                if iscons === true
                    nvcdi2 = k - Nsmooth + 1
                    # break
                    if nvcdi2 ≥ nvcdi
                        break
                    end
                end
            end
        else
            N1 = Nsmooth - 1
            vec0 = 1:N1
            vec = 2:Nsmooth
            for k in Nsmooth+1:1:length(Rdy)
                Vsign[vec0] = Vsign[vec]
                if isnan(Rdy[k])
                    if k ≤ 6
                        @show k, Rdy[k-1], Rdy[k], Rdy[k+1]
                    end
                    Rdy[k] = (Rdy[k+1] + Rdy[k-1]) / 2
                    Vsign[Nsmooth] = Int(sign(Rdy[k]))
                else
                    Vsign[Nsmooth] = Int(sign(Rdy[k]))
                end
                issame = isvecsame(Vsign, Nsmooth)             # Bool
                Vcons[vec0] = Vcons[vec]
                Vcons[Nsmooth] = abs(Rdy[k]) ≤ abstol_Rdy
                if issame === true && sum(Vcons) == Nsmooth
                    iscons = true
                else
                    iscons = false
                end
                if iscons === true
                    nvcdi2 = k - Nsmooth + 1
                    #   break
                    if nvcdi2 ≥ nvcdi
                        break
                    end
                end
            end
        end
        return nvcdi2
    end
end

# 1D `nvcdi`, when `nvcdi = 1`
function nvcfind(Rdy::AbstractVector{T}; Nsmooth::Int64=3, abstol_Rdy::T=0.5) where {T}

    Vsign = Int.(sign.(Rdy[1:Nsmooth]))
    @show Rdy[1:3Nsmooth]
    issame = isvecsame(Vsign, Nsmooth)             # Bool
    Vcons = abs.(Rdy[1:Nsmooth]) .≤ abstol_Rdy
    # sum(Vcons) / Nsmooth == 1 ? iscons = true : iscons = false
    if issame === true && sum(Vcons) == Nsmooth
        iscons = true
    else
        iscons = false
    end
    nvcd1 = 1
    if iscons === true
        return nvcd1
    else
        if Nsmooth === 2
            for k in Nsmooth+1:1:length(Rdy)
                Vsign[1] = Vsign[2]
                Vsign[2] = Int(sign(Rdy[k]))
                issame = isvecsame(Vsign, Nsmooth)             # Bool
                Vcons[1] = Vcons[2]
                Vcons[2] = abs(Rdy[k]) ≤ abstol_Rdy
                if issame === true && sum(Vcons) == Nsmooth
                    iscons = true
                else
                    iscons = false
                end
                if iscons === true
                    nvcd1 = k - Nsmooth + 1
                    break
                end
            end
        else
            N1 = Nsmooth - 1
            vec0 = 1:N1
            vec = 2:Nsmooth
            for k in Nsmooth+1:1:length(Rdy)
                Vsign[vec0] = Vsign[vec]
                Vsign[Nsmooth] = Int(sign(Rdy[k]))
                issame = isvecsame(Vsign, Nsmooth)             # Bool
                Vcons[vec0] = Vcons[vec]
                Vcons[Nsmooth] = abs(Rdy[k]) ≤ abstol_Rdy
                if issame === true && sum(Vcons) == Nsmooth
                    iscons = true
                else
                    iscons = false
                end
                if iscons === true
                    nvcd1 = k - Nsmooth + 1
                    break
                end
            end
        end
        return nvcd1
    end
end

"""
  Find the critic value of `nvcy0`

  Inputs:
  Outputs:
    nvcy0 = nvc0find(y,nv)
"""
# `nvcy0` and `nvcy1`
function nvc0find(y::AbstractVector{T}, nv::Int64) where {T}

    nvcy0 = 0
    for k in 2:nv-1
        if y[k] * y[k+1] < 0
            nvcy0 = k
            break
        end
    end
    return nvcy0
end

function isvecsame(vec::AbstractVector{T}, Nv::Int) where {T}

    1vec[1] === vec[2] ? ans = true : ans = false
    if ans === false
        return ans
    else
        for k in 3:Nv
            if vec[k] ≠ vec[1]
                ans = false
                break
            end
        end
        return ans
    end
end

"""
  Outouts:
    nvccheck!(nvc3a,LM33,order_smooth)
    nvc = nvccheck(nvc,LM,order_smooth)
    nvc = nvccheck(nvc,LM)
"""

function nvccheck!(nvc::Array{Int64},LM::Int64,order_smooth::Int64=2)

    for i in 3:2+2order_smooth
        nvc[i,:] = nvccheck(nvc[i,:],LM)
    end
end

function nvccheck(nvc::Array{Int64},LM::Int64,order_smooth::Int64=2)

    for i in 3:2+2order_smooth
        nvc[i,:] = nvccheck(nvc[i,:],LM)
    end
    return nvc
end

function nvccheck(nvc::Vector{Int64},LM::Int64)

    if LM ≥ 9
        for k in 8:LM
            nvc[k+1] < nvc[k] ? nvc[k+1] = nvc[k] : nothing
            # nvc[k+1] < nvc[k] ? nvc[k+1] = floor(Int64,(nvc[k]+nvc[k+2])/2) : nothing
        end
        # k = LM
        # nvc[k+1] < nvc[k] ? nvc[k+1] = nvc[k] : nothing
    end
    return nvc
end