
"""
  Refinement scheme: Global-Local-Level strategy (fixed shape function for multi-level, a variable-level strategy which is usually lager than 2)

  Subdivision surface refinement schemes can be broadly classified into two categories: interpolating and approximating.

    ⚈ Interpolating schemes are required to match the original position of vertices in the original mesh.
    ⚈ Approximating schemes are not; they can and will adjust these positions as needed.

  In general, approximating schemes have greater smoothness, but the user has less overall control of the outcome.

   Each iteration is often called a subdivision level, starting at zero (before any refinement occurs).

"""

"""
  Type for results returned by the self-adpative algorithm when
      uplimiting of the shape function is used in vchebyadptive procedure, the attributes are:

  - .nvlevel2: the number of the subdivision by refinement of the `kᵗʰ` grid in level-1.
              default = [1, 1, ⋯] with length `nc0`.
  - .nvlevel2i:  [nci0,⋯],(where i ∈ [3,4,5] default), which is the number of initial grid of the `iᵗʰ` grid .
              default = nvlevel, the initial number vector of level-0.
  - .nvlevel: [[],[],⋯], the collection of grid number of the second-level subdivision (including all grids in the subdivision).
  - .vlevel:  [[[],[],⋯], [[],[],⋯], ⋯]
"""

mutable struct subdivs{T <: Int}

  nvlevel2  :: Vector{T}
  nvlevel2i :: Vector{T}
  nvlevel   :: Vector{Any}   # Vector{Vector{Int}}
  vlevel    :: Vector{Any}   # Vector{Vector{Vector{T}}}
end

"""

  Outputs:
    data = subdivsinitial(nc0,  nvlevel1, vlevel1)
"""

function subdivsinitial(nc0::Int,nvlevel1::Vector{Int},vlevel1::AbstractVector{Any}) where {T <: Real}

    nvlevel2 = ones(Int,nc0-1)       # The number of subdivision in the second-level subdivision on [v[i], v[i+1]]
    nvlevel2i = nvlevel1             # The total number of grids in the first-two-level subdivision on [v[i], v[i+1]]
    nvlevel = Vector((undef),nc0-1)  # The number vector in first-level subdivision on different second-level subdivision.
    vlevel = Vector((undef),nc0-1)   # The grid vector
    for i in 1:nc0 - 1
        nvlevel[i] = [nvlevel1[i]]
        vlevel[i] = [vlevel1[i]]
    end
    subdivs(nvlevel2, nvlevel2i, nvlevel, vlevel)
end

"""
  1, Double-conservation limits: lower-order bounds and higher-order bounds `[j_low, j_high]`;
  2, Double-shape limits, default is `[shape_low,shape_high] = [0,0]` which means no refinement for Chebyshev grids.
           `shape_low` gives the lowest number of each initial interval `[nG[i],nG[i+1]]` and
           `shape_high` gives the highest number of the initial interval.

  inputs:
    j: [j_lower,j_higher], default is [0,10]
    vadaptlevels：The maximum level of adaptation for refinement grids: `nGki_re: 2^(vadaptlevels)+1`

  Outputs:
    nck, nvlevel, vlevel = vchebyHadaptive(fLnshape,vGdom,Msj,j=orderVconst;nc0=nc0,ocp=ocp,abstol=abstol,reltol=reltol,vadaptlevels=vadaptlevels)

"""

# (2,1) Single-shape limits (upper) and double conservation of order: `[j_lower,j_upper]`. level = 1
function vchebyHadaptive(fLnshape::Function,vGdom::Vector{Tb},Msj::Vector{T},j::Vector{Int}=[0,10];
    nc0::Int=15,ocp::Int=3,abstol::Float64=epsT5,reltol::Float64=1e-5,vadaptlevels::Int=6) where{T<:Real,Tb}

    if j[1] ≠ j[2]
        ~, nvlevelj1, vlevelj1 = vchebyHadaptive(fLnshape,vGdom,Msj[1],j[1];
                      nc0=nc0,ocp=ocp,abstol=abstol,reltol=reltol,vadaptlevels=vadaptlevels)
        nck1, nvlevel1, vlevel1 = vchebyHadaptive(fLnshape,vGdom,Msj[2],j[2];
                      nc0=nc0,ocp=ocp,abstol=abstol,reltol=reltol,vadaptlevels=vadaptlevels)
        for k in 1:nc0 - 1
            if nvlevel1[k] < nvlevelj1[k]
                vlevel1[k] = vlevelj1[k]
                nck1 += - nvlevel1[k] + nvlevelj1[k]
                nvlevel1[k] = nvlevelj1[k]
            end
        end
        return nck1, nvlevel1, vlevel1
    else
        # nck, nvlevel, vlevel = vchebyHadaptive(fLnshape,vGdom,Msj,j[1];
        #               nc0=nc0,ocp=ocp,abstol=abstol,reltol=reltol,vadaptlevels=vadaptlevels)
        return vchebyHadaptive(fLnshape,vGdom,Msj[1],j[1];nc0=nc0,ocp=ocp,abstol=abstol,reltol=reltol,vadaptlevels=vadaptlevels)
    end
end

"""

  inputs:
    nc0: the initial number of Chebyshev grids
    ocp: the number of the shape function for the last level
    Msj: the factor for normalization of the `jᵗʰ`-order
    vGdom: the interval for refining
    reltol: default = 1e-5, the relative tolerance when  `abstol < epsT5`
    vadaptlevels：The maximum level of adaptation for refinement grids: `nGki_re: 2^(vadaptlevels)+1`

  Outputs:
    nck, nvlevel, vlevel = vchebyHadaptive(fLnshape,vGdom,Msj,j=orderVconst;
                  nc0=nc0,ocp=ocp,abstol=abstol,reltol=reltol,vadaptlevels=vadaptlevels)

"""

# (1,1) Single-shape limits (upper) and single conservation of order: `j`. level = 1

function vchebyHadaptive(fLnshape::Function,vGdom::Vector{Tb},Msj::T,j::Int=2;
    nc0::Int=15,ocp::Int=3,abstol::Float64=epsT5,reltol::Float64=1e-5,vadaptlevels::Int=6) where{T<:Real,Tb}

    vlevel = Vector((undef),nc0-1)
    v = vCmapping(vccn(nc0;datatype=Tb),vGdom[1],vGdom[2];isinv=true)
    # Whether refining is done according to `j` and `orderVconstllimit` based on the integrals
    if vadaptlevels == 0 || j < orderVconstllimit
        for k in 1:nc0 - 1
            vlevel[k] = [v[k], v[k+1]]
        end
        return nc0, 2ones(Int,nc0-1), vlevel
    else
        nvlevel = zeros(Int,nc0-1)
        isconverged = 0
        for k in 1:nc0 - 1
            # isconverged = 0                        # Level 1, assuming the leading level is not converged.
            isconverged, vk = evalruleH(fLnshape,[v[k],v[k+1]],Msj,j,0;ocp=ocp,abstol=abstol,reltol=reltol)
            # the new `isconverged` is for all the subdivision on grids `vk`
            if vadaptlevels == 1 || isconverged == -1
                nv = length(vk)
                # @show (k,nv),ocp,isconverged
                nvlevel[k] = length(vk)
                vlevel[k] = copy(vk)
            else
                nv = length(vk)
                # Level 2: = 2^(level - 1) + 1
                isconverged == 1 ? isconvergedk = ones(Int,nv-1) : isconvergedk = zeros(Int,nv-1)
                # @show k, isconvergedk
                if vadaptlevels == 2
                    ~, vk = evalruleH(fLnshape,vk,nv,Msj,j,isconvergedk;ocp=ocp,abstol=abstol,reltol=reltol)
                    nvlevel[k] = length(vk)
                    vlevel[k] = copy(vk)
                    # Nlevel = 2
                    # @show k, Nlevel, nv, isconvergedk1
                    # whyth
                # elseif vadaptlevels == 3
                #     isconvergedk, vk = evalruleH(fLnshape,vk,nv,Msj,j,isconvergedk;ocp=ocp,abstol=abstol,reltol=reltol)
                #     # Nlevel = 2
                #     # @show k, Nlevel, nv, isconvergedk
                #     nv = length(vk)
                #     # Level 3: = 2^(level - 1) + 1
                #     if prod(isconvergedk) == 0
                #         ~, vk = evalruleH(fLnshape,vk,nv,Msj,j,isconvergedk;ocp=ocp,abstol=abstol,reltol=reltol)
                #     end
                #     nvlevel[k] = length(vk)
                #     vlevel[k] = copy(vk)
                else
                    Nlevel = 2
                    while prod(isconvergedk) == 0
                        if Nlevel ≥ vadaptlevels
                            @warn ("Maxumum number of level for refinement is reached before converging! k=",k)
                            break
                        end
                        isconvergedk, vk = evalruleH(fLnshape,vk,nv,Msj,j,isconvergedk;ocp=ocp,abstol=abstol,reltol=reltol)
                        nv = length(vk)
                        nv1 = length(isconvergedk)
                        # @show k, Nlevel, nv,nv1,isconvergedk
                        Nlevel += 1
                    end
                    isconvergedk, vk = evalruleH(fLnshape,vk,nv,Msj,j,isconvergedk;ocp=ocp,abstol=abstol,reltol=reltol)
                    nv1 = length(isconvergedk)
                    # @show k, Nlevel, nv,nv1,isconvergedk
                    nvlevel[k] = length(vk)
                    vlevel[k] = copy(vk)
                end
            end
        end
        nck = sum(nvlevel) - nc0 + 2
        return nck, nvlevel, vlevel
    end
end

"""

  Inputs:
    vkdom:
    nv: = length(vkdom)
    isconverged: default = [-1,0,1], denotes three states [-1, false, true] for whether refinement is converged.
                 `-1` means converged for no more refinement;
                 ` 1` means true but one more refinement is needed in current procedure for data stuctures and slowing-variable grids;
                 ' 0' denotes false where refinement is needed.
    ocp: the number of the shape function for the last level
    Msj: the factor for normalization of the `jᵗʰ`-order
    v2dom: = [v[k],v[k+1]], the interval for refining
    reltol: default = 1e-5, the relative tolerance when  `abstol < epsT5`

  Outputs:
    isconverged, vk = evalruleH(fLnshape,vkdom,nv,Msj,j,isconverged;ocp=ocp,abstol=abstol,reltol=reltol)
"""

# for `H-adaptive` refinement or `1/4-adaptive` for subdivision [vk,vk1]
function evalruleH(fLnshape::Function,vkdom::Vector{T},nv::Int,Msj::T,j::Int,
    isconverged::Vector{Int};ocp::Int=3,abstol::Float64=epsT5,reltol::Float64=1e-5) where{T<:Real}

    vk = Vector((undef),nv-1)
    nvlevel = Vector((undef),nv-1)
    isconvergedk = Vector((undef),nv-1)
    for k in 1:nv-1
        if isconverged[k] == 1          # no refinement, only the endpoints is needed
            vk[k] = vkdom[k:k+1]
            nvlevel[k] = 2
            isconvergedk[k] = ones(Int,1)
        else
            isconverged[k], vk[k] = evalruleH(fLnshape,vkdom[k:k+1],Msj,j,isconverged[k];ocp=ocp,abstol=abstol,reltol=reltol)
            nvlevel[k] = ocp
            if isconverged[k] == 1
                isconvergedk[k] = ones(Int,ocp-1)
            else
                isconvergedk[k] = zeros(Int,ocp-1)
            end
            # @show k, ocp, isconverged[k]
        end
    end
    nv == 2 ? nck = sum(nvlevel) : nck = sum(nvlevel) - nv + 2
    vkdom = zeros(T,nck)
    isconverged = zeros(Int,nck-1)
    s, k = 1, 1
    sk = nvlevel[k]
    # @show k,s,sk, (nv,nck,length(nvlevel)),isconvergedk[k]
    vkdom[s:sk] = vk[k]
    isconverged[s:sk-1] = isconvergedk[k]
    s = sk + 1
    for k in 2:nv-1
        sk += (nvlevel[k] - 1)
        # @show k,s,sk,isconvergedk[k]
        vkdom[s:sk] = vk[k][2:end]
        isconverged[s-1:sk-1] = isconvergedk[k]
        s = sk + 1
    end
    return isconverged, vkdom
end

"""

  Outputs:
    isconverged, vk = evalruleH(fLnshape,v2dom,Msj,j,isconverged;ocp=ocp,abstol=abstol,reltol=reltol)

"""

function evalruleH(fLnshape::Function,v2dom::Vector{Tb},Msj::T,j::Int=2,
    isconverged::Int=0;ocp::Int=3,abstol::Float64=epsT5,reltol::Float64=1e-5) where{T<:Real,Tb}

    nc1 = 2 * ocp - 1  # the higher order for error estimation
    cMsi = - 2/sqrtpi * (v2dom[1] - v2dom[2]) / Msj
    v = vCmapping(vccn(nc1;datatype=Tb),v2dom[1],v2dom[2];isinv=true)
    μ = μccn(nc1)
    # dI_k
    vk = v[1:2:end]
    # wcck = clenshawcurtisweights(μ[1:ocp])
    dMsk = cMsi * dot(clenshawcurtisweights(μ[1:ocp]), vk.^j .* fLnshape(vk))
    # dI_k1
    # wcck = clenshawcurtisweights(μ)
    dMs2k = cMsi * dot(clenshawcurtisweights(μ), v.^j .* fLnshape(v))
    δdMsk = abs(dMs2k - dMsk)
    if δdMsk < abstol
        if dMsk == 0.0 && dMs2k == 0.0
            isconverged = 1
            RδdMsk = 0.0
        else
            RδdMsk = 2δdMsk / (dMsk + dMs2k)
            if RδdMsk < reltol
                isconverged = 1
            else
                if δdMsk < 1e-20  && RδdMsk < 1e-2
                    isconverged = 1
                    # @warn ("1, The relative tolerance is bigger than `reltol`! (dM,δdM,RδdM,vk[1])=",fmtf2.([dMsk,δdMsk,RδdMsk,v2dom[1]]))
                else
                    # isconverged = 1
                    isconverged = 0
                    @warn ("2, The relative tolerance is bigger than `reltol`! (dM,δdM,RδdM,vk[1])=",fmtf2.([dMsk,δdMsk,RδdMsk,v2dom[1]]))
                end
            end
        end
    else
        # RδdMsk = 2δdMsk / (dMsk + dMs2k)
        isconverged = 0
    end
    # @show fmtf2.([dMsk,δdMsk,RδdMsk,v2dom[1]])
    return isconverged, vk
end

"""
  `vlevel → vG` when `isinv = false`, or else when `isinv = false`.

  Outputs:
    vGk = vGvlevel1(vGk,nvlevel,vlevel;isinv=false)
    vlevel = vGvlevel1(vGk,nvlevel,vlevel;isinv=true)
"""

function vGvlevel1(vGk::AbstractVector{T},nvlevel::AbstractVector{Int},vlevel::AbstractVector{Any};isinv::Bool=false) where {T<:Real}

    if isinv == 0
        vGk[1] = vlevel[1][1]
        vGk[end] = vlevel[end][end]
        nvGi = 2
        i = 0
        for k in nvlevel
            i += 1
            vGk[nvGi:nvGi+k-2] = vlevel[i][2:end]
            nvGi += k - 1
        end
        return vGk
    else
        vlevel[1][1] = vGk[1]
        vlevel[end][end] = vGk[end]
        nvGi = 2
        i = 0
        for k in nvlevel
            i += 1
            vlevel[i][2:end] = vGk[nvGi:nvGi+k-2]
            nvGi += k - 1
        end
        return vlevel
    end
end

"""
  nvlevel0 = nvlevel0invGk(nvlevel0,vG0,vlevel,nc0,nck)
"""

function nvlevel0invGk(nvlevel0,vG0,vlevel,nc0,nck)

    if nc0 == nck
        return 1:nck |> Vector{Int}
    else
        i = 1
        for j in 1:nck
            if vlevel[j] == vG0[i]
                nvlevel0[i] = j
                i += 1
            end
        end
        prod(nvlevel0 .≠ 0) == 0 ? @error("position vector of `vG0` in `vGk` is falure !!!") : 1
        return nvlevel0
    end
end


"""
  Advanced mesh refinement is a method of adapting the accuracy of a solution
  within certain sensitive or turbulent regions of simulation,
  dynamically and during the time the solution is being calculated.

  Adaptive mesh refinement can be introduced via functionals:

    Winslow function

    Modified Liao function

   The advantages of a dynamic gridding scheme:

     1. Increased computational saving and storage savings over a static grid approach.
     2. Complete control of grid resolution, compared to the fixed resolution of a static grid approach
     3. Compared to pre-tuned static meshes, the adaptive approach requires
         less detailed a priori knowledge on the evolution of the solution.
     4. The computational costs inherit properties of the physical system
"""
