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

"""
  Refinement scheme:

  Subdivision surface refinement schemes can be broadly classified into two categories: interpolating and approximating.

    ⚈ Interpolating schemes are required to match the original position of vertices in the original mesh.
    ⚈ Approximating schemes are not; they can and will adjust these positions as needed.

  In general, approximating schemes have greater smoothness, but the user has less overall control of the outcome.

   Each iteration is often called a subdivision level, starting at zero (before any refinement occurs).

"""

"""
  Type for results returned by the self-adpative algorithm when
      uplimiting of the shape function is used in vchebyadptive procedure, the attributes are:

  - issub:      [0,1,⋯], where the `iᵗʰ` grid [v[i], v[i+1] for i in 1:nc0-1] is need to be adptively updated.
  - nvlevel2: the number of the subdivision by refinement of the `kᵗʰ` grid in level-1.
              default = [1, 1, ⋯] with length `nc0`.
  - nvlevel2i:  [nci0,⋯],(where i ∈ [3,4,5] default), which is the number of initial grid of the `iᵗʰ` grid .
  - nvlevel: [[],[],⋯], the collection of grid number of the second-level subdivision.
  - vlevel:  [[[],[],⋯], [[],[],⋯], ⋯]
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
  nc0new, nck, nvlevel, vlevel = vchebyadaptive(fLnshape,vGdom,j=[j1,j2],limitshape=limitshape;nc0=nc0,vadaptlevels=vadaptlevels)
"""

# (2,2) Double-shape limits and double conservation of order: `[j_lower,j_upper]`. level = 1
function vchebyadaptive(fLnshape::Function,vGdom::Vector{T},j::Vector{Int}=[0,10],
    limitshape::Vector{Int}=[0,0];nc0::Int=15,vadaptlevels::Int=5) where{T<:Real}

    limitshape[1] > limitshape[2] ? ArgumentError(error("limitshape[1] > limitshape[2] is not permitted")) : 1
    if j[1] < orderVconstllimit && j[2] < orderVconstllimit
        return nc0, vchebyadaptive(fLnshape,vGdom,j,limitshape[1];nc0=nc0,vadaptlevels=vadaptlevels)
    else
        if j[1] == j[2]
            nck1, isconverged, nvlevel1, vlevel1 = vchebyadaptive(fLnshape,vGdom,j[1],limitshape[1];nc0=nc0,vadaptlevels=vadaptlevels)
            if limitshape[2] ≥ limitshape[1] > 0
                sub = subdivsinitial(nc0, nvlevel1, vlevel1)
                @show isconverged
                for k in 1:nc0 - 1
                    if nvlevel1[k] .> limitshape[2]
                        for nc0sub in 3:5
                            ncksub, isconvergedsub, nvlevelsub, vlevelsub = vchebyadaptive(fLnshape,[vlevel1[k][1],
                                        vlevel1[k][end]],j[1],limitshape[1];nc0=nc0sub,vadaptlevels=vadaptlevels)
                            sub.nvlevel2[k] = nc0sub - 1
                            sub.nvlevel2i[k] = ncksub
                            sub.nvlevel[k] = nvlevelsub
                            sub.vlevel[k] = vlevelsub
                            sum(nvlevelsub .> limitshape[2]) == 0 ? break : 1
                            # @show k, nc0sub, ncksub, nvlevelsub
                        end
                    end
                end
                Nsub = sum(sub.nvlevel2 .- 1)
                if Nsub == 0
                    return nc0, (nck1, nvlevel1, vlevel1)
                else
                    nc0new = nc0 + Nsub
                    vlevelre = Vector((undef),nc0new-1)
                    nvlevelre = zeros(Int,nc0new-1)
                    s = 0  # point for the neaw `nvlevel`
                    # @show length(sub.nvlevel2), nc0-1+Nissub
                    for k in 1:nc0 - 1
                        if sub.nvlevel2[k] == 1
                            s += 1
                            nvlevelre[s] = sub.nvlevel[k][1]
                            vlevelre[s] = sub.vlevel[k][1]
                        elseif sub.nvlevel2[k] ≥ 2
                            # @show k, sub.nvlevel2[k]
                            for i in 1:sub.nvlevel2[k]
                                s += 1
                                @show k,i,s,sub.nvlevel2[k],sub.nvlevel[k][i]
                                nvlevelre[s]
                                nvlevelre[s] = sub.nvlevel[k][i]
                                vlevelre[s] = sub.vlevel[k][i]
                            end
                        else
                            ryfkrgrth
                        end
                    end
                    return nc0new,  (isconverged, nck1, nvlevelre, vlevelre)
                end
            else
                return nc0, (nck1, isconverged, nvlevel1, vlevel1)
            end
            return nc0, (nck1, isconverged, nvlevel1, vlevel1)
        else
            nck1, isconverged, nvlevel1, vlevel1 = vchebyadaptive(fLnshape,vGdom,j,limitshape[1];nc0=nc0,vadaptlevels=vadaptlevels)
            @show isconverged
            if prod(isconverged) == 1
                return nc0, (nck1, isconverged, nvlevel1, vlevel1)
            else
            end
            if limitshape[2] ≥ limitshape[1] > 0
                sub = subdivsinitial(nc0, nvlevel1, vlevel1)
                for k in 1:nc0 - 1
                    if nvlevel1[k] .> limitshape[2]
                        for nc0sub in 3:5
                            ncksub, isconvergedk, nvlevelsub, vlevelsub = vchebyadaptive(fLnshape,[vlevel1[k][1],
                                                   vlevel1[k][end]],j,limitshape[1];nc0=nc0sub,vadaptlevels=vadaptlevels)
                            sub.nvlevel2[k] = nc0sub - 1
                            sub.nvlevel2i[k] = ncksub
                            sub.nvlevel[k] = nvlevelsub
                            sub.vlevel[k] = vlevelsub
                            # @show k, nc0sub,ncksub, nvlevelsub
                            # sum(nvlevelsub .> limitshape[2]) == 0 ? break : 1
                        end
                    end
                end
                Nsub = sum(sub.nvlevel2 .- 1)
                if Nsub == 0
                    return nc0, (nck1, nvlevel1, vlevel1)
                else
                    nc0new = nc0 + Nsub
                    vlevelre = Vector((undef),nc0new-1)
                    nvlevelre = zeros(Int,nc0new-1)
                    s = 0
                    for k in 1:nc0 - 1
                        if sub.nvlevel2[k] == 1
                            s += 1
                            nvlevelre[s] = sub.nvlevel[k][1]
                            vlevelre[s] = sub.vlevel[k][1]
                        elseif sub.nvlevel2[k] ≥ 2
                            for i in 1:sub.nvlevel2[k]
                                s += 1
                                # @show k,i,s,sub.nvlevel2[k],sub.nvlevel[k][i]
                                nvlevelre[s] = sub.nvlevel[k][i]
                                vlevelre[s] = sub.vlevel[k][i]
                            end
                        else
                            ryfkrgrth
                        end
                    end
                    return nc0new,  (nck1, isconverged, nvlevelre, vlevelre)
                end
            else
                return nc0, (nck1, isconverged, nvlevel1, vlevel1)
            end
        end
    end
end

"""
  1, Double-conservation limits: lower-order bounds and higher-order bounds `[j_low, j_high]`;
  2, Double-shape limits, default is `[shape_low,shape_high] = [0,0]` which means no refinement for Chebyshev grids.
           `shape_low` gives the lowest number of each initial interval `[nG[i],nG[i+1]]` and
           `shape_high` gives the highest number of the initial interval.

  inputs:
    limitshape: [lower,higher] order of the shape limits, default is [0,10] which means no bound limits.
    vadaptlevels：The maximum level of adaptation for refinement grids: `nGki_re: 2^(vadaptlevels)+1`

  Outputs:
    nck, nvlevel, vlevel = vchebyadaptive(fLnshape,vGdom,j=[j1,j2],limitshape=limitshape;nc0=nc0,vadaptlevels=vadaptlevels)
    nck, nvlevel, vlevel = vchebyadaptive(fLnshape,vGdom,orderVconst,limitshape[1];nc0=nc0,vadaptlevels=vadaptlevels)

"""

# (2,1) Single-shape limits (lowest) and double conservation of order: `[j_lower,j_upper]`. level = 1
function vchebyadaptive(fLnshape::Function,vGdom::Vector{T},j::Vector{Int}=[0,10],
              limitshape::Int=0;nc0::Int=15,vadaptlevels::Int=5) where{T<:Real}

    if j[1] ≠ j[2]
        ~, isconvergedj, nvlevelj1, vlevelj1 = vchebyadaptive(fLnshape,vGdom,j[1],limitshape;nc0=nc0,vadaptlevels=vadaptlevels)
        nck1, isconverged1, nvlevel1, vlevel1 = vchebyadaptive(fLnshape,vGdom,j[2],limitshape;nc0=nc0,vadaptlevels=vadaptlevels)
        for k in 1:nc0 - 1
            if nvlevel1[k] < nvlevelj1[k]
                vlevel1[k] = vlevelj1[k]
                nck1 += - nvlevel1[k] + nvlevelj1[k]
                nvlevel1[k] = nvlevelj1[k]
            end
        end
        return nck1, [isconvergedj isconverged1], nvlevel1, vlevel1
    else
        # nck1, nvlevel1, vlevel1 = vchebyadaptive(fLnshape,vGdom,j[2],limitshape;nc0=nc0,vadaptlevels=vadaptlevels)
        return vchebyadaptive(fLnshape,vGdom,j[2],limitshape;nc0=nc0,vadaptlevels=vadaptlevels)
    end
end

"""

  inputs:
    limitshape: The lowest bound of order of CHebyshev polynomial.
    vadaptlevels：The maximum level of adaptation for refinement grids: `nGki_re: 2^(vadaptlevels)+1`

  Outputs:
    nck, isconverged, nvlevel, vlevel = vchebyadaptive(fLnshape,vGdom,j=orderVconst,limitshape=limitshape;nc0=nc0,vadaptlevels=vadaptlevels)
    nck, isconverged, nvlevel, vlevel = vchebyadaptive(fLnshape,vGdom,0,9;nc0=nc0,vadaptlevels=vadaptlevels)
    vGk = zeros(datatype,nck)
    vGk = vGvlevel1(vGk,nvlevel,vlevel)
"""

# (1,1) Single-shape limits (lowest) and single conservation of order: `j`. level = 1
function vchebyadaptive(fLnshape::Function,vGdom::Vector{T},j::Int=2,
    limitshape::Int=0;nc0::Int=15,vadaptlevels::Int=5) where{T<:Real}

    vlevel = Vector((undef),nc0-1)
    v = vCmapping(vccn(nc0;datatype=T),vGdom[1],vGdom[2];isinv=true)
    isconverged = zeros(Bool,nc0-1)
    if j < orderVconstllimit
        for i in 1:nc0 - 1
            vlevel[i] = [v[i], v[i+1]]
        end
        return nc0, nothing, 2ones(Int,nc0-1), vlevel
    else
        nvlevel = zeros(Int,nc0-1)
        orderk = 2^(vadaptlevels+1) + 1
        vcc = vccn(orderk;datatype=T)
        nck = copy(nc0)
        nc_kup = copy(nc0)
        orderkup = 0
        vkup = Vector{(undef)}
        μ = chebyshevmoments1(Float64, orderk)
        δdMsk = 0 |> T
        δdMskup = Inf |> T
        dMsk = 0 |> T
        dMs2k = 0 |> T
        for i in 1:nc0 - 1
            cMsi = - 2/sqrtpi * (v[i] - v[i+1])
            δdMskup = Inf |> T
            k = 1
            orderk = 2^k + 1
            dk = 2^(vadaptlevels - k + 1)
            vk = vCmapping(vcc[1:dk:end],v[i],v[i+1];isinv=true)
            wcck = clenshawcurtisweights(μ[1:orderk])
            dMsk = cMsi * dot(wcck, vk.^j .* fLnshape(vk))
            for k in 1:vadaptlevels
                orderk = 2^k + 1
                order2k = 2orderk - 1
                dk = 2^(vadaptlevels - k)
                v2k = vCmapping(vcc[1:dk:end],v[i],v[i+1];isinv=true)
                wcck = clenshawcurtisweights(μ[1:order2k])
                dMs2k = cMsi * dot(wcck, v2k.^j .* fLnshape(v2k))
                δdMsk = abs(dMs2k - dMsk)
                RδdMsk = 2δdMsk / abs(dMsk + dMs2k)
                if δdMsk < epsT5
                    if δdMsk < 1e-20 && RδdMsk < 1e-7
                        nck += orderk - 2
                        vkup = vk
                        # @show i,k,δdMsk,RδdMsk
                        isconverged[i] = true
                        break
                    elseif RδdMsk < 1e-13
                        nck += orderk - 2
                        vkup = vk
                        # @show i,k,δdMsk,RδdMsk
                        isconverged[i] = true
                        break
                    end
                else
                    if k == vadaptlevels
                        if δdMsk < δdMskup
                            nck += orderk - 2
                            vkup = vk
                        else
                            nck = nc_kup
                            orderk = orderkup
                            # vk = vkup
                        end
                        # @warn(" `k` reach the maximum order, `vadaptlevels`=",vadaptlevels)
                    else
                        if δdMsk < δdMskup
                            δdMskup = copy(δdMsk)
                            nc_kup = nck + orderk - 2
                            orderkup = copy(orderk)
                            vkup = deepcopy(vk)
                        else
                            orderk = orderkup
                            nck = nc_kup + orderk - 2
                            break
                        end
                    end
                end
                vk = v2k
                dMsk = dMs2k
            end
            if limitshape > 1 && orderk < limitshape
                vlevel[i] = vCmapping(vccn(limitshape;datatype=T),v[i],v[i+1];isinv=true)
                nck += limitshape - orderk
                nvlevel[i] = limitshape
            else
                nvlevel[i] = orderk
                vlevel[i] = vkup
            end
        end
        # nck = sum(nvlevel) - nc0 + 2
        if nc0 > 2
            return nck, isconverged,nvlevel, vlevel
        elseif nc0 == 2
            rymj
            return nck, isconverged[1], nvlevel[1], vlevel[1]
        end
    end
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
        prod(nvlevel0) == 0 ? @error("position vector of `vG0` in `vGk` is falure !!!") : @show 1
        return nvlevel0
    end
end
