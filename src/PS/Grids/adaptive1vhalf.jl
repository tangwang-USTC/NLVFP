
"""
  Refinement scheme: Global-Local-Level strategy (fixed shape function for multi-level, a variable-level strategy which is usually lager than 2)

  Subdivision surface refinement schemes can be broadly classified into two categories: interpolating and approximating.

    ⚈ Interpolating schemes are required to match the original position of vertices in the original mesh.
    ⚈ Approximating schemes are not; they can and will adjust these positions as needed.

  In general, approximating schemes have greater smoothness, but the user has less overall control of the outcome.

   Each iteration is often called a subdivision level, starting at zero (before any refinement occurs).

"""


"""
"""
# [Nshape,level], 
function vHadaptive(fLnshape::Vector{Function},Nshape::Int64,
    vh0::AbstractVector{T},nc0::Int64,ocp::Int64,jVconst::Vector{Int64};
    abstol::T=epsT5,reltol::T=1e-5,vadaptlevels::Int64=4) where{T}

    num0 = zeros(Int64,Nshape)
    vh = Vector{AbstractVector{T}}(undef,Nshape)
    nvh = zeros(Int64,Nshape)
    RδdMs = zeros(T,2,Nshape)
    RδdMsk = zeros(T,2)
    k = 1
    num0[k], vh[k] = vHadaptive!(RδdMsk,fLnshape[k],vh0,nc0,ocp,jVconst[k];
                      abstol=abstol,reltol=reltol,vadaptlevels=vadaptlevels)
    RδdMs[:,k] = deepcopy(RδdMsk)
    nvh[k] = length(vh[k])

    for k in 2:Nshape
        num0[k], vh[k] = vHadaptive!(RδdMsk,fLnshape[k],vh0,nc0,ocp,jVconst[k];
                          abstol=abstol,reltol=reltol,vadaptlevels=vadaptlevels)
        nvh[k] = length(vh[k])
    end

    # Choosing the optimized grids for all orders `ℓ` owing to `num0`, `RδdMs` and `nvh`.
    if sum(num0) == 0
        isconverged = true                          # min(nvh)
        Lvh = 1
        for k in 2:Nshape
            nvh[k] < nvh[Lvh] ? Lvh = k : nothing
        end
    else
        isconverged = false                         # max(nvh)
        Lvh = 1
        for k in 2:Nshape
            nvh[k] > nvh[Lvh] ? Lvh = k : nothing
        end
    end

    return isconverged, num0, RδdMs, Lvh, vh[Lvh]
end

"""

  Inputs:
    RδdMsk: =[δdMsk,RδdMsk]
    fLnshape: `= f(v)`, a analysis shape function respective to `vsk`.
    vsk: = [vₖ,vₖ₊₁]
    ocp: (= 8 default), the number of the shape function for Clenshaw-Curtis quandrature.
    reltol: (default = 1e-5), the relative tolerance when  `abstol < epsT5`
    isconvergeds0: `= zeros(Bool,nc0-1)` default, denotes whether refinement is converged.
                 `= true` means converged for no more refinement;
                 '= fasle' denotes false where refinement is needed.


  Outputs:
    num0, vk = vHadaptive!(RδdMsk,fLnshape,vh0,nc0,ocp,jVconst;
                    abstol=abstol,reltol=reltol,vadaptlevels=vadaptlevels)
    isconverged, vk, δdMsk,RδdMsk = vHadaptive(fLnshape,vsk,nc0,ocp,j;
                    isconvergeds0=isconvergeds0,ocp=ocp,abstol=abstol,reltol=reltol)
"""

# [level,ocp], Multilayer refinement
function vHadaptive!(RδdMsk::AbstractVector{T},fLnshape::Function,
    vh0::AbstractVector{T},nc0::Int64,ocp::Int64,j::Int64;
    abstol::T=epsT5,reltol::T=1e-5,vadaptlevels::Int64=4) where{T}

    # level = 2
    isconvergeds0 = zeros(Bool,nc0-1)
    if vadaptlevels == 2
        isconvergeds0, vk, RδdMsk[1],RδdMsk[2] = vHadaptive(fLnshape,vh0,nc0,ocp,j;
                        isconvergeds0=isconvergeds0,abstol=abstol,reltol=reltol)
        num0 = length(isconvergeds0) - sum(isconvergeds0)
        return num0, vk
    else
        isconvergeds0, vk, RδdMsk[1],RδdMsk[2] = vHadaptive(fLnshape,vh0,nc0,ocp,j;
                                    isconvergeds0=isconvergeds0,abstol=abstol,reltol=reltol)
        num0 = length(isconvergeds0) - sum(isconvergeds0)
        for level in 3:vadaptlevels
            if num0 == 0 # prod(isconvergeds0) == 1
                break
            end
            nc0 = length(vk)
            isconvergeds0, vk, RδdMsk[1],RδdMsk[2] = vHadaptive(fLnshape,vk,nc0,ocp,j;
                                    isconvergeds0=isconvergeds0,abstol=abstol,reltol=reltol)
            num0 = length(isconvergeds0) - sum(isconvergeds0)
            if num0 ≥ 1 && level == vadaptlevels
                @warn("Maximun adpative level has reached before converged on all velocity axis domain.")
            end
        end
        return num0, vk
    end

end

# [], Single sub-level, [vk]
function vHadaptive(fLnshape::Function,
    vsk::AbstractVector{T},nv::Int64,ocp::Int64,j::Int64;
    isconvergeds0::Vector{Bool}=zeros(Bool,nv-1),
    abstol::T=epsT5,reltol::T=1e-5) where{T}

    if nv == 2
        if isconvergeds0[1] == true
            return isconvergeds0, vsk, 0.0,0.0
        else
            return evalruleH(fLnshape,vsk,j;ocp=ocp,abstol=abstol,reltol=reltol)
            return isconverged, vk, δdMsk,RδdMsk
        end
    else
        vk = zeros(2nv-1)
        isconvergeds = zeros(Bool,(2nv-1))
        nvk = 0
        k = 1
        if isconvergeds0[k] == true
            nvk += 2
            vk[nvk-1:nvk] = deepcopy(vsk[1:2])   # vk[1:2]
            isconvergeds[nvk-1] = true
            δdMsk,RδdMsk = 0.0,0.0
        else
            isconverged, vk2, δdMsk,RδdMsk = evalruleH(fLnshape,vsk[k:k+1],j;ocp=ocp,abstol=abstol,reltol=reltol)
            if isconverged[1] == true
                nvk += 2
                vk[nvk-1:nvk] = deepcopy(vk2)     # vk[1:2]
                isconvergeds[nvk-1] = true
            else
                nvk += 3
                vk[nvk-2:nvk] = deepcopy(vk2)      # vk[1:3]
                isconvergeds[nvk-2:nvk-1] .= false
            end
        end
        for k in 2:nv-1
            if isconvergeds0[k] == true
                nvk += 1
                vk[nvk] = vsk[k+1]
                isconvergeds[nvk-1] = true
                δdMsk1,RδdMsk1 = 0.0,0.0
            else
                isconverged, vk2, δdMsk1,RδdMsk1 = evalruleH(fLnshape,vsk[k:k+1],j;ocp=ocp,abstol=abstol,reltol=reltol)
                if isconverged[1] == true
                    nvk += 1
                    vk[nvk] = vk2[2]
                    isconvergeds[nvk-1] = true
                else
                    nvk += 2
                    vk[nvk-1:nvk] = vk2[2:3]
                    isconvergeds[nvk-2:nvk-1] .= false
                end
            end
            δdMsk = (δdMsk + δdMsk1)/2
            RδdMsk = (RδdMsk + RδdMsk1)/2
        end
        return isconvergeds[1:nvk-1], vk[1:nvk], δdMsk,RδdMsk
    end
end

"""

  Inputs:
    v2dom: = [vₖ,vₖ₊₁]

  Outputs:
    isconverged, vkdom = evalruleH(fLnshape,v2dom,j;ocp=ocp,abstol=abstol,reltol=reltol)

"""

function evalruleH(fLnshape::Function,v2dom::AbstractVector{T},j::Int64=2;
    ocp::Int64=8,abstol::T=epsT5,reltol::T=1e-5) where{T}

    nc1 = 2ocp-1  # the higher order for error estimation
    cMsi = - 2/sqrtpi * (v2dom[1] - v2dom[2])

    # giving `2ocp-1` nodes
    v = vCmapping(vccn(nc1;datatype=T),v2dom[1],v2dom[2];isinv=true)
    μ = μccn(nc1)

    # `dI_k` based on the `ocp` nodes
    vk = v[1:2:end]
    # wcck = clenshawcurtisweights(μ[1:ocp])
    dMsk = cMsi * dot(clenshawcurtisweights(μ[1:ocp]), vk.^j .* fLnshape(vk))

    # `dI_k1` based on the `2ocp-1` nodes
    # wcck = clenshawcurtisweights(μ)
    dMs2k = cMsi * dot(clenshawcurtisweights(μ), v.^j .* fLnshape(v))

    δdMsk = abs(dMs2k - dMsk)
    if dMsk == 0.0 && dMs2k == 0.0
        # println("vHremesh, 0000")
        return [true], v2dom, 0.0,0.0
    else
        RδdMsk = 2δdMsk / (dMsk + dMs2k)
        if δdMsk < abstol && RδdMsk < reltol
            # println("vHremesh, 1000")
            return [true], v2dom, δdMsk,RδdMsk
        else
            if δdMsk < 1e-20  && RδdMsk < 1e-2
                # @warn ("1,  (dM,δdM,RδdM,vk[1])=",fmtf2.([v2dom[1],dMsk,δdMsk,RδdMsk]))
                return [true], v2dom, δdMsk,RδdMsk
            else
                # @warn ("2, The relative error is bigger than `reltol`! (dM,δdM,RδdM,vk[1])=",fmtf2.([v2dom[1],dMsk,δdMsk,RδdMsk]))
                return [false,false], [v2dom[1], sum(v2dom)/2, v2dom[2]], δdMsk,RδdMsk
            end
        end
    end
    return isconverged, v2dom, δdMsk,RδdMsk
end
