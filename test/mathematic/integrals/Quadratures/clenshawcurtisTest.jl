using Xsum, AccurateArithmetic, DoubleFloats          # for higher precision
using FastTransforms, LinearAlgebra
import FastTransforms: chebyshevmoments1,chebyshevjacobimoments1,chebyshevlogmoments1

pathroot = "G:\\BaiduNetdiskWorkspace\\FP0D1VchebySelf"
include(joinpath(pathroot,"mathematics\\cheby_extrema.jl"))
include(joinpath(pathroot,"src\\Grids\\gridv.jl"))

n = 3
dtype = BigFloat
nk = zeros(Int,8)
i = 1
nk[1] = n
vcc0 = clenshawcurtisnodes(dtype,nk[i])  |> Vector{dtype}
i = 2
nk[i] = 2nk[i-1] - 1
vcc2 = clenshawcurtisnodes(dtype,nk[i])
i = 3
nk[i] = 2nk[i-1] - 1
vcc4 = clenshawcurtisnodes(dtype,nk[i])
i = 4
nk[i] = 2nk[i-1] - 1
vcc6 = clenshawcurtisnodes(dtype,nk[i])
i = 5
nk[i] = 2nk[i-1] - 1
vcc8 = clenshawcurtisnodes(dtype,nk[i])
i = 6
nk[i] = 2nk[i-1] - 1
vcc10 = clenshawcurtisnodes(dtype,nk[i])
i = 7
nk[i] = 2nk[i-1] - 1
vcc12 = clenshawcurtisnodes(dtype,nk[i])
i = 8
nk[i] = 2nk[i-1] - 1
vcc14 = clenshawcurtisnodes(dtype,nk[i])
###
i = 2
dtype = BigFloat   # No `BigFloat`
vccb = clenshawcurtisnodes(dtype,nk[i])
μb = chebyshevmoments1(dtype, nk[i])
wccb = clenshawcurtisweights(μb)             # `BigFloat` is faulty here.
ff(x) = exp(x)
@show dot(wccb, ff.(vccb)) - 2sinh(1)
##
i = 8
T = dtype = Float64
Pi = pi |> dtype
vcc = clenshawcurtisnodes(dtype,nk[i])
μ = chebyshevmoments1(dtype, nk[i])
wcc = clenshawcurtisweights(μ)
ff(x) = exp(x)
@show dot(wcc, ff.(vcc)) - 2sinh(1)

const sqrtpi = 1.772453850905516027298
const epsT5 = 5eps(Float64)
domain = [-1.0,1.0] |> Vector{dtype}        #
vGmin = 10eps(Float64) |> dtype #     # (=1e-4, default) which will affect the conservations owing to the lower boundary and the errors of vaules when v → 0 owint to CFL conditions.
# vGmin = 1e-3 #     # (=1e-4, default) which will affect the conservations owing to the lower boundary and the errors of vaules when v → 0 owint to CFL conditions.
                    # `v → 0` is for to de
vGmax = 8.0
vGdom = [vGmin,vGmax] |> Vector{dtype}        # The domain of velicity axis, [a,b]
fM0(v) = v.^2 .* exp.(-v.^2)
vGab = vGdom
function fMcc(vGab::Vector{T},vcc::Vector{T}) where{T<:Real}
    vGcc = vCmapping(vcc,vGab[1],vGab[2];isinv=true)
    return exp.(-vGcc.^2)
end

function Mscc(vGab::Vector{T},nc0::Int) where{T<:Real}

    vcc = clenshawcurtisnodes(T,nc0)
    v = vCmapping(vcc,vGab[1],vGab[2];isinv=true)
    μ = chebyshevmoments1(T, nc0)
    wcc = clenshawcurtisweights(μ)
    Ms = zeros(23)
    dxdv = - 2/sqrtpi * (vGab[1] - vGab[2])
    Mn2 = dxdv * dot(wcc, exp.(-v.^2))
    Mn1 = dxdv * dot(wcc, v .* exp.(-v.^2))
    Ms[1] = dxdv * dot(wcc, fM0.(v))
    Ms[2] = dxdv * dot(wcc, v .* fM0.(v))
    for j in 3:23
        Ms[j] = dxdv * dot(wcc, v.^(j-1) .* fM0.(v))
    end
    # return Ms, Mn1, Mn2
    return Ms
end

function Mscc(vGab::Vector{T},j::Int,nc0::Int) where{T<:Real}

    vcc = clenshawcurtisnodes(T,nc0)
    v = vCmapping(vcc,vGab[1],vGab[2];isinv=true)
    μ = chebyshevmoments1(T, nc0)
    wcc = clenshawcurtisweights(μ)
    dxdv = - 2/sqrtpi * (vGab[1] - vGab[2])
    Ms = dxdv * dot(wcc, v.^j .* fM0.(v))
    return Ms
end

function Mscc(dxdv::T,vGab::Vector{T},j::Int,nc0::Int) where{T<:Real}
    vcc = clenshawcurtisnodes(T,nc0)
    v = vCmapping(vcc,vGab[1],vGab[2];isinv=true)
    μ = chebyshevmoments1(T, nc0)
    wcc = clenshawcurtisweights(μ)
    Ms = dxdv * dot(wcc, v.^j .* fM0.(v))
    return Ms
end

function Mscc(vGab::Vector{T},j::Int,nvec::Vector{Int}) where{T<:Real}
    Ms = zeros(T,length(nvec))
    k = 0
    for i in nvec
        k += 1
        Ms[k] = Mscc(vGab,j,i)
    end
    return Ms
end


function RMsL(vGab::Vector{T},i::Int) where{T<:Real}
    Ms, Mn1, Mn2 = Mscc(vGab,nk[i])
    (Ms-Mst)./Mst
end

function RMsn(vGab::Vector{T},j::Int,nMax::Int,dnn::Int=5) where{T<:Real}
    nnvec = 15:dnn:nMax |> Vector{Int}
    Ms = Mscc(vGab,j,nnvec)
    label = string("Ms,j=",j)
    xlabel = "ncc"
    ylabel = "(Ms/Mst - 1)"
    # dMs = Ms.-Mst[j+1]
    dMs = Ms./Mst[j+1] .- 1
    pp = plot(nnvec,abs.(dMs),xlabel=xlabel,ylabel=ylabel,legend=legendtL,label=label)
    display(pp)
    dMs
end


function Msccc(vGab::Vector{T},j::Int,nc0::Int) where{T<:Real}
    dxdv = - 2/sqrtpi * (vGab[1] - vGab[2])
    # gauss-chebyshev quadrature of first kind
    vc = cheby_extrema(nc0,domain)
    wc = cheby_extremaweights(nc0,domain;GQmethod=:chebyshev)
    vG = vCmapping(vc,vGab[1],vGab[2];isinv=true)
    Msc = dxdv * dot(wc, (vG.^j .* fM0.(vG)))
    # Clenshawcurtis quadrature
    # vcc = clenshawcurtisnodes(T,nc0)
    # v = vCmapping(vc,vGab[1],vGab[2];isinv=true)
    v = vG
    μ = chebyshevmoments1(T, nc0)
    wcc = clenshawcurtisweights(μ)
    Msclen = dxdv * dot(wcc, (v.^j .* fM0.(v)))
    # Fejer quandrature
    vf = fejernodes1(T, nc0)
    μ = chebyshevmoments1(T, nc0)
    wf = fejerweights1(μ)
    vf = vCmapping(vf,vGab[1],vGab[2];isinv=true)
    Msfejer = dxdv * dot(wf, (vf.^j .* fM0.(vf)))
    @show norm(wcc - wc)
    return [Msclen; Msc; Msfejer]
end

# subdivisions: level = 0
function Msccsub(vGab::Vector{T},j::Int,ordershape::Int=5,nc_0::Int=15) where{T<:Real}

    ordershape < 3 ? ArgumentError(error("ordershape must be greater than `2`")) : 1
    vcc0 = clenshawcurtisnodes(T,nc_0)
    v = vCmapping(vcc0,vGab[1],vGab[2];isinv=true)
    Msk = zero(vGab[1])
    nc_k = 1 * nc_0
    for i in 1:nc_0 - 1
        nc_k += ordershape - 2
        vGabi = [v[i];v[i+1]]
        Msk += Mscc(vGabi,j,ordershape)
    end
    @show (ordershape, nc_k)
    # single process with `nc_k` order of Clenshaw-Curtis nodes
    vcck = clenshawcurtisnodes(T,nc_k)
    vk = vCmapping(vcck,vGab[1],vGab[2];isinv=true)
    μ = chebyshevmoments1(T, nc_k)
    wcc = clenshawcurtisweights(μ)
    dxdv = - 2/sqrtpi * (vGab[1] - vGab[2])
    Ms = dxdv * dot(wcc, vk.^j .* fM0.(vk))
    return [Ms; Msk]
end

# subdivisions: level == 1
function Msccsub10(vGab::Vector{T},j::Int,ordershape::Int=5,nc_0::Int=15,levelMax::Int=1) where{T<:Real}

    ordershape < 3 ? ArgumentError(error("ordershape must be greater than `2`")) : 1
    vcc0 = clenshawcurtisnodes(T,nc_0)
    v = vCmapping(vcc0,vGab[1],vGab[2];isinv=true)
    Msk = zero(vGab[1])
    nc_k = 1 * nc_0
    if levelMax == 1
        for i in 1:nc_0 - 1
            vGabi = [v[i];v[i+1]]
            println()
            for k in 1:ordershape
                orderk = 2^k + 1
                dMsk = Mscc(vGabi,j,orderk)
                dMs2k = Mscc(vGabi,j,2^(k+1) + 1)
                @show j, i, k, fmtf2.([v[i], dMs2k - dMsk])
                if abs(dMs2k - dMsk) < 1e-15
                    Msk += dMsk
                    nc_k += orderk - 2
                    break
                else
                    if k == ordershape
                        Msk += dMs2k
                        nc_k += 2k + 1
                    end
                end
            end
        end
    else
    end
    @show (ordershape, nc_k)
    # single process with `nc_k` order of Clenshaw-Curtis nodes
    vcck = clenshawcurtisnodes(T,nc_k)
    vk = vCmapping(vcck,vGab[1],vGab[2];isinv=true)
    μ = chebyshevmoments1(T, nc_k)
    wcc = clenshawcurtisweights(μ)
    dxdv = - 2/sqrtpi * (vGab[1] - vGab[2])
    Ms = dxdv * dot(wcc, vk.^j .* fM0.(vk))
    # display(plot(vk))
    return [Ms; Msk]
end

"""
  Msk, nc_k, nvlevel, vlevel, vGcc = MsccsubvG(vGab,j,ordershape,nc_0)
"""

function MsccsubvG(vGab::Vector{T},j::Int,ordershape::Int=5,nc_0::Int=15,levelMax::Int=1) where{T<:Real}

    ordershape < 3 ? ArgumentError(error("ordershape must be greater than `2`")) : 1
    vcc0 = clenshawcurtisnodes(T,nc_0)
    v = vCmapping(vcc0,vGab[1],vGab[2];isinv=true)
    Msk = zero(vGab[1])
    if levelMax == 1
        orderk = 2^(ordershape+1) + 1
        vcc = clenshawcurtisnodes(T,orderk)
        μ = chebyshevmoments1(T, orderk)
        nc_k = copy(nc_0)
        nc_kup = copy(nc_0)
        δdMsk = 0 |> T
        δdMskup = Inf |> T
        orderkup = 0
        dMsk = 0 |> T
        dMs2k = 0 |> T
        dMskup = 0 |> T
        vkup = Vector{(undef)}
        vlevel = Vector((undef),nc_0-1)
        nvlevel = zeros(Int,nc_0-1)
        for i in 1:nc_0 - 1
            vGabi = [v[i];v[i+1]]
            dxdvi = - 2/sqrtpi * (vGabi[1] - vGabi[2])
            # printstyled("j=",j,", i=",i,",vi=",fmtf2.(vGabi), color=:red)
            # println()
            δdMskup = Inf |> T
            for k in 1:ordershape
                orderk = 2^(k+1) + 1
                dk = 2^(ordershape - k)
                vk = vCmapping(vcc[1:dk:end],vGabi[1],vGabi[2];isinv=true)
                wcck = clenshawcurtisweights(μ[1:orderk])
                dMs2k = dxdvi * dot(wcck, vk.^j .* fM0.(vk))
                #
                orderk = 2^k + 1
                dk = 2^(ordershape - k + 1)
                vk = vCmapping(vcc[1:dk:end],vGabi[1],vGabi[2];isinv=true)
                wcck = clenshawcurtisweights(μ[1:orderk])
                dMsk = dxdvi * dot(wcck, vk.^j .* fM0.(vk))
                #
                δdMsk = abs(dMs2k - dMsk)
                # @show k, orderk, fmtf2.([δdMskup, δdMsk, δdMsk/dMs2k])
                if min(δdMsk,100 * 2δdMsk / abs(dMsk + dMs2k)) < epsT5
                    Msk += dMsk
                    nc_k += orderk - 2
                    vkup = vk
                    break
                else
                    if k == ordershape
                        if δdMsk < δdMskup
                            nc_k += orderk - 2
                            Msk += dMsk
                            vkup = vk
                        else
                            Msk += dMskup
                            nc_k = nc_kup
                            orderk = orderkup
                            # vk = vkup
                        end
                        # @warn(" `k` reach the maximum order, `ordershape`=",ordershape)
                    else
                        if δdMsk < δdMskup
                            δdMskup = copy(δdMsk)
                            nc_kup = nc_k + orderk - 2
                            dMskup = copy(dMsk)
                            orderkup = copy(orderk)
                            vkup = deepcopy(vk)
                        else
                            Msk += dMskup
                            # vk = vkup
                            orderk = orderkup
                            nc_k = nc_kup + orderk - 2
                            break
                        end
                    end
                end
            end
            nvlevel[i] = orderk
            vlevel[i] = vkup
        end
    else
    end
    # nc_k = sum(nvlevel) - nc_0 + 2
    @show (j, ordershape, nc_k), nvlevel
    vG = zeros(T,nc_k)
    vG[1] = vlevel[1][1]
    vG[end] = vlevel[end][end]
    nvGi = 2
    i = 0
    for k in nvlevel
        i += 1
        @show i,k
        vG[nvGi:nvGi+k-2] = vlevel[i][2:end]
        nvGi += k - 1
    end
    return Msk, nc_k, nvlevel, vlevel, vG
end

"""
  Msccc = Msccsub1(vGab,j,ordershape,nc_0)
  f(j) = 1 .- Msccsub1(vGab,j,ordershape,nc_0) ./ Mst[j+1]
"""

function Msccsub1(vGab::Vector{T},j::Int,ordershape::Int=5,nc_0::Int=15,levelMax::Int=1) where{T<:Real}

    ordershape < 3 ? ArgumentError(error("ordershape must be greater than `2`")) : 1
    vcc0 = clenshawcurtisnodes(T,nc_0)
    v = vCmapping(vcc0,vGab[1],vGab[2];isinv=true)
    Msk = zero(vGab[1])
    if levelMax == 1
        orderk = 2^(ordershape+1) + 1
        vcc = clenshawcurtisnodes(T,orderk)
        μ = chebyshevmoments1(T, orderk)
        nc_k = copy(nc_0)
        nc_kup = copy(nc_0)
        δdMsk = 0 |> T
        δdMskup = Inf |> T
        orderkup = 0
        dMsk = 0 |> T
        dMs2k = 0 |> T
        dMskup = 0 |> T
        vkup = Vector{(undef)}
        vlevel = Vector((undef),nc_0-1)
        nvlevel = zeros(Int,nc_0-1)
        for i in 1:nc_0 - 1
            vGabi = [v[i];v[i+1]]
            dxdvi = - 2/sqrtpi * (vGabi[1] - vGabi[2])
            # printstyled("j=",j,", i=",i,",vi=",fmtf2.(vGabi), color=:red)
            # println()
            δdMskup = Inf |> T
            for k in 1:ordershape
                orderk = 2^(k+1) + 1
                dk = 2^(ordershape - k)
                vk = vCmapping(vcc[1:dk:end],vGabi[1],vGabi[2];isinv=true)
                wcck = clenshawcurtisweights(μ[1:orderk])
                dMs2k = dxdvi * dot(wcck, vk.^j .* fM0.(vk))
                #
                orderk = 2^k + 1
                dk = 2^(ordershape - k + 1)
                vk = vCmapping(vcc[1:dk:end],vGabi[1],vGabi[2];isinv=true)
                wcck = clenshawcurtisweights(μ[1:orderk])
                dMsk = dxdvi * dot(wcck, vk.^j .* fM0.(vk))
                #
                δdMsk = abs(dMs2k - dMsk)
                # @show k, orderk, fmtf2.([δdMskup, δdMsk, δdMsk/dMs2k])
                if min(δdMsk,100 * 2δdMsk / abs(dMsk + dMs2k)) < epsT5
                    Msk += dMsk
                    nc_k += orderk - 2
                    vkup = vk
                    break
                else
                    if k == ordershape
                        if δdMsk < δdMskup
                            nc_k += orderk - 2
                            Msk += dMsk
                            vkup = vk
                        else
                            Msk += dMskup
                            nc_k = nc_kup
                            orderk = orderkup
                            # vk = vkup
                        end
                        # @warn(" `k` reach the maximum order, `ordershape`=",ordershape)
                    else
                        if δdMsk < δdMskup
                            δdMskup = copy(δdMsk)
                            nc_kup = nc_k + orderk - 2
                            dMskup = copy(dMsk)
                            orderkup = copy(orderk)
                            vkup = deepcopy(vk)
                        else
                            Msk += dMskup
                            # vk = vkup
                            orderk = orderkup
                            nc_k = nc_kup + orderk - 2
                            break
                        end
                    end
                end
            end
            nvlevel[i] = orderk
            vlevel[i] = vkup
        end
    else
    end
    # nc_k = sum(nvlevel) - nc_0 + 2
    @show (j, ordershape, nc_k), nvlevel
    # single process with `nc_k` order of Clenshaw-Curtis nodes
    vcck = clenshawcurtisnodes(T,nc_k)
    vk = vCmapping(vcck,vGab[1],vGab[2];isinv=true)
    μ = chebyshevmoments1(T, nc_k)
    wcc = clenshawcurtisweights(μ)
    dxdv = - 2/sqrtpi * (vGab[1] - vGab[2])
    Ms = dxdv * dot(wcc, vk.^j .* fM0.(vk))
    # display(plot(vk))
    return [Ms; Msk]
end

# subdivisions: level ≥ 2

nc0 = 85
vc = cheby_extrema(nc0,domain)
vG = vCmapping(vc,vGab[1],vGab[2];isinv=true)
fvG = fM0.(vG)
@show vc - clenshawcurtisnodes(dtype,nc0)

ML1 = 23   # = 1 + L_max
MLvec = 0:ML1-1 |>Vector{Int}
Mst = zeros(dtype,ML1)      # M̂s = (n̂, M̂₁, K̂, ⋯, M̂ⱼ, ⋯)
# # # #   L =  [1,  3,  5,   7,  9,    11   13    15      17     19      21]
Mst[2:2:ML1] = [1,  2,  6,  24, 120,  720, 5040, 40320, 362880, 3628800, 39916800] * 2 / Pi^0.5
# # # #   L =  [0, 2, 4,  6,   8,   10,    12,     14,      16,      18,        20,        22]
Mst[1:2:ML1] = [1, 3, 15, 105, 945, 10395, 135135, 2027025,34459425,654729075, 13749310575,316234143225] ./ 2 .^(0:11)

j = 0
nc_0 = 15
ordershape = 5
