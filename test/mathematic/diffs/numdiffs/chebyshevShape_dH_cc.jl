using ChebyshevApprox, FastTransforms, LinearAlgebra, LinearAlgebraX
using ToeplitzMatrices
using Plots, DataFrames

import FastTransforms: chebyshevmoments1

# T = Float64
# pathroot = "G:\\BaiduNetdiskWorkspace\\FP0D1VchebySelf"
# include(joinpath(pathroot,"mathematics\\cheby_extrema.jl"))
# include(joinpath(pathroot,"mathematics\\mathematic.jl"))
# include(joinpath(pathroot,"src\\Grids\\gridv.jl"))

"""
   Test the effects of Chebyshev shape (number of notes) for

     1. interpolation
     2. difference by dierrence matrices and
     3. integration.


"""

const sqrtpi = 1.772453850905516027298
const epsT5 = 5eps(Float64)

H0(v) = @. √Pi / 4 * erf(v) / v
dH0(v) = @. exp(-v^2) / 2v -  √Pi / 4v^2 * erf(v)
ddH0(v) = @. (-2(v + v^3) * exp(-v^2)  + √Pi * erf(v))/(2 * v^3)

dHLn = zero(fLn)
ddHLn = zero(fLn)
ddHLn,dHLn = dfvLg(ddHLn,dHLn,HLnt,va,nc0)

nc0 = 15
vadaptshapes = 7
j = 0       # [0, N⁺]
M = 2       # [1, 2]
vabth = 5
orders = 3
i = 6
nccc = 0

Pi = pi |> T
domain = [-1.0,1.0] |> Vector{T}        #
vGmin = 10eps(Float64) |> T #     # (=1e-4, default) which will affect the conservations owing to the lower boundary and the errors of vaules when v → 0 owint to CFL conditions.
# vGmin = 1e-3 #     # (=1e-4, default) which will affect the conservations owing to the lower boundary and the errors of vaules when v → 0 owint to CFL conditions.
                    # `v → 0` is for to de
vGmax = 8.0
vGdom = [vGmin,vGmax] |> Vector{T}
vcc = clenshawcurtisnodes(BigFloat,nc0) |> AbstractVector{T}
v = vCmapping(vcc,vGdom[1],vGdom[2];isinv=true)
orderk = 2^(vadaptshapes+1) + 1
vcc = clenshawcurtisnodes(BigFloat,orderk) |> AbstractVector{T}

vi = v[i]
vi1 = v[i+1]
if nccc > 2
    vcccc = clenshawcurtisnodes(BigFloat,nccc) |> AbstractVector{T}
    vkccc = vCmapping(vcccc, vi, vi1;isinv=true)
    vi = vkccc[i]
    vi1 = vkccc[i+1]
end
dxdvi = 2 / (vi - vi1)

"""
 chebyshev interpolation shape (orders) testting.
"""

function errDf(M,k)

    orderk = 2^k + 1
    dk = 2^(vadaptshapes - k + 1)
    vk = vCmapping(vcc[1:dk:end],vi,vi1;isinv=true)
    Dc12 = chebyshevdiff(orderk;M=2, datatype = BigFloat) |> Array{T}
    Dk = dxdvi^M * (Dc12[:,:,M] * H0(vk))

    orderk = 2^(k+1) + 1
    dk = 2^(vadaptshapes - k)
    vk = vCmapping(vcc[1:dk:end],vi,vi1;isinv=true)
    Dc12 = chebyshevdiff(orderk;M=2, datatype = BigFloat) |> Array{T}
    D2k = dxdvi^M * (Dc12[:,:,M] * H0(vk))
    #

    δDk = norm(D2k[1:2:end] - Dk)
    if M == 1
        D2kt = dH0(vk)
    elseif M == 2
        D2kt = ddH0(vk)
    end
    δDkt = norm(D2k - D2kt)

    return k, δDk, δDkt
end

edf = errDf.(M,1:7)
@show [vi, vi1]
