using ChebyshevApprox, FastTransforms
using LinearAlgebra, LinearAlgebraX
using ToeplitzMatrices
using Plots, DataFrames

import FastTransforms: chebyshevmoments1

T = Float64
pathroot = "G:\\BaiduNetdiskWorkspace\\FP0D1VchebySelf"
include(joinpath(pathroot,"mathematics\\cheby_extrema.jl"))
include(joinpath(pathroot,"mathematics\\mathematic.jl"))
include(joinpath(pathroot,"src\\Grids\\gridv.jl"))

"""
   Test the effects of Chebyshev shape (number of notes) for

     1. interpolation
     2. difference by dierrence matrices and
     3. integration.


"""


fM0(v) = exp.(-v.^2)
dfM0(v) = - 2v .* fM0.(v)
ddfM0(v) = (4v.^2 .-2) .* fM0.(v)

# fM0(v) = - v.^2     # fM0log = log.(fM0(v))
# dfM0(v) = - 2v
# ddfM0(v) = - 2one.(v)

# nc0 = 10
# vadaptshapes = 7
# j = 0       # [0, N⁺]
# M = 2       # [1, 2]
# vabth = 5
# orders = 3
# i = 7
# nccc = 0
#
# Pi = pi |> T
# domain = [-1.0,1.0] |> Vector{T}        #
# vGmin = 10eps(Float64) |> T #     # (=1e-4, default) which will affect the conservations owing to the lower boundary and the errors of vaules when v → 0 owint to CFL conditions.
# # vGmin = 1e-3 #     # (=1e-4, default) which will affect the conservations owing to the lower boundary and the errors of vaules when v → 0 owint to CFL conditions.
#                     # `v → 0` is for to de
# vGmax = 8.0
# vGdom = [vGmin,vGmax] |> Vector{T}
# vcc = clenshawcurtisnodes(BigFloat,nc0) |> AbstractVector{T}
# v = vCmapping(vcc,vGdom[1],vGdom[2];isinv=true)
# orderk = 2^(vadaptshapes+1) + 1
# vcc = clenshawcurtisnodes(BigFloat,orderk) |> AbstractVector{T}
#
# vi = v[1]
# vi1 = v[end]
# if nccc > 2
#     vcccc = clenshawcurtisnodes(BigFloat,nccc) |> AbstractVector{T}
#     vkccc = vCmapping(vcccc, vi, vi1;isinv=true)
#     vi = vkccc[i]
#     vi1 = vkccc[i+1]
# end
# dxdvi = 2 / (vi - vi1)
# cMsi = - 2/sqrtpi * (vi - vi1)

"""
 chebyshev interpolation shape (orders) testting.
"""

function erritef(vabth,orders,kk::AbstractVector{Int};isploterr = 2)

    linewidth = 3
    xlabel = string("vabth=",vabth,",orders=",orders)
    nn = length(kk)
    δDk = zeros(T,nn)
    δDkt = zeros(T,nn)
    δD2kt = zeros(T,nn)
    #
    RδDk = zeros(T,nn)
    RδDkt = zeros(T,nn)
    RδD2kt = zeros(T,nn)
    nGk = zeros(Int,nn)
    data = DataFrame()
    s = 0
    for k in kk
        s += 1
        orderk = 2^k + 1
        nGk[s] = orderk
        dk = 2^(vadaptshapes - k + 1)
        vk = vCmapping(vcc[1:dk:end],vi,vi1;isinv=true)
        va = vk * vabth
        weight = chebyshev_weights_extrema(fM0(vk),vcc[1:dk:end],orders,domain)
        chebpoly = ChebPoly(weight,orders,domain)
        chebInterp = chebyshev_evaluate(chebpoly)
        vcint = vCmapping(va,vk[1],vk[end])     # mapping: [v2[1], v2[2]]  → va
        FLnk = [chebInterp([vcint[i]]) for i in 1:orderk]
        #
        FLnkt = fM0.(va)
        orderk = 2^(k+1) + 1
        dk = 2^(vadaptshapes - k)
        vk = vCmapping(vcc[1:dk:end], vi, vi1;isinv=true)
        va = vk * vabth
        weight = chebyshev_weights_extrema(fM0(vk),vcc[1:dk:end],orders,domain)
        chebpoly = ChebPoly(weight,orders,domain)
        chebInterp = chebyshev_evaluate(chebpoly)
        vcint = vCmapping(va,vk[1],vk[end])     # mapping: [v2[1], v2[2]]  → va
        FLn2k = [chebInterp([vcint[i]]) for i in 1:orderk]
        FLn2kt = fM0.(va)
        if s == 1
            if isploterr == 0
                label = string("Fk,=",k)
                pp = plot(vk[1:2:end],FLnk,label=label,line=(linewidth,:auto),xlabel=xlabel)
                label = string("F2k")
                pp = plot!(vk,FLn2k,label=label,line=(linewidth,:auto))
                label = string("Fkt")
                pp = plot!(vk[1:2:end],FLnkt,label=label,line=(linewidth,:auto))
                label = string("F2kt")
                pp = plot!(vk,FLn2kt,label=label,line=(linewidth,:auto),legend=:bottomleft)
                display(pp)
            elseif isploterr == 1
                label = string("dFkt,=",k)
                pp = plot(vk[1:2:end],FLnk - FLnkt,label=label,line=(linewidth,:auto),xlabel=xlabel)
                label = string("dF2kt")
                pp = plot!(vk,FLn2k - FLn2kt,label=label,line=(linewidth,:auto),legend=:bottomleft)
                display(plot(pp))
            elseif isploterr == 2
                label = string("RdFkt,=",k)
                ppR = plot(vk[1:2:end],(FLnk ./ FLnkt .- 1),label=label,line=(linewidth,:auto),xlabel=xlabel)
                label = string("RdF2kt")
                ppR = plot!(vk,(FLn2k ./ FLn2kt .- 1),label=label,line=(linewidth,:auto),legend=:bottomleft)
                display(plot(ppR))
            end
        else
            if isploterr == 0
                # label = string("Fk,=",k)
                # pp = plot!(vk[1:2:end],FLnk,label=label,line=(linewidth,:auto),xlabel=xlabel)
                label = string("F2k,=",k)
                pp = plot!(vk,FLn2k,label=label,line=(linewidth,:auto))
                # label = string("Fkt")
                # pp = plot!(vk[1:2:end],FLnkt,label=label,line=(linewidth,:auto))
                label = string("F2kt")
                pp = plot!(vk,FLn2kt,label=label,line=(linewidth,:auto))
                display(pp)
            elseif isploterr == 1
                # label = string("dFkt,=",k)
                # pp = plot!(vk[1:2:end],FLnk - FLnkt,label=label,line=(linewidth,:auto))
                label = string("dF2kt,=",k)
                pp = plot!(vk,FLn2k - FLn2kt,label=label,line=(linewidth,:auto))
                display(plot(pp))
            elseif isploterr == 2
                # label = string("RdFkt,=",k)
                # ppR = plot!(vk[1:2:end],(FLnk ./ FLnkt .- 1),label=label,line=(linewidth,:auto))
                label = string("RdF2kt,=",k)
                ppR = plot!(vk,(FLn2k ./ FLn2kt .- 1),label=label,line=(linewidth,:auto))
                display(plot(ppR))
            end
        end
        a12 = FLn2k[1:2:end] - FLnk
        δDk[s] = norm(a12[2:1:end])
        a1 = FLnk - FLnkt
        δDkt[s] = norm(a1[2:1:end])
        a2 = FLn2k - FLn2kt
        δD2kt[s] = norm(a2[2:1:end])
        #
        a12 = FLn2k[1:2:end] ./ FLnk .- 1
        RδDk[s] = norm(a12[2:1:end])
        a1 = FLn2k[1:2:end] ./ FLnk .- 1
        RδDkt[s] = norm(a1[2:1:end])
        a2 = FLn2k ./ FLn2kt .- 1
        RδD2kt[s] = norm(a2[2:1:end])
        data = DataFrame([kk nGk δDk δDkt δD2kt RδDk RδDkt RδD2kt],:auto)
        rename!(data,[:k,:nGk,:Dk,:Dkt,:D2kt,:RDk2k,:RDkt,:RD2kt])
    end
    return data
end

function erritef(vabth,orders,k::Int64;isploterr = 1)

    k = 3
    orderk = 2^k + 1
    dk = 2^(vadaptshapes - k + 1)
    vk = vCmapping(vcc[1:dk:end],vi,vi1;isinv=true)
    va = vk * vabth
    weight = chebyshev_weights_extrema(fM0(vk),vcc[1:dk:end],orders,domain)
    chebpoly = ChebPoly(weight,orders,domain)
    chebInterp = chebyshev_evaluate(chebpoly)
    vcint = vCmapping(va,vk[1],vk[end])     # mapping: [v2[1], v2[2]]  → va
    FLnk = [chebInterp([vcint[i]]) for i in 1:orderk]
    FLnkt = fM0.(va)
    #
    orderk = 2^(k+1) + 1
    dk = 2^(vadaptshapes - k)
    vk = vCmapping(vcc[1:dk:end], vi, vi1;isinv=true)
    va = vk * vabth
    weight = chebyshev_weights_extrema(fM0(vk),vcc[1:dk:end],orders,domain)
    chebpoly = ChebPoly(weight,orders,domain)
    chebInterp = chebyshev_evaluate(chebpoly)
    vcint = vCmapping(va,vk[1],vk[end])     # mapping: [v2[1], v2[2]]  → va
    FLn2k = [chebInterp([vcint[i]]) for i in 1:orderk]
    FLn2kt = fM0.(va)

    if isploterr == 0
        xlabel = string("vabth=",vabth,",orders=",orders)
        label = string("Fk,=",k)
        pp = plot(vk[1:2:end],FLnk,label=label,line=(2,:auto),xlabel=xlabel)
        label = string("F2k")
        pp = plot!(vk,FLn2k,label=label,line=(2,:auto))
        label = string("Fkt")
        pp = plot!(vk[1:2:end],FLnkt,label=label,line=(2,:auto))
        label = string("F2kt")
        pp = plot!(vk,FLn2kt,label=label,line=(2,:auto),legend=:bottomleft)
        display(pp)
    else
        xlabel = string("vabth=",vabth,",orders=",orders)
        label = string("dFkt")
        pp = plot(vk[1:2:end],FLnk - FLnkt,label=label,line=(2,:auto))
        label = string("dF2kt")
        pp = plot!(vk,FLn2k - FLn2kt,label=label,line=(2,:auto))
        label = string("RdFkt")
        ppR = plot(vk[1:2:end],(FLnk ./ FLnkt .- 1),label=label,line=(2,:auto),xlabel=xlabel)
        label = string("RdF2kt")
        ppR = plot!(vk,(FLn2k ./ FLn2kt .- 1),label=label,line=(2,:auto))
        display(plot(pp,ppR,layout=(2,1)))
    end
    a12 = FLn2k[1:2:end] - FLnk
    δDk = norm(a12[2:1:end])
    a1 = FLnk - FLnkt
    δDkt = norm(a1[2:1:end])
    a2 = FLn2k - FLn2kt
    δD2kt = norm(a2[2:1:end])
    #
    a12 = FLn2k[1:2:end] ./ FLnk .- 1
    RδDk = norm(a12[2:1:end])
    a1 = FLn2k[1:2:end] ./ FLnk .- 1
    RδDkt = norm(a1[2:1:end])
    a2 = FLn2k ./ FLn2kt .- 1
    RδD2kt = norm(a2[2:1:end])
    data = DataFrame([k δDk δDkt δD2kt RδDk RδDkt RδD2kt],:auto)
    rename!(data,[:k,:Dk,:Dkt,:D2kt,:RDk,:RDkt,:RD2kt])
    return data
end

function errDf(M,k)

    orderk = 2^k + 1
    dk = 2^(vadaptshapes - k + 1)
    vk = vCmapping(vcc[1:dk:end],vi,vi1;isinv=true)
    Dc12 = chebyshevdiff(orderk;M=2, datatype = BigFloat) |> Array{T}
    Dk = dxdvi^M * (Dc12[:,:,M] * fM0(vk))

    orderk = 2^(k+1) + 1
    dk = 2^(vadaptshapes - k)
    vk = vCmapping(vcc[1:dk:end],vi,vi1;isinv=true)
    Dc12 = chebyshevdiff(orderk;M=2, datatype = BigFloat) |> Array{T}
    D2k = dxdvi^M * (Dc12[:,:,M] * fM0(vk))
    #

    δDk = norm(D2k[1:2:end] - Dk)
    if M == 1
        D2kt = dfM0(vk)
    elseif M == 2
        D2kt = ddfM0(vk)
    end
    δDkt = norm(D2k - D2kt)

    return k, δDk, δDkt
end

"""
  Inputs:
    ocp: OrderShapes
    fLnv9: = fLn(vGk[end])

  Outputs:
    fLnI = CDFshape(fLnI,dfLn,vGk,nvlevel,nc0,nck,ocp,fLnv9)
    fLnI = CDFshape(fLnI,dfLn,vGk,nvlevel,nc0,nck,ocp,fLnv0)

"""

function CDFshape(fLnI::AbstractVector{T},dfLn0::AbstractVector{T},vGk::AbstractVector{Tb},
    nvlevel::AbstractVector{Int},nc0::Int64,nck::Int64,ocp::Int64,fLnv9::T=0.0) where{T<:Real,Tb}

    if nck == nc0
        errer("No algorithm when `nck = nc0` for CDF ")
    else
        fLnI = zeros(T,nc0)
        fLnI[nc0] = 1 * fLnv9
        # μk = chebyshevmoments1(T, ocp)
        wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
        k = nc0 - 1
        nk = nvlevel[k]
        nk9 = nck
        nk1 = nk9 - (ocp - 1)
        vk = vGk[nk1:nk9]
        dxdvi = 0.5 * (vk[1] - vk[end])
        fLnI[k] = fLnI[nc0] + dxdvi * dot(wcck, dfLn[nk1:nk9])
        @show nk1, dxdvi * dot(wcck, dfLn[nk1:nk9])
        # @show k,nk, nk1,nk9, length(wcck)
        if nk > ocp
            nks = ocp
            while nk - nks > 0
                nks += (ocp - 1)
                nk9 = nk1
                nk1 = nk9 - (ocp - 1)
                vk = vGk[nk1:nk9]
                dxdvi = 0.5 * (vk[1] - vk[end])
                @show nk1,(vk[1]), dxdvi * dot(wcck, dfLn[nk1:nk9])
                fLnI[k] += dxdvi * dot(wcck, dfLn[nk1:nk9])
            end
            # @show nk, nks - nk9
        end
        for k in nc0-2:-1:1
            nk = nvlevel[k]
            nk9 = nk1
            nk1 = nk9 - (ocp - 1)
            # @show k, nk, nk1, nk9
            vk = vGk[nk1:nk9]
            dxdvi = 0.5 * (vk[1] - vk[end])
            fLnI[k] = fLnI[k+1] + dxdvi * dot(wcck, dfLn[nk1:nk9])
            @show nk1, dxdvi * dot(wcck, dfLn[nk1:nk9])
            if nk > ocp
                nks = ocp
                while nk - nks > 0
                    nks += (ocp - 1)
                    nk9 = nk1
                    nk1 = nk9 - (ocp - 1)
                    vk = vGk[nk1:nk9]
                    dxdvi = 0.5 * (vk[1] - vk[end])
                    fLnI[k] += dxdvi * dot(wcck, dfLn[nk1:nk9])
                    @show nk1, dxdvi * dot(wcck, dfLn[nk1:nk9])
                end
                # @show nk, nks - nk9
            end
        end
        return fLnI
    end
end

function CDFshape0(fLnI::AbstractVector{T},dfLn::AbstractVector{T},vGk::AbstractVector{Tb},
    nvlevel::AbstractVector{Int},nc0::Int64,nck::Int64,ocp::Int64,fLnv0::T=0.0) where{T<:Real,Tb}

    if nck == nc0
        errer("No algorithm when `nck = nc0` for CDF ")
    else
        fLnI = zeros(T,nc0)
        fLnI[1] = 1 * fLnv0
        # μk = chebyshevmoments1(T, ocp)
        wcck = clenshawcurtisweights(chebyshevmoments1(T, ocp))
        k = 1
        nk = nvlevel[k]
        nk1 = 1
        nk9 = ocp
        vk = vGk[nk1:nk9]
        dxdvi = - 0.5 * (vk[1] - vk[end])
        fLnI[2] = fLnI[1] + dxdvi * dot(wcck, dfLn[nk1:nk9])
        # @show k,nk, nk1,nk9, length(wcck)
        if nk > ocp
            nks = ocp
            while nk - nks > 0
                nks += (ocp - 1)
                nk1 = nk9
                nk9 = nk1 + ocp - 1
                vk = vGk[nk1:nk9]
                dxdvi = - 0.5 * (vk[1] - vk[end])
                fLnI[2] += dxdvi * dot(wcck, dfLn[nk1:nk9])
            end
            # @show nk, nks - nk9
        end
        for k in 2:nc0-1
            nk = nvlevel[k]
            nk1 = nk9
            nk9 = nk1 + ocp - 1
            # @show k, nk, nk1, nk9
            vk = vGk[nk1:nk9]
            dxdvi = - 0.5 * (vk[1] - vk[end])
            fLnI[k+1] = fLnI[k] + dxdvi * dot(wcck, dfLn[nk1:nk9])
            if nk > ocp
                nks = ocp
                while nk - nks > 0
                    nks += (ocp - 1)
                    nk1 = nk9
                    nk9 = nk1 + ocp - 1
                    vk = vGk[nk1:nk9]
                    dxdvi = - 0.5 * (vk[1] - vk[end])
                    fLnI[k+1] += dxdvi * dot(wcck, dfLn[nk1:nk9])
                end
                # @show nk, nks - nk9
            end
        end
        return fLnI
    end
end

fLnI = zeros(datatype,nc0)
fLnv9 = fLnt[end]
fLnI = CDFshape(fLnI,dfLn,vGk,nvlevel,nc0,nck,ordershapes,fLnv9)
errfI = fLnI - fLnt[nvlevel0]
errRfI = errfI ./ fLnt[nvlevel0]
errdf = dfLn[nvlevel0] - dfLnt[nvlevel0]
errRdf = errdf ./ dfLnt[nvlevel0]
data = [vG0 errfI errRfI  errdf errRdf fLnI dfLnt[nvlevel0] [0; nvlevel]]
datafI = DataFrame(data,:auto)
rename!(datafI,1=>:vG0,2=>:errfI,3=>:errRfI,4=>:errdf,5=>:errRdf,6=>:fLn,7=>:dfLn,8=>:nvlevel)
# # edf = errDf.(M,1:7)
# # eif = errIf.(j,1:7)
# eitf1 = erritef.(vabth,orders,1)
# eitfm1 = erritef(vabth,orders,2:5)
# @show eitfm1
# # eitf = erritef(vabth,orders,2:7)
# # @show eitf
#
# # function (nvlevel0,vlevel0,nc0,nvlevel,nck,vlevel)
# # end
