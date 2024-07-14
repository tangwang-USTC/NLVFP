
"""
  nv0itp,nvremesh,vremesh,nvgauss = remeshvgl(nv1,nitp,nitp_vth,n_vth,vmin,nv0mim,vG)

  Remesh the Gaussian_Laguerre points vG for integration,
      include point vmin  which is tend to zero.

  Candidates for s1: s1(n) = 10^(1/n), for example: s1(n = 45) = 1.0525
      n = [2000, 500,  200,  100,  75,    45,     40,     35,    30,     25,    20,     15,    13,   10]
      s1= [1.001,1.005,1.012,1.023,1.031,1.0525, 1.0593, 1.068, 1.0798, 1.0964, 1.122, 1.166, 1.194, 1.258]

  Inputs:
    nv1 = nv + 1
    nitp = dvG/dvG_new = 2N⁺, the multiple of remeshing for vG
    vmin: the minimum vaule of vremesh owing to vth[:] and vG[1]
    nv0mim:  (=0, default means no remeshing for vG <vG[1]), minimum number of remesh when vG < vG[1]
    vG: Guassian-Laguerre collections
    s1: (=1.2 default), the scale number for remeshing which will limit the ratio v[i]/v[i+1] ≤ s1

  Outputs:
   nv0itp: The number of grids where v < vG[1]
   nvremesh: Total number of the remeshing grids
   vremesh: The remeshing velocity axis grids
   nvgauss: The locations of vG in the new collection, vremesh.

  nv0itp,nvremesh,vremesh,nvgauss = remeshvgs(nv1,vmin,vG;s1=s1,nv0min=1,nitp=nitp)
  nv0itp,nvremesh,vremesh,nvgauss = remeshvgs(nv1,vmin,vG;s1=s1,nv0min=0,nitp=2)
  nv0itp,nvremesh,vremesh,nvgauss = remeshvgs(nv1,vmin,vG;s1=s1,nv0min=1,nitp=2)

"""

function remeshvgs(nv1::Int64,vmin::Real,vG::AbstractVector{T};
        s1::Real=1.1,nv0min::Int=0,nitp::Int=0,nitp_vth::Int=0,n_vth::Real=0) where {T}

    if s1 == 0
        vlog = log.(vG)
        if nv0min == 0
            if nitp == 0
                return 0,nv1,vG,1:nv1 |> Vector
            else
                Nvren,vren,nvgauss = remeshvgn(nitp,nitp_vth,n_vth,vG)
                return 0,Nvren,vren,nvgauss
            end
        else
            if nitp == 0
                nvgauss = 1:nv1 |> Vector
                ########### v < vG[1]
                ratio = (vG[2] - vG[1]) / (vG[1] - vmin)
                s1 = (vG[3] - vG[2]) / (vG[2] - vG[1])
                if ratio ≤ s1
                    return 1,nv1 + 1,[vmin; vG],nvgauss .+ 1
                else
                    nv0itp = 1
                    while ratio > s1
                        nv0itp += 1
                        ratio /= s1
                    end
                    if nv0itp == 2
                        return nv0itp,nv1+nv0itp,[vmin; (vmin+vG[1])/2; vG],nvgauss .+ 2
                    else
                        v01 = zeros(nv0itp)  # Collection of v < vG[1]
                        v01[nv0itp] = vG[1] / s1
                        for i in nv0itp-1:-1:3
                            v01[i] = v01[i+1] / s1
                        end
                        v01[2] = (v01[3] + vmin) / 2
                        v01[1] = vmin
                        return nv0itp,nv1+nv0itp,[v01; vG],nvgauss .+ nv0itp
                    end
                end
            else
                nvremesh,vremesh,nvgauss = remeshvgn(nitp,nitp_vth,n_vth,vG)
                ##################################### v < vG[1]
                ratio = vremesh[1] / vmin
                if ratio ≤ vG[2] / vG[1]
                    return 1,nvremesh+1,[vmin; vremesh],nvgauss .+ 1
                else
                    nv0itp = 1
                    s1 = vG[2] / vG[1]
                    while ratio > s1
                        nv0itp += 1
                        ratio /= s1
                    end
                    if nv0itp == 2
                        return nv0itp,nvremesh+nv0itp,[vmin; (vmin+vremesh[1])/2; vremesh],nvgauss.+nv0itp
                    else
                        v01 = zeros(nv0itp)  # Collection of v < vG[1] = vremesh[1]
                        v01[nv0itp] = vremesh[1] / s1
                        for i in nv0itp-1:-1:3
                            v01[i] = v01[i+1] / s1
                        end
                        v01[2] = (v01[3] + vmin) / 2
                        v01[1] = vmin
                        return nv0itp,nvremesh+nv0itp,[v01; vremesh],nvgauss.+nv0itp
                    end
                end
            end
        end
    else
        1.0001 ≤ s1 || ArgumentError(throw("1.0001 ≤ s1 is recommended, or else no scaling remeshing"))
        vlog = log.(vG)
        if nv0min == 0
            nv0itp = 0  # Collections for v < vG[1]
            if nitp == 0
                ratios = vG[2:nv1] ./ vG[1:nv1-1]
                nvs1 = findfirst(ratios .< s1)
                if nvs1 === nothing
                    nvs1 = nv1 - 1
                elseif nvs1 == 1
                    return nv0itp,nv1,vG,1:nv1 |> Vector
                else
                    nvs1 -= 1
                end
                nCandi = zeros(Int,nvs1)
                for i in 1:1:nvs1
                    if ratios[i] > s1
                        ri = ratios[i]
                        while ri > s1
                            nCandi[i] += 1
                            ri /= s1
                        end
                    else
                        break
                    end
                end
                Nums = sum(nCandi)         # Total number of scaling remeshing for nG
                nvremesh = Nums + nv1
                vremesh = zeros(nvremesh)
                nvgauss = ones(Int,nv1)
                i = 1
                nvgauss[i] = 1
                for i in 1:1:nvs1
                    if nCandi[i] ≥ 2
                        niC = 1:1:nCandi[i] - 1
                        vi = vG[i] * s1 .^ niC
                        vremesh[nvgauss[i]] = vG[i]
                        vremesh[nvgauss[i] .+ niC] = vi
                        vremesh[nvgauss[i] + nCandi[i]] = (vi[nCandi[i] - 1] + vG[i+1]) / 2
                        nvgauss[i+1] = 1 + nvgauss[i] + nCandi[i]
                    else
                        vremesh[nvgauss[i]] = vG[i]
                        vremesh[nvgauss[i] + 1] = (vG[i] + vG[i+1]) / 2
                        nvgauss[i+1] = 2 + nvgauss[i]
                    end
                end
                if nvs1 < nv1
                    for i in nvs1+1:1:nv1-1
                        vremesh[nvgauss[i]] = vG[i]
                        nvgauss[i+1] = 1 + nvgauss[i]
                    end
                    i = nv1
                    vremesh[nvgauss[i]] = vG[i]
                end
                return nv0itp,nvremesh,vremesh,nvgauss
            else
                Nvren,vren,nvgauss = remeshvgn(nitp,nitp_vth,n_vth,vG)
                ratios = vren[2:Nvren] ./ vren[1:Nvren-1]
                # @show ratios[1:3]
                nvs1 = findfirst(ratios .< s1)
                if nvs1 === nothing
                    nvs1 = nv1 - 1
                elseif nvs1 == 1
                    return nv0itp,Nvren,vren,nvgauss
                else
                    nvs1 -= 1
                end
                nCandi = zeros(Int,nvs1)
                for i in 1:1:nvs1
                    if ratios[i] > s1
                        ri = ratios[i]
                        while ri > s1
                            nCandi[i] += 1
                            ri /= s1
                        end
                    else
                        break
                    end
                end
                Nums = sum(nCandi)         # Total number of scaling remeshing for nG
                nvremesh = Nums + Nvren
                vremesh = zeros(nvremesh)
                nvren = ones(Int,Nvren)
                for i in 1:1:nvs1
                    if nCandi[i] ≥ 2
                        niC = 1:1:nCandi[i] - 1
                        vi = vren[i] * s1 .^ niC
                        vremesh[nvren[i]] = vren[i]
                        vremesh[nvren[i] .+ niC] = vi
                        vremesh[nvren[i] + nCandi[i]] = (vi[nCandi[i] - 1] + vren[i+1]) / 2
                        nvren[i+1] = 1 + nvren[i] + nCandi[i]
                    else
                        vremesh[nvren[i]] = vren[i]
                        vremesh[nvren[i] + 1] = (vren[i] + vren[i+1]) / 2
                        nvren[i+1] = 2 + nvren[i]
                    end
                end
                if nvs1 < Nvren
                    for i in nvs1+1:1:Nvren-1
                        vremesh[nvren[i]] = vren[i]
                        nvren[i+1] = 1 + nvren[i]
                    end
                    i = Nvren
                    vremesh[nvren[i]] = vren[i]
                end
                ############## Updating nvgauss
                ig = 1
                for i in 1:1:nvremesh
                    if abs(vremesh[i] - vG[ig]) < 1e-12
                        nvgauss[ig] = i
                        ig += 1
                    end
                end
                return nv0itp,nvremesh,vremesh,nvgauss
            end
        else
            if nitp == 0
                ratios = vG[2:nv1] ./ vG[1:nv1-1]
                nvs1 = findfirst(ratios .< s1)
                if nvs1 === nothing
                    nvs1 = nv1 - 1
                else
                    nvs1 -= 1
                end
                if nvs1 == 0
                    Nums = sum(nCandi)         # Total number of scaling remeshing for nG
                    nvremesh = nv1
                    nvgauss = 1:nv1 |> Vector
                    ########### v < vG[1]
                    ratio = vG[1] / vmin
                    if ratio ≤ s1
                        return 1,nvremesh + 1,[vmin; vremesh],nvgauss .+ 1
                        # vremesh = [vmin; vremesh]
                        # nv0itp = 1
                        # nvremesh += 1
                        # nvgauss .+= 1
                    else
                        nv0itp = 1
                        while ratio > s1
                            nv0itp += 1
                            ratio /= s1
                        end
                        if nv0itp == 2
                            return nv0itp,nvremesh+nv0itp,[vmin; (vmin+vG[1])/2; vG],nvgauss .+ 2
                        else
                            v01 = zeros(nv0itp)  # Collection of v < vG[1]
                            v01[nv0itp] = vG[1] / s1
                            for i in nv0itp-1:-1:3
                                v01[i] = v01[i+1] / s1
                            end
                            v01[2] = (v01[3] + vmin) / 2
                            v01[1] = vmin
                            return nv0itp,nvremesh+nv0itp,[v01; vG],nvgauss .+ nv0itp
                        end
                    end
                else
                    nCandi = zeros(Int,nvs1)
                    for i in 1:1:nvs1
                        if ratios[i] > s1
                            ri = ratios[i]
                            while ri > s1
                                nCandi[i] += 1
                                ri /= s1
                            end
                        else
                            break
                        end
                    end
                    Nums = sum(nCandi)         # Total number of scaling remeshing for nG
                    nvremesh = Nums + nv1
                    vremesh = zeros(nvremesh)
                    nvgauss = ones(Int,nv1)
                    i = 1
                    nvgauss[i] = 1
                    for i in 1:1:nvs1
                        if nCandi[i] ≥ 2
                            niC = 1:1:nCandi[i] - 1
                            vi = vG[i] * s1 .^ niC
                            vremesh[nvgauss[i]] = vG[i]
                            vremesh[nvgauss[i] .+ niC] = vi
                            vremesh[nvgauss[i] + nCandi[i]] = (vi[nCandi[i] - 1] + vG[i+1]) / 2
                            nvgauss[i+1] = 1 + nvgauss[i] + nCandi[i]
                        else
                            vremesh[nvgauss[i]] = vG[i]
                            vremesh[nvgauss[i] + 1] = (vG[i] + vG[i+1]) / 2
                            nvgauss[i+1] = 2 + nvgauss[i]
                        end
                    end
                    if nvs1 < nv1
                        for i in nvs1+1:1:nv1-1
                            vremesh[nvgauss[i]] = vG[i]
                            nvgauss[i+1] = 1 + nvgauss[i]
                        end
                        i = nv1
                        vremesh[nvgauss[i]] = vG[i]
                    end
                    ########### v < vG[1]
                    ratio = vG[1] / vmin
                    if ratio ≤ s1
                        return 1,nvremesh + 1,[vmin; vremesh],nvgauss .+ 1
                    else
                        nv0itp = 1
                        while ratio > s1
                            nv0itp += 1
                            ratio /= s1
                        end
                        if nv0itp == 2
                            return nv0itp,nvremesh+nv0itp,[vmin; (vmin+vG[1])/2; vremesh],nvgauss .+ 2
                        else
                            v01 = zeros(nv0itp)  # Collection of v < vG[1]
                            v01[nv0itp] = vG[1] / s1
                            for i in nv0itp-1:-1:3
                                v01[i] = v01[i+1] / s1
                            end
                            v01[2] = (v01[3] + vmin) / 2
                            v01[1] = vmin
                            return nv0itp,nvremesh+nv0itp,[v01; vremesh],nvgauss .+ nv0itp
                        end
                    end
                end
            else
                Nvren,vren,nvgauss = remeshvgn(nitp,nitp_vth,n_vth,vG)
                ratios = vren[2:Nvren] ./ vren[1:Nvren-1]
                nvs1 = findfirst(ratios .< s1)
                if nvs1 === nothing
                    nvs1 = Nvren - 1
                elseif nvs1 > 1
                    nvs1 -= 1
                end
                nCandi = zeros(Int,nvs1)
                for i in 1:1:nvs1
                    if ratios[i] > s1
                        ri = ratios[i]
                        while ri > s1
                            nCandi[i] += 1
                            ri /= s1
                        end
                    else
                        break
                    end
                end
                Nums = sum(nCandi)         # Total number of scaling remeshing for nG
                nvremesh = Nums + Nvren
                vremesh = zeros(nvremesh)
                nvren = ones(Int,Nvren)
                for i in 1:1:nvs1
                    if nCandi[i] ≥ 2
                        niC = 1:1:nCandi[i] - 1
                        vi = vren[i] * s1 .^ niC
                        vremesh[nvren[i]] = vren[i]
                        vremesh[nvren[i] .+ niC] = vi
                        vremesh[nvren[i] + nCandi[i]] = (vi[nCandi[i] - 1] + vren[i+1]) / 2
                        nvren[i+1] = 1 + nvren[i] + nCandi[i]
                    else
                        vremesh[nvren[i]] = vren[i]
                        vremesh[nvren[i] + 1] = (vren[i] + vren[i+1]) / 2
                        nvren[i+1] = 2 + nvren[i]
                    end
                end
                if nvs1 < Nvren
                    for i in nvs1+1:1:Nvren-1
                        vremesh[nvren[i]] = vren[i]
                        nvren[i+1] = 1 + nvren[i]
                    end
                    i = Nvren
                    vremesh[nvren[i]] = vren[i]
                end
                ############## Updating nvgauss
                ig = 1
                for i in 1:1:nvremesh
                    if abs(vremesh[i] - vG[ig]) < 1e-12
                        nvgauss[ig] = i
                        ig += 1
                    end
                end
                ##################################### v < vG[1]
                ratio = vremesh[1] / vmin
                if ratio ≤ s1
                    vremesh = [vmin; vremesh]
                    nv0itp = 1
                    nvremesh += 1
                    nvgauss .+= 1
                else
                    nv0itp = 1
                    while ratio > s1
                        nv0itp += 1
                        ratio /= s1
                    end
                    if nv0itp == 2
                        nvremesh += 2
                        nvgauss .+= 2
                        vremesh = [vmin; (vmin+vremesh[1])/2; vremesh]
                    else
                        v01 = zeros(nv0itp)  # Collection of v < vG[1] = vremesh[1]
                        v01[nv0itp] = vremesh[1] / s1
                        for i in nv0itp-1:-1:3
                            v01[i] = v01[i+1] / s1
                        end
                        v01[2] = (v01[3] + vmin) / 2
                        v01[1] = vmin
                        nvremesh += nv0itp
                        nvgauss .+= nv0itp
                        vremesh = [v01; vremesh]
                    end
                end
                return nv0itp,nvremesh,vremesh,nvgauss
            end
        end
    end
end

"""
  Remeshing Gaussion collection, vG, with interpolation method based on

  Inputs:
    nitp: Interpolaion number between vG[i] and vG[i+1] for all domain vG[1:nv1]
    n_vth: Refine domian [1/n_vth, n_vth] near the grid v̂ = 1
    nitp_vth: Number of interpolation for n_vth

  Outputs:
    nvremesh,vremesh,nvgauss = remeshvgn(nitp,nitp_vth,n_vth,vG)

"""

function remeshvgn(nitp::Int,nitp_vth::Int,n_vth::Real,vG::AbstractVector{T}) where {T}

    isinteger(nitp/2) || ArgumentError(throw("nitp should be 2N⁺"))
    vlog = log.(vG)
    dvlog = diff(vlog)/nitp
    nv = nv1 - 1
    isv0itp = false
    vlogitp = zeros(nv*(nitp-1))
    ######## interpolations for vG where vG[1:nv1] by nitp
    nvc = 1:nv        # 1:nv1-1
    ## v[0.5:nv1-0.5]
    for i in 1:nitp-1
        vlogitp[(i:(nitp-1):((nitp-1)*nv))] = vlog[nvc] + i * dvlog
    end
    nvitp = nv*(nitp-1)
    ############################# remeshing for v = [v,vitp(nitp)]
    nvremesh = nv1 + nvitp
    vlogremesh = zeros(T, nvremesh)
    vlogremesh[1:nitp:nvremesh] = vlog # v = vG[1:nv1]
    for i in 2:nitp
        vlogremesh[i:nitp:(nitp*nv)] = vlogitp[(i-1):(nitp-1):((nitp-1)*nv)]
    end
    ################## remeshing for v near v̂ ~ 1
    if nitp_vth == 0
        vremesh = exp.(vlogremesh)
        nvgauss = 1:nitp:nvremesh  |> collect
        vremesh[nvgauss] = vG
        return nvremesh,vremesh,nvgauss
    else
        nv_v05 = findfirst(vlogremesh .≥ log(1/n_vth))
        if nv_v05 === nothing
            nv_v05 == 1
        end
        nv_v2 = findfirst(vlogremesh .≥ log(n_vth))
        if nv_v05 < nv_v2
            dv12 = vlogremesh[nv_v05+1:nv_v2] - vlogremesh[nv_v05:nv_v2-1]
            dv12re = dv12 / nitp_vth
            Nitp = (nv_v2 - nv_v05) * (nitp_vth-1)
            vremesh = zeros(T,nvremesh + Nitp)
            vremesh[1:nv_v05-1] = vlogremesh[1:nv_v05-1]
            vremesh[nv_v2+Nitp:nvremesh+Nitp] = vlogremesh[nv_v2:nvremesh]
            Nv12 = Nitp + nv_v2 - nv_v05
            for i = 1:nitp_vth
                vremesh[nv_v05 - 1  .+ (i:nitp_vth:Nv12)] = vlogremesh[nv_v05:nv_v2-1] + (i-1) * dv12re
            end
            vremesh = exp.(vremesh)
            nvremesh = nvremesh + Nitp
            nvgs1 = 1:nitp:nv_v05-1
            ngs0 = findfirst(abs.(vremesh .- v[length(nvgs1)+1]) .< eps(Float64))
            nvgs2 = ngs0:nitp+nitp_vth:nv_v2+Nitp-1
            ngs0 = findfirst(abs.(vremesh .- v[length(nvgs1) + length(nvgs2) .+ 1]) .< eps(Float64))
            nvgauss = [nvgs1;nvgs2;ngs0:nitp:nvremesh]
            vremesh[nvgauss] = vG
        else
        end
        return nvremesh,vremesh,nvgauss
    end
end
