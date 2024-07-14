
"""
  Updating the M-functions for the `s = 1` stage in the implicit Euler method according to the relations

    Mhcsd2lk1 = Ratio_nT * (Mhcsd2lk + Δₜ * Rhcsd2lk)
      Ratio_nT = (ρk[isp] / ρk1[isp]) * (vthk[isp] / vthk1[isp])
  
  Inputs:
  Outputs:
    Mtheorems_fMab!(Mhcsd2l,dtfvL0k,vhe,ma,na,vthk,vthk1,
            uai,ns,nMod;is_MjMs_max=is_MjMs_max)
    Mtheorems_fMab!(Mhcsd2l,dtfvL0k,vhe,ma,na,vthk,
            uai,ns,nMod;is_MjMs_max=is_MjMs_max)
"""

# Mhcsd2l, 
function Mtheorems_fMab!(Mhcsd2l::Vector{Any},fvL0k::AbstractVector{Matrix{T}},
    vhe::AbstractVector{StepRangeLen},uai::Vector{AbstractVector{T}},ns::Int64,nMod::Vector{Int};
    is_MjMs_max::Bool=false) where{T}

    if is_MjMs_max
        nModmax1 = maximum(nMod) + 1
    end
    is_MjMs_max ? nMjMs = nModmax1 : nMjMs = nMod[isp] + 1
    for isp in 1:ns
        if norm(uai[isp]) ≤ epsT10
            Mhcsd2l[isp] = MsnnEvens(zeros(nMjMs),fvL0k[isp][:,1],vhe[isp],nMjMs,0;is_renorm=true)
        else
            # nMjMs = ceil(Int,2nMod[isp] / 2)     # is_nai_const = true
            Mhcsd2l01 = zeros(2nMjMs - 1)
            Mhcsd2l01[1:2:end] = MsnnEvens(Mhcsd2l01[1:2:end],fvL0k[isp][:,1],vhe[isp],nMjMs,0;is_renorm=true)
            Mhcsd2l01[2:2:end] = MsnnEvens(Mhcsd2l01[2:2:end],fvL0k[isp][:,2],vhe[isp],nMjMs-1,1;is_renorm=true)
            Mhcsd2l[isp] = deepcopy(Mhcsd2l01)
        end
    end
end

# Mhcsd2l, Msab
function Mtheorems_fMab!(Mhcsd2l::Vector{Any},fvL0k::AbstractVector{Matrix{T}},
    vhe::AbstractVector{StepRangeLen},ma::AbstractVector{T},na::AbstractVector{T},
    vthk::AbstractVector{T},uai::Vector{AbstractVector{T}},
    ns::Int64,nMod::Vector{Int};is_MjMs_max::Bool=false) where{T}

    if is_MjMs_max
        nModmax1 = maximum(nMod) + 1
    end
    for isp in 1:ns
        is_MjMs_max ? nMjMs = nModmax1 : nMjMs = nMod[isp] + 1
        if norm(uai[isp]) ≤ epsT10
            Mhcsd2l[isp] = MsnnEvens(zeros(nMjMs),fvL0k[isp][:,1],vhe[isp],nMjMs,0;is_renorm=true)
        else
            # nMjMs = ceil(Int,2nMod[isp] / 2)     # is_nai_const = true
            Mhcsd2l01 = zeros(2nMjMs - 1)
            Mhcsd2l01[1:2:end] = MsnnEvens(Mhcsd2l01[1:2:end],fvL0k[isp][:,1],vhe[isp],nMjMs,0;is_renorm=true)
            Mhcsd2l01[2:2:end] = MsnnEvens(Mhcsd2l01[2:2:end],fvL0k[isp][:,2],vhe[isp],nMjMs-1,1;is_renorm=true)
            Mhcsd2l[isp] = deepcopy(Mhcsd2l01)
        end
    end
    if is_MjMs_max
        Mhcsd2l[end] = zeros(T,nModmax1)
        if ns == 2
            Mhcsj_fM!(Mhcsd2l[end],ma,na,vthk,nModmax1,[Mhcsd2l[1] Mhcsd2l[2]])
        elseif ns == 3
            Mhcsj_fM!(Mhcsd2l[end],ma,na,vthk,nModmax1,[Mhcsd2l[1] Mhcsd2l[2] Mhcsd2l[3]])
        elseif ns == 4
            Mhcsj_fM!(Mhcsd2l[end],ma,na,vthk,nModmax1,[Mhcsd2l[1] Mhcsd2l[2] Mhcsd2l[3] Mhcsd2l[4]])
        elseif ns == 5
            Mhcsj_fM!(Mhcsd2l[end],ma,na,vthk,nModmax1,[Mhcsd2l[1] Mhcsd2l[2] Mhcsd2l[3] Mhcsd2l[4] Mhcsd2l[5]])
        else
            gbhjkl
        end
    else
        gghjjjkjks
    end
end


# Mhcsd2l + Rhcsd2l, Msab
function Mtheorems_fMab!(Mhcsd2lk::Vector{Any},dtfvL0k::AbstractVector{Matrix{T}},
    vhe::AbstractVector{StepRangeLen},ma::AbstractVector{T},na::AbstractVector{T},
    vthk::AbstractVector{T},vthk1::AbstractVector{T},uai::Vector{AbstractVector{T}},
    ns::Int64,nMod::Vector{Int};is_MjMs_max::Bool=false) where{T}

    ddddddd
    if is_MjMs_max
        nModmax1 = maximum(nMod) + 1
    end
    Mhcsd2l = deepcopy(Mhcsd2lk)
    Rhcsd2l = similar(Mhcsd2l)
    for isp in 1:ns
        # Ratio_nT = vthk[isp] / vthk1[isp]             # (ρk[isp] / ρk1[isp]) * (vthk[isp] / vthk1[isp])
        if norm(uai[isp]) ≤ epsT10
            if is_MjMs_max
                nMjMs = nModmax1
            else
                nMjMs = nMod[isp] + 1
            end
            Rhcsd2l[isp] = dtMsnnEvens(zeros(nMjMs),dtfvL0k[isp][:,1],vhe[isp],nMjMs,0;is_renorm=true)
            Mhcsd2l[isp] += Rhcsd2l[isp]
        else
            tyhkjm
            if is_MjMs_max
                nMjMs = nModmax1
            else
                nMjMs = nMod[isp] + 1
            end
            # nMjMs = ceil(Int,2nMod[isp] / 2)     # is_nai_const = true
            Rhcsd2l01 = zeros(2nMjMs - 1)
            Rhcsd2l01[1:2:end] = dtMsnnEvens(Rhcsd2l01[1:2:end],dtfvL0k[isp][:,1],vhe[isp],nMjMs,0;is_renorm=true)
            Rhcsd2l01[2:2:end] = dtMsnnEvens(Rhcsd2l01[2:2:end],dtfvL0k[isp][:,2],vhe[isp],nMjMs-1,1;is_renorm=true)
            Rhcsd2l[isp] = deepcopy(Rhcsd2l01)
        end
    end
    if is_MjMs_max
        Mhcsd2l[end] = zeros(T,nModmax1)
        if ns == 2
            Mhcsj_fM!(Mhcsd2l[end],ma,na,vthk1,nModmax1,[Mhcsd2l[1] Mhcsd2l[2]])
            Mhcsj_fM!(Mhcsd2l[end],ma,na,vthk,nModmax1,[Mhcsd2l[1] Mhcsd2l[2]])
        elseif ns == 3
            Mhcsj_fM!(Mhcsd2l[end],ma,na,vthk,nModmax1,[Mhcsd2l[1] Mhcsd2l[2] Mhcsd2l[3]])
        elseif ns == 4
            Mhcsj_fM!(Mhcsd2l[end],ma,na,vthk,nModmax1,[Mhcsd2l[1] Mhcsd2l[2] Mhcsd2l[3] Mhcsd2l[4]])
        elseif ns == 5
            Mhcsj_fM!(Mhcsd2l[end],ma,na,vthk,nModmax1,[Mhcsd2l[1] Mhcsd2l[2] Mhcsd2l[3] Mhcsd2l[4] Mhcsd2l[5]])
        else
            gbhjkl
        end
    else
        gghjjjkjks
    end
end
