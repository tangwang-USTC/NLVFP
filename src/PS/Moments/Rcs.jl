"""
    The kinetic dissipations owing to the Fokker-Planck collision term.

    Outputs:
      Rcsd2l!(Rc,errRhc,dtfvL,vhe,nMjMs,ma,na,vth,LM,ns;
            is_renorm=is_renorm,is_norm_error=is_norm_error)
"""

function Rcsd2l!(Rc::AbstractArray{T,N},errRhc::AbstractArray{T,N},dtfvL::AbstractVector{Matrix{T}},
    vhe::AbstractVector{StepRangeLen},nMjMs::Vector{Int64},ρa::AbstractVector{T},
    vth::AbstractVector{T},LM::Vector{Int},ns::Int64;
    is_renorm::Bool=true,is_norm_error::Bool=true) where{T,N}
    
    for isp in 1:ns
        a = Rc[:,:,isp]
        erra = errRhc[:,:,isp]
        RhnnEvens!(a,erra,dtfvL[isp],vhe[isp],nMjMs[isp],LM[isp];is_renorm=is_renorm) # Rh
        # @show erra
        # @show a
        # # Rc[:,:,isp] = a

        if is_norm_error
            for L in 0:LM[isp]
                L1 = L + 1
                if norm(a[:,L1]) ≥ epsT1000
                    errRhc[:,L1,isp] = erra[:,L1]
                    for nj in 1:nMjMs[isp]
                        j = L + 2(nj - 1)
                        Rc[nj,L1,isp] = a[nj,L1] * (ρa[isp] * vth[isp]^j)
                        # @show L,j,Rc[nj,L1,isp]
                        # @show (ρa[isp] * vth[isp]^j)
                    end
                else
                    Rc[:,L1,isp] .= 0.0
                    errRhc[:,L1,isp] .= 0.0
                end
            end
        else
            for L in 0:LM[isp]
                L1 = L + 1
                if norm(a[:,L1]) ≥ epsT1000
                    for nj in 1:nMjMs[isp]
                        j = L + 2(nj - 1)
                        w = ρa[isp] * vth[isp]^j
                        Rc[nj,L1,isp] = a[nj,L1] * w
                        errRhc[nj,L1,isp] = erra[nj,L1] * w
                    end
                else
                    Rc[:,L1,isp] .= 0.0
                    errRhc[:,L1,isp] .= 0.0
                end
            end
        end
    end
end

function Rcsd2l!(Rc::AbstractArray{T,N},errRhc::AbstractArray{T,N},dtfvL::AbstractVector{Matrix{T}},
    vhe::Vector{AbstractVector{T}},nvG::Vector{Int64},nMjMs::Vector{Int64},ρa::AbstractVector{T},
    vth::AbstractVector{T},LM::Vector{Int},ns::Int64;
    is_renorm::Bool=true,is_norm_error::Bool=true) where{T,N}
    
    for isp in 1:ns
        a = Rc[:,:,isp]
        erra = errRhc[:,:,isp]
        RhnnEvens!(a,erra,dtfvL[isp],vhe[isp],nvG[isp],nMjMs[isp],LM[isp];is_renorm=is_renorm) # Rh
        if is_norm_error
            for L in 0:LM[isp]
                L1 = L + 1
                if norm(a[:,L1]) ≥ epsT1000
                    errRhc[:,L1,isp] = erra[:,L1]
                    for nj in 1:nMjMs[isp]
                        j = L + 2(nj - 1)
                        Rc[nj,L1,isp] = a[nj,L1] * (ρa[isp] * vth[isp]^j)
                    end
                else
                    Rc[:,L1,isp] .= 0.0
                    errRhc[:,L1,isp] .= 0.0
                end
            end
        else
            for L in 0:LM[isp]
                L1 = L + 1
                if norm(a[:,L1]) ≥ epsT1000
                    for nj in 1:nMjMs[isp]
                        j = L + 2(nj - 1)
                        w = ρa[isp] * vth[isp]^j
                        Rc[nj,L1,isp] = a[nj,L1] * w
                        errRhc[nj,L1,isp] = erra[nj,L1] * w
                    end
                else
                    Rc[:,L1,isp] .= 0.0
                    errRhc[:,L1,isp] .= 0.0
                end
            end
        end
    end
end

function Rcsd2l!(Rc::AbstractArray{T,N},dtfvL::AbstractVector{Matrix{T}},
    vhe::Vector{AbstractVector{T}},nvG::Vector{Int64},nMjMs::Vector{Int64},ρa::AbstractVector{T},
    vth::AbstractVector{T},LM::Vector{Int},ns::Int64;
    is_renorm::Bool=true,is_norm_error::Bool=true) where{T,N}
    dddddd
    for isp in 1:ns
        a = Rc[:,:,isp]
        RhnnEvens!(a,dtfvL[isp],vhe[isp],nvG[isp],nMjMs[isp],LM[isp];is_renorm=is_renorm) # Rh
        for L in 0:LM[isp]
            L1 = L + 1
            for nj in 1:nMjMs[isp]
                j = L + 2(nj - 1)
                Rc[nj,L1,isp] = a[nj,L1] * (ρa[isp] * vth[isp]^j)
            end
        end
    end
end

function Rcsd2l!(Rc::AbstractArray{T,N},errRhc::AbstractArray{T,N},dtfvL::AbstractVector{Matrix{T}},
    vhe::AbstractVector{T},nvG::Int64,nMjMs::Vector{Int64},ρa::AbstractVector{T},
    vth::AbstractVector{T},LM::Vector{Int},ns::Int64;
    is_renorm::Bool=true,is_norm_error::Bool=true) where{T,N}
    gggggg
    for isp in 1:ns
        a = Rc[:,:,isp]
        erra = errRhc[:,:,isp]
        RhnnEvens!(a,erra,dtfvL[isp],vhe,nvG,nMjMs[isp],LM[isp];is_renorm=is_renorm) # Rh
        if is_norm_error
            errRhc[:,:,isp] = erra
            for L in 0:LM[isp]
                L1 = L + 1
                for nj in 1:nMjMs[isp]
                    j = L + 2(nj - 1)
                    Rc[nj,L1,isp] = a[nj,L1] * (ρa[isp] * vth[isp]^j)
                end
            end
        else
            for L in 0:LM[isp]
                L1 = L + 1
                for nj in 1:nMjMs[isp]
                    j = L + 2(nj - 1)
                    w = ρa[isp] * vth[isp]^j
                    Rc[nj,L1,isp] = a[nj,L1] * w
                    errRhc[nj,L1,isp] = erra[nj,L1] * w
                end
            end
        end
    end
end

function Rcsd2l!(Rc::AbstractArray{T,N},errRhc::AbstractArray{T,N},dtfvL::AbstractVector{Matrix{T}},
    vhe::AbstractVector{T},nMjMs::Vector{Int64},ρa::AbstractVector{T},
    vth::AbstractVector{T},LM::Vector{Int},ns::Int64;
    is_renorm::Bool=true,is_norm_error::Bool=true) where{T,N}
    jjjjjj
    for isp in 1:ns
        a = Rc[:,:,isp]
        erra = errRhc[:,:,isp]
        RhnnEvens!(a,erra,dtfvL[isp],vhe,nMjMs[isp],LM[isp];is_renorm=is_renorm) # Rh
        if is_norm_error
            errRhc[:,:,isp] = erra
            for L in 0:LM[isp]
                L1 = L + 1
                for nj in 1:nMjMs[isp]
                    j = L + 2(nj - 1)
                    Rc[nj,L1,isp] = a[nj,L1] * (ρa[isp] * vth[isp]^j)
                end
            end
        else
            for L in 0:LM[isp]
                L1 = L + 1
                for nj in 1:nMjMs[isp]
                    j = L + 2(nj - 1)
                    w = ρa[isp] * vth[isp]^j
                    Rc[nj,L1,isp] = a[nj,L1] * w
                    errRhc[nj,L1,isp] = erra[nj,L1] * w
                end
            end
        end
    end
end

"""
    The kinetic dissipations owing to the Fokker-Planck collision term.

    Outputs:
      Rcsd2l!(Rc,Rhc,nMjMs,ρa,vth,LM,ns)
      RRcsd2l!(RRcs,Rc,njMs,ρa,Ia,Ka)
"""

function Rcsd2l!(Rc::AbstractArray{T,N},Rhc::AbstractArray{T,N},nMjMs::Vector{Int64},
    ρa::AbstractVector{T},vth::AbstractVector{T},LM::Vector{Int},ns::Int64) where{T,N}
  
    for isp in 1:ns
        for L in 0:LM[isp]
            L1 = L + 1
            for nj in 1:nMjMs[isp]
                j = L + 2(nj - 1)
                Rc[nj,L1,isp] = Rhc[nj,L1,isp] * (ρa[isp] * vth[isp]^j)
            end
        end
    end
end

function RRcsd2l!(RRcs::AbstractArray{T,N2},Rc::AbstractArray{T,N},njMs::Int64,
    ρa::AbstractVector{T},Ia::AbstractVector{T},Ka::AbstractVector{T}) where{T,N,N2}
    
    ρs = sum_kbn(ρa)
    # nas = sum_kbn(na)
    # nhS = na / nas
    # ms = ρs / nas
    # mhS = ma / ms
    # ρhS = ρa / ρs
    Is = sum_kbn(Ia)
    us = Is ./ ρs
    Ks = sum_kbn(Ka)
    vSth = (2 / 3 * (2Ks ./ ρs - us^2))
    # dtvSth = 2/3 / (vSth * ρs) * (sum_kbn(dtKa) - Is / ρs * sum_kbn(dtIa))
    # RdtvSth = dtvSth / vSth
    RRcs[:,:] = sum(Rc;dims=3)[:,:,1]         # Rcs
    for LL1 in 1:LM1
        LL = LL1 - 1
        for nj in 1:njMs
            # j = LL + 2(nj - 1)
            # RRcs[nj,LL1] /=  ρs * vSth^(j)


            # RRcs[nj,LL1] /=  ρs * vSth^(nj)
            RRcs[nj,LL1] /=  ρs * vSth^(nj + LL)
            # RRcs[nj,LL1] /=  ρs * vSth^(nj + LL/2)
            # RRcs[nj,LL1] /=  ρs * vSth^(nj + LL)

            # RRcs[nj,LL1] /=  ρs * vSth^((j + LL))
            # RRcs[nj,LL1] /=  ρs * vSth^((j + LL)/2)
            # RRcs[nj,LL1] /=  ρs * vSth^(j/2)
            # RRcs[nj,LL1] /=  ρs * vSth^(j/2 + LL)
            
            # RRcs[:,LL1] ./= abs(RRcs[end,LL1])
        end
    end
end

"""
    The kinetic dissipations owing to the Fokker-Planck collision term.

    Outputs:
      dtMhcsd2l!(dtMhc,Mhc,Rhc,w3k,nMjMs,LM,ns)
"""

function dtMhcsd2l!(dtMhc::AbstractArray{TA,N},Mhc::AbstractArray{TA,N},Rhc::AbstractArray{TA,N},
    w3k::AbstractVector{T},nMjMs::Vector{Int64},LM::Vector{Int},ns::Int64) where{T,TA,N}
  
    for isp in 1:ns
        for L in 0:LM[isp]
            L1 = L + 1
            for nj in 1:nMjMs[isp]
                j = L + 2(nj - 1)
                dtMhc[isp][nj,L1] = Rhc[isp][nj,L1] - j * w3k[isp] * Mhc[isp][nj,L1]
            end
        end
    end
end

