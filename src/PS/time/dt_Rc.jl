
"""
  Limit the timestep to satisfy the perturbation assumption of 
  the variables `nIK` of all spices.

  Inputs:
    dtk: = τ₀ / nτ
    dtIK: dtIa, dtKa

  Outputs:
    dtk = dt_Rc(dtk,Rc,Mc,LMk,nModk,nMjMs,ns;
                rtol_DnIK=rtol_DnIK,dtk_order_Rc=dtk_order_Rc)
"""

# [nMod,nj,LM1,ns]
function dt_Rc(dtk::T,Rc::AbstractArray{T,N},Mc::AbstractArray{T,N},
    LMk::Vector{Int64},nModk::Vector{Int64},nMjMs::Vector{Int64},ns::Int64;
    rtol_DnIK::T=0.1,dtk_order_Rc::Symbol=:mid,) where{T<:Real,N}

    dtk1 = 1dtk
    for isp in 1:ns
        if dtk_order_Rc == :max
            jMax = min(2nModk[isp], nMjMs[isp])
        else # if dtk_order_Rc == :mid
            if nModk[isp] == 1
                jMax = 2
            else
                jMax = min(nModk[isp]+1, nMjMs[isp])
                if isodd(jMax)
                    jMax += 1
                end
            end
        end
        for L1 in 1:LMk[isp]
            if isodd(L1)
                for j in 2:2:jMax
                    nj = j / 2 + 1 |> Int64
                    if Mc[nj,L1,isp] ≥ epsT1000
                        RdtM = Rc[nj,L1,isp] / Mc[nj,L1,isp]
                        if RdtM * dtk > rtol_DnIK
                            dtk = rtol_DnIK / RdtM
                        end
                    end
                end
            else
                for j in 2:2:jMax
                    nj = j / 2 + 1|> Int64
                    if abs(Mc[j,L1,isp]) ≥ epsT1000
                        RdtM = Rc[nj,L1,isp] / Mc[nj,L1,isp]
                        if RdtM * dtk > rtol_DnIK
                            dtk = rtol_DnIK / RdtM
                        end
                    end
                end
            end
        end
        if dtk < dtk1
            @warn("dt_Rc: The timestep is decided by dtRc,",isp)
        end
        asdfvbnm
    end
    return dtk
end