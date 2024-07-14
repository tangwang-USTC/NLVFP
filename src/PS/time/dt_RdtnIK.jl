
"""
  Limit the timestep to satisfy the perturbation assumption of 
  the variables `nIK` of all spices.

  Inputs:
    dtk: = τ₀ / nτ
    dtIK: dtIa, dtKa

  Outputs:
    dtk = dt_RdtnIK(dtk,dtIKk,IKk,nModk1,ns;rtol_DnIK=rtol_DnIK)
"""

# [nMod,KI,ns]
function dt_RdtnIK(dtk::T,dtIKk::AbstractArray{T,N},IKk::AbstractArray{T,N},
    nModk1::Vector{Int64},ns::Int64;rtol_DnIK::T=0.1) where{T<:Real,N}

    # Computing the timestep according to ratio `dtK / K` and `dtI / I`
    RdtM = 0.0
    # @show IKk[1,2,:]
    dtk1 = 1dtk
    for isp in 1:ns
        for k in 1:nModk1[isp]
            #### relative increament: ΔK / K， dtk = min(dtk, rtol_DnIK / (abs(dtIKk) / IKk))
            RdtM = abs(dtIKk[k,1,isp]) / IKk[k,1,isp]
            if RdtM * dtk > rtol_DnIK
                dtk = rtol_DnIK / RdtM
            end

            #### relative increament: ΔI / I
            Iabs = abs(IKk[k,2,isp])
            if Iabs ≥ epsn6
                RdtM = abs(dtIKk[k,2,isp]) / Iabs
                if RdtM * dtk > rtol_DnIK
                    dtk = rtol_DnIK / RdtM
                end
            else
                if Iabs ≥ epsn10
                    RdtM = abs(dtIKk[k,2,isp]) / Iabs
                    if RdtM * dtk > rtol_DnIK
                        dtk = rtol_DnIK / RdtM
                    end
                else
                    @warn("Iabs < epsn10")
                    # dtk = min(dtk, dtk_limit)
                end
            end
        end
    end
    # dtk ≥ dtk1 || @warn("The timestep is decided by `dt_RdtnIK`!")
    dtk ≥ dtk1 || printstyled("1: The timestep is decided by `dtKIa/KIa`!",color=:purple,"\n")
    
    return dtk
end

# [KI,ns]
function dt_RdtnIK(dtk::T,dtIKk::AbstractArray{T,N},Ik::AbstractVector{T},
    Kk::AbstractVector{T},ns::Int64;rtol_DnIK::T=0.1) where{T<:Real,N}

    # Computing the timestep according to ratio `dtK / K` and `dtI / I`
    RdtM = 0.0
    dtk1 = 1dtk
    for isp in 1:ns
        #### relative increament: ΔK / K
        RdtM = abs(dtIKk[1,isp]) / Kk[isp]
        if RdtM * dtk > rtol_DnIK
            dtk = rtol_DnIK / RdtM
        end

        #### relative increament: ΔI / I
        Iabs = abs(Ik[isp])
        if Iabs ≥ epsn6
            RdtM = abs(dtIKk[2,isp]) / Iabs
            if RdtM * dtk > rtol_DnIK
                dtk = rtol_DnIK / RdtM
            end
        else
            if Iabs ≥ epsn10
                RdtM = abs(dtIKk[2,isp]) / Iabs
                if RdtM * dtk > rtol_DnIK
                    dtk = rtol_DnIK / RdtM
                end
            else
                # @warn("Iabs < epsn10")
                # @show fmtf2(Ik[isp])
                # dtk = min(dtk, dtk_limit)
            end
        end
    end
    dtk ≥ dtk1 || printstyled("2: The timestep is decided by `dtKIa/KIa`!",color=:purple,"\n")
    # iujikghyjmn
    return dtk
end

"""
  Limit the timestep to satisfy the perturbation assumption of 
  the variables `nIK` of all sub-components.

  Inputs:
    dtk: = τ₀ / nτ
    dtIKs: [dtIas, dtKas]

  Outputs:
    dtk = dt_RdtnIK2(dtk,dtIKs,uk,IkL, KkL; rtol_DnIK=rtol_DnIK)
"""

# `ns=2` or `nMod=2`, IkL = Vector{Float64,2}
# [IK,2]
function dt_RdtnIK2(dtk::T,dtIKs::AbstractArray{T,N},uk::AbstractVector{T},
    IkL::AbstractVector{T},KkL::AbstractVector{T};rtol_DnIK::T=0.1) where{T<:Real,N}
    
    # Computing the timestep according to ratio `dtK / K` and `dtI / I`
    RdtM = 0.0
    uab = sum(abs.(uk))
    dtk1 = 1dtk
    if uab ≥ atol_u
        dtki = 0.0
        dtk_copy = 1dtk
        Ruab = abs(uk[1] - uk[2]) / uab
        @show dtk,Ruab
        for isp in 1:2 
            @show isp
            #### relative increament: ΔK / K
            RdtM = abs(dtIKs[2,isp]) / KkL[isp]
            if RdtM * dtk_copy > rtol_DnIK
                dtki = rtol_DnIK / RdtM
                @show "K", fmtf4.([dtki,KkL[isp],dtIKs[2,isp]])
                dtk < dtki || (dtk = dtki)
            end

            #### relative increament: ΔI / I, where `ΔI = ∂ₜI * Δt`
            if Ruab ≥ rtol_u
                Iabs = abs(IkL[isp]) + epsT
                if Iabs ≥ epsn6
                    RdtM = abs(dtIKs[1,isp]) / Iabs
                    # @show 6, dtIKs[1,isp], Iabs,RdtM,rtol_DnIK / RdtM
                    if RdtM * dtk_copy > rtol_DnIK
                        dtki = rtol_DnIK / RdtM
                        @show "I6", fmtf4.([dtki,IkL[isp],dtIKs[1,isp]])
                        dtk < dtki || (dtk = dtki)
                    end
                else
                    if Iabs ≥ epsn8
                        RdtM = abs(dtIKs[1,isp]) / Iabs
                        # @show 8, dtIKs[1,isp], Iabs,RdtM,rtol_DnIK / RdtM
                        if RdtM * dtk_copy > rtol_DnIK
                            dtki = rtol_DnIK / RdtM
                            @show "I8", fmtf4.([dtki,IkL[isp],dtIKs[1,isp]])
                            dtk < dtki || (dtk = dtki)
                        end
                    else
                        if Iabs ≥ epsn10
                            # @show 10, dtIKs[1,isp], Iabs,RdtM,rtol_DnIK / RdtM
                            RdtM = abs(dtIKs[1,isp]) / Iabs
                            if RdtM * dtk_copy > rtol_DnIK
                                dtki = rtol_DnIK / RdtM
                                @show "I10", fmtf4.([dtki,IkL[isp],dtIKs[1,isp]])
                                dtk < dtki || (dtk = dtki)
                            end
                        else
                            if abs(dtIKs[1,isp]) ≥ epsn10
                                if Iabs ≥ epsn12
                                    dtk = min(dtk, 0.3 * Iabs / abs(dtIKs[1,isp]))
                                else
                                    dtk = min(dtk, 0.3 * 1e-12 / abs(dtIKs[1,isp]))
                                end
                                dtki = rtol_DnIK / RdtM
                                @show "I12", fmtf4.([dtki,IkL[isp],dtIKs[1,isp]])
                                dtk < dtki || (dtk = dtki)
                            else
                            end
                        end
                    end
                end
                # @show 11, isp, dtk
            else
            end

            #### relative change rate: ∂ₜI / I
        end
        # @show 9999999, dtk
    else
        for isp in 1:2
            RdtM = abs(dtIKs[2,isp]) / KkL[isp]
            if RdtM * dtk > rtol_DnIK
                dtk = rtol_DnIK / RdtM
            end
        end
    end
    dtk ≥ dtk1 || printstyled("3: The timestep is decided by `dtKIa/KIa`!",color=:purple,"\n")
    
    return dtk
end

# [I,K,2]
function dt_RdtnIK2(dtk::T,dtIs::AbstractVector{T},dtKs::AbstractVector{T},uk::AbstractVector{T},
    IkL::AbstractVector{T},KkL::AbstractVector{T};rtol_DnIK::T=0.1) where{T<:Real}
    
    # Computing the timestep according to ratio `dtK / K` and `dtI / I`
    RdtM = 0.0
    uab = sum(abs.(uk))
    dtk1 = 1dtk
    # @show dtk1
    if uab ≥ atol_u
        dtki = 0.0
        dtk_copy = 1dtk
        Ruab = abs(uk[1] - uk[2]) / uab
        @show dtk,Ruab
        for isp in 1:2 
            @show isp
            #### relative increament: ΔK / K
            RdtM = abs(dtKs[isp]) / KkL[isp]
            if RdtM * dtk_copy > rtol_DnIK
                dtki = rtol_DnIK / RdtM
                @show "K", fmtf4.([dtki,KkL[isp],dtKs[isp]])
                dtk < dtki || (dtk = dtki)
            end

            #### relative increament: ΔI / I, where `ΔI = ∂ₜI * Δt`
            if Ruab ≥ rtol_u
                Iabs = abs(IkL[isp])
                if Iabs ≥ epsn6
                    RdtM = abs(dtIs[isp]) / Iabs
                    # @show 6, dtIs[isp], Iabs,RdtM,rtol_DnIK / RdtM
                    if RdtM * dtk_copy > rtol_DnIK
                        dtki = rtol_DnIK / RdtM
                        @show "I6", fmtf4.([dtki,IkL[isp],dtIs[isp]])
                        dtk < dtki || (dtk = dtki)
                    end
                else
                    if Iabs ≥ epsn8
                        RdtM = abs(dtIs[isp]) / Iabs
                        # @show 8, dtIs[isp], Iabs,RdtM,rtol_DnIK / RdtM
                        if RdtM * dtk_copy > rtol_DnIK
                            dtki = rtol_DnIK / RdtM
                            @show "I8", fmtf4.([dtki,IkL[isp],dtIs[isp]])
                            dtk < dtki || (dtk = dtki)
                        end
                    else
                        if Iabs ≥ epsn10
                            # @show 10, dtIs[isp], Iabs,RdtM,rtol_DnIK / RdtM
                            RdtM = abs(dtIs[isp]) / Iabs
                            if RdtM * dtk_copy > rtol_DnIK
                                dtki = rtol_DnIK / RdtM
                                @show "I10", fmtf4.([dtki,IkL[isp],dtIs[isp]])
                                dtk < dtki || (dtk = dtki)
                            end
                        else
                            if abs(dtIs[isp]) ≥ epsn10
                                if Iabs ≥ epsn12
                                    dtk = min(dtk, 0.3 * Iabs / abs(dtIs[isp]))
                                else
                                    dtk = min(dtk, 0.3 * 1e-12 / abs(dtIs[isp]))
                                end
                                dtki = rtol_DnIK / RdtM
                                @show "I12", fmtf4.([dtki,IkL[isp],dtIs[isp]])
                                dtk < dtki || (dtk = dtki)
                            else
                            end
                        end
                    end
                end
                # @show 11, isp, dtk
            else
            end

            #### relative change rate: ∂ₜI / I
        end
        # @show 9999999, dtk
    else
        for isp in 1:2
            RdtM = abs(dtKs[isp]) / KkL[isp]
            if RdtM * dtk > rtol_DnIK
                dtk = rtol_DnIK / RdtM
            end
        end
    end
    dtk == dtk1 || printstyled("4: The timestep is decided by `dtKIa/KIa`!",color=:purple,"\n")
    # @show dtk
    # @show dtk == dtk1
    # uhyjkhj
    return dtk
end

"""
  Outputs:
    IK2_lagrange!(IkL, KkL, ρk, ukL, vthk)
    IK2_lagrange!(Ikl, Kkl, ρkl, ua, vth)
    IKL, KkL = IK_lagrange(ρk,ukL,vthk)
"""
# [ns*nMod = 2]
function IK2_lagrange!(IkL::AbstractVector{T},KkL::AbstractVector{T},
    ρk::AbstractVector{T},ukL::AbstractVector{T},vthk::AbstractVector{T}) where{T}

    for k in 1:2
        IkL[k], KkL[k] = IK_lagrange(ρk[k],ukL[k],vthk[k])
    end
end

# []
function IK_lagrange(ρk::T,ukL::T,vthk::T) where{T}

    return ρk * ukL, 0.5 * ρk * (1.5 * vthk^2 + ukL^2)
    # IkL = ρk * ukL
    # KkL = 0.5 * ρk * (1.5 * vthk^2 + ukL^2)
end
