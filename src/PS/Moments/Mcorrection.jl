
"""
  `Mc3correction!` procedure is implemented to correct the conservation moments `I, K` 
  according to conservation laws during Fokker-Planck collision process.

  Inputs:
  Outputs:
    Mconservations!(Ik1,Kk1,Ik,Kk;atol_IKTh=atol_IKTh,
        reltol_DMs=reltol_DMs,reltol_DMs_err=reltol_DMs_err,
        is_corrections=is_corrections,residualMethod_FP0D=residualMethod_FP0D)
"""

function Mconservations!(Ik1::AbstractVector{T},Kk1::AbstractVector{T},
    Ik::AbstractVector{T},Kk::AbstractVector{T};
    atol_IKTh::T=epsT1000,reltol_DMs::T=5e-3,reltol_DMs_err::T=1e-1,
    is_corrections::Vector{Bool}=[true,false,false],residualMethod_FP0D::Int64=1) where{T}
    
    @show sum(Kk1) / sum(Kk) - 1
    if norm(IK1) + norm(Ik) ≤ atol_IKTh
        if is_corrections[2]
            IK1 .= 0.0
        end
    else
        if Iab0 ≤ epsTe6
            Iks = sum(abs.(Ik1))
            if Iks  ≤ epsTe6
                @show 0,sum(Ik1) - Iab0
            else
                @show 0,(sum(Ik1) - Iab0) / Iks
            end
        elseif norm(Ik1) ≥ epsT100
            @show 0, sum(Ik1) / Iab0 - 1
        end
        
        DIk = Ik1 - Ik
        errDIk = abs(DIk[1] - DIk[2])
        if errDIk ≤ epsT 
            RerrDIk = abs(sum(DIk))
            if norm(DIk) > atol_IKTh
                if RerrDIk > atol_IKTh
                    if RerrDIk > 0.01
                        @warn("The time step is so large that the convergence and efficiency of the optimization process of `fvL` may be poor!", RerrDIk)
                    end
                    if RerrDIk > reltol_DMs 
                        printstyled("`RerrDIk ≥ 1e-1` which means the algorithm is not convergent!!!",color=:red,"\n")
                    end
                end
            end
            # Corrections to satisfy the conservation laws by applying a posterior analysis.
            # Updating the parameters `Ihk` according to the `I` at `kᵗʰ` step
            if is_corrections[2]
              if RerrDIk > epsT
                  if residualMethod_FP0D == 1
                      if RerrDIk > epsT
                          DIk .-= sum(DIk) / 2
                          Ik1 = Ik + DIk
                      end
                  elseif residualMethod_FP0D == 2
                      ertyui
                  end
              end
            end
        else
            RerrDIk = abs(sum(DIk)) / errDIk
            if norm(DIk) > atol_IKTh
                if RerrDIk > atol_IKTh
                    if RerrDIk > 0.01
                        @warn("The time step is so large that the convergence and efficiency of the optimization process of `fvL` may be poor!", RerrDIk)
                    end
                    if RerrDIk > reltol_DMs 
                        printstyled("`RerrDIk ≥ 1e-1` which means the algorithm is not convergent!!!",color=:red,"\n")
                    end
                end
            end
            # Corrections to satisfy the conservation laws by applying a posterior analysis.
            # Updating the parameters `Ihk` according to the `I` at `kᵗʰ` step
            if is_corrections[2]
              if RerrDIk > epsT
                  if residualMethod_FP0D == 1
                      if RerrDIk > epsT
                          DIk .-= sum(DIk) / 2
                          Ik1 = Ik + DIk
                      end
                  elseif residualMethod_FP0D == 2
                      ertyui
                  end
              end
            end
        end
    end

    DKk = Kk1 - Kk
    errDKk = abs(DKk[1] - DKk[2])
    if errDKk ≤ epsT
        RerrDKk = abs(sum(DKk))
    else
        RerrDKk = abs(sum(DKk)) / errDKk
    end
    if norm(DKk) > atol_IKTh
        if RerrDKk > atol_IKTh
            if RerrDKk > 0.01
                @warn("The time step is so large that the convergence and efficiency of the optimization process of `fvL` may be poor!", RerrDKk)
            end
            if RerrDKk > reltol_DMs 
                printstyled("`RerrDKk ≥ 1e-1` which means the algorithm is not convergent!!!",color=:red,"\n")
            end
        end
    end
    # Updating the parameters `Ka` according to the first three moments `K` at `kᵗʰ` step
    if is_corrections[3]
        if RerrDKk > epsT
            if residualMethod_FP0D == 1
                if RerrDKk > epsT
                    DKk .-= sum(DKk) / 2
                    Kk1 = Kk + DKk
                end
            elseif residualMethod_FP0D == 2
                yujk
            end
        end
    end

    RDIk = DIk ./ Ik
    RDKk = DKk ./ Kk
    @show fmtf2.(DKk), fmtf2.(DIk)
    @show fmtf2.(RDIk), fmtf2.(RDKk)
    @show fmtf2.(sum(DKk)), fmtf2.(sum(DIk))
    @show fmtf2.(RerrDIk), fmtf2.(RerrDKk)

end

# if is_corrections[2] == true
#     for isp in nsp_vec
#         Mck1[1, 2, isp] = Ik1[isp]
#     end
# end

# @show sum(Kk1) / Kab0 - 1
# if is_corrections[3] == true
#     for isp in nsp_vec
#         Mck1[2, 1, isp] = Kk1[isp] / CMcKa
#     end
# end
      

function RDnIKTcheck(;reltol_DMs::T=5e-3,reltol_DMs_err::T=1e-1) where{T}

    RdMsk[2,:] = Kk1 ./ Kk .- 1
    if norm(Ik) ≤ epsTe6
        # if norm(Ik) ≤ epsT
        # else 
        #     # @show Ik1 - Ik
        # end
    else
        # RDIk1 = Ik1 ./ (Ik .+ epsT) .- 1
        RdMsk[1,:] = Ik1 ./ (Ik .+ epsT) .- 1
    end
    RdMsk[3,:] = abs.(vthk1 ./ vthk .- 1)
    
    # Checking for the quantities of `Rvthk1`, `RKak1` and `RIak1`
    if is_checking_nIKT_dt
        # Checking for the quantities of `RDvthk1`
        if norm(RdMsk[3,:]) ≥ reltol_DMs
            @warn("The change of `Rdvthk ≥ reltol_DMs` is so big, reduce `dtk` please!",RdMsk[3,:])
            if norm(RdMsk[3,:]) ≥ reltol_DMs_err
                @error("The convergence criterion of the algorithm will be false owing the `Rdvthk ≥ reltol_DMs_err`!!!")
            end
        end
    
        # Checking for the quantities of `RDKk1`
        if norm(RdMsk[2,:]) ≥ reltol_DMs
            @warn("The change of `RDKk1 ≥ reltol_DMs` is so big, reduce `dtk` please!",RdMsk[2,:])
            if norm(RdMsk[2,:]) ≥ reltol_DMs_err
                @error("The convergence criterion of the algorithm will be false owing the `RDKk1 ≥ reltol_DMs_err`!!!")
            end
        end
    
        # Checking for the quantities of `RDIk1`
        if norm(Ik) ≤ epsTe6
        else
            if norm(RdMsk[1,:]) ≥ reltol_DMs
                @warn("The change of `RDIk1 ≥ reltol_DMs` is so big, reduce `dtk` please!",RdMsk[1,:])
                if norm(RdMsk[1,:]) ≥ reltol_DMs_err
                    @error("The convergence criterion of the algorithm will be false owing the `RDIk1 ≥ reltol_DMs_err`!!!")
                end
            end
        end
    end
end

"""
  `MXcorrection!` procedure is implemented to correct the transfer along coarse-fine grid interfaces,
  to ensure that the amount of any conserved quantity leaving one cell
  exactly balances the amount entering the bordering cell.

"""