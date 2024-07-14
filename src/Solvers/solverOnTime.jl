

"""
  The solver of ODE about time `t`

    ∂ₜf(t) = g(f(t))

  Inputs:

  Outputs:
    alg = solverOnTime()
"""

function solverOnTime()

    alg = VCAB3()      # St =307.1s, RdK ∈ [1e-2 ~ 1e-3], 1e-2, stability
    # alg = Midpoint()   # St =171.9s, RdK ∈ [1e-2 ~ 1e-3], 1e-2,  good ************
    # alg = Heun()       # St =176.7s, RdK ∈ [1e-1 ~ 1e-2], 1e-2
    # # alg = Ralston()   # St =182.5s, RdK ∈ [1e-2 ~ 1e-3], 1e-2 *******
    # alg = RK4()        # 大量回代,                               worse
    # alg = BS3()        # St =188.2s, RdK ∈ [1e-1 ~ 1e-4], 1e-2
    # if alg == BS3()
    #     algname = BS3
    # end
    # alg = BS5()        # St =261.3s, RdK ∈ [1e-2 ~ 1e-4], 1e-3
    # alg = OwrenZen3()  # St =180.3s, RdK ∈ [1e-1 ~ 1e-4], 1e-2
    # alg = OwrenZen4()  # St =361.6s, RdK ∈ [1e-1 ~ 1e-4], 1e-2
    # alg = DP5()        # St =269.5s, RdK ∈ [1e-2 ~ 1e-4], 1e-3
    # # # RK54
    # alg = Tsit5()      # St =265.7s, RdK ∈ [1e-2 ~ 1e-4], 1e-3
    # if alg == Tsit5()
    #     algname = Tsit5
    # end
    # alg = Vern6()      # St =802s, RdK ∈ [1e-2 ~ 1e-4], 1e-2
    # alg = Vern7()       # ManifoldProjection(conservationFun)
    # alg = ExplicitRK(tableau = constructDormandPrince())
    # if alg == ExplicitRK(tableau = constructDormandPrince())
    #     algname = ExplicitRK_constructDormandPrince
    # end
    if alg == Tsit5()
        algname = Tsit5
    elseif alg == BS3()
        algname = BS3
    elseif alg == ExplicitRK(tableau = constructDormandPrince())
        algname = ExplicitRK_constructDormandPrince
    else
        algname = alg
    end
    @show alg
    return alg
end
