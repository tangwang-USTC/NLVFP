"""
  Applying the fixed time step explicit Newton method to solve the ODE problems.
    The fixed time step is decided by the characteristic time scale of the relaxation process and
    the relative rate of change of the moments respect to time.

    `RdtM = M⁻¹Δₜ∂ₜM ≪ 1` and is `≤ 1e-3` defaultly.
  
  Updating `vth` and `ûa` according to the values of `Ka` and `Ia` as:

    vth = √(2/3 * (2Ka / ρa - (Ia / ρa)^2))
    ûa = √(3/2) × Ia / √(2ρa * Ka - Ia^2)

  Constraints:

    K̂a = 3/2 + ûa²

  1

  Inputs:
    residualMethod_FP0D::Int ∈ [1,2]. When `residualMethod_FP0D=1` denotes absorbing the residuals by using the "dichotomy method"; or else
                                     `residualMethod_FP0D=2` denotes absorbing the residuals with a geometric ratio  等比残差吸收
    1
  Outouts:

"""

fvL0ek = copy(fvL0e)
pstk = deepcopy(pst0)
fvL0ek = FP0D2VExplicitNewton!(fvL0ek, pstk)

fvL0ek1 = copy(fvL0ek)
pstk1 = deepcopy(pstk)
fvL0ek1 = FP0D2VExplicitNewton!(fvL0ek1, pstk1)

fvL0ek2 = copy(fvL0ek1)
pstk2 = deepcopy(pstk1)
fvL0ek2 = FP0D2VExplicitNewton!(fvL0ek2, pstk2)

####################################################
# fvL0ek = copy(fvL0e)
# pstk = deepcopy(pst0)
# fvL0ek, dtfvL0ek1opt = FP0D2VExplicitNewton!(fvL0ek, pstk)

# fvL0ek1 = copy(fvL0ek)
# pstk1 = deepcopy(pstk)
# fvL0ek1, dtfvL0ek1opt1 = FP0D2VExplicitNewton!(fvL0ek1, pstk1)

# fvL0ek2 = copy(fvL0ek1)
# pstk2 = deepcopy(pstk1)
# fvL0ek2, dtfvL0ek1opt2 = FP0D2VExplicitNewton!(fvL0ek2, pstk2)

# fvL0ek3 = copy(fvL0ek2)
# pstk3 = deepcopy(pstk2)
# fvL0ek3, dtfvL0ek1opt2 = FP0D2VExplicitNewton!(fvL0ek3, pstk3)

# fvL0ek4 = copy(fvL0ek3)
# pstk4 = deepcopy(pstk3)
# fvL0ek4, dtfvL0ek1opt2 = FP0D2VExplicitNewton!(fvL0ek4, pstk4);
3