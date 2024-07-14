
include(joinpath(pathroot, "src/Solvers/newAlgorithms/RKos.jl"))
include(joinpath(pathroot, "src/Solvers/newAlgorithms/MsRKos.jl"))

"""
  Testing the convergence of Composite Runge-Kutta algorithm (CRK);
  Compare the accuracy of CRK algorithm with the standard RK algorithms.

  CRK55 ~ RK5/Lawson5 which have the same weights (bᵢ)
"""

alg_RKos = :MsRKo6s6           # Only has `5.1`-order convergence 
# alg_RKos = :RK5              
# alg_RKos = :Lawson5            # which have the same weights (bᵢ) as ":RK5"

# alg_RKos = :MsRKo5s5         # Only has `4.7`-order convergence      
# alg_RKos = :RungeFirst5      # Only has `4.3`-order convergence   
 
# alg_RKos = :Hammingo4s3     # Only has `3.5`-order convergence  
# alg_RKos = :Adamso4s3       # Only has `3.5`-order convergence 
# alg_RKos = :RK438           # Only has `3.3`-order convergence, ":RK438" is better than ":RK4" and ":RK42" according to the integral errors
# alg_RKos = :RK4             # Only has `3.2`-order convergence 
# alg_RKos = :RK42          # Only has `3.2`-order convergence    
# alg_RKos = :Adamso4s4       # Only has `3.1`-order convergence   
# alg_RKos = :Milneo4s4       # Only has `3.1`-order convergence   
# alg_RKos = :LobattoIIIA4    # Only has `3.1`-order convergence     
# alg_RKos = :Kutta3
# alg_RKos = :SSPRK3

x0, xend = 0.0, 1e-8 * 2^0
is_fx_polynomial = true
nnn = 6
if is_fx_polynomial
    fx(x) = x.^nnn
    Ix(x) = x.^(nnn + 1) / (nnn + 1)
else
  if nnn == 1
    fx(x) = exp.(-x)
    Ix(x) = - exp.(-x)
  elseif nnn == 2
    ghhhh
    fx(x) =  exp.(-x.^2)
    Ix(x) =  pi / sqrt2 * erf.(x / sqrt2)
  else
    fx(x) = x .* exp.(-x.^2)
    Ix(x) = -0.5 * exp.(-x.^2)
  end
end

hx = xend - x0

is_MsRK = false
is_enbedded = false
if alg_RKos == :RK5
  A, c, b, o, s, rs = construct_RK5()
  nx = rs
  dx = (xend - x0) / (nx - 1)
  xvec = zeros(s)
  xvec[1] = x0
  xvec[2] = x0 + dx
  xvec[3:end] = x0 .+ dx:dx:hx
elseif alg_RKos == :Lawson5
  A, c, b, o, s, rs = construct_Lawson5()
  nx = rs
  dx = (xend - x0) / (nx - 1)
  xvec = zeros(s)
  xvec[1] = x0
  xvec[2] = x0 + dx / 3
  xvec[3:end] = x0 .+ dx:dx:hx
elseif alg_RKos == :RungeFirst5
  A, c, b, o, s, rs = construct_RungeFirst5()
  nx = rs
  dx = (xend - x0) / (nx - 1)

  xvec = zeros(s)
  xvec[1] = x0
  xvec[2] = x0 + 1dx
  xvec[3] = x0 + 2dx
  xvec[4] = x0 + 5dx
  xvec[5] = x0 + 3dx
  xvec[6] = x0 + 4dx
elseif alg_RKos == :RK4
  A, c, b, o, s, rs = construct_RK4()
  nx = rs
  dx = (xend - x0) / (nx - 1)

  xvec = zeros(s)
  xvec[1] = x0
  xvec[2] = x0 + dx
  xvec[3:end] = x0 .+ dx:dx:hx
elseif alg_RKos == :RK438
  A, c, b, o, s, rs = construct_RK438()
  nx = rs
  dx = (xend - x0) / (nx - 1)

  xvec = zeros(s)
  xvec[:] = x0:dx:xend
elseif alg_RKos == :RK42
  A, c, b, o, s, rs = construct_RK42()
  nx = rs
  dx = (xend - x0) / (nx - 1)

  xvec = zeros(s)
  xvec[1] = x0
  xvec[2] = x0 + 1dx
  xvec[3] = x0 + 2dx
  xvec[end] = xend
elseif alg_RKos == :LobattoIIIA4
  A, c, b, b2, o, s, rs = construct_LobattoIIIA4()
  is_enbedded = true
  nx = rs
  dx = (xend - x0) / (nx - 1)

  xvec = zeros(s)
  xvec[:] = x0:dx:xend
elseif alg_RKos == :Kutta3
  A, c, b, o, s, rs = construct_Kutta3()
  nx = rs
  dx = (xend - x0) / (nx - 1)

  xvec = zeros(rs)
  xvec[:] = x0:dx:xend
elseif alg_RKos == :SSPRK3
  A, c, b, o, s, rs = construct_SSPRK3()
  nx = rs
  dx = (xend - x0) / (nx - 1)

  xvec = zeros(rs)
  xvec[1] = x0
  xvec[2] = x0 + xend
  xvec[3] = x0 + dx
else
    if alg_RKos == :Adamso4s3
      A, b, d, o, s, rs, (nb, nd) = construct_Adamso4s3()
    elseif alg_RKos == :Adamso4s4
      A, b, d, o, s, rs, (nb, nd) = construct_Adamso4s4()
    elseif alg_RKos == :Hammingo4s3
      A, b, d, o, s, rs, (nb, nd) = construct_Hammingo4s3()
    elseif alg_RKos == :Milneo4s4
      A, b, d, o, s, rs, (nb, nd) = construct_Milneo4s4()
    elseif alg_RKos == :MsRKo6s6
      A, b, d, o, s, rs, (nb, nd) = construct_MsRKo6s6()
    elseif alg_RKos == :MsRKo5s5
      A, b, d, o, s, rs, (nb, nd) = construct_MsRKo5s5()
    # elseif alg_RKos == :MsRKo5s5b1d6
    #   A, b, d, o, s, rs, (nb, nd) = construct_MsRKo5s5b1d6()
    # elseif alg_RKos == :MsRKo6s6b1d7
    #   A, b, d, o, s, rs, (nb, nd) = construct_MsRKo6s6b1d7()
    end
    hx /= s
    is_MsRK = true
    nx = rs
    dx = (xend - x0) / (nx - 1)

    xvec = zeros(rs)
    xvec[:] = x0:dx:xend
# else
#   eghb
end

# The theoretic value
y0 = Ix(x0)
yend = Ix(xend)
I_theory = yend - y0

fvec = fx.(xvec)
if is_MsRK
    yvec = Ix.(xvec)
    yend_RKos = dot(yvec, d) + hx * dot(fvec,b)
    Rerry = yend_RKos / yend - 1
    if is_enbedded
      yend_RKos2 = dot(yvec, d) + hx * dot(fvec,b2)
      Rerry2 = yend_RKos2 / yend - 1
      RDy = yend_RKos2 / yend_RKos - 1
      @show 2, alg_RKos, fmtf4.([Rerry, Rerry2, RDy]);
    else
      @show 2, alg_RKos, Rerry;
    end
else
    y0_RKos = copy(y0)
    yend_RKos = y0_RKos + hx * dot(fvec,b)
    Rerry = yend_RKos / yend - 1
    if is_enbedded
      yend_RKos2 = y0_RKos + hx * dot(fvec,b2)
      Rerry2 = yend_RKos2 / yend - 1
      RDy = yend_RKos2 / yend_RKos - 1
      @show alg_RKos, fmtf4.([Rerry, Rerry2, RDy]);
    else
      @show alg_RKos, Rerry;
    end
end
