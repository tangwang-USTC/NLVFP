"""
# When `is_nai_const = false`

  Optimization of the amplitude function `fhl0` by using the `King` functions 
    to finding the optimized parameters `(n̂ₛ,ûₛ,v̂ₜₕₛ) = (nai,uai,vthi)`. 

  The general moments is renormalized as: 
    
    `M̂ⱼₗᵐ*`= M̂ⱼₗᵐ / CjLL2(j,L)`.

  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

  Inputs:
    Mhcsl0: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.

  Outputs:
    king_fM!(out, x; Mhcsl0=Mhcsl0, nMod=nMod)
  
"""

function king_fM!(out, x; Mhcsl0::AbstractVector=[1.0, 1.0, 0.0], nModel::Int64=1)

  nh = x[1:2:end]
  vhth = x[2:2:end]
  vhth2 = vhth .^ 2
  nj = 1
  # (l,j) = (0,0)
  out[nj] = sum(nh) - Mhcsl0[nj]
  nj += 1
  # (l,j) = (0,2)
  out[nj] = sum(nh .* vhth2) - Mhcsl0[nj]
  if nModel ≥ 2
    nj += 1
    # (l,j) = (0,4) = (0,2(nj-1))
    out[nj] = sum(nh .* vhth2 .^ 2) - Mhcsl0[nj]

    nj += 1
    # (l,j) = (0,6)
    out[nj] = sum(nh .* vhth2 .^ (nj - 1)) - Mhcsl0[nj]
    for kM in 3:nModel
      nj += 1
      # (l,j) = (0,2(nj-1))
      out[nj] = sum(nh .* vhth2 .^ (nj - 1)) - Mhcsl0[nj]

      nj += 1
      # (l,j) = (0,2(nj-1))
      out[nj] = sum(nh .* vhth2 .^ (nj - 1)) - Mhcsl0[nj]
    end
  end
  # #### Restrained by the relation `Ta = 2/3 * (Ka - Ek) / na`
  # nj += 1
  # out[nj] = sum(nh .* vhth.^2) + 1/3 * Ih^2 - sum(nh .* uh.^2)) - 1
  #         = sum(nh .* vhth.^2) + 1/3 * ((sum(nh .* uh))^2 - sum(nh .* uh.^2)) - 1
end

function king_fM_g!(J, x; nModel::Int64=1)

  fill!(J, 0.0)
  nh = x[1:2:end]
  vhth = x[2:2:end]
  nj = 1
  # (l,j) = (0,0)
  for s in 1:nModel
    J[nj, 2(s-1)+1] = 1.0
  end

  nj += 1
  # (l,j) = (0,2)    2
  for s in 1:nModel
    s2 = 2(s - 1)
    J[nj, s2+1] = vhth[s]^2
    J[nj, s2+2] = 2nh[s] * vhth[s]
  end

  if nModel ≥ 2
    nj += 1
    (l, j) = (0, 4)   # nj = 3
    for s in 1:nModel
      s2 = 2(s - 1)
      J[nj, s2+1] = vhth[s]^4
      J[nj, s2+2] = 4nh[s] * vhth[s]^3
    end

    nj += 1
    (l, j) = (0, 6)   # nj = 4
    for s in 1:nModel
      s2 = 2(s - 1)
      J[nj, s2+1] = vhth[s]^j
      J[nj, s2+2] = j * nh[s] * vhth[s]^(j - 1)
    end

    for kM in 3:nModel
      nj += 1
      j = 2(nj - 1)
      for s in 1:nModel
        s2 = 2(s - 1)
        J[nj, s2+1] = vhth[s] .^ j
        J[nj, s2+2] = j * nh[s] * vhth[s] .^ (j - 1)
      end

      nj += 1
      j = 2(nj - 1)
      for s in 1:nModel
        s2 = 2(s - 1)
        J[nj, s2+1] = vhth[s] .^ j
        J[nj, s2+2] = j * nh[s] * vhth[s] .^ (j - 1)
      end
    end
    # Q,R = qr(J)
    # dataJ = DataFrame(J,:auto)
    # dataQ = DataFrame(Q,:auto)
    # dataR = DataFrame(R,:auto)
    # @show dataJ
    # @show dataQ
    # @show dataR
  end
end


"""

  Inputs:
    Mhcsl0: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.

  Outputs:
    nais,uais,vthis = fl0king01_fM(nais,uais,vthis,Mhcsl0,nModel;
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            x_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
            p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
"""

function fl0king01_fM(nai::AbstractVector{T}, vthi::AbstractVector{T}, Mhcsl0::AbstractVector{T}, nModel::Int64;
  is_vthi_const::Bool=false,
  optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
  is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
  p_tol::Float64=1e-27, f_tol::Float64=1e-27, g_tol::Float64=1e-27,
  p_noise_rel::Float64=0e-3, p_noise_abs::Float64=0e-15) where {T}

  # The parameter limits for MCF plasma.
  x0 = zeros(2nModel)
  lbs = zeros(2nModel)
  ubs = zeros(2nModel)
  x0[1:2:end] .= nai

  lbs[1:2:end] .= nai
  ubs[1:2:end] .= nai
  if is_vthi_const
    if nModel == 1
      # vhthlimit = 1.0
      lbs[2:2:end] .= 1.0
      ubs[2:2:end] .= 1.0
      x0[2:2:end] .= 1.0
    else
      hhhhhhhhudd
    end
  else
    vhthlimit = min(vhthMax, Nspan_optim_nuTi * maximum(vthi))
    lbs[2:2:end] .= 1 / vhthlimit
    ubs[2:2:end] .= vhthlimit
    x0[2:2:end] .= vthi
  end

  # if is_nai_const
  #   if p_noise_rel ≠ 0.0
  #     p_noise_ratio = rand(nModel) * p_noise_rel
  #     x0[2:2:end] .*= (1.0 .+ p_noise_ratio)
  #   end
  #   if p_noise_abs ≠ 0.0
  #     x0[2:2:end] .+= p_noise_abs * rand(nModel)
  #   end
  # else
  #   if p_noise_rel ≠ 0.0
  #     p_noise_ratio = rand(2nModel) * p_noise_rel
  #     x0 .*= (1.0 .+ p_noise_ratio)
  #   end
  #   if p_noise_abs ≠ 0.0
  #     x0 .+= p_noise_abs * rand(2nModel)
  #   end
  # end

  res = fl0king01_fM(copy(x0), Mhcsl0, nModel; lbs=lbs, ubs=ubs,
    optimizer=optimizer, factor=factor, autodiff=autodiff,
    is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)

  if NL_solve == :LeastSquaresOptim
    xfit = res.minimizer         # the vector of best model1 parameters
    niter = res.iterations
    is_converged = res.converged
    xssr = res.ssr                         # sum(abs2, fcur)
  elseif NL_solve == :NLsolve
    xfit = res.zero         # the vector of best model1 parameters
    niter = res.iterations
    is_converged = res.f_converged
    xssr = res.residual_norm                         # sum(abs2, fcur)
  elseif NL_solve == :JuMP
    fgfgg
  end

  naifit = xfit[1:2:end]
  vthifit = xfit[2:2:end]

  # T̂ = ∑ₖ (n̂ₖv̂ₜₕₖ²).
  Thfit = sum(naifit .* vthifit .^ 2)

  # yfit0 = zeros(2nModel)
  # king_fM!(yfit0, x0; Mhcsl0=Mhcsl0, nModel=nModel)
  # @show yfit0

  yfit = zeros(2nModel)
  king_fM!(yfit, xfit; Mhcsl0=Mhcsl0, nModel=nModel)

  # Jfit = zeros(2nModel, 2nModel)
  # king_fM_g!(Jfit, x0; nModel=nModel)
  # @show paraM(Jfit);

  Dnh = sum(naifit) - 1
  DTh = Thfit .- 1
  if norm([Dnh, DTh, maximum(abs.(yfit))]) ≤ epsT1000
    printstyled("(Dnh,DTh,yfit)=", (Dnh, DTh, maximum(abs.(yfit))); color=:green)
  else
    printstyled("(Dnh,DTh,yfit)=", (Dnh, DTh, maximum(abs.(yfit))); color=:red)
  end
  # println()
  # @show Thfit - 1
  # @show Thfit ./ Rvth^2
  # @show Thfit .* Rvth^2
  # @show Rvth^2

  # println()
  # @show fmtf8.(lbs)
  # @show fmtf8.(x0)
  # @show fmtf8.(xfit)
  # @show fmtf8.(ubs)
  if is_converged
    @show is_converged, niter, xssr
  else
    printstyled("is_converged, niter, xssr=", (is_converged, niter, xssr); color=:red)
    println()
  end
  if norm([Dnh, DTh]) ≥ epsT1000
    @warn("` norm([Dnh,DTh]) ≥ epsT1000` which denotes the time step is so large that the convergence cannot be reached!", [Dnh, DTh])
    @show fmtf2.(xfit)
    @show fmtf8.(xfit - x0)
  end
  # The evaluation method according to the following parameters:
  return naifit, vthifit
end

"""

  Inputs:
    x0::Vector, x0[1:2:end] = nai[isp], x0[2:2:end] = vthi[isp];
    lbs::Vector, lbs[1:2:end] =
    ubs::Vector, ubs[1:2:end] = 
    Mhcsl0::Vector, where `Mhcsl0 = Ml0`;
    nModel::Integer

  res = fl0king01_fM(x0,Mhcsl0,nModel;lbs=lbs,ubs=ubs,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            x_tol=p_tol,f_tol=f_tol,g_tol=g_tol)
"""

function fl0king01_fM(x0::AbstractVector{T}, Mhcsl0::AbstractVector{T}, nModel::Int64;
  lbs::AbstractVector{T}=[0.0, 1.0, 0.0], ubs::AbstractVector{T}=[nhMax, 1.0, uhMax],
  optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
  is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
  p_tol::Float64=1e-27, f_tol::Float64=1e-27, g_tol::Float64=1e-27) where {T}

  king01_fM!(out, x) = king_fM!(out, x; Mhcsl0=Mhcsl0, nModel=nModel)
  if NL_solve == :LeastSquaresOptim
    if is_Jacobian
      J!(J, x) = king_fM_g!(J, x; nModel=nModel)
      nls = LeastSquaresProblem(x=x0, (f!)=king01_fM!, (g!)=J!, output_length=length(x0), autodiff=autodiff)
    else
      nls = LeastSquaresProblem(x=x0, (f!)=king01_fM!, output_length=length(x0), autodiff=autodiff)
    end
    res = optimize!(nls, optimizer(factor), iterations=maxIterKing, show_trace=show_trace,
      x_tol=p_tol, f_tol=f_tol, g_tol=g_tol, lower=lbs, upper=ubs)
  elseif NL_solve == :NLsolve
    if is_Jacobian
      Js!(J, x) = king_fM_g!(J, x; nModel=nModel)
      nls = OnceDifferentiable(king01_fM!, Js!, x0, similar(x0))
      if NL_solve_method == :trust_region
        res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
          iterations=maxIterKing, show_trace=show_trace)
      elseif NL_solve_method == :newton
        res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
          iterations=maxIterKing, show_trace=show_trace)
      end
    else
      nls = OnceDifferentiable(king01_fM!, x0, similar(x0))
      if NL_solve_method == :trust_region
        res = nlsolve(nls, x0, method=NL_solve_method, factor=1.0, autoscale=true, xtol=p_tol, ftol=f_tol,
          iterations=maxIterKing, show_trace=show_trace, autodiff=:forward)
      elseif NL_solve_method == :newton
        res = nlsolve(nls, x0, method=NL_solve_method, xtol=p_tol, ftol=f_tol,
          iterations=maxIterKing, show_trace=show_trace, autodiff=:forward)
      end
    end
  elseif NL_solve == :JuMP
    gyhhjjmj
  else
    esfgroifrtg
  end
  return res
end


