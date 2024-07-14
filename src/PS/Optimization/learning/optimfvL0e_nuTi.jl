"""
  Optimization of the amplitude function `fhl0` by using the `King` functions 
    to finding the optimized parameters `(n̂ₛ,ûₛ,v̂ₜₕₛ) = (nai,uai,vthi)`. 

  The general moments is renormalized as: 
    
    `M̂ⱼₗᵐ*`= M̂ⱼₗᵐ / CjLL2(j,L)`.

  Notes: `{M̂₁}/3 = Î ≠ û`, generally. Only when `nMod = 1` gives `Î = û`.

  Inputs:
    Mhcsl01: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.

  Outputs:
  
"""

function king!(out, x; Mhcsl01::AbstractVector=[1.0, 1.0, 0.0], nModel::Int64=1)

  nh = x[1:3:end]
  uh = x[2:3:end]
  vhth = x[3:3:end]
  uvth = uh ./ vhth
  vhth2 = vhth .^ 2
  uvth2 = uvth .^ 2
  nj = 1
  # (l,j) = (0,0)
  out[nj] = sum(nh) - Mhcsl01[nj]
  nj += 1
  # (l,j) = (1,1)
  out[nj] = sum(nh .* uh) - Mhcsl01[nj]
  nj += 1
  # (l,j) = (0,2)
  out[nj] = sum((nh .* vhth2) .* (1 .+ 2 / 3 * uvth2)) - Mhcsl01[nj]
  
  if nModel ≥ 2
    nj += 1
    # (l,j) = (1,3)
    out[nj] = sum((nh .* uh .* vhth2) .* (1 .+ 2 / 5 * uvth2)) - Mhcsl01[nj]

    nj += 1
    # (l,j) = (0,4)
    l = 0
    j = 4
    out[nj] = sum((nh .* vhth2 .^ 2) .* (1 .+ 4 / 3 * uvth2 .+ 4 / 15 * uvth2 .^ 2)) - Mhcsl01[nj]

    nj += 1
    # (l,j) = (1,5)
    l = 1 - l
    j += 1
    out[nj] = sum((nh .* uh .* vhth2 .^ 2) .* (1 .+ 4 / 5 * uvth2 .+ 4 / 35 * uvth2 .^ 2)) - Mhcsl01[nj]
    for kM in 3:nModel
      for i in 1:3
        nj += 1
        l = 1 - l
        j += 1
        N = (j - l) / 2 |> Int
        k = 1:N
        if l === 0
          ck = [2^k * binomial(N, k) / prod(3:2:2k+1) for k in 1:N]
          out[nj] = sum((nh .* vhth .^ j) .* (1 .+ [sum(ck .* uvth[s] .^ (2k)) for s in 1:nModel])) - Mhcsl01[nj]
        elseif l === 1
          ck = [2^k * binomial(N, k) / prod(5:2:2(l+k)+1) for k in 1:N]
          out[nj] = sum((nh .* uh .* vhth .^ (j - l)) .* (1 .+ [sum(ck .* uvth[s] .^ (2k)) for s in 1:nModel])) - Mhcsl01[nj]
        end
      end
    end
  end
  # #### Restrained by the relation `Ta = 2/3 * (Ka - Ek) / na`
  # nj += 1
  # out[nj] = sum(nh .* vhth.^2) + 1/3 * Ih^2 - sum(nh .* uh.^2)) - 1
  #         = sum(nh .* vhth.^2) + 1/3 * ((sum(nh .* uh))^2 - sum(nh .* uh.^2)) - 1
end

function king_g!(J, x; nModel::Int64=1)

  fill!(J, 0.0)
  nh = x[1:3:end]
  uh = x[2:3:end]
  vhth = x[3:3:end]
  nj = 1
  # (l,j) = (0,0)
  for s in 1:nModel
    J[nj, 3(s-1)+1] = 1.0
  end
  nj += 1
  # (l,j) = (1,1)
  for s in 1:nModel
    s3 = 3(s - 1)
    J[nj, s3+1] = uh[s]
    J[nj, s3+2] = nh[s]
  end
  nj += 1
  # (l,j) = (0,2)
  for s in 1:nModel
    s3 = 3(s - 1)
    J[nj, s3+1] = vhth[s]^2 + 2 / 3 * uh[s]^2
    J[nj, s3+2] = 4 / 3 * nh[s] * uh[s]
    J[nj, s3+3] = 2nh[s] * vhth[s]
  end
  if nModel ≥ 2
    nj += 1
    # (l,j) = (1,3)
    for s in 1:nModel
      s3 = 3(s - 1)
      J[nj, s3+1] = uh[s] * (vhth[s]^2 + 2 / 5 * uh[s]^2)
      J[nj, s3+2] = nh[s] * (vhth[s]^2 + 6 / 5 * uh[s]^2)
      J[nj, s3+3] = 2nh[s] * uh[s] * vhth[s]
    end
    nj += 1
    (l, j) = (0, 4)
    for s in 1:nModel
      s3 = 3(s - 1)
      J[nj, s3+1] = vhth[s]^4 + 4 / 3 * uh[s]^2 * vhth[s]^2 + 4 / 15 * uh[s]^4
      J[nj, s3+2] = 8 / 3 * nh[s] * uh[s] * (vhth[s]^2 + 2 / 5 * uh[s]^2)
      J[nj, s3+3] = 4nh[s] * vhth[s] * (vhth[s]^2 + 2 / 3 * uh[s]^2)
    end
    uvth = uh ./ vhth

    nj += 1
    (l, j) = (1, 5)
    for s in 1:nModel
      s3 = 3(s - 1)
      uvths = uh[s] / vhth[s]
      J[nj, s3+1] = uh[s] * vhth[s]^4 * (1 + 4 / 5 * uvths^2 + 4 / 35 * uvths^4)
      J[nj, s3+2] = nh[s] * vhth[s]^4 * (1 + 12 / 5 * uvths^2 + 4 / 7 * uvths^4)
      J[nj, s3+3] = 4nh[s] * uh[s] * vhth[s]^3 * (1 + 2 / 5 * uvths^2)
    end
    for kM in 3:nModel
      for i in 1:3
        nj += 1
        l = 1 - l
        j += 1
        N = (j - l) / 2 |> Int
        k = 1:N
        if l === 0
          ck = [2^k * binomial(N, k) / prod(3:2:2k+1) for k in 1:N]
          for s in 1:nModel
            s3 = 3(s - 1)
            J[nj, s3+1] = vhth[s]^(j - l) * (1 + sum(ck .* uvth[s] .^ (2k)))
            J[nj, s3+2] = nh[s] * vhth[s]^(j - l) * sum(2k .* ck .* uvth[s] .^ (2k .- 1))
            J[nj, s3+3] = j * nh[s] * vhth[s]^(j - 1) * (1 + sum((1 .- 2 / j * k) .* ck .* uvth[s] .^ (2k)))
          end
        elseif l === 1
          ck = [2^k * binomial(N, k) / prod(5:2:2(l+k)+1) for k in 1:N]
          # ck = [2^k * binomial(N, k) / prod((2l+3):2:2(l+k)+1) for k in 1:N]
          for s in 1:nModel
            s3 = 3(s - 1)
            J[nj, s3+1] = uh[s] * vhth[s]^(j - l) * (1 + sum(ck .* uvth[s] .^ (2k)))
            J[nj, s3+2] = nh[s] * vhth[s]^(j - l) * (l + sum((2k .+ l) .* ck .* uvth[s] .^ (2k)))
            J[nj, s3+3] = nh[s] * uh[s] * vhth[s]^(j - 2) * ((j - l) - sum((2k .+ (l - j)) .* ck .* uvth[s] .^ (2k)))
          end
        end
      end
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

"""

  Inputs:
    Mhcsl01: = M̂ⱼₗᵐ*, which is the renormalized general kinetic moments.

  Outputs:
    nais,uais,vthis = fl0king01(nais,uais,vthis,Mhcsl01,nModel;is_nai_const=is_nai_const,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            x_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
            p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
"""

function fl0king01(nai::AbstractVector{T}, uai::AbstractVector{T},
  vthi::AbstractVector{T}, Mhcsl01::AbstractVector{T}, nModel::Int64;
  is_nai_const::Bool=false,is_vthi_const::Bool=false,
  optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
  is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
  p_tol::Float64=1e-27, f_tol::Float64=1e-27, g_tol::Float64=1e-27,
  p_noise_rel::Float64=0e-3, p_noise_abs::Float64=0e-15,
  is_dtuh::Bool=false,Nspan_optim_du::Float64=1e-2) where {T}

  # The parameter limits for MCF plasma.
  x0 = zeros(3nModel)
  lbs = zeros(3nModel)
  ubs = zeros(3nModel)
  x0[1:3:end] .= nai
  if is_nai_const
    lbs[1:3:end] .= nai
    ubs[1:3:end] .= nai
  else
    nhlimit = min(nhMax, Nspan_optim_nuTi * maximum(abs.(nai)))
    lbs[1:3:end] .= 0.0
    ubs[1:3:end] .= nhlimit
  end
  if is_dtuh
    uhlimit = min(uhMax, Nspan_optim_nuTi * maximum(abs.(uai)) + Nspan_optim_du)
    x0[2:3:end] .= uai.+ epsT100
  else
    uhlimit = min(uhMax, Nspan_optim_nuTi * maximum(abs.(uai)))
    x0[2:3:end] .= uai
  end
  lbs[2:3:end] .= -uhlimit
  ubs[2:3:end] .= uhlimit
  if is_vthi_const
    if nModel == 1
      # vhthlimit = 1.0
      lbs[3:3:end] .= 1.0
      ubs[3:3:end] .= 1.0
      x0[3:3:end] .= 1.0
    else
      hhhhhhhhudd
    end
  else
    vhthlimit = min(vhthMax, Nspan_optim_nuTi * maximum(abs.(vthi)))
    lbs[3:3:end] .= 1 / vhthlimit
    ubs[3:3:end] .= vhthlimit
    x0[3:3:end] .= vthi
  end

  # if is_nai_const
  #   if p_noise_rel ≠ 0.0
  #     p_noise_ratio = rand(nModel) * p_noise_rel
  #     x0[2:3:end] .*= (1.0 .+ p_noise_ratio)
  #     x0[3:3:end] .*= (1.0 .+ p_noise_ratio)
  #   end
  #   if p_noise_abs ≠ 0.0
  #     x0[2:3:end] .+= p_noise_abs * rand(nModel)
  #     x0[3:3:end] .+= p_noise_abs * rand(nModel)
  #   end
  # else
  #   if p_noise_rel ≠ 0.0
  #     p_noise_ratio = rand(3nModel) * p_noise_rel
  #     x0 .*= (1.0 .+ p_noise_ratio)
  #   end
  #   if p_noise_abs ≠ 0.0
  #     x0 .+= p_noise_abs * rand(3nModel)
  #   end
  # end

  res = fl0king01(x0, Mhcsl01, nModel; lbs=lbs, ubs=ubs,
    optimizer=optimizer, factor=factor, autodiff=autodiff,
    is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol)
  xfit = res.minimizer         # the vector of best model1 parameters
  niter = res.iterations
  is_converged = res.converged
  xssr = res.ssr                         # sum(abs2, fcur)

  naifit = xfit[1:3:end]
  # nhafit = sum(naifit)
  xfit[2:3:end][naifit.≤epsMs] .= 0.0
  uhafit = sum(naifit .* xfit[2:3:end])
  # T̂ = ∑ₖ n̂ₖ (v̂ₜₕₖ² + 2/3 * ûₖ²) - 2/3 * û², where `û = ∑ₖ(n̂ₖûₖ) / ∑ₖ(n̂ₖ)`
  Thfit = sum(naifit .* (xfit[3:3:end] .^ 2 + 2 / 3 * xfit[2:3:end] .^ 2)) - 2 / 3 * uhafit .^ 2
  # Th = sum(naifit .* xfit[3:3:end].^2) + 2/3 * (sum(naifit .* xfit[2:3:end].^2) - uhafit.^2) - 1
  yfit = zeros(3nModel)
  king!(yfit, xfit; Mhcsl01=Mhcsl01, nModel=nModel)
  Dnh = sum(naifit) - 1
  DTh = Thfit .- 1
  if norm([Dnh, DTh, maximum(abs.(yfit))]) ≤ epsT1000
    printstyled("(Dnh,DTh,yfit)=", (Dnh, DTh, maximum(abs.(yfit))); color=:green)
  else
    printstyled("(Dnh,DTh,yfit)=", (Dnh, DTh, maximum(abs.(yfit))); color=:red)
  end
  println()
  if is_converged
    @show is_converged, niter, xssr
  else
    printstyled("is_converged, niter, xssr=",(is_converged, niter, xssr);color=:red)
    println()
  end

  if Dnh ≥ epsT1000
    @warn("` Dnh ≥ epsT1000` which denotes the time step is so large that the convergence cannot be reached!",Dnh)
  end

  if DTh ≥ epsT1000
    @warn("` norm([Dnh,DTh]) ≥ epsT1000` which denotes the time step is so large that the convergence cannot be reached!",DTh)
  end
  
    # The evaluation method according to the following parameters:
  # `is_converged, niter, xssr, nhafit, Thfit,maximum(abs.(yfit))`
  
  return naifit, xfit[2:3:end], xfit[3:3:end]
end

"""

  Inputs:
    x0::Vector, x0[1:3:end] = nai[isp], x0[2:3:end] = uai[isp], x0[3:3:end] = vthi[isp];
    lbs::Vector, lbs[1:3:end] =
    ubs::Vector, ubs[1:3:end] = 
    Mhcsl01::Vector, where `Mhcsl01[1:2:end] = Ml0`, `Mhcsl01[2:3:end] = Ml1`;
    nModel::Integer

  res = fl0king01!(x0,Mhcsl01,nModel;lbs=lbs,ubs=ubs,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            x_tol=p_tol,f_tol=f_tol,g_tol=g_tol)
"""

function fl0king01(x0::AbstractVector{T}, Mhcsl01::AbstractVector{T}, nModel::Int64;
  lbs::AbstractVector{T}=[0.0, 1.0, 0.0], ubs::AbstractVector{T}=[nhMax, 1.0, uhMax],
  optimizer=Dogleg, factor=QR(), autodiff::Symbol=:central,
  is_Jacobian::Bool=true, show_trace::Bool=false, maxIterKing::Int64=200,
  p_tol::Float64=1e-27, f_tol::Float64=1e-27, g_tol::Float64=1e-27) where {T}

  king01!(out, x) = king!(out, x; Mhcsl01=Mhcsl01, nModel=nModel)
  if is_Jacobian
    king01_g!(J, x) = king_g!(J, x; nModel=nModel)
    nls = LeastSquaresProblem(x=x0, (f!)=king01!, (g!)=king01_g!, output_length=length(x0), autodiff=autodiff)
  else
    nls = LeastSquaresProblem(x=x0, (f!)=king01!, output_length=length(x0), autodiff=autodiff)
  end
  res = optimize!(nls, optimizer(factor), iterations=maxIterKing, show_trace=show_trace,
    x_tol=p_tol, f_tol=f_tol, g_tol=g_tol, lower=lbs, upper=ubs)
  return res
end

