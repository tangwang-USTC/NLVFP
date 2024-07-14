"""
"""

# Applying the Explicit Newton method to solve the ODE equations
dt = 1e-0           # (=1e2 default) Which is normalized by `td ~ 1e10` for MCF plasma.
# `1e0 ~ 1e4` is proposed to be used for most situation.

# 1D
isp33 = isp3
isp33 = iFv3
L1m = 2
Rdata_limit = 5.0
# 2D
## Algorithm `RdtfvL`: Optimizations for `dtfvL`.
if 12 == 1
    order_dvdtf = 2         # (=2, default), `∈ [-1, 1, 2]` which denotes `[BackwardDiff, ForwardDiff, CentralDiff]`
    Nsmooth = 3             # (=3, default), which is `∈ N⁺ + 1`, the number of points to smooth the function `dtfvL`.
    order_smooth_itp = 0    # (=0, default), (0,1) → (y=Rdtf,Rd1y), the order of function to be extrapolated to smooth the function `dtfvL`.
    order_smooth = 2        # (=2, default), which is `∈ N⁺`, the highest order of smoothness to smooth the function `dtfvL`.
    order_nvc_itp = 2   # (=2, default), (1,2,3,23) → (nvcd1,nvcd2,nvcd3,max(nvcd2,nvcd3))
    is_boundv0 = zeros(Bool,order_smooth) 
    is_boundv0[1] = true    # (::Vector{Bool}=[true,false], default). 
                            # When it is `true`, the value `Rdiy(v[1]=0.0)` will be `true`.
                            # It is `true` when `v[1] == 0.0` and the value `Rdiy[1] = 0.0`
    k_dtf = 2               # (=2, default), the order of the Spline Interpolations for `dtfLn(v̂→0)`
    Nitp = 10               # (=10, default), the number of gridpoints to generate the interpolating function for `dtfLn(v̂→0)`
    nvc0_limit = 4          # (=4, default), `nvc0_limit ∈ N⁺` which is the lower bound of
                            # `nvc(order_nvc_itp)` to applying to the extrapolation for `dtfLn(v̂→0)`
    L1nvc_limit = 3         # (=3, default), `L1nvc_limit ∈ N⁺` which is the lower bound of `L` to applying to the extrapolation for `dtfLn(v̂→0)`
    if order_smooth == 2
        abstol_Rdy = [0.95, 0.5]      # ∈ (0, 1.0 → 10). 
                            # When `abstol_Rdy > 0.5,` the change of `dtfvL` will be very sharp and localized.
                            # When `û → 0.0`, the higher-order components 
                            # When `û ≥ 0.5 ≫ 0.0`, lower value , `abstol_Rdy < 0.5` is proposed.
    elseif order_smooth == 3
        abstol_Rdy = [0.45, 0.35, 0.35]
    else
        erfghjmk
    end
end
LM33 = LM[isp33]
@show LM1, LM33, L1m
Lvec = 1:LM33+1
Lvec1 = 2:LM33+2
# 2D
if 1 == 1

  RdtfvL0ek = dtfvL0e ./ fvL0e
  [fvL0e[1, L1, isp3] ≠ 0.0 || (RdtfvL0ek[1, L1, isp3] = 2RdtfvL0ek[2, L1, isp3] - RdtfvL0ek[3, L1, isp3]) for L1 in 1:LM1]
  [fvL0e[1, L1, iFv3] ≠ 0.0 || (RdtfvL0ek[1, L1, iFv3] = 2RdtfvL0ek[2, L1, iFv3] - RdtfvL0ek[3, L1, iFv3]) for L1 in 1:LM1]

  L1vec = 1:LM1
  dataf = [[0; vGe] [(L1vec)'; fvL0e[:, L1vec, isp33]] [0; vGe] 0:nvG]
  dataRdtf = [[0; vGe] [(L1vec)'; RdtfvL0ek[:, L1vec, isp33]] [0; vGe] 0:nvG]
  datadtf = [[0; vGe] [(L1vec)'; dtfvL0e[:, L1vec, isp33]] [0; vGe] 0:nvG]

  Rd1y = zeros(nvG + 1, LM33 + 4)
  yy = copy(dtfvL0e[:, Lvec, isp33])
  Rd1y[1, Lvec1] = Lvec
  Rd1y[2:end, [1, LM33 + 3]] .= vGe
  Rd1y[2:end, Lvec1] .= 0.0
  Rd1y[2:end, LM33+4] = 1:nvG

  Rd3y = copy(Rd1y)
  Rd3y[2:end, Lvec1] .= 0.0

  Rd2y = copy(Rd1y)
  Rd2y[2:end, Lvec1] .= 0.0

  # Rd1y[2:end, Lvec1] = RdpdtfvL0CDS(Rd1y[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0[1])
  # Rd2y[2:end, Lvec1] = RdpdtfvL0CDS(Rd2y[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0[2])
  # Rd3y[2:end, Lvec1] = RdpdtfvL0CDS(Rd3y[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0[3])
  if order_smooth == 2
    Rd2y[2:end, Lvec1],Rd1y[2:end, Lvec1] = RdpdtfvL0CDS(Rd2y[2:end, Lvec1],Rd1y[2:end, Lvec1],
                       yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
  else
    Rd3y[2:end, Lvec1],Rd2y[2:end, Lvec1],Rd1y[2:end, Lvec1] = RdpdtfvL0CDS(Rd3y[2:end, Lvec1],Rd2y[2:end, Lvec1],Rd1y[2:end, Lvec1],
                       yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
  end

  a = Rd1y[2:end, Lvec1]
  a[abs.(a).>Rdata_limit] .= Rdata_limit
  Rd1y[2:end, Lvec1] = a

  a = Rd2y[2:end, Lvec1]
  a[abs.(a).>Rdata_limit] .= Rdata_limit
  Rd2y[2:end, Lvec1] = a

  a = Rd3y[2:end, Lvec1]
  a[abs.(a).>Rdata_limit] .= Rdata_limit
  Rd3y[2:end, Lvec1] = a

  datasm = zeros(nvG, 9)
  datanames = [:n, :v, :fL, :dtfL, :Rdtf, :Rd1y, :Rd2y, :Rd3y, :nn]
  datas = DataFrame(datasm, datanames)
  datas[:, 1] = 1:nvG
  datas[:, 2] = vGe
end

function datasL1(datas, LL1)

  i = 3
  datas[:, i] = dataf[2:end, LL1]
  i += 1
  datas[:, i] = datadtf[2:end, LL1]
  i += 1
  datas[:, i] = dataRdtf[2:end, LL1]
  i += 1
  datas[:, i] = Rd1y[2:end, LL1]
  i += 1
  datas[:, i] = Rd2y[2:end, LL1]
  i += 1
  datas[:, i] = Rd3y[2:end, LL1]
  i += 1
  datas[:, i] = datas[:, 1]
  return datas
end

# 2D plotting
if 2 == 2
  pfvL(n1, n9, LL1, LL9) = plot(vGe[n1:n9], dataf[n1:n9, LL1:LL9], line=(2, :auto), label=(LL1-1:LL9-1)')
  pfvL(n1, n9, LL1) = plot(vGe[n1:n9], dataf[n1:n9, LL1], line=(2, :auto), 
                           xlabel=string("n1,n9=",(n1-1, n9)), label=string("fL_", LL1 - 1))
  if isp33 == 1
    pdtfvL(n1, n9, LL1, LL9) = plot(vGe[n1:n9], datadtf[n1:n9, LL1:LL9], line=(2, :auto), label=(LL1-1:LL9-1)', ylabel="dₜfₐ")
    pdtfvL(n1, n9, LL1) = plot(vGe[n1:n9], datadtf[n1:n9, LL1], line=(2, :auto), label=string("dₜfL"), ylabel="dₜfₐ")
  else
    pdtfvL(n1, n9, LL1, LL9) = plot(vGe[n1:n9], datadtf[n1:n9, LL1:LL9], line=(2, :auto), label=(LL1-1:LL9-1)', ylabel="dₜfᵦ")
    pdtfvL(n1, n9, LL1) = plot(vGe[n1:n9], datadtf[n1:n9, LL1], line=(2, :auto), label=string("dₜfL"), ylabel="dₜfᵦ")
  end
  pRdtfvL(n1, n9, LL1) = plot(vGe[n1:n9], dataRdtf[n1:n9, LL1], line=(2, :auto), label=string("RdₜfL_", LL1 - 1))
  pRd1y(n1, n9, LL1) = plot(vGe[n1:n9], Rd1y[n1:n9, LL1], line=(2, :auto),
                            xlabel=string("nvc1=",nvc[3,LL1-1,isp33]), label=string("Rd1y"))
  pRd2y(n1, n9, LL1) = plot(vGe[n1:n9], Rd2y[n1:n9, LL1], line=(2, :auto),
                            xlabel=string("nvc2=",nvc[5,LL1-1,isp33]), label=string("Rd2y"))
  if order_smooth == 3
    pRd3y(n1, n9, LL1) = plot(vGe[n1:n9], Rd3y[n1:n9, LL1], line=(2, :auto),
                              xlabel=string("nvc3=",nvc[7,LL1-1,isp33]), label=string("Rd3y"))
  else
    pRd3y(n1, n9, LL1) = plot(vGe[n1:n9], 0Rd2y[n1:n9, LL1], line=(2, :auto),label=string("Rd3y"))
  end

  pfpdtfpRdtf(n1, n9, LL1) = display(plot(
    pfvL(n1, n9, LL1), pdtfvL(n1, n9, LL1), pRdtfvL(n1, n9, LL1),
    pRd1y(n1, n9, LL1), pRd2y(n1, n9, LL1), pRd3y(n1, n9, LL1), 
    layout=(2, 3)))
end
# 1D
nvcL1 = zeros(Int64,2(order_smooth+1))  # `[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3]`
yfLn = fvL0e[:,L1m,isp33]
ydtfLn = dtfvL0e[:,L1m,isp33]
# Rdtf = ydtfLn ./ yfLn
# if yfLn[1] == 0.0
#   Rdtf[1] = 2Rdtf[2] - Rdtf[3]
# end
nvcL1 = nvcfind(nvcL1,ydtfLn,yfLn,nvG,L1m;orders=order_dvdtf,is_boundv0=is_boundv0,
              Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)
# 1
nvcL1[3] > 6 ? n11 = nvcL1[3] - 5 : n11 = nvcL1[3]
@show nvcL1, nvG, n11
@show datasL1(datas,L1m)[n11:n11+27,:]


# 2D 
nvc = zeros(Int64,2(order_smooth+1),LM33+1)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
nvc = nvcfind(nvc,dtfvL0e[:,1:LM33+1,isp33],fvL0e[:,1:LM33+1,isp33],nvG,LM33;
            orders=order_dvdtf,is_boundv0=is_boundv0,Nsmooth=Nsmooth,
            order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)

function pfpdtfpRdtf(Lvec)
    
    if typeof(Lvec) == Int64
        LL1 = Lvec
        n1 = nvc[5,LL1]
        n9 = nvc[5,LL1] + 15
        # n1 = max(2,nvc[1,LL1])
        # n9 = max(7,nvc[5,LL1])
        # n9 ≤ n1 ? n9 = nvc[4,LL1-1] : 1
        display(plot(pfvL(n1, n9, LL1), pdtfvL(n1, n9, LL1), pRdtfvL(n1, n9, LL1),
               pRd1y(n1, n9, LL1), pRd2y(n1, n9, LL1), pRd3y(n1, n9, LL1), 
               layout=(2, 3)))
    else
        for LL1 in Lvec
            # n1 = max(2,nvc[1,LL1])
            # n9 = max(7,nvc[5,LL1])
            # n1 = nvc[1,LL1-1] + 1
            # n9 = nvc[5,LL1-1]
            # n9 ≤ n1 ? n9 = nvc[4,LL1-1] : 1
            n1 = nvc[5,LL1]
            n9 = nvc[5,LL1] + 15
            # @show LL1-1,n1,n9
            display(plot(pfvL(n1, n9, LL1), pdtfvL(n1, n9, LL1), pRdtfvL(n1, n9, LL1),
                   pRd1y(n1, n9, LL1), pRd2y(n1, n9, LL1), pRd3y(n1, n9, LL1), 
                   layout=(2, 3)))
      end
    end
end

function datasL1nvc(datas, LL1)
  
  n1 = nvc[1,LL1-1] + 1
  n9 = nvc[5,LL1-1]
  n9 ≤ n1 ? n9 = nvc[4,LL1-1] : 1
  nv19 = n1+1:n9
  i = 3
  datas[nv19, i] = dataf[nv19, LL1]
  i += 1
  datas[nv19, i] = datadtf[nv19, LL1]
  i += 1
  datas[nv19, i] = dataRdtf[nv19, LL1]
  i += 1
  datas[nv19, i] = Rd1y[nv19, LL1]
  i += 1
  datas[nv19, i] = Rd2y[nv19, LL1]
  i += 1
  datas[nv19, i] = datas[nv19, 1]
  return datas[nv19,:]
end

pfpdtfpRdtf(nvc[1,L1m]+1,nvc[5,L1m],L1m)

[Lvec'; nvc]
