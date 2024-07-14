"""
  Optimizations `dtfvL` according to 
    `y(v̂→0) → Cₗᵐ`
    `Rd1y(v̂→0) → 0`
  
  where
    
    y = dtfvL ./ fvL
"""

## Algorithm `RdtfvL`: Optimizations for `dtfvL`.
if 1 == 1
    order_dvdtf = 2         # (=2, default), `∈ [-1, 1, 2]` which denotes `[BackwardDiff, ForwardDiff, CentralDiff]`
    Nsmooth = 3             # (=3, default), which is `∈ N⁺ + 1`, the number of points to smooth the function `dtfvL`.
    order_smooth = 3        # (=3, default), which is `∈ N⁺`, the highest order of smoothness to smooth the function `dtfvL`.
    order_smooth_itp = 1    # (=1, default), (0,1) → (y=Rdtf,Rd1y), the order of function to be extrapolated to smooth the function `dtfvL`.
    order_nvc_itp = 3       # (=2, default), (1,2,3,N⁺≥4) → (nvcd1,nvcd2,nvcd3,max(nvcd2,nvcd3))
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
        abstol_Rdy = [0.35, 0.35]      # ∈ (0, 1.0 → 10). 
                            # When `abstol_Rdy > 0.5,` the change of `dtfvL` will be very sharp and localized.
                            # When `û → 0.0`, the higher-order components 
                            # When `û ≥ 0.5 ≫ 0.0`, lower value , `abstol_Rdy < 0.5` is proposed.
    elseif order_smooth == 3
        abstol_Rdy = [0.35, 0.35, 0.10]
    else
        erfghjmk
    end
end
isp33 = iFv3
# isp33 = isp3
LM33 = LM[isp33]
Lvec = 1:LM33+1
Lvec1 = 2:LM33+2
# 2D dtfvL
if 2 == 2
  Rdata_limit = 5.0
  RdtfvL0ek = dtfvL0e0 ./ fvL0e
  [fvL0e[1, L1, isp3] ≠ 0.0 || (RdtfvL0ek[1, L1, isp3] = 2RdtfvL0ek[2, L1, isp3] - RdtfvL0ek[3, L1, isp3]) for L1 in 1:LM1]
  [fvL0e[1, L1, iFv3] ≠ 0.0 || (RdtfvL0ek[1, L1, iFv3] = 2RdtfvL0ek[2, L1, iFv3] - RdtfvL0ek[3, L1, iFv3]) for L1 in 1:LM1]

  L1vec = 1:LM1
  dataf = [[0; vGe] [(L1vec)'; fvL0e[:, L1vec, isp33]] [0; vGe] 0:nvG]
  dataRdtf = [[0; vGe] [(L1vec)'; RdtfvL0ek[:, L1vec, isp33]] [0; vGe] 0:nvG]
  datadtf = [[0; vGe] [(L1vec)'; dtfvL0e0[:, L1vec, isp33]] [0; vGe] 0:nvG]
end
# Optimizations
if 1 == 1
  yy = copy(dtfvL0e0[:, Lvec, isp33])
  Rd1y = zeros(nvG + 1, LM33 + 4)
  Rd1y[1, Lvec1] = Lvec
  Rd1y[2:end, [1, LM33 + 3]] .= vGe
  Rd1y[2:end, Lvec1] .= 0.0
  Rd1y[2:end, LM33+4] = 1:nvG
  # Rd1y[2:end, Lvec1] = RdpdtfvL0CDS(Rd1y[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0[1])
  # Rd2y[2:end, Lvec1] = RdpdtfvL0CDS(Rd2y[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0[2])
  # Rd3y[2:end, Lvec1] = RdpdtfvL0CDS(Rd3y[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0[3])
  if order_smooth == 1
    Rd1y[2:end, Lvec1] = RdpdtfvL0CDS(Rd1y[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0[1])
  elseif order_smooth == 2
    Rd2y = copy(Rd1y)
    Rd2y[2:end, Lvec1] .= 0.0
    Rd2y[2:end, Lvec1],Rd1y[2:end, Lvec1] = RdpdtfvL0CDS(Rd2y[2:end, Lvec1],Rd1y[2:end, Lvec1],
                       yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
    a = Rd2y[2:end, Lvec1]
    a[abs.(a).>Rdata_limit] .= Rdata_limit
    Rd2y[2:end, Lvec1] = a
  else
    Rd2y = copy(Rd1y)
    Rd2y[2:end, Lvec1] .= 0.0

    Rd3y = copy(Rd1y)
    Rd3y[2:end, Lvec1] .= 0.0
    Rd3y[2:end, Lvec1],Rd2y[2:end, Lvec1],Rd1y[2:end, Lvec1] = RdpdtfvL0CDS(Rd3y[2:end, Lvec1],Rd2y[2:end, Lvec1],Rd1y[2:end, Lvec1],
                       yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
    a = Rd2y[2:end, Lvec1]
    a[abs.(a).>Rdata_limit] .= Rdata_limit
    Rd2y[2:end, Lvec1] = a

    a = Rd3y[2:end, Lvec1]
    a[abs.(a).>Rdata_limit] .= Rdata_limit
    Rd3y[2:end, Lvec1] = a
  end

  a = Rd1y[2:end, Lvec1]
  a[abs.(a).>Rdata_limit] .= Rdata_limit
  Rd1y[2:end, Lvec1] = a

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
  if order_smooth ≥ 2
    datas[:, i] = Rd2y[2:end, LL1]
  end
  i += 1
  if order_smooth ≥ 3
    datas[:, i] = Rd3y[2:end, LL1]
  end
  i += 1
  datas[:, i] = datas[:, 1]
  return datas
end

nsp_vec = 1:ns
iFv33 = nsp_vec[nsp_vec .≠ isp33][1]
LM33 = LM[isp33]
vabth33 = vth[isp33] / vth[iFv33]
@show isp33, LM1, LM33, vabth33, Ek0

dtfvLa0 = copy(dtfvL0e0[:, :, isp33])
dtfvLb0 = copy(dtfvL0e0[:, :, iFv33])

nvc3 = zeros(Int64, 2(order_smooth + 1), LM1, ns)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
dtfvL0e3D = copy(dtfvL0e0)
optimdtfvL0e!(nvc3, dtfvL0e3D, fvL0e, vGe, nvG, ns, LM;
    orders=order_dvdtf, is_boundv0=is_boundv0,
    Nsmooth=Nsmooth, order_smooth=order_smooth, abstol_Rdy=abstol_Rdy,
    k=k_dtf, Nitp=Nitp, order_smooth_itp=order_smooth_itp, order_nvc_itp=order_nvc_itp,
    nvc0_limit=nvc0_limit, L1nvc_limit=L1nvc_limit)
dtMsnnE33D = zeros(datatype, njMs, LM1, ns)
dtMsnnE33D = MsnnEvens(dtMsnnE33D, dtfvL0e3D, vGe, njMs, LM, LM1, ns; is_renorm=is_renorm)
dtfvLa = copy(dtfvL0e3D[:, :, isp33])
dtfvLb = copy(dtfvL0e3D[:, :, iFv33])

############################# The smoothed datas
if 1 == 1
  ys = copy(dtfvL0e3D[:, Lvec, isp33])
  Rd1ys = zeros(nvG + 1, LM33 + 4)
  Rd1ys[1, Lvec1] = Lvec
  Rd1ys[2:end, [1, LM33 + 3]] .= vGe
  Rd1ys[2:end, Lvec1] .= 0.0
  Rd1ys[2:end, LM33+4] = 1:nvG
  if order_smooth == 1
    Rd1ys[2:end, Lvec1] = RdpdtfvL0CDS(Rd1ys[2:end, Lvec1], ys, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0[1])
  elseif order_smooth == 2
    Rd2ys = copy(Rd1ys)
    Rd2ys[2:end, Lvec1] .= 0.0
    Rd2ys[2:end, Lvec1],Rd1ys[2:end, Lvec1] = RdpdtfvL0CDS(Rd2ys[2:end, Lvec1],Rd1ys[2:end, Lvec1],
                       ys, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
    a = Rd2ys[2:end, Lvec1]
    a[abs.(a).>Rdata_limit] .= Rdata_limit
    Rd2ys[2:end, Lvec1] = a
  else
    Rd2ys = copy(Rd1ys)
    Rd2ys[2:end, Lvec1] .= 0.0

    Rd3ys = copy(Rd1ys)
    Rd3ys[2:end, Lvec1] .= 0.0
    Rd3ys[2:end, Lvec1],Rd2ys[2:end, Lvec1],Rd1ys[2:end, Lvec1] = RdpdtfvL0CDS(Rd3ys[2:end, Lvec1],Rd2ys[2:end, Lvec1],Rd1ys[2:end, Lvec1],
                       ys, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
    a = Rd2ys[2:end, Lvec1]
    a[abs.(a).>Rdata_limit] .= Rdata_limit
    Rd2ys[2:end, Lvec1] = a

    a = Rd3ys[2:end, Lvec1]
    a[abs.(a).>Rdata_limit] .= Rdata_limit
    Rd3ys[2:end, Lvec1] = a
  end

  a = Rd1ys[2:end, Lvec1]
  a[abs.(a).>Rdata_limit] .= Rdata_limit
  Rd1ys[2:end, Lvec1] = a
end

# Plotting
nvec = vGe[2] .< vGe .< 5.73
yy0 = dtfvL0e0 ./ fvL0e
yy0[1, 2:end, :] = 2yy0[2, 2:end, :] - yy0[3, 2:end, :]
yys = dtfvL0e3D ./ fvL0e
yys[1, 2:end, :] = 2yys[2, 2:end, :] - yys[3, 2:end, :]
yy0a = yy0[:, :, isp33]
xlabel = "v̂"
pRdtfas(L1) = display(plot(vGe[nvec], yys[nvec, L1, isp33], line=(2, :auto), label=string("Rdₜf,L=", L1 - 1), xlabel=xlabel))
pdtfas(L1) = display(plot(vGe[nvec], dtfvLa[nvec, L1], line=(2, :auto), label=string("dₜf,L=", L1 - 1), xlabel=xlabel))

pdtfas00(L1) = plot(vGe[nvec], dtfvLa0[nvec, L1], line=(2, :auto), label=string("dₜf0,L=", L1 - 1))
pdtfas0(L1) = plot(vGe[nvec], dtfvLa[nvec, L1], line=(2, :auto),
    xlabel=string("nvc2=", (nvc3[5, L1, isp33])), label=string("dₜf,L=", L1 - 1))
pRdtfas00(L1) = plot(vGe[nvec], yy0[nvec, L1, isp33], line=(2, :auto), label=string("Rdₜf0,L=", L1 - 1))
pRdtfas0(L1) = plot(vGe[nvec], yys[nvec, L1, isp33], line=(2, :auto), label=string("Rdₜfs,L=", L1 - 1))
pfas0(L1) = plot(vGe[nvec], fvL0e[nvec, L1, isp33], line=(2, :auto), label=string("f,L=", L1 - 1), xlabel=xlabel)
pdt_Rdtfas(L1) = display(plot(pdtfas00(L1), pRdtfas0(L1), pdtfas0(L1), pfas0(L1), layout=(2, 2)))
display(pdt_Rdtfas(LM[isp33] + 1))

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
  pRdtfvL(n1, n9, LL1) = plot(vGe[n1:n9], dataRdtf[n1:n9, LL1], line=(2, :auto), 
                            xlabel=string("v̂(y=0)=",fmtf2(vGe[nvc3[4,LL1,isp33]])),label=string("RdₜfL_", LL1 - 1))
  pRd1y(n1, n9, LL1) = plot(vGe[n1:n9], Rd1y[n1:n9, LL1], line=(2, :auto),
                            xlabel=string("nvc1=",nvc3[3,LL1-1,isp33]), label=string("Rd1y"))
  pRd2y(n1, n9, LL1) = plot(vGe[n1:n9], Rd2y[n1:n9, LL1], line=(2, :auto),
                            xlabel=string("nvc2=",nvc3[5,LL1-1,isp33]), label=string("Rd2y"))
  if order_smooth == 3
    pRd3y(n1, n9, LL1) = plot(vGe[n1:n9], Rd3y[n1:n9, LL1], line=(2, :auto),
                              xlabel=string("nvc3=",nvc3[7,LL1-1,isp33]), label=string("Rd3y"))
  else
    pRd3y(n1, n9, LL1) = plot(vGe[n1:n9], 0Rd2y[n1:n9, LL1], line=(2, :auto),label=string("Rd3y"))
  end
  pRd1ys(n1, n9, LL1) = plot(vGe[n1:n9], Rd1ys[n1:n9, LL1], line=(4, :auto,:green),
                            xlabel=string("nvc1=",nvc3[3,LL1-1,isp33]), label=string("Rd1ys"))
  pRd2ys(n1, n9, LL1) = plot(vGe[n1:n9], Rd2ys[n1:n9, LL1], line=(2, :auto),
                            xlabel=string(fmtf1.(vGe[nvc3[3:2:end,LL1,isp33]])),  label=string("Rd2ys"))
  if order_smooth == 3
    pRd3ys(n1, n9, LL1) = plot(vGe[n1:n9], Rd3ys[n1:n9, LL1], line=(2, :auto),
                            xlabel=string("nvc3=",nvc3[7,LL1-1,isp33]), label=string("Rd3ys"))
  else
    pRd3ys(n1, n9, LL1) = plot(vGe[n1:n9], 0Rd2ys[n1:n9, LL1], line=(2, :auto),label=string("Rd3ys"))
  end
  pyys(n1, n9, LL1) = plot(vGe[n1:n9], yys[n1:n9, LL1-1,isp33], line=(4, :auto,:red), 
                            xlabel=string("vabth=",fmtf2(vabth33)),label=string("ys"))
  
  pfpdtfpRdtf(n1, n9, LL1) = display(plot(
    pfvL(n1, n9, LL1), pdtfvL(n1, n9, LL1), pRdtfvL(n1, n9, LL1),
    pRd1y(n1, n9, LL1), pRd2y(n1, n9, LL1), pRd3y(n1, n9, LL1), 
    # pRd1ys(n1, n9, LL1), pRd2ys(n1, n9, LL1), pRd3ys(n1, n9, LL1), 
    pRd1ys(n1, n9, LL1), pRd2ys(n1, n9, LL1), pyys(n1, n9, LL1), 
    layout=(3, 3)))
end

function datasL1nvc(datas, LL1)
  
  n1 = nvc3[1,LL1-1,isp33] + 1
  n9 = nvc3[5,LL1-1,isp33]
  n9 ≤ n1 ? n9 = nvc3[4,LL1-1,isp33] : 1
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

# pfpdtfpRdtf(nvc3[1,L1m,isp33]+1,nvc3[5,L1m,isp33],L1m)

nvc3a = [Lvec'; nvc3[:,Lvec,isp33]]

@show isp33, LM1, LM33, vabth33
@show fmtf2.(vGe[nvc3[3:2:end,L1,isp33]])
