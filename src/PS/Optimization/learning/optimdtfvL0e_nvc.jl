"""
"""

# Applying the Explicit Newton method to solve the ODE equations
dt = 1e-0           # (=1e2 default) Which is normalized by `td ~ 1e10` for MCF plasma.
# `1e0 ~ 1e4` is proposed to be used for most situation.

isp33 = isp3
isp33 = iFv3
is_boundv0 = true
# is_boundv0 = false
L1m = 2
vmin = 0.5
vmax = 1.85
k_dtf = 2
s_dtf = 1e-2
order_dvdtf = 2               # [-1, 1, 2]
                             # [BackwardDiff, ForwardDiff, CentralDiff]
Rdata_limit = 5.0
Nsmooth = 3            # ∈ N⁺ + 1
order_smooth = 3       # ∈ N⁺, 
if order_smooth == 2
    abstol_Rdy = [0.95, 0.5]      # ∈ (0, 1.0 → 10). 
                        # When `abstol_Rdy > 0.5,` the change of `dtfvL` will be very sharp and localized.
                        # When `û → 0.0`, the higher-order components 
                        # When `û ≥ 0.5 ≫ 0.0`, lower value , `abstol_Rdy < 0.5` is proposed.
elseif order_smooth == 3
    abstol_Rdy = [0.95, 0.5, 0.5]
else
    erfghjmk
end

LM33 = LM[isp33]
@show LM1, LM33, L1m
# 2D
if 1 == 1
  Lvec = 1:LM33+1
  Lvec1 = 2:LM33+2

  RdtfvL0ek = dtfvL0e ./ fvL0e
  [fvL0e[1, L1, isp3] ≠ 0.0 || (RdtfvL0ek[1, L1, isp3] = 2RdtfvL0ek[2, L1, isp3] - RdtfvL0ek[3, L1, isp3]) for L1 in 1:LM1]
  [fvL0e[1, L1, iFv3] ≠ 0.0 || (RdtfvL0ek[1, L1, iFv3] = 2RdtfvL0ek[2, L1, iFv3] - RdtfvL0ek[3, L1, iFv3]) for L1 in 1:LM1]

  L1vec = 1:LM1
  dataf = [[0; vGe] [(L1vec)'; fvL0e[:, L1vec, isp33]] [0; vGe] 0:nvG]
  dataRdtf = [[0; vGe] [(L1vec)'; RdtfvL0ek[:, L1vec, isp33]] [0; vGe] 0:nvG]
  datadtf = [[0; vGe] [(L1vec)'; dtfvL0e[:, L1vec, isp33]] [0; vGe] 0:nvG]

  dvy = zeros(nvG + 1, LM33 + 4)
  yy = copy(dtfvL0e[:, Lvec, isp33])
  dvy[1, Lvec1] = Lvec
  dvy[2:end, [1, LM33 + 3]] .= vGe
  dvy[2:end, LM33+4] = 1:nvG
  dvy[2:end, Lvec1] = dvdtfvL0CDS(dvy[2:end, Lvec1], yy, nvG, dvGe, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=true)

  dy = copy(dvy)
  dy[2:end, Lvec1] .= 0.0
  dy[2:end, Lvec1] = ddtfvL0CDS(dy[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)

  Rdvy = copy(dvy)
  Rdvy[2:end, Lvec1] .= 0.0
  Rdvy[2:end, Lvec1] = RdvdtfvL0CDS(Rdvy[2:end, Lvec1], yy, nvG, dvGe, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
  a = Rdvy[2:end, Lvec1]
  a[abs.(a).>Rdata_limit] .= Rdata_limit
  Rdvy[2:end, Lvec1] = a

  Rddvy = copy(dvy)
  Rddvy[2:end, Lvec1] .= 0.0
  Rddvy[2:end, Lvec1] = RddvdtfvL0CDS(Rddvy[2:end, Lvec1], yy, nvG, dvGe, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
  a = Rddvy[2:end, Lvec1]
  a[abs.(a).>Rdata_limit] .= Rdata_limit
  Rddvy[2:end, Lvec1] = a

  Rdy = copy(dvy)
  Rdy[2:end, Lvec1] .= 0.0
  Rdy[2:end, Lvec1] = RddtfvL0CDS(Rdy[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
  a = Rdy[2:end, Lvec1]
  a[abs.(a).>Rdata_limit] .= Rdata_limit
  Rdy[2:end, Lvec1] = a

  Rddy = copy(dvy)
  Rddy[2:end, Lvec1] .= 0.0
  Rddy[2:end, Lvec1] = RdddtfvL0CDS(Rddy[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
  a = Rddy[2:end, Lvec1]
  a[abs.(a).>Rdata_limit] .= Rdata_limit
  Rddy[2:end, Lvec1] = a

  Rd2y = copy(dvy)
  Rd2y[2:end, Lvec1] .= 0.0
  Rd2y[2:end, Lvec1] = RdRddtfvL0CDS(Rd2y[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
  a = Rd2y[2:end, Lvec1]
  a[abs.(a).>Rdata_limit] .= Rdata_limit
  Rd2y[2:end, Lvec1] = a

  Rd3y = copy(dvy)
  Rd3y[2:end, Lvec1] .= 0.0
  Rd3y[2:end, Lvec1] = RdRdRddtfvL0CDS(Rd3y[2:end, Lvec1], yy, nvG, fvL0e[:, Lvec, isp33], LM33; orders=order_dvdtf, is_boundv0=is_boundv0)
  a = Rd3y[2:end, Lvec1]
  a[abs.(a).>Rdata_limit] .= Rdata_limit
  Rd3y[2:end, Lvec1] = a

  datasm = zeros(nvG, 12)
  datanames = [:n, :v, :fL, :dtfL, :Rdtf, :dvy, :dy, :Rdvy, :Rdy, :Rd2y, :Rd3y, :nn]
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
  datas[:, i] = dvy[2:end, LL1]
  i += 1
  datas[:, i] = dy[2:end, LL1]
  i += 1
  datas[:, i] = Rdvy[2:end, LL1]
  i += 1
  datas[:, i] = Rdy[2:end, LL1]
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
  pfvL(n1, n9, LL1, LL9) = plot(vGe[n1:n9], neps * dataf[n1:n9, LL1:LL9], line=(2, :auto), label=(LL1-1:LL9-1)')
  pfvL(n1, n9, LL1) = plot(vGe[n1:n9], neps * dataf[n1:n9, LL1], line=(2, :auto), label=string("fL_", LL1 - 1))
  if isp33 == 1
    pdtfvL(n1, n9, LL1, LL9) = plot(vGe[n1:n9], neps * datadtf[n1:n9, LL1:LL9], line=(2, :auto), label=(LL1-1:LL9-1)', ylabel="dₜfₐ")
    pdtfvL(n1, n9, LL1) = plot(vGe[n1:n9], neps * datadtf[n1:n9, LL1], line=(2, :auto), label=string("dₜfL_", LL1 - 1), ylabel="dₜfₐ")
  else
    pdtfvL(n1, n9, LL1, LL9) = plot(vGe[n1:n9], neps * datadtf[n1:n9, LL1:LL9], line=(2, :auto), label=(LL1-1:LL9-1)', ylabel="dₜfᵦ")
    pdtfvL(n1, n9, LL1) = plot(vGe[n1:n9], neps * datadtf[n1:n9, LL1], line=(2, :auto), label=string("dₜfL_", LL1 - 1), ylabel="dₜfᵦ")
  end
  pRdtfvL(n1, n9, LL1) = plot(vGe[n1:n9], dataRdtf[n1:n9, LL1], line=(2, :auto), label=string("RdₜfL_", LL1 - 1))
  pdvy(n1, n9, LL1) = plot(vGe[n1:n9], dvy[n1:n9, LL1], line=(2, :auto), label=string("dvy_", LL1 - 1))
  pdy(n1, n9, LL1) = plot(vGe[n1:n9], dy[n1:n9, LL1], line=(2, :auto), label=string("dy_", LL1 - 1))
  pRdvy(n1, n9, LL1) = plot(vGe[n1:n9], Rdvy[n1:n9, LL1], line=(2, :auto), label=string("Rdvy_", LL1 - 1))
  pRdy(n1, n9, LL1) = plot(vGe[n1:n9], Rdy[n1:n9, LL1], line=(2, :auto),xlabel=string("n1,n9=",(n1-1, n9)), label=string("Rdy_", LL1 - 1))
  pRddvy(n1, n9, LL1) = plot(vGe[n1:n9], Rddvy[n1:n9, LL1], line=(2, :auto), label=string("Rddvy_", LL1 - 1))
  pRddy(n1, n9, LL1) = plot(vGe[n1:n9], Rddy[n1:n9, LL1], line=(2, :auto), label=string("Rddy_", LL1 - 1))
  pRd2y(n1, n9, LL1) = plot(vGe[n1:n9], Rd2y[n1:n9, LL1], line=(2, :auto), label=string("Rd2y_", LL1 - 1))
  pRd3y(n1, n9, LL1) = plot(vGe[n1:n9], Rd3y[n1:n9, LL1], line=(2, :auto), label=string("Rd3y_", LL1 - 1))

  pfpdtfpRdtf(n1, n9, LL1) = display(plot(
    pfvL(n1, n9, LL1), pdtfvL(n1, n9, LL1), pRdtfvL(n1, n9, LL1),
    pdvy(n1, n9, LL1), pdy(n1, n9, LL1), pRdvy(n1, n9, LL1),
    pRdy(n1, n9, LL1), pRd2y(n1, n9, LL1), pRd3y(n1, n9, LL1), 
    layout=(3, 3)))
end
# 1D
nvcL1 = zeros(Int64,2(order_smooth+1))  # `[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3]`
yfLn = fvL0e[:,L1m,isp33]
ydtfLn = dtfvL0e[:,L1m,isp33]
nvcL1 = nvcfind(nvcL1,ydtfLn,yfLn,nvG,L1m;orders=order_dvdtf,is_boundv0=is_boundv0,
              Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)
# 1
nvcL1[3] > 6 ? n11 = nvcL1[3] - 5 : n11 = nvcL1[3]
@show nvcL1, nvG, n11
@show datasL1(datas,L1m)[n11:n11+27,:];

Rdtf = ydtfLn ./ yfLn
if yfLn[1] == 0.0
  Rdtf[1] = 2Rdtf[2] - Rdtf[3]
end
if nvcL1[3] ≥ 2
    vec0 = nvcL1[3]
    vec9 = min(vec0+15,nvG)
    vecitp = vec0:vec9
    y = copy(Rdtf)
    itpDL = Dierckx.Spline1D(vGe[vecitp],y[vecitp];k=k_dtf,s=s_dtf,bc="extrapolate")  # [extrapolate, nearest]
    y[1:vec0-1] = itpDL.(vGe[1:vec0-1])
end


# 2D 
nvc = zeros(Int64,2(order_smooth+1),LM33+1)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
nvc = nvcfind(nvc,dtfvL0e[:,1:LM33+1,isp33],fvL0e[:,1:LM33+1,isp33],nvG,LM33;
            orders=order_dvdtf,is_boundv0=is_boundv0,Nsmooth=Nsmooth,
            order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)

function pfpdtfpRdtf(Lvec)
    
    if typeof(Lvec) == Int64
        LL1 = Lvec
        n1 = nvc[1,LL1] + 1
        n9 = nvc[5,LL1]
        # n1 = max(2,nvc[1,LL1])
        # n9 = max(7,nvc[5,LL1])
        n9 ≤ n1 ? n9 = nvc[4,LL1-1] : 1
        display(plot(pfvL(n1, n9, LL1), pdtfvL(n1, n9, LL1), pRdtfvL(n1, n9, LL1),
               pdvy(n1, n9, LL1), pdy(n1, n9, LL1), pRdvy(n1, n9, LL1),
               pRdy(n1, n9, LL1), pRd2y(n1, n9, LL1), pRd3y(n1, n9, LL1), 
               layout=(3, 3)))
    else
        for LL1 in Lvec
            # n1 = max(2,nvc[1,LL1])
            # n9 = max(7,nvc[5,LL1])
            n1 = nvc[1,LL1-1] + 1
            n9 = nvc[5,LL1-1]
            n9 ≤ n1 ? n9 = nvc[4,LL1-1] : 1
            # @show LL1-1,n1,n9
            display(plot(pfvL(n1, n9, LL1), pdtfvL(n1, n9, LL1), pRdtfvL(n1, n9, LL1),
                   pdvy(n1, n9, LL1), pdy(n1, n9, LL1), pRdvy(n1, n9, LL1),
                   pRdy(n1, n9, LL1), pRd2y(n1, n9, LL1), pRd3y(n1, n9, LL1), 
                   layout=(3, 3)))
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
  datas[nv19, i] = dvy[nv19, LL1]
  i += 1
  datas[nv19, i] = dy[nv19, LL1]
  i += 1
  datas[nv19, i] = Rdvy[nv19, LL1]
  i += 1
  datas[nv19, i] = Rddy[nv19, LL1]
  i += 1
  datas[nv19, i] = Rdy[nv19, LL1]
  i += 1
  datas[nv19, i] = Rd2y[nv19, LL1]
  i += 1
  datas[nv19, i] = datas[nv19, 1]
  return datas[nv19,:]
end

pfpdtfpRdtf(nvc[1,L1m]+1,nvc[5,L1m],L1m)

[Lvec'; nvc]
