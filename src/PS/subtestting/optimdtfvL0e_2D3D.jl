"""
"""

isp33 = isp3
# isp33 = iFv3
is_boundv0 = false      # (= false, default). 
                        # It is `true` when `v[1] == 0.0` and the value `Rdiy[1] = 0.0`
order_dvδtf = 2         # (=2, default), `∈ [-1, 1, 2]` which denotes `[BackwardDiff, ForwardDiff, CentralDiff]`
Nsmooth = 2             # (=2, default), which is `∈ N⁺ + 1`, the number of points to smooth the function `δtfvL`.
order_smooth = 2        # (=2, default), which is `∈ N⁺`, the highest order of smoothness to smooth the function `δtfvL`.
order_smooth_itp = 2    # (1,2,3,23) → (nvcd1,nvcd2,nvcd3,max(nvcd2,nvcd3))
k_δtf = 2               # (=2, default), the order of the Spline Interpolations for `δtfLn(v̂→0)`
Nitp = 10               # (=10, default), the number of gridpoints to generate the interpolating function for `δtfLn(v̂→0)`
nvc0_limit = 4          # (=4, default), `nvc0_limit ∈ N⁺` which is the lower bound of
                        # `nvc(order_smooth_itp)` to applying to the extrapolation for `δtfLn(v̂→0)`
L1nvc_limit = 3         # (=2, default), `L1nvc_limit ∈ N⁺` which is the lower bound of `L` to applying to the extrapolation for `δtfLn(v̂→0)`
if order_smooth == 2
    abstol_Rdy = [0.95, 0.5]      # ∈ (0, 1.0 → 10). 
                        # When `abstol_Rdy > 0.5,` the change of `δtfvL` will be very sharp and localized.
                        # When `û → 0.0`, the higher-order components 
                        # When `û ≥ 0.5 ≫ 0.0`, lower value , `abstol_Rdy < 0.5` is proposed.
elseif order_smooth == 3
    abstol_Rdy = [0.95, 0.5, 0.5]
else
    erfghjmk
end
LM33 = LM[isp33]
@show isp33,LM33
Lvec = 1:LM33+1
# 3D 
nvc3 = zeros(Int64,2(order_smooth+1),LM1,ns)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
δtfa3 = copy(δtfvL0e)
optimdtfvL0e!(nvc3,δtfa3,fvL0e,vGe,nvG,ns,LM;
            orders=order_dvδtf,is_boundv0=is_boundv0,
            Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy,
            k=k_δtf,Nitp=Nitp,order_smooth_itp=order_smooth_itp,
            nvc0_limit=nvc0_limit,L1nvc_limit=L1nvc_limit)

# 2D 
nvc = zeros(Int64,2(order_smooth+1),LM33+1)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
δtfa = copy(δtfvL0e[:, Lvec, isp33])
optimdtfvL0e!(nvc,δtfa,fvL0e[:,Lvec,isp33],vGe,nvG,LM33;
            orders=order_dvδtf,is_boundv0=is_boundv0,
            Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy,
            k=k_δtf,Nitp=Nitp,order_smooth_itp=order_smooth_itp,nvc0_limit=nvc0_limit)

dtMsnnE32 = zeros(datatype,njMs,LM1,ns)
dtMsnnE32 = MsnnEvens(dtMsnnE32,δtfa3,vGe,njMs,LM,LM1,ns;is_renorm=is_renorm)

[Lvec'; nvc]
yy0 = δtfvL0e ./ fvL0e
yy0[1,2:end,:] = 2yy0[2,2:end,:] - yy0[3,2:end,:]
yy = δtfa3 ./ fvL0e
yy[1,2:end,:] = 2yy[2,2:end,:] - yy[3,2:end,:]

# Plotting
nvec = vGe .< 6.99
xlabel = "v̂"
pRdtfas(L1) = display(plot(vGe[nvec],yy[nvec,L1,isp33],line=(2,:auto),label=string("Rdₜf,L=",L1-1),xlabel=xlabel))
pdtfas(L1) = display(plot(vGe[nvec],δtfa[nvec,L1],line=(2,:auto),label=string("dₜf,L=",L1-1),xlabel=xlabel))

pdtfas0(L1) = plot(vGe[nvec],δtfa[nvec,L1],line=(2,:auto),label=string("dₜf,L=",L1-1))
pRdtfas0(L1) = plot(vGe[nvec],yy[nvec,L1,isp33],line=(2,:auto),label=string("Rdₜf,L=",L1-1))
pfas0(L1) = plot(vGe[nvec],fvL0e[nvec,L1,isp33],line=(2,:auto),label=string("f,L=",L1-1),xlabel=xlabel)
pdt_Rdtfas(L1) = display(plot(pdtfas0(L1),pRdtfas0(L1),pfas0(L1),layout=(3,1)))

