"""
"""

# Applying the Explicit Newton method to solve the ODE equations
dt = 1e-0           # (=1e2 default) Which is normalized by `td ~ 1e10` for MCF plasma.
# `1e0 ~ 1e4` is proposed to be used for most situation.

isp33 = isp3
isp33 = iFv3
is_boundv0 = true
# is_boundv0 = false
vmin = 0.5
vmax = 1.85
order_dvdtf = 2          # [-1, 1, 2]
                        # [BackwardDiff, ForwardDiff, CentralDiff]
Nsmooth = 3             # ∈ N⁺ + 1
order_smooth = 3        # ∈ N⁺, 
k_dtf = 2
Nitp = 10               # number of interpolating grids.
order_smooth_itp = 2    # (1,2,3) → (nvcd1,nvcd2,nvcd3)
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
@show LM1, LM33
Lvec = 1:LM33+1
# 2D 
yy = dtfvL0e[:, Lvec, isp33] ./ fvL0e[:, Lvec, isp33]
yy[1,2:end] = yy[2,2:end]
yy0 = copy(yy)
nvc = zeros(Int64,2(order_smooth+1),LM33+1)  # `[[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3], ⋯ ]`
nvc = nvcfind(nvc,yy,fvL0e[:,Lvec,isp33],nvG,LM33;
            orders=order_dvdtf,is_boundv0=is_boundv0,Nsmooth=Nsmooth,
            order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)

[Lvec'; nvc]

yy = RdtfvL0interp(yy,vGe,nvG,nvc[[2;3:2:end],:],LM33;k=k_dtf,Nitp=Nitp,order_smooth_itp=order_smooth_itp)
dtfa = yy .* fvL0e[:, Lvec, isp33]

nvec = vGe .< 0.99
xlabel = "v̂"
pRdtfas(L1) = display(plot(vGe[nvec],yy[nvec,L1],line=(2,:auto),label=string("Rdₜf,L=",L1-1),xlabel=xlabel))
pdtfas(L1) = display(plot(vGe[nvec],dtfa[nvec,L1],line=(2,:auto),label=string("dₜf,L=",L1-1),xlabel=xlabel))

pdtfas0(L1) = plot(vGe[nvec],dtfa[nvec,L1],line=(2,:auto),label=string("dₜf,L=",L1-1))
pRdtfas0(L1) = plot(vGe[nvec],yy[nvec,L1],line=(2,:auto),label=string("Rdₜf,L=",L1-1))
pfas0(L1) = plot(vGe[nvec],fvL0e[nvec,L1,isp33],line=(2,:auto),label=string("f,L=",L1-1),xlabel=xlabel)
pdt_Rdtfas(L1) = display(plot(pdtfas0(L1),pRdtfas0(L1),pfas0(L1),layout=(3,1)))

# # nvec = vGe .< 2.0
# pdtff = plot(vGe[nvec],dtfa[nvec,:],line=(2,:auto),label=Lvec')
# pyy = plot(vGe[nvec],yy[nvec,:],line=(2,:auto),label=Lvec',xlabel=xlabel,legend=false)
# display(plot(pdtff,pyy,layout=(2,1)))
