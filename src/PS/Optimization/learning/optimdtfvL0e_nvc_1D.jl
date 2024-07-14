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
k_dtf = 2               # (=2, default) The order of the interpolations in the optimization process. For most cases, `k_dtf = 2` is the best value.
order_dvdtf = 2          # [-1, 1, 2]
                        # [BackwardDiff, ForwardDiff, CentralDiff]
Nsmooth = 3             # ∈ N⁺ + 1
order_smooth = 3        # ∈ N⁺, 
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

L1m = 10
Nitp = 10      # The number of grid points to be used to generalize the interpolations.
               # The value is responsible for the accuracy of the interpolation process and
               # the proposed value is `6 ~ 10` for the optimization process.


nvcL1 = zeros(Int64,2(order_smooth+1))  # `[nvcy00, nvcy0, nvcd1, nvcy1, nvcd2, nvcy2, nvcd3, nvcy3]`
yfLn = fvL0e[:,L1m,isp33]
ydtfLn = dtfvL0e[:,L1m,isp33]
nvcL1 = nvcfind(nvcL1,ydtfLn,yfLn,nvG,L1m;orders=order_dvdtf,is_boundv0=is_boundv0,
              Nsmooth=Nsmooth,order_smooth=order_smooth,abstol_Rdy=abstol_Rdy)
# 1
nvcL1[3] > 6 ? n11 = nvcL1[3] - 5 : n11 = nvcL1[3]
@show nvcL1, nvG, n11
# @show datasL1(datas,L1m)[n11:n11+27,:];

# ############################################    Interpolations for `dtfvL` when `i < nvci`
Rdtf = ydtfLn ./ yfLn
if yfLn[1] == 0.0
  Rdtf[1] = 2Rdtf[2] - Rdtf[3]
end
vec0 = nvcL1[5]
vec9 = min(vec0+Nitp,nvG)
@show (LM1, LM[isp33], L1m), (vec0,vec9),vec9-vec0
if vec0 ≥ 2
    vecitp = vec0:vec9
    y1 = copy(Rdtf)
    itpDL = Dierckx.Spline1D(vGe[vecitp],y1[vecitp];k=1,bc="extrapolate")  # [extrapolate, nearest]
    y1[1:vec0-1] = itpDL.(vGe[1:vec0-1])
    y2 = copy(Rdtf)
    itpDL = Dierckx.Spline1D(vGe[vecitp],y2[vecitp];k=2,bc="extrapolate")
    y2[1:vec0-1] = itpDL.(vGe[1:vec0-1])
    y3 = copy(Rdtf)
    itpDL = Dierckx.Spline1D(vGe[vecitp],y3[vecitp];k=3,bc="extrapolate")
    y3[1:vec0-1] = itpDL.(vGe[1:vec0-1])
    if Nitp ≥ 4
        y4 = copy(Rdtf)
        itpDL = Dierckx.Spline1D(vGe[vecitp],y4[vecitp];k=4,bc="extrapolate")
        y4[1:vec0-1] = itpDL.(vGe[1:vec0-1])
        if Nitp ≥ 5
            y5 = copy(Rdtf)
            itpDL = Dierckx.Spline1D(vGe[vecitp],y5[vecitp];k=5,bc="extrapolate")
            y5[1:vec0-1] = itpDL.(vGe[1:vec0-1])
            ys = [y1 y2 y3 y4 y5]
        end
    end
end

# Nitp, y2[1],y3[1]

# err_y = y - Rdtf

ys1max = 1.2 * maximum(abs.(ys[1,:]))
y0vec = copy(Rdtf)
veclimit = abs.(Rdtf) .> ys1max
y0vec[veclimit] .= ys1max * sign(ys[1,5])
label = (1:5)'
# plotting
if 1 == 1
    vecp = vGe .< 0.1
    py = plot(vGe[vecp], ys[vecp, :], ylabel="RdtfvL",
        line=(2, :auto), label=label, legend=legendtR)
    py = plot!(vGe[vecp], y0vec[vecp], line=(2, :auto), label="0")
    perry = plot(vGe[vecp], y1[vecp] - y3[vecp],
        line=(2, :auto), label=string("y1-y3,L=", L1m - 1))
    perry = plot!(vGe[vecp], y2[vecp] - y3[vecp], line=(2, :auto), label=string("y2-y3"))
    perry = plot!(vGe[vecp], y2[vecp] - y1[vecp], line=(2, :auto), label=string("y2-y1"))
    perry = plot!(vGe[vecp], y2[vecp] - y4[vecp], line=(2, :auto), label=string("y4-y3"))
    perry = plot!(vGe[vecp], y2[vecp] - y5[vecp], line=(2, :auto), label=string("y5-y3"), legend=legendtR)

    vecp = vGe .< 3.0
    py5 = plot(vGe[vecp], ys[vecp, :], xlabel="v̂", ylabel="RdtfvL", line=(2, :auto), legend=false)
    py5 = plot!(vGe[vecp], y0vec[vecp], line=(2, :auto), label="0")
    perry5 = plot(vGe[vecp], y1[vecp] - y3[vecp], xlabel="v̂",
        line=(2, :auto), legend=false)
    perry5 = plot!(vGe[vecp], y2[vecp] - y3[vecp], line=(2, :auto))
    perry5 = plot!(vGe[vecp], y2[vecp] - y1[vecp], line=(2, :auto))
    perry5 = plot!(vGe[vecp], y2[vecp] - y4[vecp], line=(2, :auto))
    perry5 = plot!(vGe[vecp], y2[vecp] - y5[vecp], line=(2, :auto))

    display(plot(py, perry, py5, perry5, layout=(2, 2)))
end

