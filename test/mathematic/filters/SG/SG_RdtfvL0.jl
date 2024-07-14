using SavitzkyGolay

LL1 = 5
y0 = copy(RdtfvL0[:, LL1, isp3])
isinfy = isinf.(y0)
ninf = sum(isinfy)
isnany = isnan.(y0)
nnan = sum(isnany)
ny = length(y0)
if ny === (ninf+nnan)
    1
else
    vv = vG0[ninf+nnan+1:ny]
    y = y0[ninf+nnan+1:ny]
    # y = sg.y
    p_order = 1     # which is ∈ N⁺
    w_size = 3      # (≥ p_order + 2), a odd number which is ∈ N⁺.
    w_size - p_order ≥ 2 || (w_size = p_order + 2)
    isodd(w_size) || (w_size += 1)
    sgfilter = SGolay(w_size, p_order)
    # sg = savitzky_golay(y, w_size, p_order)
    sg = sgfilter(y)
    Rdy = (y./sg.y .-1)
    [y sg.y y - sg.y Rdy]

    @show w_size,p_order,LL1,norm(y - sg.y)
    pRdy = plot(vv,Rdy,label="Rdy")
    py = plot(vv, [y sg.y], ylabel=string("RδₜfvL0, L =",LL1-1),
              xlabel="v̂",line=(2,:auto),label=["Original signal" "Filtered signal"])
    display(plot(pRdy,py,layout=(2,1)))
end
