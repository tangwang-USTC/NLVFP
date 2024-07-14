

sss = 5
In = ones(sss)

μs, DPs2, Mμs, Muns = Dμ(sss-1;datatype = datatype)
cs0 = (μs .+ 1) / 2
DPs = DPs2[:,:,1]
paraM(DPs)

As, bs, cs = construct_GL(sss)
Asn = inv(As)


dts = - 1e-0

Deffn = Asn - dts * DPs
paraM(As)
paraM(Deffn)

Deff = inv(Deffn)
# paraM(Deff)

RMks = Deff * (Asn * In)

RMks = 1.0 .+ LinRange(0.0,0.1,sss)

DMk1s = RMks .- 1.0  |> Vector{Float64}

# DMk1s = dts * (As * (DPs * (In + DMk1s)))

Mk1s = deepcopy(RMks)

Mk1s_up = deepcopy(Mk1s)
Mk1s = In + dts * (As * (DPs * Mk1s))
Rerr = Mk1s ./ Mk1s_up .- 1.0

for ii in 1:10
    @show ii, norm(Rerr)
    if norm(Rerr) ≤ epsT100
        break
    end
    Mk1s = In + dts * (As * (DPs * Mk1s))
    Mk1s_up = deepcopy(Mk1s)
end
ppM = plot(Mk1s .- 1,line=(2,:auto))
ppDM = plot!(DMk1s,line=(2,:auto))
display(ppM)


