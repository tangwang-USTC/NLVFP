using ChebyshevApprox, SpecialFunctions, GaussQuadrature
using ToeplitzMatrices, LinearAlgebra, LinearAlgebraX
using Plots,DataFrames

path = "K:\\BaiduNetdiskWorkspace\\VFP0D3V"
cd(path)
include(joinpath(path,"mathematics\\mathematic.jl"))
##################           ChybyshevApprox
nG = 125              # Number of Gauss-Laguerre pionts
α = 0.0
endptv = neither
# Chebyshev grids
kind = 2
nc = nG              # Affect the Interpolations and differences
L = 2                # factor for function: `f(v) = v^L * exp^(-v^2)`
orders = 51           # Affect the Interpolations and differences
                     #  but not the differences by matrices `Dc`
vdomain = [1.0,-1.0] # [0, ∞] → [1, -1]
## Mapping vdomain ro `dom` with vG
vG, wG = laguerre(nG,α, endptv)
dom = [vG[1], vG[nG]]
# Mapping to domain `x ∈ [-1,1]` according to transform `v(x) = 0.5(a-b)*(x+1) +b`
#  where `a,b` is the lower and higher boundary of domain `v ∈ [a,b]`.
b = vG[nG]
a = vG[1]
b = 10             # Generally, `f(vG > 10s) ≈ 0`
a = 0              # (default)
## Target Functions where `v ∈ [0,∞]`
f0(v) = v^L * exp(-v^2)
df0(v) = (L * v^(L-1) - 2 * v^(L+1)) * exp(-v^2)
If0(v) = 0.5((-1)^(L+1) * gamma((L+1)/2,1) + (-1)^L * gamma((L+1)/2,v^2))
plot(vG,f0.(vG))
## Mapping to domoin `x ∈ [1,-1]` by transforms: `v(x) = 0.5(a-b)*(x+1) +b`
if a == 0
    vGc(x,b) = b  * (1-x) / 2.0         # `b` is the upper-boundary of domain `dom`,
                                      # which should be `∞` in theory. Here, `a=0`.
    f(x,b) = vGc(x,b)^L * exp(-vGc(x,b)^2.0)
    # ML0(x,b) = - 2/√π * b * ∫₁⁻¹dx (v²(x,b) * exp(- v²(x,b)))
    # df = ∂f(x,b)/∂x
    df(x,b) = 2.0^(-1-L) * b^L * (1.0-x)^(L-1) * (2L - b^2.0 * (x-1)^2.0) * exp(-(vGc(x,b))^2.0)
    if L - 2 == 0
        If(x,b) = -1 + erfc(- vGc(x,b)) - 2/√π * vGc(x,b) * exp(-(vGc(x,b))^2)
    elseif L - 2 == 2
        If(x,b) = (-6erf(- vGc(x,b)) - 2/√π * vGc(x,b)* (6+b^2*(x-1)^2) * exp(-(vGc(x,b))^2)) / 4
    elseif L - 2 == - 2
        If(x,b) = - 2erf(- vGc(x,b))
    end
    f(x) = f(x,b)
    df(x) = df(x,b)
    If(x) = If(x,b)
else
    v(x,a,b) = 0.5(a - b)  * (1+x) + b  # `a` is the lower-boundary and `b` is the upper-boundary of domain `dom`,
                                        # which should be `∞` in theory.
    f(x,a,b) = v(x,a,b)^L * exp(-v(x,a,b)^2.0)
    # ML0(x,a,b) =  ∫₋₁¹dx (v²(x,a,b) * exp(- v²(x,a,b)))
    # df = ∂f(x,a,b)/∂x
    df(x,a,b) = - 2.0^(-1-L) * b^L * (1.0-x)^(L-1) * (2L - b^2.0 * (x-1)^2.0) * exp(-(v(x,a,b))^2.0)

    f(x) = f(x,a,b)
    df(x) = df(x,a,b)
    If(x) = If(x,a,b)
end

println()
printstyled("kind=",kind,",factor,L=",L,",nc=",nc,",orders=",orders,",vdom=",vdomain;color=:yellow)
println()

## Gauss nodes and weight for GaussQuadrature
if 1 == 1
    if kind == 1
        w(x) = 1 / √(1 - x^2)
        node_type = chebyshev_nodes
        wc = π / nc * ones(nc)  # for the first kind of Gauss-Chebyshev quandrature
        wc ./= w.(vc)
    elseif kind == 2
        node_type = chebyshev_extrema
        # # rate = sin.(π * ((1:nc) .-1)/(nc-1)) ./ √(1 .- vc.^2)
        # wc = π / (nc - 1) * sin.(π * ((1:nc) .-1)/(nc-1)).^2  # for the first kind of Gauss-Chebyshev quandrature
        w(x) = √(1 - x^2)
        # wvc = w.(vc)
        # dataw = [wc wvc wc ./ wvc.^2]
        # # wc = wc / w(v), however, w(vᵢ) maybe has a zero value.
        # wc ./= wvc.^2
        # wc[1] = wc[2]
        # wc[nc] = wc[nc-1]
        # wc .*= wvc
        ## Taht is:
        wc = π / (nc - 1) * sin.(π * ((1:nc) .-1)/(nc-1))
    else
        error("No procedure for difference when using this kind ")
    end
    # ## Gauss-Chebyshev Quandrature
    # vcg, wc = chebyshev_quadrature(node_type,nc,vdomain)
    # Shifted Chebyshev polynomials
    if kind == 1
        vc = chebyshev_nodes(nc,vdomain) # First
    elseif kind == 2
        vc = chebyshev_extrema(nc,vdomain)    # Second
    else
        # vc = chebyshev_extended(nc)   # Third
        # vc = vertesi_nodes(nc)        # Fourth
    end
end
vcG(x) = 0.5(a - b) * (x .+ 1) .+ b # mapping `vc` to `vG` in domain `dom`.
vG = vcG.(vc)
# if vG[1] > vG[nG]
#     vG = reverse(vG)
# end

vcint = -0.9:0.1:0.9 |> Vector{Float64}
vGint = vcG.(vcint)
# weights for Chebyshev polynomials
if 1 == 1
    ft = f.(vc)
    poly = chebyshev_polynomial(orders,vc)  # Tₙ(x) where L = 0:orders
    ## weight = zeros(orders + 1), which will used in the tensor product polynomian.
    if kind == 1
        weight = chebyshev_weights(ft,vc,orders,vdomain)
        # weight = chebyshev_weights_threaded(ftc,vc,orders,vdomain)
    elseif kind == 2
        weight = chebyshev_weights_extrema(ft,vc,orders,vdomain)
        # weight = chebyshev_weights_extrema_threaded(ftc,vc,orders,vdomain)
    else
        weight = chebyshev_weights_vertesi(ft,vc,orders,vdomain)
        weight = chebyshev_weights_extended(ft,vc,orders,vdomain)
    end
end
xlabel = string("Chebyshev grids: vc")
xlabelG = string("Velocity axis grids: vG")
# title = string("f(x) = x^n * exp(-x^2)")
title = string("f(x) = xⁿ exp(-x²)")
vG7 = 1e-4 .< vGint .< 7.51
## Structures and evaluating `f(x)`
if 1 == 1
    chebpoly = ChebPoly(weight,orders,vdomain)
    # evaluating
    chebInterp = chebyshev_evaluate(chebpoly)
    # point = [vcint[2]]
    # fcp = chebyshev_evaluate(weight,point,orders,vdomain)
    fc = zero(vcint)
    ncint = length(vcint)
    for i in 1:ncint
        fc[i] = chebInterp([vcint[i]])
    end
    ftc = f.(vcint)
    Rerr_f = (fc - ftc) ./ ftc
    isInf = isinf.(Rerr_f)
    Rerr_f[isInf] .= fc[isInf]
    isNan = isnan.(Rerr_f)
    Rerr_f[isNan] .= fc[isNan]
    label = string("Rel_err_f")
    pRerrf = plot(vGint[vG7],abs.(Rerr_f[vG7]).+eps(Float64),yaxis=:log10,label=label,xlabel=xlabelG,title=title)
    fs = DataFrame([vGint vcint ftc fc Rerr_f],:auto)
    rename!(fs,[1=>:vG,2=>:vc,3=>:ft,4=>:fc,5=>:Rerr_f])
    label = string("ft")
    pf = plot(vGint, ftc,label=label,xlabel=xlabelG)
    label = string("fc")
    pf = plot!(vGint, fc,label=label,title=title)
    # display(pf)
    @show  norm(Rerr_f)
end
## Evaluating `f(x)` throught `log(f(x))`
if 1 == 1
    ft = f.(vc)
    ftlog = log.(ft)
    ## weight = zeros(orders + 1), which will used in the tensor product polynomian.
    if kind == 1
        weight = chebyshev_weights(ftlog,vc,orders,vdomain)
        # weight = chebyshev_weights_threaded(ftlog,vc,orders,vdomain)
    elseif kind == 2
        weight = chebyshev_weights_extrema(ftlog,vc,orders,vdomain)
        # weight = chebyshev_weights_extrema_threaded(ftlog,vc,orders,vdomain)
    else
        weight = chebyshev_weights_vertesi(ftlog,vc,orders,vdomain)
        weight = chebyshev_weights_extended(ftlog,vc,orders,vdomain)
    end
    chebpoly = ChebPoly(weight,orders,vdomain)
    # evaluating function
    chebInterp = chebyshev_evaluate(chebpoly)
    # point = [vc[2]]
    # fcp = chebyshev_evaluate(weight,point,orders,vdomain)
    fclog = zero(vcint)
    for i in 1:ncint
        fclog[i] = chebInterp([vcint[i]])
    end
    fc = exp.(fclog)
    ftc = f.(vcint)
    Rerr_flog = (fc - ftc) ./ ftc
    isInf = isinf.(Rerr_f)
    Rerr_flog[isInf] .= fc[isInf]
    isNan = isnan.(Rerr_f)
    Rerr_flog[isNan] .= fc[isNan]
    label = string("Rel_err_flog")
    pRerrf = plot(vGint[vG7],abs.(Rerr_f[vG7]).+eps(Float64),yaxis=:log10,label=label,xlabel=xlabelG,title=title)
    fslog = DataFrame([vGint vcint ftc fc Rerr_flog],:auto)
    rename!(fslog,[1=>:vG,2=>:vc,3=>:ft,4=>:fc,5=>:Rerr_f])
    label = string("ft")
    pf = plot(vGint, ftc,label=label,xlabel=xlabelG)
    label = string("fclog")
    pf = plot!(vGint, fc,label=label,title=title)
    # display(pf)
    @show  norm(Rerr_flog)
end
## derivatives
vG7 = 1e-4 .< vG .< 7.51
if 1 == 1
    ftc = f.(vc)
    dft = df.(vc)
    # # dfc22 = chebyshev_derivative(chebpoly,vc,1)
    # dfc = zero(vc)
    # for i in 1:nc
    #     point2 = [vc[i]]
    #     # deriv = chebyshev_derivative(weight,vc,point2,orders,vdomain)
    #     dfc[i] = chebyshev_derivative(chebpoly,point2,1)
    # end
    Dc = chebyshevdiff(nc,M=1)
    dfc = Dc * ftc
    if kind == 1
        error("When `kind = 1`, the difference matrix is absent now. Adding it!!!")
    end
    Rerr_df = (dft -dfc) ./ dft
    isInf = isinf.(Rerr_df)
    Rerr_df[isInf] .= dfc[isInf]
    isNan = isnan.(Rerr_df)
    Rerr_df[isNan] .= dfc[isNan]
    label = string("Rel_err_df")
    pRerrdf = plot(vc[vG7], abs.(Rerr_df[vG7]).+eps(Float64),yaxis=:log10,label=label,xaxis=(:flip))
    dfs = DataFrame([vG vc dft dfc Rerr_df],:auto)
    rename!(dfs,[1=>:vG,2=>:vc,3=>:dft,4=>:dfc,5=>:Rerr_df])
    label = string("dft")
    # pdf = plot(vc, dft,label=label)
    pdf = plot(vc, dft,label=label,xaxis=(:flip))
    label = string("dfc")
    pdf = plot!(vc, dfc,label=label)
    @show norm(Rerr_df)
end
# display(plot(pf,pdf,layout=(2,1)))

## gradients
# grad = zeros(nc)
# for i in 1:nc
#     point = [vc[i]]
#     # grad1 = chebyshev_gradient(weight,point,orders,vdomain)
#     @show chebyshev_gradient(chebpoly,point)
# end

## Quandrature
if 1 == 1
    Ift = reverse(If.(reverse(vc)))
    cI = 4π / π^(3/2) * (b/2)
    Ifc = reverse(cI * cumsum(wc .* f.(reverse(vc))))
    # Ifc = cumsum(wc .* f.(reverse(vc)))
    Rerr_If = (Ifc - Ift) ./ Ift
    isInf = isinf.(Rerr_If)
    Rerr_If[isInf] .= Ifc[isInf]
    isNan = isnan.(Rerr_If)
    Rerr_If[isNan] .= Ifc[isNan]
    label = string("Rel_err_If")
    pRerrIf = plot(vc[vG7], abs.(Rerr_If[vG7]).+eps(Float64),xlabel=xlabel,yaxis=:log10,label=label,xaxis=(:flip))
    Ifs = DataFrame([vG vc Ift Ifc Rerr_If],:auto)
    rename!(Ifs,[1=>:vG,2=>:vc,3=>:Ift,4=>:Ifc,5=>:Rerr_If])
    label = string("Ift")
    pIf = plot(vc, Ift,label=label,xaxis=(:flip))
    label = string("Ifc")
    pIf = plot!(vc, Ifc,label=label,xlabel=xlabel)
    @show norm(Rerr_If)
end

display(plot(pf,pdf,pIf,layout=(3,1)))
display(plot(pRerrf,pRerrdf,pRerrIf,layout=(3,1)))
