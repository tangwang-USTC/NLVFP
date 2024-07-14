using ChebyshevApprox, SpecialFunctions, GaussQuadrature
using ToeplitzMatrices, LinearAlgebra, LinearAlgebraX
using Plots


path = "K:\\BaiduNetdiskWorkspace\\VFP0D3V"
cd(path)
include(joinpath(path,"mathematics\\mathematic.jl"))
##################           ChybyshevApprox
nG = 75              # Affect the Interpolations and differences
vdomain = [-1.0,1.0]
kind = 2
nc = nG
order = 53           # Affect the Interpolations and differences
                     #  but not the differences by matrices `Dc`

## Target Functions
f(v) = v^2
df(v) = 2v
If(v) = (v^3 .+ 1) / 3

println()
printstyled("kind=",kind,",nc=",nc,",order=",order,",vdom=",vdomain;color=:yellow)
println()

## Gauss nodes and weight for GaussQuadrature
if 1 == 1
    if kind == 1
        w(v) = 1 / √(1 - v^2)
        node_type = chebyshev_nodes
        wgc = π / nc * ones(nc)  # for the first kind of Gauss-Chebyshev quandrature
    elseif kind == 2
        w(v) = √(1 - v^2)
        node_type = chebyshev_extrema
        wgc = π / (nc - 1) * sin.((((1:nc) .-1)/(nc-1))*pi).^2  # for the first kind of Gauss-Chebyshev quandrature
    else
        error("No procedure for difference when using this kind ")
    end
    # ## Gauss-Chebyshev Quandrature
    # vcg, wgc = chebyshev_quadrature(node_type,nc,vdomain)
    # Shifted Chebyshev polynomials
    if kind == 1
        vc = chebyshev_nodes(nc,vdomain) # First
    elseif kind == 2
        vc = chebyshev_extrema(nc)    # Second
    else
        # vc = chebyshev_extended(nc)   # Third
        # vc = vertesi_nodes(nc)        # Fourth
    end
end
# weights for Chebyshev polynomials
if 1 == 1
    ftc = f.(vc)
    poly = chebyshev_polynomial(order,vc)  # Tₙ(x) where n = 0:order
    ## weight = zeros(order + 1), which will used in the tensor product polynomian.
    if kind == 1
        weight = chebyshev_weights(ftc,vc,order,vdomain)
        # weight = chebyshev_weights_threaded(ftc,vc,order,vdomain)
    elseif kind == 2
        weight = chebyshev_weights_extrema(ftc,vc,order,vdomain)
        # weight = chebyshev_weights_extrema_threaded(ftc,vc,order,vdomain)
    else
        weight = chebyshev_weights_vertesi(ftc,vc,order,vdomain)
        weight = chebyshev_weights_extended(ftc,vc,order,vdomain)
    end
end
## Structures and evaluating
if 1 == 1
    chebpoly = ChebPoly(weight,order,vdomain)
    # evaluating
    chebInterp = chebyshev_evaluate(chebpoly)
    # point = [vc[2]]
    # fcp = chebyshev_evaluate(weight,point,order,vdomain)
    fc = zero(vc)
    for i in 1:nc
        fc[i] = chebInterp([vc[i]])
    end
    err_f = (fc - ftc) ./ ftc
    isInf = isinf.(err_f)
    err_f[isInf] .= fc[isInf]
    isNan = isnan.(err_f)
    err_f[isNan] .= fc[isNan]
    label = string("ft")
    pf = plot(vc, ftc,label=label)
    label = string("fc")
    pf = plot!(vc, fc,label=label)
    # display(pf)
    @show  norm(err_f)
end
## derivatives
if 1 == 1
    dft = df.(vc)
    # # dfc22 = chebyshev_derivative(chebpoly,vc,1)
    # dfc = zero(vc)
    # for i in 1:nc
    #     point2 = [vc[i]]
    #     # deriv = chebyshev_derivative(weight,vc,point2,order,vdomain)
    #     dfc[i] = chebyshev_derivative(chebpoly,point2,1)
    # end
    Dc = chebyshevdiff(nc,M=1)
    dfc = reverse(Dc * ftc)
    if kind == 1
        error("When `kind = 1`, the difference matrix is absent now. Adding it!!!")
    end
    err_df = (dft -dfc) ./ dft
    isInf = isinf.(err_df)
    err_df[isInf] .= dfc[isInf]
    isNan = isnan.(err_df)
    err_df[isNan] .= dfc[isNan]
    dfs = DataFrame([err_df dft dfc],:auto)
    rename!(dfs,[1=>:err_df,2=>:dft,3=>:dfc])
    label = string("dft")
    pdf = plot(vc, dft,label=label)
    label = string("dfc")
    pdf = plot!(vc, dfc,label=label)
    @show norm(err_df)
end

## gradients
# grad = zeros(nc)
# for i in 1:nc
#     point = [vc[i]]
#     # grad1 = chebyshev_gradient(weight,point,order,vdomain)
#     @show chebyshev_gradient(chebpoly,point)
# end

## Quandrature
if 1 == 1
    Ift = If.(vc)
    wgc = wgc ./ w.(vc)
    wgc[1] = 0.0
    wgc[nc] = 0.0
    Ifc = cumsum(wgc .* f.(vc))
    err_If = (Ifc - Ift) ./ Ift
    isInf = isinf.(err_If)
    err_If[isInf] .= Ifc[isInf]
    isNan = isnan.(err_If)
    err_If[isNan] .= Ifc[isNan]
    @show norm(err_If[nc])
    Ifs = DataFrame([err_If Ift Ifc],:auto)
    rename!(Ifs,[1=>:err_If,2=>:Ift,3=>:Ifc])
    label = string("Ift")
    pIf = plot(vc, Ift,label=label)
    label = string("Ifc")
    pIf = plot!(vc, Ifc,label=label,legend=false)
end

display(plot(pf,pdf,pIf,layout=(3,1)))
