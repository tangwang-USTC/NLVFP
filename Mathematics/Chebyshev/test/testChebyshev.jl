# using FastChebInterp, SpecialFunctions, GaussQuadrature
# using Plots

# function maxerror(n::Int)
#     vG1 = vG[1]
#     vG9 = vG[nG]

#     f(v) = erf(v) ./ v
#     dfv(v) = exp.(-v.^2) ./ v - erf.(v) ./ v.^2
#     vcheb = chebpoints(n,vG1,vG9)
#     fcheb = chebinterp(f.(vcheb),vG1,vG9)

#     ft = f.(vG)
#     fc = fcheb.(vG)
#     dft = dfv.(vG)
#     dfc = zero(vG)
#     for i in 1:nG
#         ~, dfc[i] = chebgradient(fcheb,vG[i])
#     end
#     merrordf = maximum(dfc - dft)
#     # dataf = [ft fc]
#     merrorf = maximum(fc - ft)
#     return merrorf, merrordf
# end

# function maxerror(nvec::Vector{Int})
#     vG1 = vG[1]
#     vG9 = vG[nG]
#     v = vG
#         """
#           for function:  `f(v) = vⁿ erf(v)`
#               f(v) is ok when n ∈ []
#               ∂ᵥf(v) is ok when n ∈ [] when error < 1e-10
#                                     [1,2] when error < 1e-5

#               Bad for all.
#         """
#         # nn = 1
#         # f(v) = 1 / v^nn
#         # df(v) = - nn / v^(nn-1)
#         # title = string("1/v^",nn)
#     # elseif case == 2
#         # nn = 1
#         # f(v) = log(1 / v^nn)   # ∂ᵥlog(v^-n) = - n * v^n * v^(-n - 1) = n / v
#         # df(v) = - nn / v
#         # title = string("log(1/v^",nn,")")
#     # elseif case == 3
#         """
#           for function:  `f(v) = vⁿ erf(v)`

#               ∂ᵥf(v) is ok when n ∈ [-1,0,1]

#               f(v) is ok when n ∈ [-1:2] when error < 1e-10
#                                     [3,4] when error < 1e-5
#         """
#         # nn =  1                  # nn ∈ [1]   for f(v)
#         # f(v) = v^nn * erf(v)
#         # df(v) = 2/√π * exp(-v^2) * v^nn + nn * erf(v) / v^(nn - 1)
#         # title = string("v^",nn," × erf(v)")

#         """
#           for function:  `f(v) = vⁿ erf(v)`

#               ∂ᵥf(v) is ok when n ∈ []

#               f(v) is ok when n ∈ [] when error < 1e-10
#                                   [] when error < 1e-5
#         """
#     #     nn =  - 2                  # nn ∈ [1]   for f(v)
#     #     f(v) = v^nn * erf(v)
#     #     df(v) = 2/√π * exp(-v^2) * v^nn + nn * erf(v) / v^(nn - 1)
#     #     title = string("v^",nn," × erf(v)")
#     # # elseif case == 5
#     #     f(v) = sin(v)
#     #     df(v) = cos(v)
#     #     title = string("sin(v)")
#     # elseif case == 6
#         """
#           for function:  `f(v) = vⁿ exp(v)`

#               ∂ᵥf(v) is ok when n ∈ [0:9]

#               f(v) is ok when n ∈ [0:11] when error < 1e-10
#                                     [-1, 12:20] when error < 1e-5
#         """
#         α = 5
#         f(v) = v^α * exp(-v^2)           # ok
#         df(v) = α * v^(α-1) * exp(-v^2) - 2v^(α+1) * exp(-v^2)
#         if α == 0
#             If(v) = √π / 2 * erf(v)
#         else
#             If(v) = 0.5 * v^(1+α) * (v^2)^(-0.5-0.5α) * (gamma(0.5(1+α))-gamma(0.5(1+α),v^2))
#         end
#         title = string("v^",α," × exp(-v^2)")

#     """
#       for function:  `f(v) = vⁿ exp(v)`

#           ∂ᵥf(v) is ok when n ∈ [0:9]

#           f(v) is ok when n ∈ [0:11] when error < 1e-10
#                               [-1, 12:20] when error < 1e-5
#     """
#     #
#     # α = 0
#     #
#     # f(v) = log(v^α * exp(-v^2))
#     # # f(v) = (v^α * exp(-v^2))           # ok
#     # df(v) = α / v - 2v
#     # if α == 0
#     #     If(v) = √π / 2 * erf(v)
#     # else
#     #     If(v) = 0.5 * v^(1+α) * (v^2)^(-0.5-0.5α) * (gamma(0.5(1+α))-gamma(0.5(1+α),v^2))
#     # end
#     # title = string("v^",α," × exp(-v^2)")
#     ##############  end
#     dfc = zero(vG)
#     Ifc = zero(vG)
#     N = length(nvec)
#     merrorf = zeros(N)
#     merrordf = zeros(N)
#     for i in 1:N
#         n = nvec[i]
#         vcheb = chebpoints(n,vG1,vG9)
#         fcc = f.(vcheb)
#         @show fcc
#         ############
#         # n1 = findfirst(vcheb .> 3)
#         # @show n1
#         # islogfInf = isinf.(fcc[1:n1]) .== 1
#         # nc = findlast(islogfInf)
#         # if nc ≠ nothing
#         #     itpDL = Spline1D(v[nc+1:n1+5],fcc[nc+1:n1+5];k=k,bc="extrapolate")
#         #     fcc[1:nc] = itpDL.(vG[1:nc])
#         # end
#         # islogfInf = isinf.(fcc[n1:n]) .== 1
#         # nc = findfirst(islogfInf)
#         # @show nc
#         # if nc ≠ nothing
#         #     itpDL = Spline1D(v[1:n1+nc-1],fcc[1:n1+nc-1];k=k,bc="extrapolate")
#         #     fcc[n1+nc:n] = itpDL.(vG[n1+nc:n])
#         # end
#         ###############
#         fcheb = chebinterp(fcc,vG1,vG9)
#         for i in 1:nG
#             ~, dfc[i] = chebgradient(fcheb,vG[i])
#         end
#         merrordf[i] = maximum(dfc - df.(v))
#         # dataf = [f fc]
#         merrorf[i] = maximum(fc - f.(v))
#     end
#     return merrorf, merrordf, title
# end

# nG = 77
# vG, w1v = laguerre(nG,0.0)
# nvec= 5:5:500 |> Vector{Int}
# errorf, errordf,title = maxerror(nvec)
# xlabel=string("nC: Number of Chebyshev points")
# ylabel = string("Error between ChebInterp and Theory value")
# label = string("error_f")
# perr = plot(nvec,errorf)
# perr = plot(nvec,errorf,label=label,yscale=:log10)
# # perr = plot(nvec,errorf,label=label,title=title,xlabel=xlabel,ylabel=ylabel)
# # xlabel!(xlabel)
# # yscale=:log10
# label = string("Error_df")
# # perr = plot!(nvec,errordf,label=label,yscale=:log10)
# perr = plot!(nvec,errordf,label=label,yscale=:log10,title=title,xlabel=xlabel,ylabel=ylabel)
# display(perr)
