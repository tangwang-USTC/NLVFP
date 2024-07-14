
vec01 = 0:0.01:1

dunnodes = 0.2
unnodes = 0:dunnodes:1

ncheby = 6
subk = vCmapping(vccn(ncheby+1;),0.0,1.0;isinv=true)[1:ncheby] * dunnodes

dcheby = dunnodes / ncheby
subgrids = 0:dcheby:1 |> Vector
for kk in 1:length(unnodes)-1
    subgrids[(1:ncheby) .+ (kk-1)*ncheby] = unnodes[kk] .+ subk
end

yy = 1.5
label = L"\hat{v}"
p01 = plot(vec01,yy * one.(vec01),xticks=false,label=label)
p01 = plot!(vec01,1 .+ yy * one.(vec01),xticks=false,label=label)
p01 = plot!(vec01,-1 .+ yy * one.(vec01),xticks=false,label=label)
label = "Chebyshev grids in the subinterval"
scatter!(subgrids,yy * one.(subgrids),color=:black,label=label)
label = "nodes"
scatter!(nodes,yy * one.(nodes),color=:red,label=label)

ii = 2
annotate!(unnodes[ii], yy+0.1, L"\hat{v}_{\alpha-1}")
ii = 3
annotate!(unnodes[ii], yy+0.1, L"\hat{v}_{\alpha}")
ii = 4
annotate!(unnodes[ii], yy+0.1, L"\hat{v}_{\alpha+1}")
ii = 5
annotate!(unnodes[ii], yy+0.1, L"\hat{v}_{\alpha+2}")

