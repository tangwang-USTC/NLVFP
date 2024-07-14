
k = 2
n = 96
α = 0.0 |> BigFloat
α = 0.0 |> Float64
endptv = left
# endptv = neither

v , w = laguerre(n,α,endptv)
fv2 = v.^2
f = exp.(-v.^2)
logf = log.(f)
nc = findfirst(isinf.(logf) .==1 )
if nc ≠ nothing
    itpDL = Spline1D(v[1:nc-1],logf[1:nc-1];k=k,bc="extrapolate")
    logf[nc:n] = itpDL.(v[nc:n])
end
f0 = exp.(logf)
@show norm(f0-f)

DL = laguerrediff(v;M=2)   # GL is DLp but GL-Radau is DLf
DL1 = DL[:,:,1]
DL2 = DL[:,:,2]
DLp = laguerrePolydiff(v)
DLf = laguerreFundiff(v)      # DLf[k,j] = = M * DLp[k,j] - 0.5δₖⱼ
                              #             M = exp(-vₖᵃ/2) / exp(-vⱼᵃ/2)

dft = -2v .* f                                 # 一阶导数理论值
dfg = DLp * f                                   # 直接求解
dflog =  f .* DLp * logf                        # 对数函数求解
dfdata = [v dft dflog dfg]  |> Array{Float64}
dfs = DataFrame(dfdata,:auto)
dropmissing!(dfs)
rename!(dfs,[1 => :v, 2=>:dft,3=>:dflog,4=>:dfg])
@show dfs
























1
