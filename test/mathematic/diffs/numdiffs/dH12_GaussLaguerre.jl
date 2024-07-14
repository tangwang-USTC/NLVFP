DL = laguerrediff(vG;M=2)   # GL is DLp but GL-Radau is DLf
DL1 = DL[:,:,1]
DL2 = DL[:,:,2]
DLp = laguerrePolydiff(vG)
DLf = laguerreFundiff(vG)      # DLf[k,j] = = M * DLp[k,j] - 0.5δₖⱼ
                              #             M = exp(-vₖᵃ/2) / exp(-vⱼᵃ/2)

## For Maxwellian distribution where f(v) = fM = C * exp(-v²)
L1 = 1
isp3 = 1
H = HvL[:,L1,isp3]
dHvL = zero(HvL)
dHvL = dH012g(dfvL,fvL,fvLlog,DLp,vG,nG,nvG1,LM;k=k)
logf = fvLlog[:,L1,isp3]
dft = -2vG .* f                                 # 一阶导数理论值
dfgp = DLp * f                                   # 直接求解
dfgf = DLf * f                                   # 直接求解
dlogf = DLp * logf                              # 对数函数的一阶导数
# ∂ᵥf = f⁻¹∂ᵥlog(f)
dflog =  f .* dlogf                        # ∂ᵥf  = - 2v * exp(-v⁻²)
dfdata = [vG dft dflog dfgf dfgp]  |> Array{Float64}
dfs = DataFrame(dfdata,:auto)
dropmissing!(dfs)
rename!(dfs,[1 => :vG, 2=>:dft,3=>:dflog,4=>:dfgf,5=>:dfgp])
# @show dfs
#######################
# ∂ᵥ²f = ∂ᵥf × (v⁻¹ + ∂ᵥ(log(-v⁻¹∂ᵥf)))
ddft = (4vG.^2 .- 2) .* f
ddfgp = DLp * dft
ddfgf = DLp * dft
dfv1 = dflog ./ vG                         # ∂ᵥf/v = -2exp(-v⁻²)
logdfv1 = log.(- dfv1)                   # log(-∂ᵥf/v) = log(2) - v²
nc = findfirst(isinf.(logdfv1) .==1 )
k = 2
if nc ≠ nothing
    itpDL = Spline1D(vG[1:nc-1],logdfv1[1:nc-1];k=k,bc="extrapolate")
    logdfv1[nc:nG] = itpDL.(vG[nc:nG])
end
dlogdfv1t = DLp * (log(2) .- vG.^2)                # ∂ᵥlog(-∂ᵥf/v) = -2v
dlogdfv1 = DLp * logdfv1                # ∂ᵥlog(-∂ᵥf/v) = -2v
ddflog = dfv1 + dflog .* dlogdfv1
ddfdata = [vG ddft ddflog ddfgf ddfgp dfv1 dflog .* dlogdfv1]  |> Array{Float64}
ddfs = DataFrame(ddfdata,:auto)
dropmissing!(ddfs)
rename!(ddfs,[1 => :vG, 2=>:ddft,3=>:ddflog,4=>:ddfgf,5=>:ddfgp,6=>:dfv1])
@show ddfs
