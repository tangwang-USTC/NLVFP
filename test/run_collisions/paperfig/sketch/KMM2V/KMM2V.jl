gr()


# θ = range(0, 2π; length = 100)
# ρ = range(0, 120; length = 50)
# z = sin.(ρ ./ 10) .* (cos.(θ))'

# heatmap(θ, ρ, z; projection = :polar, color = :cividis, right_margin = 2 * Plots.mm)


theta = pi / 2 * μ

isp333 = 1
rr = vhk[isp333]

veccc = rr .< 2

LLL1 = 0

fvL00 = fvL[isp333][veccc,:]

if LLL1 == 0 || LLL1 > LM[isp333]
    zz = fvL00 * Mun
    heatmap(theta, rr[veccc], zz; projection = :polar, color = :cividis, right_margin = 2 * Plots.mm)
    
    theta2 = (pi / 2 * μ .+ pi)
    pht0 = heatmap!(theta2, rr[veccc], reverse(zz;dims=2); projection = :polar, color = :cividis, right_margin = 2 * Plots.mm)
    
    # contour(theta, rr[veccc], zz; projection = :polar, color = :cividis, right_margin = 2 * Plots.mm)
    
    # contour(theta, rr[veccc], zz, levels=10, color=:turbo, clabels=true, cbar=false, lw=2)
    
    title = L"$ \hat{f} (\hat{v}, θ)$"
    title!(title)
    display(pht0)

    nlevels = 10
    contour(theta, rr[veccc], zz, levels=nlevels, color=:turbo, clabels=true, cbar=false, lw=2)
    theta2 = (pi / 2 * μ .+ pi)
    pcont0 = contour!(theta2, rr[veccc], reverse(zz;dims=2), levels=nlevels, color=:turbo, clabels=true, cbar=false, lw=2)
    title = L"$ \hat{f} (\hat{v}, θ)$"
    title!(title)
    display(pcont0)
else
    fvL000 = 0.0fvL[isp333][veccc,:]
    fvL000[:,LLL1] = fvL00[:,LLL1]
    
    # fvL000 = fvL[isp333][veccc,:]
    # fvL000[:,1:12] .= 0.0
    
    zz = fvL000 * Mun
    heatmap(theta, rr[veccc], zz; projection = :polar, color = :cividis, right_margin = 2 * Plots.mm)
    theta2 = (pi / 2 * μ .+ pi)
    heatmap!(theta2, rr[veccc], reverse(zz;dims=2); projection = :polar, color = :cividis, right_margin = 2 * Plots.mm)
    
    title = L"$ \hat{f}_l^0 (\hat{v}, θ)$, $l = $" *string(LLL1-1)
    title!(title)
end