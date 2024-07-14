using Plots, BasisFunctionExpansions
using ReverseDiff
using DSP            # filt

# using Test, Statistics, Random, LinearAlgebra, Statistics
# using BenchmarkTools
"""
  The final choice of this number is a tradeoff between reconstruction
  bias and variance, where a high number of basis functions can model
  the signal in great detail, but may increase the variance if data is sparse.

"""

nk0 = nc0
LL1 = 1
y_isp = 1
if nk0 == nc0
    vkv = zeros(nk0-1,1)
    vk0 = vkv[:,1] = vG0[2:end]
    yk0 = fvL[nvlevel0,LL1,y_isp][2:end]
    nn = 1
    is_yk = yk .> eps(Float64) * 10.0 ^(- nn)
    vk = vk0[is_yk]
    yk = yk0[is_yk]
    # yk = vk.^2
    nk = length(vk[:,1])               # Number of basis functions
    # nk = 77
    if nk > length(vk[:,1])
    end
    #
    vc,gammas = get_centers_automatic(vk,nk,false)

    rbf = UniformRBFE(vk,nk; normalize=true)     # Approximate using radial basis functions with constant width
    acts = rbf.activation
    mus = rbf.μ
    sigma = rbf.σ
    # rbf = MultiRBFE(vkv,nk; normalize=true)     #
    # rbf = MultiDiagonalRBFE(vkv,nk; normalize=true)     #
    @time bfa = BasisFunctionApproximation(yk,vk,rbf,1e-2) # Create approximation object
    lcs = bfa.linear_combination
    yfit = bfa(vk)     # Reconstruct the function using approximation object
    # dvy(vk) = ReverseDiff.gradient(bfa,vk)
    dvy(vkv) = ForwardDiff.jacobian(bfa,vkv)
    dvyk = diag(dvy(vk))
    erry = yfit - yk
    Rerry = yfit ./ yk .- 1

    label = string("y,nk=",nk)
    py = plot(vk,yk,label=label,line=(1,:auto))
    label = "y_fit"
    py = plot!(vk,yfit,label=label,line=(1,:auto))
    label = "err_y"
    perry = plot(vk,erry,label=label,line=(1,:auto))
    label = "Rerr_y"
    pRerry = plot(vk,Rerry,label=label,line=(1,:auto))
    # display(plot(py,perry,pRerry,layout=(3,1)))
    if 1 == 1
        vkk = vGk[2:end]
        ykk = fvL[:,LL1,y_isp][2:end]
        yfitk = bfa(vkk)     # Reconstruct the function using approximation object
        erryk = yfitk - ykk
        Rerryk = yfitk ./ ykk .- 1
        # label = "FLn"
        # scatter(vkk,ykk,label=label)
        # label = "FLn_fit"
        # scatter!(vkk,ykk,label=label)
        label = string("yk,nk=",nk)
        pyk = plot(vkk,ykk,label=label,line=(1,:auto))
        label = "yk_fit"
        pyk = plot!(vkk,yfitk,label=label,line=(1,:auto))
        label = "err_yk"
        perryk = plot(vkk,erryk,label=label,line=(1,:auto))
        label = "Rerr_yk"
        pRerryk = plot(vkk,Rerryk,label=label,line=(1,:auto))
        display(plot(py,pyk,perry,perryk,pRerry,pRerryk,layout=(3,2)))
    end
elseif nk0 == nck
    dk = 1
    vk = vGk[2:end][1:dk:end]
    yk = HLnt[2:end][1:dk:end]
    nk = length(vk)               # Number of basis functions
    rbf = UniformRBFE(vk,nk, normalize=true)     # Approximate using radial basis functions with constant widt
    @time bfa = BasisFunctionApproximation(yk,vk,rbf,0.0) # Create approximation object
    yfit = bfa(vk)     # Reconstruct the function using approximation object
    # label = "FLn"
    # scatter(vk,yk,label=label)
    # label = "FLn_fit"
    # scatter!(vk,yk,label=label)
    dvy(v) = ForwardDiff.jacobian(bfa,v)
    dvyk = diag(dvy(vk))

    label = string("y,dk,nk=",(dk,nk))
    py = plot(vk,yk,label=label,line=(1,:auto))
    label = "y_fit"
    py = plot!(vk,yfit,label=label,line=(1,:auto))

    label = "err_y"
    erry = yfit - yk
    perry = plot(vk,erry,label=label,line=(1,:auto))

    label = "Rerr_y"
    Rerry = yfit ./ yk .- 1
    pRerry = plot(vk,Rerry,label=label,line=(1,:auto))

    # display(plot(py,perry,pRerry,layout=(3,1)))
    if 1 == 1
        vkk = vGk[2:end]
        ykk = HLnt[2:end]

        yfitk = bfa(vkk)     # Reconstruct the function using approximation object
        # label = "FLn"
        # scatter(vkk,ykk,label=label)
        # label = "FLn_fit"
        # scatter!(vkk,ykk,label=label)
        label = string("yk,nk=",nk)
        pyk = plot(vkk,ykk,label=label,line=(1,:auto))
        label = "yk_fit"
        pyk = plot!(vkk,yfitk,label=label,line=(1,:auto))

        label = "err_yk"
        erryk = yfitk - ykk
        perryk = plot(vkk,erryk,label=label,line=(1,:auto))

        label = "Rerr_yk"
        Rerryk = yfitk ./ ykk .- 1
        pRerryk = plot(vkk,Rerryk,label=label,line=(1,:auto))

        display(plot(py,pyk,perry,perryk,pRerry,pRerryk,layout=(3,2)))
    end
end
