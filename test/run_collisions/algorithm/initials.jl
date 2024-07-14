1 
printstyled(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"; color=:magenta)
include(joinpath(pathroot, "src/PS/subtestting/plot_testting.jl"))
## parameters: Experimental moments
is_qconservation = false # = 1 (default, conservation is on)
include(joinpath(pathroot, "src/PS/nuT/initial_nuTs.jl"))
include(joinpath(pathroot, "src/PS/nuT/nuTs_SI.jl"))
include(joinpath(pathroot, "src/PS/nuT/nuTs_dimensionless.jl")) 
include(joinpath(pathroot, "src/PS/nuT/nuTs_submoments_initial.jl"))
include(joinpath(pathroot,"test/run_collisions/datas/titlesnIK.jl"))

# if is_MultiCase && iCase == 1
#     include(joinpath(pathroot,path_paper,"MultiCases.jl"))
# end

# multi-scale of time evolution process
include(joinpath(pathroot,"test/run_collisions/datas/tau_initial.jl"))

include(joinpath(pathroot,"test/run_collisions/algorithm/js_qqoo.jl"))

if is_plot_only == false
    ## meshgrids 𝕧 = (v, μ) = (vc, ℓ = 0:LM)  = (vG, ℓ = 0:LM)
    include(joinpath(pathroot, "test/run_collisions/algorithm/meshgrids_v.jl"))
    
    isp3 = 1
    L1 = 1
    L = L1 - 1
    if 1 == 1
        sum(abs.(u0)) == 0 ? L1 = 1 : nothing
        L1 ≤ 0 ? L1 = 1 : nothing
        ℓ = L1 - 1
        nsp_vec3 = 1:ns
        iFv3 = nsp_vec3[nsp_vec3.≠isp3][1]
        nvplot = vhk[isp3] .< 100
    end
    ## Meshgrids on angular dimensions of velocity space
    # LM .= LM1 - 1
    ## Calculating expansion coefficients `fvL`
    include(joinpath(pathroot, "src/PS/subtestting/fvL_testting.jl"))
    
    # Checking the meshgrids
    include(joinpath(pathroot, "test/run_collisions/algorithm/Mc_testting.jl"))
    
    is_solve_t0 = true
    is_solve_dtf = true
    if is_solve_t0 
        include(joinpath(pathroot, "src/PS/subtestting/submoments_testting.jl"))
        if is_solve_dtf
            # yscales, ys_scale = normalfLn(fLn1[nvlevel0[isp3]], L, uai[isp3], nMod[isp3])
    
            #  # nvlevele = nvlevel0[nvlevele0]
            iFv3 = nsp_vec3[nsp_vec3.≠isp3][1]
            vabth = vth[isp3] / vth[iFv3]
            mM = ma[isp3] / ma[iFv3]
            va = vhk[isp3] * vabth                       # vabth * vhk
            vb = vhk[iFv3] / vabth                       # vhk / vabth
            va0 = va[nvlevel0[isp3]]
            vb0 = vb[nvlevel0[iFv3]]
            nvaplot = va .< 30
            fvL0a = fvL0e[isp3]
            fvL0b = fvL0e[iFv3]
    
            ## δₜf,aa: the Fokker-Planck collisions
            # if is_extrapolate_FLn_initial
                include(joinpath(pathroot, "src/PS/subtestting/dtfvL_testting.jl"))
            # else
            #     include(joinpath(pathroot, "run_collisions/algorithm/dtfvL_nIK_testting.jl"))
            # end
            if is_fvL_CP == false
                include(joinpath(pathroot, "test/run_collisions/algorithm/dtMc_testting.jl"))
            end
    
            println()
            gridsnv = string("nnv0,(nvG,nc0,nck),ocp,vdm = ", (nnv0, (nvG, nc0, nck), ocp, vGdom[2, :]))
            @show gridsnv
    
            # include(joinpath(pathroot, "src/PS/subtestting/dtMsNEven_testting.jl"))
    
            include(joinpath(pathroot,"test/run_collisions/datas/Ms_dtMs_tk0.jl"))
    
            # which is normalized by `nd`, `md * Id = nd * vd` and `Kd = md * nd * vd^2` respectively.
    
            # println("6,dtMsEven")
            # if nMod0[isp3] == 1 && u0[isp3] == 0
            #     include(joinpath(pathroot,"src/PS/subtestting/dtMsEven_testting.jl"))
            # else
            #     include(joinpath(pathroot,"src/PS/subtestting/dtMsEven_u0_testting.jl"))
            # end
            
        end
    end
    
    # dtMsnnE3
    # ρa .* (3/2 * 1/2 * vth.^2 + 1/2 * ua.^2)
    # (2 * ρa .* vth.^2 .* dtMc2[njMs+1,1,:] + (4/3) * (ua) .* dtnIKsc2[2,:]) 
    # ## ########   Solve the FP0D2V problem by applying the normalized distribution function `f̂vL(v̂)` and thermal velocity `vₜₕ`
    
end