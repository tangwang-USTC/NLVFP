
include(joinpath(pathroot,"test/run_collisions/paras_phys.jl"))

1 
printstyled(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"; color=:magenta)

## parameters: Experimental moments
is_qconservation = false # = 1 (default, conservation is on)
include(joinpath(pathroot, "src/PS/nuT/initial_nuTs.jl"))
include(joinpath(pathroot, "src/PS/nuT/nuTs_SI.jl"))
include(joinpath(pathroot, "src/PS/nuT/nuTs_dimensionless.jl")) 
include(joinpath(pathroot, "src/PS/nuT/nuTs_submoments_initial.jl"))


# multi-scale of time evolution process
include(joinpath(pathroot,"test/run_collisions/datas/tau_initial.jl"))

if is_plot_only == false
    nMod = copy(nMod0)
    nai = deepcopy(nai0)     # `n̂a = naᵢ / na`
    uai = deepcopy(uai0)     # `ûa = uaᵢ / vth`
    vthi = deepcopy(vthi0)    # `v̂th = vathᵢ / vth`
end

include(joinpath(pathroot,"test/run_collisions/parasTaTb_sol_ode.jl"))

include(joinpath(pathroot,"test/run_collisions/sol_MsIKTaTb.jl"))

