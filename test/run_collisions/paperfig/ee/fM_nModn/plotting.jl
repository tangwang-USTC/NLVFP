


# include(joinpath(pathroot,"test/run_collisions/paperfig/ee/fM_nMod1/TaTb_ns2.jl"))
include(joinpath(pathroot,path_paper,"TaTb_ns2.jl"))
include(joinpath(pathroot,path_paper,"edtnIKTs.jl"))
include(joinpath(pathroot,path_paper,"CnIKs.jl"))

if is_MultiCase
    tplotM[iCase] = deepcopy(tplot[tvec])
    tplotMTaTb[iCase] = deepcopy(tplotTaTb[tvecTaTb])
    if iCase == NCase
        include(joinpath(pathroot,path_paper,"edtnIKTs_MCases.jl"))
        include(joinpath(pathroot,path_paper,"CnIKs_MCases.jl"))
    end
end
