


linetypes =  [:solid, :dot,  :dash,   :dashdot, :dashdotdot]
linecolors = [:orange, :green,:blue,   :yellow,  :red ]
linemakers = [:circle,:star5,:diamond,:xcross,  :rtriangle ]

include(joinpath(pathroot,path_paper,"edtnIKTs.jl"))
include(joinpath(pathroot,path_paper,"CnIKs.jl"))
include(joinpath(pathroot,path_paper,"TeTDTA.jl"))

if is_MultiCase
    tplotM[iCase] = deepcopy(tplot[tvec])
    if iCase == NCase
        include(joinpath(pathroot,path_paper,"edtnIKTs_MCases.jl"))
        include(joinpath(pathroot,path_paper,"CnIKs_MCases.jl"))
    end
    if is_Case_C01 && is_enforce_errdtnIKab
        include(joinpath(pathroot,path_paper,"CnIKs_MCasesC01.jl"))
    end
end
