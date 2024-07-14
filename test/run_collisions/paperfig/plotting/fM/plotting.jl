
# linetypes =  [:solid,:dash,   :dashdot,  :dot,  :dashdotdot]
# linecolors = [:orange, :green, :gray,  :red, :blue, :purple, :magenta, :lightgreen, :lightblue, :yellow, :gray, :purple, :magenta ]
# linemakers = [:circle,:star5,:diamond,:xcross,  :rtriangle ]
# linetypes =  [:solid]
# linecolors = [:orange]
# linemakers = [:circle]
include(joinpath(pathroot,path_paper_fM,"edtnIKTs.jl"))
include(joinpath(pathroot,path_paper_fM,"CnIKs.jl"))
include(joinpath(pathroot,path_paper_fM,"RDMck1.jl"))
include(joinpath(pathroot,path_paper_fM,"RDMck1NK3.jl"))

if is_MultiCase
    tplotM[iCase] = deepcopy(tplot[tvec])
    if is_sol_Brag
        tplotMTaTb[iCase] = deepcopy(tplotTaTb[tvecTaTb])
    end
    if iCase == NCase 
        if MultiType == :nnv
            include(joinpath(pathroot,path_paper_fM,"edtnIKTs_MCases.jl"))
            include(joinpath(pathroot,path_paper_fM,"CnIKs_MCases.jl"))
            if is_Case_C01 && is_enforce_errdtnIKab
                include(joinpath(pathroot,path_paper_fM,"CnIKs_MCasesC01.jl"))
            end
        elseif MultiType == :dt
            include(joinpath(pathroot,path_paper_fM,"edtnIKTs_MCases_dt.jl"))
            include(joinpath(pathroot,path_paper_fM,"CnIKs_MCases_dt.jl"))
            if is_Case_C01 && is_enforce_errdtnIKab
                include(joinpath(pathroot,path_paper_fM,"CnIKs_MCasesC01_dt.jl"))
            end
        else
        end
    end
    if is_moments_out
        if isfile(file_Ms_errMhcopla) && iCase == NCase
            include(joinpath(pathroot,path_paper_fM,"errMhcop.jl"))
            if MultiType == :nnv
                include(joinpath(pathroot,path_paper_fM,"errMhcop_MCases_nnv.jl"))
            elseif MultiType == :dt
                include(joinpath(pathroot,path_paper_fM,"errMhcop_MCases_dt.jl"))
            elseif MultiType == :NK
                include(joinpath(pathroot,path_paper_fM,"errMhcop_MCases_NK.jl"))
            else
                edfgbn 
            end
        end
    end
end

if iCase == NCase
    include(joinpath(pathroot,path_paper_fM,"TaTbns2.jl"))
else
    include(joinpath(pathroot,path_paper_fM,"TaTbns2.jl"))
end