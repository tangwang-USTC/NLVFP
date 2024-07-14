
# include("weightfunctions.jl")   # /Es/FP0d2V/fl0

include("optimfvL0e_fMNKCnhC.jl")
include("optimfvL0e_fMNKC.jl")
include("optimfvL0e_fM_nhC.jl")

include("optimfvL0e_fMNK.jl")
include("optimfvL0e_fM_nh.jl")
include("optimfvL0e_fMNKfix.jl")
include("optimfvL0e_fMNKCfix.jl")

include("optimfvL0e_fDM.jl")
include("fit_TRMs.jl")

include("dvdtfvL0eFDM.jl")
include("dtfvL0eInterp.jl")
include("optimdtfvL0e.jl")
include("preprocess.jl")

include("nModk_update_Ms!.jl")
include("nModk_update_IK!.jl")
include("convergence.jl")