

include("cheby_extrema.jl")
include("constants\\constants.jl")
include("dichotomy.jl")
include("diffmatrix.jl")
include("diffmatrixCheby.jl")
include("Matrix.jl")
include("physicsfuns.jl")
include("ButcherTableaus\\ButcherTableaus.jl")

# include("Isimpson1D.jl")
# include("precondChebydiff.jl")
# include("PDE2bc2Chebyshev.jl")

function paraM(A)
    n, m = size(A)
    if n == m
        @show isposdef(A), issymmetric(A), ishermitian(A)
        @show size(A),rank(A), fmtf2.([cond(A), det(A), tr(A)])
        if rank(A) == n
            println("Matrix `A` is regular!")
        elseif rank(A) < length(A[:,1])
            println("Matrix `A` is singular!")
        end
    else
        @show isposdef(A) size(A),rank(A), fmtf2(cond(A))
        if rank(A) == min(n,m)
            println("Matrix `A` is regular!")
        elseif rank(A) < length(A[:,1])
            println("Matrix `A` is singular!")
        end
    end
end
