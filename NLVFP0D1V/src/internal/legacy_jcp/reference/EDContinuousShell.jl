module EDContinuousShell

using LinearAlgebra
import ..C1PositiveShellClosure
const Shell=C1PositiveShellClosure

struct Selector
    count::Int
    gram::Vector{Matrix{Float64}}
    shifted::Vector{Matrix{Float64}}
    margin::Float64
    padding::Float64
end

function Selector(n; margin=.22,padding=.20)
    prior=Shell.gamma_moments(n+1,Float64)
    function whitened(size,shift)
        L=cholesky(Symmetric(Shell.hankel(prior,size,shift))).L
        [Matrix(Symmetric(L \ Shell.hankel(
            [(k<j ? 0. : binomial(k,j)*prior[k+1]) for k in 0:n],size,shift)/L')) for j in 2:n]
    end
    Selector(n,whitened(n÷2+1,0),whitened((n+1)÷2,1),margin,padding)
end

function centered(departures)
    [sum((-1)^(k-j)*binomial(k,j)*departures[j-1] for j in 2:k) for k in 2:length(departures)+1]
end

function admissible(s,c,d)
    function test(mats)
        A=Matrix{Float64}(I,size(first(mats),1),size(first(mats),2))*(1-s.margin)
        for (i,B) in enumerate(mats)
            A .+= (c[i]/d^(i+1)).*B
        end
        isposdef(Symmetric(A))
    end
    test(s.gram) && test(s.shifted)
end

function choose_d(s,c)
    scale=maximum(abs(x)^(1/(i+1)) for (i,x) in enumerate(c))
    iszero(scale) && return 0.
    hi=min(.92,2scale)
    while !admissible(s,c,hi)
        hi==.92 && throw(DomainError(c,"No common width with a positive Stieltjes margin below the maximum contraction"))
        hi=min(.92,2hi)
    end
    lo=0.
    for _ in 1:29
        mid=(hi+lo)/2
        if admissible(s,c,mid)
            hi=mid
        else
            lo=mid
        end
    end
    hi+s.padding*hi*(1-hi)
end

function reconstruct(s,departures)
    if all(iszero,departures)
        return (;weights=fill(inv(s.count),s.count),widths=ones(s.count),shifts=zeros(s.count))
    end
    c=centered(departures)
    d=choose_d(s,c)
    m=vcat(big(1.),big(1.),1 .+ BigFloat.(departures))
    Shell.invert(m,s.count,s.count;fixed_d=d)
end

end
