# The spectral test bounds the sum of the pencil perturbations directly.
# The existing majorant adds their norms, which can reject opposite-sign terms.
function spectral_contraction(selector,c)
    all(iszero,c) && return 0.
    function admissible(d)
        weights=[x/d^(j+1) for (j,x) in enumerate(c)]
        for matrices in (selector.gram,selector.shifted)
            matrix=sum(weights[j]*matrices[j] for j in eachindex(c))
            maximum(abs,eigvals(Symmetric(matrix)))<=MATRIX_RADIUS || return false
        end
        true
    end
    upper=.95
    admissible(upper) || throw(DomainError(upper,"Exact two-sided pencil bound requires excessive width contraction"))
    lower=upper/2
    while admissible(lower)
        upper=lower
        lower/=2
    end
    for _ in 1:32
        d=(lower+upper)/2
        if admissible(d)
            upper=d
        else
            lower=d
        end
    end
    upper
end

function spectral_rule(selector,departures)
    n=selector.count
    all(iszero,departures) && return (;weights=fill(inv(n),n),widths=ones(n),shifts=zeros(n))
    d=spectral_contraction(selector,EDContinuousShell.centered(departures))
    target=vcat(big(1.),big(1.),1 .+ BigFloat.(departures))
    C1PositiveShellClosure.invert(target,n,n;fixed_d=d)
end
