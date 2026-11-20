module C1PositiveShellClosure

using LinearAlgebra

gamma_moments(n,T)=cumprod(T[1; [j+T(.5) for j in 1:n]])
hankel(m,n,shift)=[m[i+j-1+shift] for i in 1:n,j in 1:n]

function remove_broadening(m,d,prior)
    r=length(m)-1
    [prior[p+1]*sum(binomial(p,k)*(d-1)^(p-k)*m[k+1] for k in 0:p)/d^p for p in 0:r]
end

function invert(target,count,retained; fixed_d=nothing, floor_fraction=0.04, coupling=1.0)
    2<=retained<=2count-1 || throw(ArgumentError("Invalid shell moment count"))
    m=target[1:retained+1]
    if maximum(abs.(m.-1))<=16eps()
        return (;weights=fill(inv(count),count),widths=ones(count),shifts=zeros(count))
    end
    b=retained÷2+1
    c=(retained+1)÷2
    d,j=setprecision(BigFloat,160) do
        prior=gamma_moments(retained+1,BigFloat)
        m128=BigFloat.(m)
        small=BigFloat("0.0001")
        large=BigFloat("0.98")
        gram_floor=floor_fraction*hankel(prior,b,0)
        shifted_floor=floor_fraction*hankel(prior,c,1)
        function admissible(d)
            q=remove_broadening(m128,d,prior)
            isposdef(Symmetric(hankel(q,b,0)-gram_floor)) &&
                isposdef(Symmetric(hankel(q,c,1)-shifted_floor))
        end
        if fixed_d===nothing
            admissible(large)||error("No positive Stieltjes pencil at retained order $(2retained)")
            for _ in 1:28
                mid=(small+large)/2
                if admissible(mid)
                    large=mid
                else
                    small=mid
                end
            end
            d=large+(1-large)*large/16
        else
            d=BigFloat(fixed_d)
        end
        q=remove_broadening(m128,d,prior)
        l=cholesky(Symmetric(hankel(q,b,0))).L
        if iseven(retained)
            push!(q,zero(BigFloat))
            j0=(l\hankel(q,b,1))/l'
            lead=Symmetric(j0[1:end-1,1:end-1])
            t=j0[1:end-1,end]
            # Higher moments complete the exact retained block. The free
            # diagonal follows the Gaussian value subject to positive support.
            j0[end,end]=max(2b-0.5,dot(t,lead\t)+0.5)
        else
            j0=(l\hankel(q,b,1))/l'
        end
        Float64(d),Matrix{Float64}(Symmetric(j0))
    end
    for k in b+1:count
        t=zeros(k-1)
        t[end]=coupling*sqrt((k-1)*(k-0.5))
        j=[j t;t' max(2k-0.5,dot(t,j\t)+0.5)]
    end
    rule=eigen(Symmetric(j))
    ispositive=first(rule.values)>0. && all(!iszero,rule.vectors[1,:])
    ispositive || error("Nonpositive shell Jacobi completion")
    weights=rule.vectors[1,:].^2
    (;weights=weights./sum(weights),widths=fill(sqrt(1-d),count),shifts=sqrt.(d.*rule.values))
end

end
