

function RichardsonG(As::AbstractVector)
    m = length(As)        # order
    if m ≥ 2
        Cs = zeros(m)
        j =  1
        Cs[j] = (-1)^(m+j) / prod(1:m - j )
        for j in 2:m
            Cs[j] =  (-1)^(m+j) * j^(m-1) / prod(1:j-1) / prod(1:m - j )
        end
        return sum(As.*Cs)
    else
        throw(ArgumentError(" number of elements must ≥ 2 "))
    end
end

function RichardsonG(As::AbstractArray{T,N}) where {T,N}
    if N == 2
        println("N=",N)
        m,nsp = size(As)        # order
        if m ≥ 2
            Cs = zeros(m,nsp)
            for isp in 1:nsp
                j =  1
                Cs[j,isp] = (-1)^(m+j) / gamma( m - j + 1)
                for j in 2:m
                    Cs[j,isp] =  (-1)^(m+j)* j^(m-1)/ gamma(j) / gamma( m - j + 1)
                end
            end
            return sum(Cs .* As,dims=1)
        else
            throw(ArgumentError(" number of elements must ≥ 2 "))
        end
    end

end
function pisum(n::Int)
    sum(1 ./(1:n).^2)
end
As = [pisum(4), pisum(8), pisum(12), pisum(16)]
# As = [1,1.25,1.361111111,1.423611111,1.4636111111111111]
# As = [1,1.25,1.361111111,1.423611111]
# sum(1 ./(1:n)) - log(n)
# As = [1,0.8068528194400547,0.7347210446652235,0.6970389722134425]
S = RichardsonG(As)

println("S=",S)
# As = repeat(As,1,2)
# S = RichardsonG(As)
# println("S=",S)
