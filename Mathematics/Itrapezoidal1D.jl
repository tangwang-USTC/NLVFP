"""
Fix step length, h = constant of function f(x)
"""
function trapezoidal1D(f::AbstractVector,h::Number)
    nf = length(f)
    I = h*(sum(f) -0.5(f[1] + f[nf]))
    if isodd(nf) == 1
        # I₂ = 2h*(sum(f[1:2:nf])-0.5(f[1] + f[nf]))
        I₂ = h*(2sum(f[1:2:nf])-(f[1] + f[nf]))
    else
        I₂ = h*(2sum(f[1:2:nf-1])-f[1] +0.5(f[nf]- f[nf-1]))
    end
    E  = I - I₂  # error
    return I,E
end

## different steps hx = diff(x)
"""
Variable step length, h = [...h[i]...] of function  f(x)
"""
function trapezoidal1D(x::AbstractVector,f::AbstractVector)
    hx = diff(x,dims = 1)
    nf = length(f)
    I = 0
    for i = 1:nf-1
        I = I + hx[i] * 0.5(f[i] + f[i+1])
    end
    return I
end

# function trapezoidal1D(f::AbstractArrays{T,N},x::AbstracArrays{T,N};dim::Integer) where {T,N}
#     hx = diff(x,dims = dim)
#     # identify the steps are the same: hx[i]
# end
