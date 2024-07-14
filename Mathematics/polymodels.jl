

function polymodel(v::AbstractVector{T},p::AbstractVector{T}) where{T}
    np = length(p)
    if np == 2
        return p[1] .+ p[2] * v
    elseif np == 3
        return p[1] .+ p[2] * v + p[3] * v.^2
    elseif np == 4
        return p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3
    elseif np == 5
        return p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4
    elseif np == 6
        return p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4 + p[6] * v.^5
    elseif np == 7
        return p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4 + p[6] * v.^5 + p[7] * v.^6
    elseif np == 8
        return p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4 + p[6] * v.^5 + p[7] * v.^6 + p[8] * v.^7
    elseif np == 9
        return p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4 + p[6] * v.^5 + p[7] * v.^6 + p[8] * v.^7 + p[9] * v.^8
    elseif np == 10
        return p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4 + p[6] * v.^5 + p[7] * v.^6 + p[8] * v.^7 + p[9] * v.^8 + p[10] * v.^9
    end
end


function polyfun(p::AbstractVector{T}) where{T}
    np = length(p)
    if np == 2
        return fitfun(v) = p[1] .+ p[2] * v
    elseif np == 3
        return fitfun(v) = p[1] .+ p[2] * v + p[3] * v.^2
    elseif np == 4
        return fitfun(v) = p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3
    elseif np == 5
        return fitfun(v) = p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4
    elseif np == 6
        return fitfun(v) = p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4 + p[6] * v.^5
    elseif np == 7
        return fitfun(v) = p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4 + p[6] * v.^5 + p[7] * v.^6
    elseif np == 8
        return fitfun(v) = p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4 + p[6] * v.^5 + p[7] * v.^6 + p[8] * v.^7
    elseif np == 9
        return fitfun(v) = p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4 + p[6] * v.^5 + p[7] * v.^6 + p[8] * v.^7 + p[9] * v.^8
    elseif np == 10
        return fitfun(v) = p[1] .+ p[2] * v + p[3] * v.^2 + p[4] * v.^3 + p[5] * v.^4 + p[6] * v.^5 + p[7] * v.^6 + p[8] * v.^7 + p[9] * v.^8 + p[10] * v.^9
    end
end



function jaco_polymodel(v::AbstractVector{T},p::AbstractVector{T}) where {T}
    J = Array{T}(undef,length(v),length(p))
    # J[:,j] = ∂nodel/∂p[j]
    J[:,1] = 0v
    J[:,2] = v
    J[:,3] = v.^2
    J[:,4] = v.^3
    J[:,5] = v.^4
    J
end
