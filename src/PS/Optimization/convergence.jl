
function order_converg(errorn1::T, errorn::T) where{T}

    return log(errorn1 / errorn) / log(2)
end


function order_converg(error::AbstractVector{T}) where{T}

    return log.(error[1:end-1] ./ error[2:end]) / log(2)
end


function order_converg(errorn1::AbstractVector{T}, errorn::AbstractVector{T}) where{T}

    return log.(errorn1 ./ errorn) / log(2)
end

