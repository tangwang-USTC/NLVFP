"""
  Order matrices of `𝕏ⱼⁿᵐ`, `𝕆ⱼⁿᵐ` for analytical results of Shkarofsky integrals:

    Cⱼᵖ = [ûᵦ⁰, ûᵦ¹, ûᵦ², ⋯, ûᵦⁿ] (ℂⱼᵖ × 𝕏ⱼⁿᵐ) [v̂⁰, v̂¹, v̂², ⋯, v̂ᵐ]ᵀ

  where

    𝕏ⱼⁿᵐ = v̂ᵦₜₕₛ .^ 𝕆ⱼⁿᵐ
"""


"""
  Intputs:
    L:

  Outputs:
    orderXL = orderXL0nm(L)
    orderXL = orderXL2nm(L)
    orderXL = orderXL1nm(L)
    orderXL = orderXLn1nm(L)
"""

function orderXL0nm(L::Int64)

    if L == 0
        n = 1
        m = 1
        orderXL = zeros(n,m)
        orderXL .= NaN

        orderXL[1,1] = 0
    elseif L == 1
        n = 3
        m = 2
        orderXL = zeros(n,m)
        orderXL .= NaN

        orderXL[1,1] = 2
        orderXL[3,1] = 0
        orderXL[2,2] = 0
    else
        n = 2L + 1
        m = L + 1
        orderXL = zeros(n,m)
        orderXL .= NaN
        for k in 1:m
            for i in k:2:n - (k - 1)
                orderXL[n-i+1,k] = i - k
            end
        end
    end
    return orderXL
end

function orderXL2nm(L::Int64)

    if L == 0
        n = 3
        m = 3
        orderXL = zeros(n,m)
        orderXL .= NaN

        orderXL[1,1] = 2
        orderXL[3,1] = 0
        orderXL[2,2] = 0
        orderXL[1,3] = 0
    elseif L == 1
        n = 5
        m = 4
        orderXL = zeros(n,m)
        orderXL .= NaN

        orderXL[1,1] = 4
        orderXL[3,1] = 2
        orderXL[5,1] = 0
        orderXL[2,2] = 2
        orderXL[4,2] = 0
        orderXL[1,3] = 2
        orderXL[3,3] = 0
        orderXL[2,4] = 0
    else
        n = 2L + 3
        m = L + 3
        orderXL = zeros(n,m)
        orderXL .= NaN
        for k in 1:m
            if k ≤ 2
                for i in k:2:n
                    orderXL[n-i+1,k] = i - k
                end
            else
                for i in k:2:n - k + 3
                    orderXL[n-i+1,k] = i - k
                end
            end
        end
    end
    return orderXL
end

function orderXL1nm(L::Int64)

    if L == 0
        n = 1
        m = 1
        orderXL = zeros(n,m)
        orderXL .= NaN
    elseif L == 1
        n = 1
        m = 2
        orderXL = zeros(n,m)
        orderXL .= NaN

        orderXL[1,2] = 0
    else
        n = L
        m = 2L - 1
        orderXL = zeros(n,m)
        orderXL .= NaN
        for k in 1:m
            if k ≤ n
                for i in n - (k - 1):2:n
                    orderXL[n-i+1,k] = i + n - k - 1
                end
            else
                for i in k - n + 1:2:n
                    orderXL[n-i+1,k] = i + n - k - 1
                end
            end
        end
    end
    return orderXL
end

function orderXLn1nm(L::Int64)

    if L == 0
        n = 2
        m = 2
        orderXL = zeros(n,m)
        orderXL .= NaN

        orderXL[1,2] = 0
        orderXL[2,1] = 0
    elseif L == 1
        n = 2
        m = 1
        orderXL = zeros(n,m)
        orderXL .= NaN

        orderXL[2,1] = 0
    else
        n = L
        m = 2L - 3
        orderXL = zeros(n,m)
        orderXL .= NaN
        if L == 2
            orderXL[2,1] = 2
        else
            for k in 1:m
                if k ≤ n
                    for i in n - (k - 1):2:n
                        orderXL[n-i+1,k] = i + n - k - 1
                    end
                else
                    for i in k - n + 1:2:n
                        orderXL[n-i+1,k] = i + n - k - 1
                    end
                end
            end
        end
    end
    return orderXL
end


"""
  Intputs:
    L:

  Outputs:
    orderXL0nm!(orderXL,n,m)
    orderXL2nm!(orderXL,n,m)
    orderXL1nm!(orderXL,n,m)
    orderXLn1nm!(orderXL,n,m)
"""

function orderXL0nm!(orderXL::Matrix{T},n::Int64,m::Int64) where{T}

    for k in 1:m
        for i in k:2:n - (k - 1)
            orderXL[n-i+1,k] = i - k
        end
    end
end

function orderXL2nm!(orderXL::Matrix{T},n::Int64,m::Int64) where{T}

    for k in 1:m
        if k ≤ 2
            for i in k:2:n
                orderXL[n-i+1,k] = i - k
            end
        else
            for i in k:2:n - k + 3
                orderXL[n-i+1,k] = i - k
            end
        end
    end
end

function orderXL1nm!(orderXL::Matrix{T},n::Int64,m::Int64) where{T}

    for k in 1:m
        if k ≤ n
            for i in n - (k - 1):2:n
                orderXL[n-i+1,k] = i + n - k - 1
            end
        else
            for i in k - n + 1:2:n
                orderXL[n-i+1,k] = i + n - k - 1
            end
        end
    end
end

function orderXLn1nm!(orderXL::Matrix{T},n::Int64,m::Int64) where{T}

    if L == 0
    elseif L == 1
    elseif L == 2
        orderXL[2,1] = 2
    else
        for k in 1:m
            if k ≤ n
                for i in n - (k - 1):2:n
                    orderXL[n-i+1,k] = i + n - k - 1
                end
            else
                for i in k - n + 1:2:n
                    orderXL[n-i+1,k] = i + n - k - 1
                end
            end
        end
    end
end
