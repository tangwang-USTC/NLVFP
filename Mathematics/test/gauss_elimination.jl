function small_enough(v::Number, epsilon=1e-8)
    return abs(v - 0.0) < epsilon
end

"""
  gauss_elimination(A,b)
"""

function gauss_elimination(x::Array, b::Array)
    A = hcat(x, b)
    row_num = size(A, 1)

    # elimination
    for i in 1:row_num-1
        pivot = A[i, i]
        for j in i+1:row_num
            base = A[j, i] / pivot
            A[j,:] = A[j,:] - (base .* A[i,:])
        end
    end

    if small_enough(A[row_num, row_num])
        error("Can not elimination.")
    end

    # feedback
    b = A[:, end]
    b[end] /= A[end, end-1]     # last line
    for i in row_num-1:-1:1
        pivot = A[i,i]
        b[i] -= sum(A[i,i+1:end-1] .* b[i+1:end])
        b[i] /= pivot
    end
    return A[1:row_num,1:row_num], b
end
