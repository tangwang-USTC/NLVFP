

"""
Explicit SSP method of order 2 using 2 stages.
"""
function construct_SSPRK22(T::Type = Float64)
    A = [0 0
         1 0]
    c = [0; 1]
    α = [1 // 2; 1 // 2]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (ExplicitRKTableau(A, c, α, 2, stability_size = -2.0))
end

"""
Explicit SSP method of order 3 using 3 stages.
"""
function construct_SSPRK33(T::Type = Float64)
    A = [0 0 0
         1 0 0
         1//4 1//4 0]
    c = [0; 1; 1 // 2]
    α = [1 // 6; 1 // 6; 2 // 3]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (ExplicitRKTableau(A, c, α, 3, stability_size = 2.5127453266183286))
end

"""
Explicit SSP method of order 3 using 4 stages.
"""
function construct_SSPRK43(T::Type = Float64)
    A = [0 0 0 0
         1//2 0 0 0
         1//2 1//2 0 0
         1//6 1//6 1//6 0]
    c = [0; 1 // 2; 1; 1 // 2]
    α = [1 // 6; 1 // 6; 1 // 6; 1 // 2]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (ExplicitRKTableau(A, c, α, 3, stability_size = 5.149486147774043))
end

"""
Explicit SSP method of order 4 using 10 stages.
"""
function construct_SSPRK104(T::Type = Float64)
    A = [0 0 0 0 0 0 0 0 0 0
         1//6 0 0 0 0 0 0 0 0 0
         1//6 1//6 0 0 0 0 0 0 0 0
         1//6 1//6 1//6 0 0 0 0 0 0 0
         1//6 1//6 1//6 1//6 0 0 0 0 0 0
         1//15 1//15 1//15 1//15 1//15 0 0 0 0 0
         1//15 1//15 1//15 1//15 1//15 1//6 0 0 0 0
         1//15 1//15 1//15 1//15 1//15 1//6 1//6 0 0 0
         1//15 1//15 1//15 1//15 1//15 1//6 1//6 1//6 0 0
         1//15 1//15 1//15 1//15 1//15 1//6 1//6 1//6 1//6 0]
    c = [0; 1 // 6; 1 // 3; 1 // 2; 2 // 3; 1 // 3; 1 // 2; 2 // 3; 5 // 6; 1]
    α = [1 // 10; 1 // 10; 1 // 10; 1 // 10; 1 // 10; 1 // 10; 1 // 10; 1 // 10; 1 // 10;
         1 // 10]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (ExplicitRKTableau(A, c, α, 4, stability_size = 13.917047464637367))
end
