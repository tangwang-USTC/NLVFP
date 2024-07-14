
"""
  Outputs:
    A, b, c = construct_GL(s)
"""

function construct_GL(s::Int64,T::Type = Float64)

    if s == 1
        # A, b, c = construct_GL2(T)
        return construct_GL2(T)
    elseif s == 2
        return construct_GL4(T)
    elseif s == 3
        return construct_GL6(T)
    elseif s == 4
        return construct_GL8(T)
    elseif s == 5
        return construct_GL10(T)
    elseif s == 6
        return construct_GL12(T)
    elseif s == 7
        return construct_GL14(T)
    elseif s == 8
        return construct_GL16(T)
    else
        # return construct_GL2(T)
        vgiii
    end
end

"""
Gauss-Legendre Order 2.
"""
function construct_GL2(T::Type = Float64)

    A = zeros(T,1,1)
    A[1,1] = T(1 // 2)
    c = [1 // 2]
    b = [1]
    b = map(T, b)
    c = map(T, c)
    # s, o = 1, 2
    return A, b, c
    # return (ImplicitRKTableau(A, c, b, 1, 2))
end

"""
Gauss-Legendre Order 4.
"""
function construct_GL4(T::Type = Float64)

    c = [(3 - sqrt(3)) / 6; (3 + sqrt(3)) / 6]
    A = [1/4   (3 - 2 * sqrt(3))/12;
         (3 + 2 * sqrt(3))/12   1/4]
    b = [1 / 2; 1 / 2]
    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    # s, o = 2, 2s
    return A, b, c
end

"""
Gauss-Legendre Order 6.
"""
function construct_GL6(T::Type = Float64)

    c = [(5 - sqrt(15)) / 10;     1 / 2;     (5 + sqrt(15)) / 10]
    A = [5/36   (10 - 3 * sqrt(15))/45   (25 - 6 * sqrt(15))/180;
         (10 + 3 * sqrt(15))/72   2/9     (10 - 3 * sqrt(15))/72;
         (25 + 6 * sqrt(15))/180 (10 + 3 * sqrt(15))/45     5/36]
    b = [5 / 18;                 4 / 9;                   5 / 18]
    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    # s, o = 3, 2s
    return A, b, c
end

"""
Gauss-Legendre Order 8.
"""
function construct_GL8(T::Type = Float64)
    
    sqrt30 = sqrt(30)

    w1 = 1 / 8 - sqrt30 / 144
    w2 = 0.5 * sqrt((15 + 2 * sqrt30) / 35)
    w3 = w2 * (1 / 6 + sqrt30 / 24)
    w4 = w2 * (1 / 21 + 5 * sqrt30 / 168)
    w5 = w2 - 2w3

    w1s = 1 / 8 + sqrt30 / 144
    w2s = 0.5 * sqrt((15 - 2 * sqrt30) / 35)
    w3s = w2s * (1 / 6 - sqrt30 / 24)
    w4s = w2s * (1 / 21 - 5 * sqrt30 / 168)
    w5s = w2s - 2w3s

    c = [  0.5 - w2;     0.5 - w2s;     0.5 + w2s;    0.5 + w2  ]
    A = [     w1     w1s - w2 + w4s  w1s - w3 - w4s   w1 - w5   ;
         w1 - w3s + w4     w1s         w1s - w5s   w1 - w3s - w4;
         w1 + w3s + w4  w1s + w5s         w1s      w1 + w3s - w4;
            w1 + w5  w1s + w3 + w4s  w1s + w3 - w4s     w1      ]
    b = 2.0 * [w1;         w1s;           w1s;          w1      ]
    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    # s, o = 4, 2s
    return A, b, c
end

"""
Gauss-Legendre Order 10.
"""
function construct_GL10(T::Type = Float64)
    
    sqrt70 = sqrt(70)
    c12 = 32 / 225

    w1 = (322 - 13 * sqrt70) / 3600
    w2 = 0.5 * sqrt((35 + 2 * sqrt70) / 63)
    w3 = w2 * ((452 + 59 * sqrt70) / 3240)
    w4 = w2 * ((64 + 11 * sqrt70) / 1080)
    w5 = 8w2 * ((23 - sqrt70) / 405)
    w6 = w2 - 2w3 - w5
    w7 = w2 * ((308 - 23 * sqrt70) / 960)

    w1s = (322 + 13 * sqrt70) / 3600
    w2s = 0.5 * sqrt((35 - 2 * sqrt70) / 63)
    w3s = w2s * ((452 - 59 * sqrt70) / 3240)
    w4s = w2s * ((64 - 11 * sqrt70) / 1080)
    w5s = 8w2s * ((23 + sqrt70) / 405)
    w6s = w2s - 2w3s - w5s
    w7s = w2s * ((308 + 23 * sqrt70) / 960)

    c = [0.5 - w2;     0.5 - w2s;         0.5;     0.5 + w2s;    0.5 + w2]
    A = [     w1     w1s - w3 + w4s    c12 - w5  w1s - w3 - w4s   w1 - w5   ;
         w1 - w3s + w4     w1s         c12 - w5s   w1s - w6s   w1 - w3s - w4;
            w1 + w7     w1s + w7s         c12      w1s - w7s      w1 - w7   ;
         w1 + w3s + w4  w1s + w6s      c12 + w5s      w1s      w1 + w3s - w4;
            w1 + w6  w1s + w3 + w4s    c12 - w5  w1s + w3 - w4s     w1      ]
    b = 2.0 * [w1;         w1s;           c12;        w1s;          w1      ]
    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    # s, o = 5, 2s
    return A, b, c
end
