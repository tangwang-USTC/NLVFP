

"""
  General linear methods: Multistep Runge-Kuuta methods (MSRK)
  Tableau of the multistep multistage (MSMS) algorithm according to the
  "Adams-Bashforth-Moulton" algorithm
  
"""

function construct_Adamso4s3(T::Type = Float64)

    s = 3                 # Stage number
    o = 4                 # Order
    rs = 4                # regular stage (h / N)

    A = [  ]
    b = [ 1;  -5;  19;  9 ] // 24        # Coefficients of the derivatives `kᵢ`
    d = [ 0;   0;   1;  0 ]              # Coefficients of the function values `yᵢ`
    nb, nd = 4, 1

    A = map(T, A)
    b = map(T, b)
    d = map(T, d)
    return A, b, d, o, s, rs, (nb, nd)
end

function construct_Hammingo4s3(T::Type = Float64)

    s = 3                 # Stage number
    o = 4                 # Order
    rs = 4                # regular stage (h / N)

    A = [  ]
    b = [ 0;  -1;   2;  1 ] * 3 // 8     # Coefficients of the derivatives `kᵢ`
    d = [ -1;  0;   9;  0] // 8          # Coefficients of the function values `yᵢ`
    nb, nd = 3, 2

    A = map(T, A)
    b = map(T, b)
    d = map(T, d)
    return A, b, d, o, s, rs, (nb, nd)
end

function construct_Adamso4s4(T::Type = Float64)

    s = 4                 # Stage number
    o = 4                 # Order
    rs = 5                # regular stage (h / N)

    A = [  ]
    b = [ -9; 37; -59;  55; 0 ] // 24        # Coefficients of the derivatives `kᵢ`
    d = [ 0;   0;   0;  1;  0 ]              # Coefficients of the function values `yᵢ`
    nb, nd = 4, 1

    A = map(T, A)
    b = map(T, b)
    d = map(T, d)
    return A, b, d, o, s, rs, (nb, nd)
end

function construct_Milneo4s4(T::Type = Float64)

    s = 4                 # Stage number
    o = 4                 # Order
    rs = 5                # regular stage (h / N)

    A = [  ]
    b = [-9;  37; -59; 55; 0 ] // 24     # Coefficients of the derivatives `kᵢ`
    d = [ 0;   0;   0;  1; 0 ]              # Coefficients of the function values `yᵢ`
    nb, nd = 3, 1

    A = map(T, A)
    b = map(T, b)
    d = map(T, d)
    return A, b, d, o, s, rs, (nb, nd)
end

# `A(α)` stability
function construct_MsRKo5s5(T::Type = Float64)

    s = 5                 # Stage number
    o = 5                 # Order
    rs = s + 1            # regular stage (h / N)

    A = [  ]
    b = [-144;     0;    0;    0;    0; 1644 ] // 3725     # Coefficients of the derivatives `kᵢ`
    d = [ 0;    -267;  952;  -1548; 1608;  0 ] // 745     # Coefficients of the function values `yᵢ`
    nb, nd = 2, 4

    A = map(T, A)
    b = map(T, b)
    d = map(T, d)
    return A, b, d, o, s, rs, (nb, nd)
end

# `A(α)` stability, Only has `5.1`-order convergence
function construct_MsRKo6s6(T::Type = Float64)

    s = 6                 # Stage number
    o = 6                 # Order
    rs = s + 1            # regular stage (h / N)

    A = [  ]
    b = [600;     0;        0;      0;     0;    0;  8820 ] // 21509     # Coefficients of the derivatives `kᵢ`
    d = [ 0;   6984;   -28575;  54800;  -63900; 52200;  0 ] // 21509     # Coefficients of the function values `yᵢ`
    nb, nd = 2, 5

    A = map(T, A)
    b = map(T, b)
    d = map(T, d)
    return A, b, d, o, s, rs, (nb, nd)
end

# # BDF method
# function construct_MsRKo5s5b1d6(T::Type = Float64)

#     s = 5                 # Stage number
#     o = 5                 # Order
#     rs = s + 1            # regular stage (h / N)

#     A = [  ]
#     b = [  0;       0;      0;    0;    0;    1 ]         # Coefficients of the derivatives `kᵢ`
#     d = [ -1//5; 5//4; -10//3;    5;   -5;  137//60 ]     # Coefficients of the function values `yᵢ`
#     nb, nd = 1, 6

#     A = map(T, A)
#     b = map(T, b)
#     d = map(T, d)
#     return A, b, d, o, s, rs, (nb, nd)
# end

# function construct_MsRKo6s6b1d7(T::Type = Float64)

#     s = 6                 # Stage number
#     o = 6                 # Order
#     rs = s + 1            # regular stage (h / N)

#     A = [  ]
#     b = [  0;       0;      0;      0;    0;  0;    1 ]         # Coefficients of the derivatives `kᵢ`
#     d = [ 1//6; -6//5;  15//4; -20//3; 15/2; -6; 147//60 ]     # Coefficients of the function values `yᵢ`
#     nb, nd = 1, rs

#     A = map(T, A)
#     b = map(T, b)
#     d = map(T, d)
#     return A, b, d, o, s, rs, (nb, nd)
# end


function construct_DPRKN8(T2::Type = Float64)

    c1 = convert(T2, 1 // 20)
    c2 = convert(T2, 1 // 10)
    c3 = convert(T2, 3 // 10)
    c4 = convert(T2, 1 // 2)
    c5 = convert(T2, 7 // 10)
    c6 = convert(T2, 9 // 10)
    c7 = convert(T2, 1)
    c8 = convert(T2, 1)
end