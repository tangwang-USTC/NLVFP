

"""
  Tableau of RK4, the explicit Runge-Kutta method of order four
  
"""

function construct_RK4(T::Type = Float64)

    s = 4                 # Stage number
    o = 4                 # Order
    rs = 3                # regular stage (h / N)

    A = [ 0      0      0      0
          1//2   0      0      0
          0      1//2   0      0
          0      0      1      0]
    c = [0,     1//2,  1//2,   1]
    b = [1 // 6; 1 // 3; 1 // 3; 1 // 6]

    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    return A, c, b, o, s, rs
end

function construct_RK438(T::Type = Float64)

    s = 4                 # Stage number
    o = 4                 # Order
    rs = 4                # regular stage (h / N)

    A = [ 0      0      0      0
          1//3   0      0      0
         -1//3   1      0      0
          1      -1     1      0]
    c = [0;   1 // 3; 2 // 3;  1]
    b = [1 // 8; 3 // 8; 3 // 8; 1 // 8]

    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    return A, c, b, o, s, rs
end

function construct_RK42(T::Type = Float64)

    s = 4                 # Stage number
    o = 4                 # Order
    rs = 5                # regular stage (h / N)

    A = [ 0      0      0      0
          1//4   0      0      0
          0      1//2   0      0
          1      -2     1      0]
    c = [0,     1//4,  1//2,   1]
    b = [1//6,  0,     2//3,  1//6]

    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    return A, c, b, o, s, rs
end


"""
  Tableau of RK5, the explicit Runge-Kutta method of order five
  
  such as Lawson's 5th order scheme, Runge's First 5 order scheme;

"""

function construct_RK5(T::Type = Float64)

    s = 6                 # Stage number
    o = 5                 # Order
    rs = 5                # regular stage (h / N)

    A = zeros(T, 6, 6)
    c = zeros(T, 6)
    b = zeros(T, 6)

    c[2] = 1 // 4
    c[3] = 1 // 4
    c[4] = 1 // 2
    c[5] = 3 // 4
    c[6] = 1

    A[2, 1] = 1 // 4
    A[3, 1] = 1 // 8
    A[3, 2] = 1 // 8
    A[4, 3] = 1 // 2
    A[5, 1] = 3 // 16
    A[5, 2] = -3 // 8
    A[5, 3] = 3 // 8
    A[5, 4] = 9 // 16
    A[6, 1] = -3 // 7
    A[6, 2] = 8 // 7
    A[6, 3] = 6 // 7
    A[6, 4] = -12 // 7
    A[6, 5] = 8 // 7

    b[1] = 7 // 90
    # b[2] = 0
    b[3] = 16 // 45
    b[4] = 2 // 15
    b[5] = 16 // 45
    b[6] = 7 // 90

    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    return A, c, b, o, s, rs
end

function construct_Lawson5(T::Type = Float64)

    s = 6                 # Stage number
    o = 5                 # Order
    rs = 5                # regular stage (h / N)

    A = zeros(T, 6, 6)
    c = zeros(T, 6)
    b = zeros(T, 6)

    c[2] = 1 // 12
    c[3] = 1 // 4
    c[4] = 1 // 2
    c[5] = 3 // 4
    c[6] = 1

    A[2, 1] = 1 // 12
    A[3, 1] = -1 // 8
    A[3, 2] = 3 // 8
    A[4, 1] = 3 // 5
    A[4, 2] = -9 // 10
    A[4, 3] = 4 // 5
    A[5, 1] = 39 // 80
    A[5, 2] = -9 // 20
    A[5, 3] = 3 // 20
    A[5, 4] = 9 // 16
    A[6, 1] = -59 // 35
    A[6, 2] = 66 // 35
    A[6, 3] = 48 // 35
    A[6, 4] = -12 // 7
    A[6, o] = 8 // 7

    b[1] = 7 // 90
    # b[2] = 0
    b[3] = 16 // 45
    b[4] = 2 // 15
    b[5] = 16 // 45
    b[6] = 7 // 90

    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    return A, c, b, o, s, rs
end

function construct_RungeFirst5(T::Type = Float64)

    s = 6                 # Stage number
    o = 5                 # Order
    rs = 6                # regular stage (h / N)
    
    A = zeros(T, 6, 6)
    c = zeros(T, 6)
    b = zeros(T, 6)

    c[2] = 1 // 5
    c[3] = 2 // 5
    c[4] = 1
    c[5] = 3 // 5
    c[6] = 4 // 5

    A[2, 1] = 1 // 5
    # A[3, 1] = 0
    A[3, 2] = 2 // 5
    A[4, 1] = 9 // 4
    A[4, 2] = -5
    A[4, 3] = 15 // 4
    A[5, 1] = -63 // 100
    A[5, 2] = 9 // 5
    A[5, 3] = -13 // 20
    A[5, 4] = 2 // 25
    A[6, 1] = -6 // 25
    A[6, 2] = 4 // 5
    A[6, 3] = 2 // 15
    A[6, 4] = 8 // 75
    # A[6, o] = 0

    b[1] = 17 // 144
    # b[2] = 0
    b[3] = 25 // 36
    b[4] = 1 // 72
    b[5] = -25 // 72
    b[6] = 25 // 48

    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    return A, c, b, o, s, rs
end



"""
  Tableau of RK4, the implicit Runge-Kutta method of order 4
  
  such as LobattoIIIA4;

"""


function construct_LobattoIIIA4(T::Type = Float64)

    s = 3                 # Stage number
    o = 4                 # Order
    rs = 3                # regular stage (h / N)

    A = [ 0      0      0
          5//24 1//3 -1//24
          1//6  2//3  1//6 ]
    c = [  0;   1//2;   1  ]
    b = [ 1//6; 2//3; 1//6 ]
    b2 =[-1//2;  2;  -1//2 ]

    A = map(T, A)
    c = map(T, c)
    b = map(T, b)
    b2 = map(T, b2)
    return A, c, b, b2, o, s, rs
end


function construct_Kutta3(T::Type = Float64)

    s = 2                 # Stage number
    o = 3                 # Order
    rs = 3                # regular stage (h / N)

    A = [ 0      0      0
          1//2   0      0
          -1     2      0]
    c = [ 0,   1//2,    1]
    b = [1//6; 2//3; 1//6]

    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    return A, c, b, o, s, rs
end


function construct_SSPRK3(T::Type = Float64)

    s = 2                 # Stage number
    o = 3                 # Order
    rs = 3                # regular stage (h / N)

    A = [ 0      0      0
          1      0      0
         1//4  1//4     0]
    c = [ 0,   1,    1//2]
    b = [1//6; 1//6; 4//6]

    A = map(T, A)
    b = map(T, b)
    c = map(T, c)
    return A, c, b, o, s, rs
end
