
"""
Implicit Euler Method
"""
function construct_ImplicitEuler(T::Type = Float64)
    A = Matrix{T}(undef, 1, 1)
    A[1] = 1
    c = [1]
    α = [1]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (ImplicitRKTableau(A, c, α, 1))
end

"""
Order 2 Midpoint Method
"""
function construct_MidpointRule(T::Type = Float64)
    A = Matrix{T}(undef, 1, 1)
    A[1] = 1 // 2
    c = [1 // 2]
    α = [1]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (ImplicitRKTableau(A, c, α, 2))
end

function Tableau_KraaijevangerSpijker(::Type{T}=Float64) where {T}
    a = @big [[ 1/2   0   ]
              [-1/2   2   ]]
    b = @big  [-1/2,  3/2 ]
    c = @big  [ 1/2,  3/2 ]
    o = 2

    Tableau{T}(:KraaijevangerSpijker, o, a, b, c)
end

function Tableau_QinZhang(::Type{T}=Float64) where {T}
    a = @big [[ 1/4   0   ]
              [ 1/2   1/4 ]]
    b = @big  [ 1/2,  1/2 ]
    c = @big  [ 1/4,  3/4 ]
    o = 2

    Tableau{T}(:QinZhang, o, a, b, c)
end

"""
Order 2 Trapezoidal Rule (LobattoIIIA2)
Tableau_CrankNicolson = Tableau_CN
"""
function construct_TrapezoidalRule(T::Type = Float64)
    A = [0 0
         1//2 1//2]
    c = [0; 1]
    α = [1 // 2; 1 // 2]
    αEEst = [1; 0]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    αEEst = map(T, αEEst)
    return (ImplicitRKTableau(A, c, α, 2, αEEst = αEEst, adaptiveorder = 1))
end

"""
LobattoIIIB Order 2 method
"""
function construct_LobattoIIIB2(T::Type = Float64)
    A = [1//2 0
         1//2 0]
    c = [0; 1]
    α = [1 // 2; 1 // 2]
    αEEst = [1; 0]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    αEEst = map(T, αEEst)
    return (ImplicitRKTableau(A, c, α, 2, αEEst = αEEst, adaptiveorder = 1))
end

"""
LobattoIIIC Order 2 method

"""
function construct_LobattoIIIC2(T::Type = Float64)
    A = [1//2 -1//2
         1//2 1//2]
    c = [0; 1]
    α = [1 // 2; 1 // 2]
    αEEst = [1; 0]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    αEEst = map(T, αEEst)
    return (ImplicitRKTableau(A, c, α, 2, αEEst = αEEst, adaptiveorder = 1))
end

"""
LobattoIIIC* Order 2 method

"""
function construct_LobattoIIICStar2(T::Type = Float64)
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
LobattoIIID Order 2 method

"""
function construct_LobattoIIID2(T::Type = Float64)
    A = [1//2 1//2
         -1//2 1//2]
    c = [0; 1]
    α = [1 // 2; 1 // 2]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (ImplicitRKTableau(A, c, α, 2))
end

"""
RadauIA Order 3 method

"""
function construct_RadauIA3(T::Type = Float64)
    A = [1//4 -1//4
         1//4 5//12]
    c = [0; 2 // 3]
    α = [1 // 4; 3 // 4]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (ImplicitRKTableau(A, c, α, 3))
end

"""
RadauIIA Order 3 method

"""
function construct_RadauIIA3(T::Type = Float64)
    A = [5//12 -1//12
         3//4 1//4]
    c = [1 // 3; 1]
    α = [3 // 4; 1 // 4]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (ImplicitRKTableau(A, c, α, 3))
end

"""
Tableau of symmetric and symplectic three-stage, 4th order Runge-Kutta method

```julia
Tableau_SRK3(::Type{T}=Float64) where {T}
```
The constructor takes one optional argument, that is the element type of the tableau.

"""
function Tableau_SRK3(::Type{T}=Float64) where {T}
    a = @big [[ 5/36         2/9        5/36-√15/10 ]
              [ 5/36         2/9        5/36        ]
              [ 5/36+√15/10  2/9        5/36        ]]
    b = @big  [ 5/18,        4/9,       5/18        ]
    c = @big  [ 1/2-√15/10,  1/2,       1/2+√15/10  ]
    o = 4

    Tableau{T}(:SRK3, o, a, b, c; R∞=-1)
end

"""
LobattoIIIA Order 4 method
"""
function construct_LobattoIIIA4(T::Type = Float64)
    A = [0 0 0
         5//24 1//3 -1//24
         1//6 2//3 1//6]
    c = [0; 1 // 2; 1]
    α = [1 // 6; 2 // 3; 1 // 6]
    αEEst = [-1 // 2; 2; -1 // 2]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    αEEst = map(T, αEEst)
    return (ImplicitRKTableau(A, c, α, 4, αEEst = αEEst, adaptiveorder = 2))
end

"""
LobattoIIIB Order 4 method

"""
function construct_LobattoIIIB4(T::Type = Float64)
    A = [1//6 -1//6 0
         1//6 1//3 0
         1//6 5//6 0]
    c = [0; 1 // 2; 1]
    α = [1 // 6; 2 // 3; 1 // 6]
    αEEst = [-1 // 2; 2; -1 // 2]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    αEEst = map(T, αEEst)
    return (ImplicitRKTableau(A, c, α, 4, αEEst = αEEst, adaptiveorder = 2))
end

"""
LobattoIIIC Order 4 method

"""
function construct_LobattoIIIC4(T::Type = Float64)
    A = [1//6 -1//3 1//6
         1//6 5//12 -1//12
         1//6 2//3 1//6]
    c = [0; 1 // 2; 1]
    α = [1 // 6; 2 // 3; 1 // 6]
    αEEst = [-1 // 2; 2; -1 // 2]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    αEEst = map(T, αEEst)
    return (ImplicitRKTableau(A, c, α, 4, αEEst = αEEst, adaptiveorder = 2))
end

"""
LobattoIIIC* Order 4 method

"""
function construct_LobattoIIICStar4(T::Type = Float64)
    A = [0 0 0
         1//4 1//4 0
         0 1 0]
    c = [0; 1 // 2; 1]
    α = [1 // 6; 2 // 3; 1 // 6]
    αEEst = [-1 // 2; 2; -1 // 2]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    αEEst = map(T, αEEst)
    return (ImplicitRKTableau(A, c, α, 4, αEEst = αEEst, adaptiveorder = 2))
end

"""
LobattoIIID Order 4 method

"""
function construct_LobattoIIID4(T::Type = Float64)
    A = [1//6 0 -1//6
         1//12 5//12 0
         1//2 1//3 1//6]
    c = [0; 1 // 2; 1]
    α = [1 // 6; 2 // 3; 1 // 6]
    αEEst = [-1 // 2; 2; -1 // 2]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    αEEst = map(T, αEEst)
    return (ImplicitRKTableau(A, c, α, 4, αEEst = αEEst, adaptiveorder = 2))
end

"""
RadauIA Order 5 method

"""
function construct_RadauIA5(T::Type = Float64)
    A = [1//9 (-1 - sqrt(6))/18 (-1 + sqrt(6))/18
         1//9 11 // 45+7 * sqrt(6) / 360 11 // 45-43 * sqrt(6) / 360
         1//9 11 // 45+43 * sqrt(6) / 360 11 // 45-7 * sqrt(6) / 360]
    c = [0; 3 // 5 - sqrt(6) / 10; 3 // 5 + sqrt(6) / 10]
    α = [1 // 9; 4 // 9 + sqrt(6) / 36; 4 // 9 - sqrt(6) / 36]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (ImplicitRKTableau(A, c, α, 5))
end

"""
RadauIIA Order 5 method

"""
function construct_RadauIIA5(T::Type = Float64)
    sq6 = sqrt(6)
    A = [11 // 45-7sq6 / 360 37 // 225-169sq6 / 1800 -2 // 225+sq6 / 75
         37 // 225+169sq6 / 1800 11 // 45+7sq6 / 360 -2 // 225-sq6 / 75
         4 // 9-sq6 / 36 4 // 9+sq6 / 36 1//9]
    c = [2 // 5 - sq6 / 10; 2 / 5 + sq6 / 10; 1]
    α = [4 // 9 - sq6 / 36; 4 // 9 + sq6 / 36; 1 // 9]
    A = map(T, A)
    α = map(T, α)
    c = map(T, c)
    return (ImplicitRKTableau(A, c, α, 5))
end
