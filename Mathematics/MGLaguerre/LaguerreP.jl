# using SpecialFunctions

"""
 Values of Generalized Laguerre Polynomials:
    Lᵅₙ(v) = laguerreP(n,x,alpha = α) , n ≥ 0 , α ≥ -1

  The correspond weights will be:

    wₐ(v) = v^α * e^(-v)

  Coded by Wang yanpeng, USTC, 2020.10.07.
  Copyright (c) 2020, Wang Yanpeng.
  All rights reserved.

 first terms
  laguerrePoly(0,α,x) = 1;
  laguerrePoly(1,α,x) = 1 + α - x;
  laguerrePoly(2,α,x) = 2 + 3*α + α^2 - 4*x - 2*α*x + x^2)/2;
  laguerrePoly(3,α,x) = 6 + 11*α + 6*α^2 + α^3 - 18*x - 15*α*x - 3*α^2*x +
                   9*x^2 + 3*α*x^2 - x^3)/6.
 Recursion:
  laguerrePoly(0,α,x) = 1
  laguerrePoly(1,α,x) = 1 + α + x

 if 2 ⩽ n
  laguerrePoly(n,α,x) = ( (α+2n-1-x) * L(n-1,α,x) + (1-n -α) * L(n-2,α,x) ) / n

"""

function laguerrePoly(n::Integer,x::Real;α::Real=0.0)

    # Compute the values of General Laguerre Polynomials
    if n==0
        return 1         # GL(0,α,x)
    elseif n==1
        return 1.0 + α - x       # GL(1,α,x)
    elseif n > 1
        poly1st = 1.0 |> typeof(x)        # L0 = 1
        if α == 0.0
            GL = 1 - x # GL(1,0.0,x)  , i = 1 + 1
            polyn =  1 |> T
            for i = 1:n-1
                polyn = ( (2i+1-x)* GL - i * poly1st) / (i + 1)
                poly1st = GL
                GL = polyn
            end
            return GL   # GL(n,α,x)
        else
            GL = α+ 1 - x # GL(1,0.0,x)  , i = 1 + 1
            polyn =  1 |> T
            for i = 1:n-1
                polyn = ( (α+ 2i+1-x).* GL - (α+ i) * poly1st) / (i + 1)
                poly1st = GL
                GL = polyn
            end
            return GL   # GL(n,α,x)
        end
    else
        return 0
    end
end

function laguerrePoly(n::Integer,x::AbstractVector{T};α::Real=0.0) where{T}

    nx = length(x)
    # Compute the values of General Laguerre Polynomials
    if n==0
        return ones(T,nx)         # GL(0,α,x)
    elseif n==1
        return 1 + α .- x       # GL(1,α,x)
    elseif n>1
        poly1st = ones(T, nx)        # L0 = 1
        if α == 0.0
            GL = 1 .- x # GL(1,0.0,x)  , i = 1 + 1
            polyn = ones(T, nx)
            for i = 1:n-1
                polyn = ( (2i+1 .-x).* GL - i * poly1st) / (i + 1)
                poly1st = GL
                GL = polyn
            end
            return GL   # GL(n,α,x)
        else
            GL = 1 + α .- x # GL(1,α,x)  , i = 1 + 1
            for i = 1:n-1
                polyn = ( (α+2i+1 .-x).* GL - (α+i) * poly1st) / (i + 1)
                poly1st = GL
                GL = polyn
            end
            return GL   # GL(n,α,x)
        end
    else
    end
end


function laguerrePolym(n::Integer,x::Real;α::Real=0.0)

    # Compute the values of General Laguerre Polynomials
    if n==0
        return 1.0         # GL(0,α,x)
    elseif n==1
        return [1.0, 1.0 + α - x]       # GL(1,α,x)
    elseif n > 1
        GL = zeros(typeof(x), n+1 )
        GL[1] = 1.0        # GL(0,α,x)  , i = 0 + 1
        GL[2] = 1.0 + α - x # GL(1,α,x)  , i = 1 + 1
        if α == 0.0
            for i = 2:n
                GL[i+1] = ( (α+2i-1 -x) * GL[i] + (1-α-i) * GL[i-1] )/i
            end
            return GL
        else
            for i = 2:n
                GL[i+1] = ( (α+2i-1 -x) * GL[i] + (1-α-i) * GL[i-1] )/i
            end
            return GL
        end
    else
    end
end

function laguerrePolym(n::Integer,x::AbstractVector{T};α::Real=0.0) where{T}

    nx = length(x)
    # Compute the values of General Laguerre Polynomials
    if n==0
        return ones(T,nx)         # GL(0,α,x)
    elseif n==1
        GL = ones(T, nx, n+1 )
        GL[:,2] = 1 + α .- x # GL(1,α,x)  , i = 1 + 1
    elseif n>1
        GL = ones(T, nx, n+1 )
            # GL[:,1] .= 1        # GL(0,α,x)  , i = 0 + 1
        if α == 0.0
            GL[:,2] = 1 .- x # GL(1,α,x)  , i = 1 + 1
            for i = 2:n
                GL[:,i+1] = ( (2i-1 .-x).*GL[:,i] + (1-i).*GL[:,i-1] )/i
            end
            return GL
        else
            GL[:,2] = 1 + α .- x # GL(1,α,x)  , i = 1 + 1
            for i = 2:n
                GL[:,i+1] = ( (α+2i-1 .-x).*GL[:,i] + (1-α-i).*GL[:,i-1] )/i
            end
            return GL   # GL(n,α,x)
        end
    else
    end
end

"""
  General Gauss-Laguerre functions (GLFs) for modified Gauss-Laguerre points and weights

   Lᵅₙ(v) = e^(-v/2) * Lᵅₙ(v)
   w_α(v) = v^α                 # e^(-v) → Lᵅₙ(v)Lᵅₙ(v)

"""

function laguerreFun(n::Integer,x::Real;α::Real=0.0)

    # Compute the values of General Laguerre Polynomials
    if n==0
        return exp(-x/2.0)         # GL(0,α,x)
    elseif n==1
        return (1.0 + α - x) * exp(-x/2.0)       # GL(1,α,x)
    elseif n > 1
        poly1st = exp(-x/2.0)       # L0 = 1
        GL = (1.0 + α - x) * exp(-x/2.0) # GL(1,0.0,x)  , i = 1 + 1
        polyn = 1.0 |> typeof(x)
        if α == 0.0
            for i = 1:n-1
                polyn = ( (2i+1-x)* GL - i * poly1st) / (i + 1)
                poly1st = GL
                GL = polyn
            end
            return GL   # GL(n,α,x)
        else
            for i = 1:n-1
                polyn = ( (α+ 2i+1.0-x).* GL - (α+ i) * poly1st) / (i + 1)
                poly1st = GL
                GL = polyn
            end
            return GL   # GL(n,α,x)
        end
    else
        return 0
    end
end

function laguerreFun(n::Integer,x::AbstractVector{T};α::Real=0.0) where{T}

    nx = length(x)
    # Compute the values of General Laguerre Polynomials
    if n==0
        return exp.(-x/2)         # GL(0,α,x)
    elseif n==1
        return (1.0 + α .- x) .* exp.(-x/2)       # GL(1,α,x)
    elseif n>1
        poly1st = exp.(-x/2)        # L0 = 1
        GL = (1 + α .- x).*exp.(-x/2) # GL(1,α,x)  , i = 1 + 1
        polyn = ones(T, nx)
        if α == 0.0
            for i = 1:n-1
                polyn = ( (2i+1 .-x).* GL - i * poly1st) / (i + 1)
                poly1st = GL
                GL = polyn
            end
            return GL   # GL(n,α,x)
        else
            for i = 1:n-1
                polyn = ( (α+2i+1 .-x).* GL - (α+i) * poly1st) / (i + 1)
                poly1st = GL
                GL = polyn
            end
            return GL   # GL(n,α,x)
        end
    else
    end
end

function laguerreFunm(n::Integer,x::Real;α::Real=0.0)

    # Compute the values of General Laguerre Polynomials
    if n==0
        return exp(-x/2.0)         # GL(0,α,x)
    elseif n==1
        return [1.0, 1.0 + α - x] * exp(-x/2.0)       # GL(1,α,x)
    elseif n > 1
        GL = zeros(typeof(x), n+1 )
        GL[1] = exp(-x/2.0)        # GL(0,α,x)  , i = 0 + 1
        GL[2] = (1.0 + α - x)*exp(-x/2.0) # GL(1,α,x)  , i = 1 + 1
        if α == 0.0
            for i = 2:n
                GL[i+1] = ( (2i-1 -x) * GL[i] + (1-i) * GL[i-1] )/i
            end
            return GL
        else
            for i = 2:n
                GL[i+1] = ( (α+2i-1 -x) * GL[i] + (1-α-i) * GL[i-1] )/i
            end
            return GL
        end
    else
    end
end

function laguerreFunm(n::Integer,x::AbstractVector{T};α::Real=0.0) where{T}

    nx = length(x)
    # Compute the values of General Laguerre Polynomials
    if n==0
        return exp.(-x/2)         # GL(0,α,x)
    elseif n==1
        GL = ones(T, nx, n+1 )
        GL[:,1] = exp.(-x/2)        # GL(0,α,x)  , i = 0 + 1
        GL[:,2] = (1.0 + α .- x).*exp.(-x/2) # GL(1,α,x)  , i = 1 + 1
    elseif n>1
        GL = ones(T, nx, n+1 )
        GL[:,1] = exp.(-x/2)        # GL(0,α,x)  , i = 0 + 1
        GL[:,2] = (1.0 + α .- x).*exp.(-x/2) # GL(1,α,x)  , i = 1 + 1
        if α == 0.0
            for i = 2:n
                GL[:,i+1] = ( (2i-1 .-x).*GL[:,i] + (1-i).*GL[:,i-1] )/i
            end
            return GL
        else
            for i = 2:n
                GL[:,i+1] = ( (α+2i-1 .-x).*GL[:,i] + (1-α-i).*GL[:,i-1] )/i
            end
            return GL   # GL(n,α,x)
        end
    else
    end
end
