

"""
Fix step length, h = constant of function f(x)
"""
function simpson1D(f::AbstractVector,h::Number)
    nf = length(f)
    if isodd(nf) == 1
        I = h/3*(f[1] + f[nf]+ 4sum(f[2:2:nf-1]) + 2sum(f[3:2:nf-2]))
        I₂ = h*(sum(f) -0.5(f[1] + f[nf]))  # trapezoidal
    else
        I = h/3*(f[1] + f[nf-1]+ 4sum(f[2:2:nf-2]) + 2sum(f[3:2:nf-3]))
        I = I + 0.5(f[nf-1] + f[nf])  # endpoint with trapezoidal method
        I₂ = h*(2sum(f[1:2:nf-1])-f[1] +0.5(f[nf]- f[nf-1]))  # trapezoidal
    end
    E  = I - I₂  # error
    return I,E
end

## different steps hx = diff(x)
"""
Variable step length, h = [...h[i]...] of function  f(x)
  Compound Simpson method, double half-dividing process is done to keep nf = 4N⁺+1

  h[4i:4(i+1)] = hi

"""

function simpson1Dvc(f::AbstractVector,x::AbstractVector)
    hx = diff(x,dims = 1)
    nf = length(f)
    nf > 5 && isinteger((nf-1)/4) || throw(ArgumentError("
                                  meshs dons't mach hsimpson cond.: nf=4N⁺+1"))
    I = 0   # Simpson intgration
    I₂ = 0  #
    for i = 2:2:nf-1
        I = I + hx[i]/3*(f[i-1] + 4f[i]+ f[i+1])
    end
    for i = 3:4:nf-2
        I₂ = I₂ + (hx[i]+hx[i+1])/3*(f[i-2] + 4f[i]+ f[i+2])
    end

    # if abs(I) == 0
    #     if I₂ == 0
    #         E = 0
    #         warning("I_new is zero")
    #     else
    #         E  = (I - I₂)./1 # Relative Error
    #         warning("I_new is zero")
    #     end
    # else
    #     E  = (I - I₂)./1
    # end
    E  = (I - I₂)
    return I,I₂
end



## different steps hx = diff(x)
"""
Variable step length, h = [...h[i]...] of function  f(x)

  double half-dividing process is done to keep nf = 4N⁺+1
  h[4i:4(i+1)] = hi

Cumulative Integration: I = I[1:length(x)] = integrate(f(x),x,inv)
    when inv = 0, upper boundace integration
         inv = 1, lower boundace integration
"""

function cIsimpsonvc(f::AbstractVector,x::AbstractVector;inv::Bool=false)
    if inv == 0
        nf = length(f)
        hx = diff(x,dims = 1)
        nf > 5 && isinteger((nf-1)/4) || throw(ArgumentError("
                      meshs dons't mach hsimpson cond.: nf=4N⁺+1"))
        I = zeros(nf)
        # i = 1 , I[i] = 0
        # I[3:2:nf]
        for i = 2:2:nf-1
            I[i+1] = I[i-1] + hx[i]/3*(f[i-1] + 4f[i]+ f[i+1])
        end
        # I[2:2:nf-1] with interpolations of DataInterpolations.jl
        interp = QuadraticInterpolation(I[1:2:nf],x[1:2:nf])
        I[2:2:nf-1] = interp.(x[2:2:nf-1])
        return I
    else
        nf = length(f)
        hx = diff(x,dims = 1)
        nf > 5 && isinteger((nf-1)/4) || throw(ArgumentError("
                      meshs dons't mach hsimpson cond.: nf=4N⁺+1"))
        I = zeros(nf)
        # i = nf , I[i] = 0
        # I[nf-2:2:1]
        for i = nf-1:-2:2
            I[i-1] = I[i+1] + hx[i]/3*(f[i-1] + 4f[i]+ f[i+1])
        end
        # I[2:2:nf-1] with interpolations of DataInterpolations.jl
        interp = QuadraticInterpolation(I[1:2:nf],x[1:2:nf])
        I[2:2:nf-1] = interp.(x[2:2:nf-1])
        return I
    end
end

function cIsimpsonv(f::AbstractVector,x::AbstractVector;inv::Bool=false)
    if inv == 0
        nf = length(f)
        hx = diff(x,dims = 1)
        I = zeros(nf)
        # i = 1 , I[i] = 0
        # I[3:2:nf]
        for i = 2:2:nf-1
            I[i+1] = I[i-1] + hx[i]/3*(f[i-1] + 4f[i]+ f[i+1])
        end
        # I[2:2:nf-1] with interpolations of DataInterpolations.jl
        interp = QuadraticInterpolation(I[1:2:nf],x[1:2:nf])
        I[2:2:nf-1] = interp.(x[2:2:nf-1])
        return I
    else
    end
end

    # return I
    #
    # # I₂ = zeros(nf)
    # # # i = 1 , I₂[i] = 0
    # # # I[5:4:nf]
    # # for i = 3:4:nf-2
    # #     I₂[i+2] = I₂[i-2] + (hx[i]+hx[i+1])/3*(f[i-2] + 4f[i]+ f[i+2])
    # # end
    # # # I[(2:4)*(1:(nf-1)/4] with interpolations
    # #
    # # E  = I - I₂ # error
    # # println(nf)
    # return I,E
