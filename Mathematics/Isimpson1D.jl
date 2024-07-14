

## different steps hx = diff(x)
"""
Variable step length, 𝓱 = [...h[i]...] of function  f(x)

  nf = length(x) ≥ 3

  x = 0 may be a singular point, which can be replaced by ϵ → 0.

"""

function simpson1Dv(x::AbstractVector,f::AbstractVector)

    Nx = length(x)     # number of points
    # Nh = Nx - 1        # number of  subintervals
    In = zero(x)
    hn = diff(x)
    I = 0.0
    k = 1
    dI = 0.0
    if iseven(Nx - 1) == 1
        for i in 1:((Nx - 1) / 2 |> Int)
            k = 2i
            dI = (2 - hn[k]/hn[k-1]) * f[k-1]
            dI += (hn[k] + hn[k-1])^2 / (hn[k] * hn[k-1]) * f[k]
            dI += (2 - hn[k-1]/hn[k]) * f[k+1]
            dI *= (hn[k-1] + hn[k]) / 6
            I += dI
        end
        return I
    else
        for i in 1:((Nx - 2) / 2 |> Int)
            k = 2i
            dI = (2 - hn[k]/hn[k-1]) * f[k-1]
            dI += (hn[k] + hn[k-1])^2 / (hn[k] * hn[k-1]) * f[k]
            dI += (2 - hn[k-1]/hn[k]) * f[k+1]
            dI *= (hn[k-1] + hn[k]) / 6
            I += dI
        end
        dI = ((2hn[Nx-1] + 3hn[Nx-1] * hn[Nx-2]) / (6(hn[Nx-2] + hn[Nx-1]))) * f[Nx]
        dI += ((hn[Nx-1]^2 + 3hn[Nx-1] * hn[Nx-2]) / (6hn[Nx-2])) * f[Nx-1]
        dI += (hn[Nx-1]^3 / (6hn[Nx-2] * (hn[Nx-2] + hn[Nx-1]))) * f[Nx-2]
        I += dI
        return I
    end
end



## different steps hx = diff(x)
"""
Variable step length, h = [...h[i]...] of function  f(x)

Cumulative Integration: I = I[1:length(x)] = integrate(f(x),x,inv)
    when inv = 0, upper boundace integration
         inv = 1, lower boundace integration

  Inputs:
    f: f[xᵢ]
    x: [xᵢ]
    I0: = In[x₀],(=0.0 default) the initial value which is decided by the boundary condition
    k: th k of integration for no Simpson's points.
       when k < 2 , the low order is for endpoints and even points (no Simpson's points);
       or else, Spline1D is used for those points.

    inv: (=flase default), whether integrate function f(x) for the right point to the left end point.
"""

function cIsimpson1Dv(x::AbstractVector,f::AbstractVector;I0::Float64=0.0,k::Int=3,inv::Bool=false)

    Nx = length(x)     # number of points
    # Nh = Nx - 1        # number of  subintervals
    if inv == 0
        In = zero(x)
        hn = diff(x)
        if isodd(Nx) == 1
            ######## Compute the vaule of I[2N⁺+1]
            i = 1
            In[i] = I0
            i = 3
            dI = (2 - hn[i-1]/hn[i-2]) * f[i-2]
            dI += (hn[i-1] + hn[i-2])^2 / (hn[i-1] * hn[i-2]) * f[i-1]
            dI += (2 - hn[i-2]/hn[i-1]) * f[i]
            dI *= (hn[i-2] + hn[i-1]) / 6
            In[i] += dI
            for i = 5:2:Nx
                dI = (2 - hn[i-1]/hn[i-2]) * f[i-2]
                dI += (hn[i-1] + hn[i-2])^2 / (hn[i-1] * hn[i-2]) * f[i-1]
                dI += (2 - hn[i-2]/hn[i-1]) * f[i]
                dI *= (hn[i-2] + hn[i-1]) / 6
                In[i] =  In[i-2] + dI
            end
            if k ≥ 2
                spl = Spline1D(x[1:2:Nx],In[1:2:Nx];k=k)
                In[2:2:Nx-1] = evaluate(spl,x[2:2:Nx-1])
            else
                i = 2  # Trapezoidal rule
                In[i] = hn[i-1] * (f[1] + f[2]) / 2
                for i = 4:2:Nx-1
                    In[i] = (In[i-1] + In[i+1])/2
                end
            end
            return In
        else
            i = 1
            In[i] = I0
            i = 3
            dI = (2 - hn[i-1]/hn[i-2]) * f[i-2]
            dI += (hn[i-1] + hn[i-2])^2 / (hn[i-1] * hn[i-2]) * f[i-1]
            dI += (2 - hn[i-2]/hn[i-1]) * f[i]
            dI *= (hn[i-2] + hn[i-1]) / 6
            In[i] += dI
            for i = 5:2:Nx-1
                dI = (2 - hn[i-1]/hn[i-2]) * f[i-2]
                dI += (hn[i-1] + hn[i-2])^2 / (hn[i-1] * hn[i-2]) * f[i-1]
                dI += (2 - hn[i-2]/hn[i-1]) * f[i]
                dI *= (hn[i-2] + hn[i-1]) / 6
                In[i] =  In[i-2] + dI
            end
            if k ≥ 2
                spl = Spline1D(x[1:2:Nx],In[1:2:Nx];k=k)
                In[2:2:Nx] = evaluate(spl,x[2:2:Nx])
            else
                i = 2  # Trapezoidal rule
                In[i] = hn[i-1] * (f[1] + f[2]) / 2
                for i = 4:2:Nx-1
                    In[i] = (In[i-1] + In[i+1])/2
                end
                i = Nx
                dI = ((2hn[i-1] + 3hn[i-1] * hn[i-2]) / (6(hn[i-2] + hn[i-1]))) * f[i]
                dI += ((hn[i-1]^2 + 3hn[i-1] * hn[i-2]) / (6hn[i-2])) * f[i-1]
                dI += (hn[i-1]^3 / (6hn[i-2] * (hn[i-2] + hn[i-1]))) * f[i-2]
                In[i] = In[i-1] + dI
            end
            return In
        end
    else
        In = zero(x)
        hn = zero(x)
        hn[2:Nx] = diff(x)
        if isodd(Nx) == 1
            ######## Compute the vaule of I[2N⁺+1]
            i = Nx
            In[i] = I0
            i = Nx - 2
            dI = (2 - hn[i+1]/hn[i+2]) * f[i+2]
            dI += (hn[i+1] + hn[i+2])^2 / (hn[i+1] * hn[i+2]) * f[i+1]
            dI += (2 - hn[i+2]/hn[i+1]) * f[i]
            dI *= (hn[i+2] + hn[i+1]) / 6
            In[i] += dI
            for i = Nx-4:-2:1
                dI = (2 - hn[i+1]/hn[i+2]) * f[i+2]
                dI += (hn[i+1] + hn[i+2])^2 / (hn[i+1] * hn[i+2]) * f[i+1]
                dI += (2 - hn[i+2]/hn[i+1]) * f[i]
                dI *= (hn[i+2] + hn[i+1]) / 6
                In[i] =  In[i+2] + dI
            end
            if k ≥ 2
                spl = Spline1D(x[1:2:Nx],In[1:2:Nx];k=k)
                In[2:2:Nx-1] = evaluate(spl,x[2:2:Nx-1])
            else
                i = Nx - 1  # Trapezoidal rule
                In[i] = hn[i+1] * (f[1] + f[2]) / 2
                for i = Nx-3:-2:2
                    In[i] = (In[i+1] + In[i+1])/2
                end
            end
            return In
        else   # method = 1
            i = Nx
            In[i] = I0
            i = Nx - 3
            dI = (2 - hn[i+1]/hn[i+2]) * f[i+2]
            dI += (hn[i+1] + hn[i+2])^2 / (hn[i+1] * hn[i+2]) * f[i+1]
            dI += (2 - hn[i+2]/hn[i+1]) * f[i]
            dI *= (hn[i+2] + hn[i+1]) / 6
            In[i] += dI
            for i = Nx-5:-2:1
                dI = (2 - hn[i+1]/hn[i+2]) * f[i+2]
                dI += (hn[i+1] + hn[i+2])^2 / (hn[i+1] * hn[i+2]) * f[i+1]
                dI += (2 - hn[i+2]/hn[i+1]) * f[i]
                dI *= (hn[i+2] + hn[i+1]) / 6
                In[i] =  In[i+2] + dI
            end
            nvec = [1:2:Nx-3;Nx]
            spl = Spline1D(x[nvec],In[nvec];k=k)
            nvec = [2:2:Nx-2;Nx-1]
            In[nvec] = evaluate(spl,x[nvec])
            return In
        end
    end
end


"""
   Method 2 for 222222

       i = Nx
       In[i] = I0
       i = Nx - 2
       dI = (2 - hn[i+1]/hn[i+2]) * f[i+2]
       dI += (hn[i+1] + hn[i+2])^2 / (hn[i+1] * hn[i+2]) * f[i+1]
       dI += (2 - hn[i+2]/hn[i+1]) * f[i]
       dI *= (hn[i+2] + hn[i+1]) / 6
       In[i] += dI
       for i = Nx-4:-2:2
           dI = (2 - hn[i+1]/hn[i+2]) * f[i+2]
           dI += (hn[i+1] + hn[i+2])^2 / (hn[i+1] * hn[i+2]) * f[i+1]
           dI += (2 - hn[i+2]/hn[i+1]) * f[i]
           dI *= (hn[i+2] + hn[i+1]) / 6
           In[i] =  In[i+2] + dI
       end
       if k ≥ 2
           spl = Spline1D(x[2:2:Nx],In[2:2:Nx];k=k)
           In[1:2:Nx-1] = evaluate(spl,x[1:2:Nx-1])
       else
           i = Nx - 1  # Trapezoidal rule
           In[i] = hn[i+1] * (f[1] + f[2]) / 2
           for i = 3:2:Nx-3
               In[i] = (In[i+1] + In[i+1])/2
           end
           i = 1
           dI = ((2hn[i+1] + 3hn[i+1] * hn[i+2]) / (6(hn[i+2] + hn[i+1]))) * f[i]
           dI += ((hn[i+1]^2 + 3hn[i+1] * hn[i+2]) / (6hn[i+2])) * f[i+1]
           dI += (hn[i+1]^3 / (6hn[i+2] * (hn[i+2] + hn[i+1]))) * f[i+2]
           In[i] = In[i+1] + dI
       end

"""
