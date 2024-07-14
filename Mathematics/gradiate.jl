using FiniteDifferences, LinearAlgebra

"""
   Gradient of scalar multi-variable function, f(x,y)

     ∇f(x,y) = ∇f([x...],[y...]ᵀ) = ∑ êᵢ* ∂f/∂xᵢ

    (Df1...) = Df(fdm, f, xs...,q = order_q)

     mesh[:,:] = x .* y

   * if fdm = central_fdm(p,q), the derivative near boundaries is calculated by forward and backward schemes

 Example:
   c51 = central_fdm(5,1)
   c51.grid
   c51.coefs : coefficients
   c51.condition
   c51.factor
   c51.


   nx = 3
   ny = 4
   f(x,y) = y.^3 .* x.^2
   fy(x,y)  = 3 * y.^2 .* x.^2
   fx(x,y)  = 2 * y.^3 .* x
   x = collect(LinRange(1,3,nx))
   y = collect(LinRange(1,5,ny)')
   df = Df(c51,f,x,y,q = order_q)            # ∇f([x...],[y...]ᵀ)
   df = Df(c51,(x,y)->f(x,y),x,y,q = order_q)

"""

function Df(fdm, f, xs...;q = order_q)
    dims = length(xs)
    Jf = jacobian(fdm, f, xs...; len=nothing)
    nx = zeros(Int,dims)
    for i in 1:dims
        nx[i] = length(xs[i])
    end
    nxy = prod(nx)
    if dims == 2
        Df1 = zeros(nx[1],nx[2])
        Df2 = zeros(nx[1],nx[2])
        k = 0
        for j in 1:nx[2]
            for i in 1:nx[1]
                k += 1
                Df1[i,j] = Jf[1][k,i]
                Df2[i,j] = Jf[2][k,j]
            end
        end
        ##########################################
        # left boundary x0               # ok
        fdm = forward_fdm(3,q)
        Jf,~ = fDf(fdm, f, xs[1][1:4],xs[2])
        Df1[1:3,:] = Jf[1:3,:]
        # # # left boundary y0     #
        y4 = zeros(1,4)
        y4[:] = xs[2][1:4]
        ~,Jf = fDf(fdm, f, xs[1],y4)
        Df2[:,1:3] = Jf[:,1:3]
        # # right boundary, x9        # ok
        fdm = backward_fdm(3,q)
        x_num = length(xs[1])
        Jf,~ = fDf(fdm, f, xs[1][x_num-3:x_num],xs[2])
        Df1[x_num-2:x_num,:] = Jf[2:4,:]
        # # right boundary y9
        x_num = length(xs[2])
        y4 = zeros(1,4)
        y4[:] = xs[2][x_num-3:x_num]
        ~,Jf = fDf(fdm, f, xs[1],y4)
        Df2[:,x_num-2:x_num] = Jf[:,2:4]
        return (Df1,Df2)
    elseif dims == 1
        df =  diag(Jf[1])
        ########################
        x_num = length(xs[1])
        # left boundary
        fdm = forward_fdm(3,q)
        Jf = fDf(fdm, f, xs[1][1:4])
        df[1:3] = Jf[1:3]
        # right boundary
        fdm = backward_fdm(3,q)
        Jf = fDf(fdm, f, xs[1][x_num-3:x_num])
        df[x_num-2:x_num] = Jf[2:4]
        return df
    else
        println("dims ?? ")
    end
end

function fDf(fdm, f, xs...)
    dims = length(xs)
    Jf = jacobian(fdm, f, xs...; len=nothing)
    nx = zeros(Int,dims)
    for i in 1:dims
        nx[i] = length(xs[i])
    end
    nxy = prod(nx)
    if dims == 2
        Df1 = zeros(nx[1],nx[2])    # df/dx
        Df2 = zeros(nx[1],nx[2])    # df/dy
        k = 0
        for j in 1:nx[2]
            for i in 1:nx[1]
                k += 1
                Df1[i,j] = Jf[1][k,i]
                Df2[i,j] = Jf[2][k,j]
            end
        end
        return (Df1,Df2)
    elseif dims == 1
        df =  diag(Jf[1])
        # fdm = forward_fdm(3,1)
        # Jf = jacobian(fdm, f, xs[1:4])
        # df[1:3] = Jf[1:3]
        return df
    else
        println("dims ?? ")
    end
end

function Df3(fdm, f, xs...)
    dims = length(xs)
    if dims == 3
    else
        return Df(fdm, f, xs...)
    end
end


"""
# E2(x,y) = sin.(x) .* cos.(y)
@time dExyc = Df(c51,(x,y)->E2(x,y),x,x2,q = order_q)
# @time dExyf = Df(f51,(x,y)->E2(x,y),x,x2,q = order_q)
# @time dExyb = Df(b51,(x,y)->E2(x,y),x,x2,q = order_q)
# dEx = cos.(x) .* cos.(x2)
# dEy = -sin.(x) .* sin.(x2)
# ddEx = -sin.(x) .* cos.(x2)
# ddEy = -sin.(x) .* cos.(x2)
# errxf = dEx - dExyf[1]
# erryf = dEy - dExyf[2]
# errx = dEx - dExyc[1]
# erry = dEy - dExyc[2]
"""
