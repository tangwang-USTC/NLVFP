"""
  dfv = dfvCheby(fL0,Dc,dxdv,vG,vc,nc,orders,domain;is_negative=true,k=k)
"""

function dfvCheby(fL0::AbstractVector{T},Dc::AbstractArray{T,N},dxdv::T,
               vG::AbstractVector{T},vc::AbstractVector{T},nc::Int,orders::Int,
               domain::Vector{T};is_negative::Bool=true,k::Int=2) where {T,N}

    if is_negative == false
        L1 = 1
    else
        L1  = 2 # GvL(L) ≤ 0
    end
    fL0log = GvLlogTrans(fL0,vG,vc,nc,orders,domain,L1;k=k,iflag=true)
    return fL0 .* ((Dc * fL0log) * dxdv)
end

"""
  Higher-precision difference matrices for derivatives with chebyshev notes.
"""

errDc(n) = chebyshevdiff(n;M=2, datatype = BigFloat) - chebyshevdiff(n;M=2, datatype = Float64)
Dc12n3 = chebyshevdiff(3;M=2, datatype = BigFloat) |> Array{datatype}
Dc12n5 = chebyshevdiff(5;M=2, datatype = BigFloat) |> Array{datatype}
Dc12n9 = chebyshevdiff(9;M=2, datatype = BigFloat) |> Array{datatype}
Dc12n17 = chebyshevdiff(17;M=2, datatype = BigFloat) |> Array{datatype}  # Difference matrices of first kind of Chebyshev polynomials
Dc12n3b = chebyshevdiff(3;M=2, datatype = BigFloat)
Dc12n5b = chebyshevdiff(5;M=2, datatype = BigFloat)
Dc12n9b = chebyshevdiff(9;M=2, datatype = BigFloat)
Dc12n17b = chebyshevdiff(17;M=2, datatype = BigFloat)  # Difference matrices of first kind of Chebyshev polynomials

function Dc1n(n::Int;datatype::DataType=Float64)

    if datatype == BigFloat
        if n == 3
            return Dc12n3b[:,:,1]
        elseif n == 5
            return Dc12n5b[:,:,1]
        elseif n == 9
            return Dc12n9b[:,:,1]
        elseif n == 17
            return Dc12n17b[:,:,1]
        else
            return chebyshevdiff(n;M=1, datatype = BigFloat)
        end
    else
        if n == 3
            return Dc12n3[:,:,1]
        elseif n == 5
            return Dc12n5[:,:,1]
        elseif n == 9
            return Dc12n9[:,:,1]
        elseif n == 17
            return Dc12n17[:,:,1]
        else
            return chebyshevdiff(n;M=1, datatype = BigFloat) |> Array{datatype}
        end
    end
end

function Dc2n(n::Int;datatype::DataType=Float64)

    if datatype == BigFloat
        if n == 3
            return Dc12n3b[:,:,2]
        elseif n == 5
            return Dc12n5b[:,:,2]
        elseif n == 9
            return Dc12n9b[:,:,2]
        elseif n == 17
            return Dc12n17b[:,:,2]
        else
            return  chebyshevdiff(n;M=2, datatype = BigFloat)[:,:,2]
        end
    else
        if n == 3
            return Dc12n3[:,:,2]
        elseif n == 5
            return Dc12n5[:,:,2]
        elseif n == 9
            return Dc12n9[:,:,2]
        elseif n == 17
            return Dc12n17[:,:,2]
        else
            return  chebyshevdiff(n;M=2, datatype = BigFloat)[:,:,2] |> Array{datatype}
        end
    end
end

errvcc(n) = cheby_extrema(n,big.(domain)) - cheby_extrema(n,Float64.(domain))
 vccn3b = cheby_extrema(3,big.(domain))
 vccn5b = cheby_extrema(5,big.(domain))
 vccn9b = cheby_extrema(9,big.(domain))
 vccn10b = cheby_extrema(10,big.(domain))
 vccn11b = cheby_extrema(11,big.(domain))
 vccn12b = cheby_extrema(12,big.(domain))
 vccn13b = cheby_extrema(13,big.(domain))
 vccn15b = cheby_extrema(15,big.(domain))
 vccn17b = cheby_extrema(17,big.(domain))
 vccn33b = cheby_extrema(33,big.(domain))
  vccn3 = cheby_extrema(3,big.(domain)) |> AbstractVector{datatype}
  vccn5 = cheby_extrema(5,big.(domain)) |> AbstractVector{datatype}
  vccn9 = cheby_extrema(9,big.(domain)) |> AbstractVector{datatype}
  vccn10 = cheby_extrema(10,big.(domain)) |> AbstractVector{datatype}
  vccn11 = cheby_extrema(11,big.(domain)) |> AbstractVector{datatype}
  vccn12 = cheby_extrema(12,big.(domain)) |> AbstractVector{datatype}
  vccn13 = cheby_extrema(13,big.(domain)) |> AbstractVector{datatype}
  vccn15 = cheby_extrema(15,big.(domain)) |> AbstractVector{datatype}
  vccn17 = cheby_extrema(17,big.(domain)) |> AbstractVector{datatype}
  vccn33 = cheby_extrema(33,big.(domain)) |> AbstractVector{datatype}

function vccn(n::Int;datatype::DataType=Float64)

    if datatype == BigFloat
        if n == 3
            return vccn3b
        elseif n == 5
            return vccn5b
        elseif n == 9
            return vccn9b
        elseif n == 10
            return vccn10b
        elseif n == 11
            return vccn11b
        elseif n == 12
            return vccn12b
        elseif n == 13
            return vccn13b
        elseif n == 15
            return vccn15b
        elseif n == 17
            return vccn17b
        elseif n == 33
            return vccn33b
        else
            return cheby_extrema(n,big.(domain))
        end
    else
        if n == 3
            return vccn3
        elseif n == 5
            return vccn5
        elseif n == 9
            return vccn9
        elseif n == 10
            return vccn10
        elseif n == 11
            return vccn11
        elseif n == 12
            return vccn12
        elseif n == 13
            return vccn13
        elseif n == 15
            return vccn15
        elseif n == 17
            return vccn17
        elseif n == 33
            return vccn33
        else
            return cheby_extrema(n,big.(domain)) |> AbstractVector{datatype}
        end
    end
end

 μccn3 = chebyshevmoments1(Float64,3) |> AbstractVector{datatype}
 μccn4 = chebyshevmoments1(Float64,4) |> AbstractVector{datatype}
 μccn5 = chebyshevmoments1(Float64,5) |> AbstractVector{datatype}
 μccn6 = chebyshevmoments1(Float64,6) |> AbstractVector{datatype}
 μccn7 = chebyshevmoments1(Float64,7) |> AbstractVector{datatype}
 μccn8 = chebyshevmoments1(Float64,8) |> AbstractVector{datatype}
 μccn9 = chebyshevmoments1(Float64,9) |> AbstractVector{datatype}
 μccn10 = chebyshevmoments1(Float64,10) |> AbstractVector{datatype}
 μccn11 = chebyshevmoments1(Float64,11) |> AbstractVector{datatype}
 μccn12 = chebyshevmoments1(Float64,12) |> AbstractVector{datatype}
 μccn13 = chebyshevmoments1(Float64,13) |> AbstractVector{datatype}
 μccn14 = chebyshevmoments1(Float64,14) |> AbstractVector{datatype}
 μccn15 = chebyshevmoments1(Float64,15) |> AbstractVector{datatype}
 μccn16 = chebyshevmoments1(Float64,16) |> AbstractVector{datatype}
 μccn17 = chebyshevmoments1(Float64,17) |> AbstractVector{datatype}
 μccn18 = chebyshevmoments1(Float64,18) |> AbstractVector{datatype}
 μccn20 = chebyshevmoments1(Float64,20) |> AbstractVector{datatype}
 μccn33 = chebyshevmoments1(Float64,33) |> AbstractVector{datatype}

function μccn(n::Int)
    if n == 3
        μcc = μccn3
    elseif n == 4
        μcc = μccn4
    elseif n == 5
        μcc = μccn5
    elseif n == 6
        μcc = μccn6
    elseif n == 7
        μcc = μccn7
    elseif n == 8
        μcc = μccn8
    elseif n == 9
        μcc = μccn9
    elseif n == 10
        μcc = μccn10
    elseif n == 11
        μcc = μccn11
    elseif n == 12
        μcc = μccn12
    elseif n == 13
        μcc = μccn13
    elseif n == 15
        μcc = μccn15
    elseif n == 16
        μcc = μccn16
    elseif n == 17
        μcc = μccn17
    elseif n == 18
        μcc = μccn18
    elseif n == 20
        μcc = μccn20
    elseif n == 33
        μcc = μccn33
    else
        μcc = chebyshevmoments1(Float64,n) |> AbstractVector{datatype}
    end
    return μcc
end

function errxw2(L1)
    xb, wb = GaussQuadrature.legendre(BigFloat,L1, neither)
    xf, wf = GaussQuadrature.legendre(Float64,L1, neither)
    return Float64.(xb - xf), Float64.(wb - wf)
end

function errDP(L1)
    mub, DPb, Mμb, Munb = Dμ(L1;datatype = BigFloat)
    muf, DPf, Mμf, Munf = Dμ(L1;datatype = Float64)
    return Float64.(DPb - DPf)
end
function errMun(L1)
    mub, DPb, Mμb, Munb = Dμ(L1;datatype = BigFloat)
    muf, DPf, Mμf, Munf = Dμ(L1;datatype = Float64)
    return Float64.(Munb - Munf)
end
