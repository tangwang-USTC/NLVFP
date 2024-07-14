"""
  Laguerre polynomials of vG with legnth n

    nv1 = n + 1
    α = α |> Float64
    vg, wg = gausslaguerre(nv1,α)          # FastGaussQuadrature
    vG,w1 =  laguerre(nv1,α,endpt)         # GuassQuadrature
       where endpt in [neither, left]

    # Lₙᵅ(vG) = Lₙ(vG,α) = laguerreP(:,vG,;α=0.0)  # //scr//mathematics

    Ln0 = zeros(datatype,nv1,nv1)
    for i in 0:nv1-1
        Ln0[i + 1,:] = laguerreP(i,vG;α=α)
    end

    Mv = Ln0 .* w1
    Mvn = Ln0ᵀ = transpose(Ln0)

    Differential matrix:

      # Differential matrix, dᵏLₙᵃ/dvᵏ  / vⁿ
      # D(1) = dᵏLₙᵃ/dvᵏ, M = 1,2
      # b = 1     # vG = vG / b
      DL = laguerrediff(vG; M = 2, b = 1)
      vG,w1,Mv, Mvn = laguerreMv(nv1;α=α,endptv=endptv)

"""

function laguerreMv(nv1::Int64;α::AbstractFloat = 0.0, endptv=neither)

    vG, w1 = laguerre(nv1,α, endptv)
    T = typeof(α)
    # Coefficients
    Mv = zeros(T,nv1,nv1)
    Mvn = zeros(T,nv1,nv1)
    for i in 0:nv1-1
        Mvn[:,i + 1] = laguerrePoly(i,vG;α=α)
        Mv[i + 1,:] = hcat(Mvn[:,i + 1] .* w1)
    end
    return vG,w1,Mv, Mvn
end

"""

 μ, w2, Mμ, Mun = LegendreMμ0(L)
 mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(L)
 μ, w2, Mμ, Mun, Mun1, Mun2 = LegendreMμ012(L)
 μ, DP, Mμ, Mun = Dμ(L)

   Legendre polynomials of μ = cos(θ)

   Pₗ(μ) , L = 0:ℓM , m = 0

"""
function LegendreMμ0(L::Int64; datatype=Float64)

    if datatype == Float64
        μ, w2 = gausslegendre(L + 1)    # DataType = Float64
    else
        μ, w2 = GaussQuadrature.legendre(datatype,L + 1, neither)
        # [neither, both, left, right] for endpoints
    end
    # pl0 = Plm(ℓ,0,μ) , ℓ=0:L, m=0
    Lv = 0:L
    pl0 = Plm(Lv,0,μ)
    Lw = transpose(Lv .+ 1//2)     # = (2ℓ + 1)/2 = w2inv(L,0)
    Mμ = pl0 .* (w2 .* Lw)         # Mμ = pl0 .* w₂ᴸ,  w₂ᴸ = [w2,ℓ]
    Mun = transpose(pl0)
    return μ, w2, Mμ, Mun
end

function LegendreMu012(L::Int64; datatype=Float64)

    if datatype == Float64
        μ, w2 = gausslegendre(L + 1)    # DataType = Float64
    else
        μ, w2 = GaussQuadrature.legendre(datatype,L + 1, neither)
        # [neither, both, left, right] for endpoints
    end
    Lv = 0:L
    Lw = transpose(Lv .+ 1//2)     # = (2ℓ + 1)/2 = w2inv(L,0)
    # pl0 = Plm(ℓ,0,μ) , ℓ=0:L, m=0
    pl0 = Plm(Lv,0,μ)
    Mμ = pl0 .* (w2 .* Lw)         # Mμ = pl0 .* w₂ᴸ,  w₂ᴸ = [w2,ℓ]
    Mun = transpose(pl0)
    #  pl1 = Plm(ℓ,1,μ) , ℓ=0:L, m=0
    #  pl2 = Plm(ℓ,2,μ) , ℓ=0:L, m=0
    if L ≥ 1
        pl0 = Plm(Lv,1,μ)
        Mun1 = transpose(pl0[:,2:end])
        if L ≥ 2
            pl0 = Plm(Lv,2,μ)
            Mun2 = transpose(pl0[:,3:end])
        else
            Mun2 = Array{Float64}(undef)
        end
    else
        Mun1 = Array{Float64}(undef)
        Mun2 = Array{Float64}(undef)
    end
    mu = reshape(μ,1,L+1)
    return mu, Mμ, Mun, Mun1, Mun2
end

function LegendreMμ012(L::Int64; datatype=Float64)

    if datatype == Float64
        μ, w2 = gausslegendre(L + 1)    # DataType = Float64
    else
        μ, w2 = GaussQuadrature.legendre(datatype,L + 1, neither)
        # [neither, both, left, right] for endpoints
    end
    Lv = 0:L
    Lw = transpose(Lv .+ 1//2)     # = (2ℓ + 1)/2 = w2inv(L,0)
    # pl0 = Plm(ℓ,0,μ) , ℓ=0:L, m=0
    pl0 = Plm(Lv,0,μ)
    Mμ = pl0 .* (w2 .* Lw)         # Mμ = pl0 .* w₂ᴸ,  w₂ᴸ = [w2,ℓ]
    Mun = transpose(pl0)
    #  pl1 = Plm(ℓ,1,μ) , ℓ=0:L, m=0
    #  pl2 = Plm(ℓ,2,μ) , ℓ=0:L, m=0
    if L ≥ 1
        pl0 = Plm(Lv,1,μ)
        Mun1 = transpose(pl0[:,2:end])
        if L ≥ 2
            pl0 = Plm(Lv,2,μ)
            Mun2 = transpose(pl0[:,3:end])
        else
            Mun2 = Array{Float64}(undef)
        end
    else
        Mun1 = Array{Float64}(undef)
        Mun2 = Array{Float64}(undef)
    end
    return μ, w2, Mμ, Mun, Mun1, Mun2
end

function Dμ(L::Int64; datatype=Float64)

    if datatype == Float64
        μ, w2 = gausslegendre(L + 1)    # DataType = Float64
    else
        μ, w2 = GaussQuadrature.legendre(datatype,L + 1, neither)
        # [neither, both, left, right] for endpoints
    end
    Lv = 0:L
    # pl0 = Plm(ℓ,0,μ) , ℓ=0:L, m=0
    pl0 = Plm(Lv,0,μ)  # = AssociatedLegendrePolynomials.legendre(LegendreUnitNorm(),Lv,0,μ)
    Lw = transpose(Lv .+ 1//2)     # = (2ℓ + 1)/2 = w2inv(L,0)
    Mμ = pl0 .* (w2 .* Lw)         # Mμ = pl0 .* w₂ᴸ,  w₂ᴸ = [w2,ℓ]
    Mun = transpose(pl0) |> Array

    # Differential matrix, dᵏ/dμᵏ , M = 1,2
    # b = 1     # μ = μ / b
    M = 2
    DP = legendrediff(μ; M = M ,b = 1)    # first two order derivatives
    for ik in 1:M
        DP[:,:,ik] = transpose(DP[:,:,ik]) # due to f(v,μ)
    end
    return μ, DP, Mμ, Mun
end

"""
  fvu = fvL * Mun
  fvL = fvu * Mμ

  μ, w2, plmμ, Mμ, Mun = LegendreMμ(L; datatype=Float64)

   Legendre polynomials of μ = cos(θ)

   Pₗᵐ(μ) , L = 0:ℓM , m = 0:L

   plmμ[:,:,1] = Pₗ⁰(μ) = Plm(Lv,0,μ)

"""

function LegendreMμ(L::Int64; datatype=Float64)
    L1 = L + 1
    Lv = 0:L
    μ, w2 = GaussQuadrature.legendre(datatype,L1, neither)
    # [neither, both, left, right] for endpoints
    # plmμ = zeros(datatype,L1,L1,L1)
    # plmμ = Plm(Lv,Lv,μ)  # Plm(ℓ,m,μ) , ℓ=0:L, m=0:ℓ
    plmμ = Plm(Lv,Lv,μ)
    #### plmμ[μᵢ,L,m] = Plm(ℓ,m,μ)
    Mμ = zeros(datatype,L1,L1,L1)
    Mun = zeros(datatype,L1,L1,L1)
    # Mμ[:,:,m1] = Pₗᵐ(μ) * w₂ᴸ =  Pₗᵐ(μ) * (w2 .* transpose(w2⁻¹))
    for m = 0:L
        m1 = m + 1
        w2⁻¹ = w2inv(L,m; datatype = datatype)
        Mμ[:,:,m1] = plmμ[:,:,m1] .*  (w2 .* transpose(w2⁻¹))
        Mun[:,:,m1] = transpose(plmμ[:,:,m1])
    end
    return μ, w2, Mμ, Mun
end

# weight of matrix Mv⁻¹ for normalization Mv⁻¹Mv
function w1inv(nv::Int64)
    j = 0:nv
    w1⁻¹ = @. 1 // ( (j+1) * (j+2))
    return w1⁻¹
    # j = 0
    # prodj = big(1)
    # w1⁻¹[1] =  1 / gamma( α + 1)
    # for j in 1:nv
    #     if j ≤ 168
    #         prodj = j * prodj
    #         w1⁻¹[j+1] = prodj / gamma(j + α + 1)
    #     else
    #         prodj = j * prodj
    #         w1⁻¹[j+1] = prodj / gamma(big(j) + α + 1)
    #     end
    # end
    # return w1⁻¹
end

# weight of matrix Mμ⁻¹ for normalization Mμ⁻¹Mμ for m-order Pₗᵐ(μ)

# w2⁻¹ = (2ℓ+1)/2 * (ℓ-m)!
function w2inv0(L::Int64, m::Int64; datatype=Float64)
    L1 = L + 1
    w2⁻¹ = zeros(datatype,L1)
    for j = m+1:L1
        w2⁻¹[j] = j + 1//2 * (j-m)
    end
    return w2⁻¹
end

# w2⁻¹ = (2ℓ+1)/2 * (ℓ-m)! / (ℓ+m)!
function w2inv(L::Int64, m::Int64; datatype=Float64)
    m ≤ L || ArgumentError(throw(" m must no more than L"))
    L1 = L + 1
    w2⁻¹ = zeros(datatype,L1)
    if m == 0
        return (0:L) .+ 1 // 2
    elseif m == 1
        Lv = 2:L
        # ℓ = 0  w2⁻¹[1] = 0
        w2⁻¹[2] = 3 // 4
        w2⁻¹[Lv .+ 1] = @. (1+ 1//2Lv) // (Lv+1)
        return w2⁻¹
    elseif m == L
        # L = big(L)
        # ℓ = 0:L-1   w2⁻¹[:] = 0
        L2 = 2L |> datatype
        w2⁻¹[L + 1] =  (1//2 + 1//4L) / gamma(L2)
        return w2⁻¹
    else
        j = m
        mb = m |> datatype
        w2⁻¹[j+1] = (j + 1/2) /2mb / gamma(2mb)
        j = m + 1
        w2⁻¹[j+1] = (j+1/2) /(2mb+1) / gamma(2mb+1)
        for j = m +2:L
                w2⁻¹[j+1] = (j + 1/2)/(j+m) * gamma(j-mb+1)/ gamma(j+mb)
        end
        return w2⁻¹
    end
end
