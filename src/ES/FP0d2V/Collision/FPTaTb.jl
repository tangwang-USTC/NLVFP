
function dfdtTaTb(T0,para,t)
    dT = zeros(4)
    A = (2π)^0.5

    isp = 1
    iFv = 2
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] = A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])
    iFv = 3
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] += A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])
    iFv = 4
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] += A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])

    isp = 2
    iFv = 1
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] = A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])
    iFv = 3
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] += A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])
    iFv = 4
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] += A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])

    isp = 3
    iFv = 1
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] = A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])
    iFv = 2
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] += A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])
    iFv = 4
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] += A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])

    isp = 4
    iFv = 1
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] = A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])
    iFv = 2
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] += A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])
    iFv = 3
    m2 = [m0[isp], m0[iFv]]
    Zq2 = [Zq[isp], Zq[iFv]]
    n2 = [n0[isp], n0[iFv]]
    T2 = [T0[isp], T0[iFv]]
    dT[isp] += A * FPTaTb(1,2,m2,Zq2,n2,T2) * (T2[2] - T2[1])
    return dT
end

"""
   νT = FPTaTb(isp,iFv,m0,Zq,n0,T0)

"""

# νT   [1/s]

function FPTaTb(isp::Int,iFv::Int,m0::AbstractVector,
    Zq::AbstractVector,n0::AbstractVector,T0::AbstractVector)

    mDa = m0 / Da    # for normalization
    cT = (mDa[isp] * mDa[iFv])^0.5 / (mDa[isp] * T0[iFv] + mDa[iFv] * T0[isp])^1.5
    ## lnA
    # lnAab = lnA(isp, m0, n20 * n0[iFv], Tk * T0[iFv])
    lnAab = lnA()
    return 4.41720911682e2 * cT * (Zq[isp]* Zq[iFv])^2 * n0[iFv] * lnAab
end  # νT
# println("t=",1/νT)

# # When `nMod_uotput = 1` and `j ∈ ℓ:2:∞`
# function DMhj_fDM(uh::AbstractVector{T},j::Int64) where{T}

#     if j == 0
#         return 0 * uh
#     elseif j == 2
#         return 2/3 * uh.^2
#     elseif j == 4
#         return 4/3 * uh.^2 + 4/15 * uh.^4
#     elseif j == 6
#         return 2 * uh.^2 + 4/5 * uh.^4 + 8/105 * uh.^6
#     else
#         N = j / 2 |> Int    # `ℓ == 0`
#         uh2 = uh.^2
#         DMh = 0uh
#         for k in 1:N
#             ck = 2^k * binomial(N, k) / prod(3:2:2k+1)
#             DMh += ck .* uh2 .^ k
#         end
#         return DMh
#     end
# end

# function DMhj_fDM!(DMhcj::AbstractArray{T,N},uh::AbstractVector{T},njMs::Int64) where{T,N}

#     for kj in 1:njMs
#         j = 2(kj - 1)
#         DMhcj[:,kj] = DMhj_fDM(uh,j)
#     end
# end

# # When `nMod_uotput = 2` of the single spice, which means `ma=mb`, `Za = Zb`
# function Mhj_fM_nMod(na,nb,Ta,Tb,j::Int64)

#     nab = na + nb
#     nha = na ./ nab
#     nhb = nb ./ nab
#     Tab = (na .* Ta + nb .* Tb) ./ nab
#     Tha = Ta ./ Tab
#     Thb = Tb ./ Tab
#     # Mhcj = nha .* Tha .^(j/2) + nhb .* Thb .^(j/2)
#     return nha .* Tha .^(j/2) + nhb .* Thb .^(j/2)
# end

# function Mhj_fM_nMod!(Mhcj,na,nb,Ta,Tb,njMs::Int64)

#     nab = na + nb
#     nha = na ./ nab
#     nhb = nb ./ nab
#     Tab = (na .* Ta + nb .* Tb) ./ nab
#     Tha = Ta ./ Tab
#     Thb = Tb ./ Tab
#     for kj in 1:njMs
#         # j = 2(kj - 1)
#         Mhcj[:,kj] = nha .* Tha .^(kj - 1) + nhb .* Thb .^(kj - 1)
#     end
# end

# # for different spices

# function Mhsj_fDM(nah,nbh,uah::AbstractVector{T},ubh::AbstractVector{T},
#     Tah::AbstractVector{T},Tbh::AbstractVector{T},j::Int64) where{T}

#     if j == 0
#         Mh = ones(T,length(Tah))
#     elseif j == 2
#         Mh = nah .* Tah .* (1 .+ 2/3 * uah.^2 ./ Tah)
#         Mh += nbh .* Tbh .* (1 .+ 2/3 * ubh.^2 ./ Tbh)
#     elseif j == 4
#         uTah = uah.^2 ./ Tah 
#         Mh = nah .* Tah.^2 .* (1 .+ 4/3 * uTah .+ 4/15 * uTah.^2)
#         uTah = ubh.^2 ./ Tbh 
#         Mh += nbh .* Tbh.^2 .* (1 .+ 4/3 * uTah .+ 4/15 * uTah.^2)
#     elseif j == 6
#         uTah = uah.^2 ./ Tah 
#         Mh = nah .* Tah.^3 .* (1 .+ 2 * uTah .+ 4/5 * uTah.^2 .+ 8/105 * uTah.^3)
#         uTah = ubh.^2 ./ Tbh 
#         Mh += nbh .* Tbh.^3 .* (1 .+ 2 * uTah .+ 4/5 * uTah.^2 .+ 8/105 * uTah.^3)
#     else
#         N = j / 2 |> Int    # `ℓ == 0`
#         uTah = uah.^2 ./ Tah 
#         Mh = ones(T,length(Tah))
#         for k in 1:N
#             ck = 2^k * binomial(N, k) / prod(3:2:2k+1)
#             Mh += ck .* uTah .^ k
#         end
#         Mh .*= nah .* Tah .^N
        
#         uTah = ubh.^2 ./ Tbh 
#         Mbh = ones(T,length(Tah))
#         for k in 1:N
#             ck = 2^k * binomial(N, k) / prod(3:2:2k+1)
#             Mbh += ck .* uTah .^ k
#         end
#         Mh += nbh .* Tbh .^N .* Mbh
#     end
#     return Mh
# end

# function Mhsj_fDM!(Mhcsj,nah,nbh,uah,ubh,Tah,Tbh,njMs::Int64)

#     for kj in 1:njMs
#         j = 2(kj - 1)
#         Mhcsj[:,kj] = Mhsj_fDM(nah,nbh,uah,ubh,Tah,Tbh,j) 
#     end
# end


# function (Mhcsj,ma,nat,uat,Kat,vatht;ns::Int64=2,njMs::Int64=2)

#     # Mhcsj = zeros(Nt,njMs)
#     nast = sum(nat)
#     naht = nat / nast
#     ρast = sum(ma .* nat)
#     mast = ρast / nast
#     uast = sum(ma .* nat .* uat) / ρast
#     Kast = sum(Kat)

#     vthst = (2 * ((2Kast - ρast .* uast.^2) / 3nast) / mast).^0.5
#     uaht = uat ./ vthst
#     Taht = (vatht ./ vthst).^2
    
#     Mhsj_fDM!(Mhcsj,naht,uaht,Taht,njMs)
# end




# function Mhcsj_fDM(na::AbstractVector{T},ua::AbstractVector{T},vth::AbstractVector{T},j::Int64,Mhcj::AbstractVector{T}) where{T}

#     nhas = na / sum(na)
#     # uas = sum(nhas .* ua)
#     Kρs32 = sum(nhas .* (vth.^2 + 1.5 * ua.^2))
#     # vsth = (Kρs32 - 2/3 * uas^2) .^ 0.5
#     vsth = (Kρs32 - 2/3 * (sum(nhas .* ua))^2) .^ 0.5
#     return sum(nhas .* (vth / vsth) .^j .* Mhcj)
# end

# # M̂a[j], 
# function Mhcsj_fDM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},ua::AbstractVector{T},
#     vth::AbstractVector{T},njMs::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

#     nhas = na / sum(na)
#     # uas = sum(nhas .* ua)
#     Kρs32 = sum(nhas .* (vth.^2 + 1.5 * ua.^2))
#     # vsth = (Kρs32 - 2/3 * uas^2) .^ 0.5
#     vsth = (Kρs32 - 2/3 * (sum(nhas .* ua))^2) .^ 0.5
#     kj = 1
#     Mhcsj[kj] = sum(nha .* Mhcj[kj,:])
#     for kj in 2:njMs
#         # j = 2(kj - 1)
#         Mhcsj[kj] = sum(nhas .* (vth / vsth) .^(2(kj - 1)) .* Mhcj[kj,:])
#     end
# end

# # M̂a[t],
# function Mhcsj_fDM!(Mhcsj::AbstractVector{T},na::AbstractVector{T},ua::AbstractArray{T,N},
#     vth::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

#     nhas = na / sum(na)
#     for k in 1:Nt
#         Kρs32 = sum(nhas .* (vth[k,:].^2 + 1.5 * ua[k,:].^2))
#         vsth = (Kρs32 - 2/3 * (sum(nhas .* ua[k,:]))^2) .^ 0.5
#         Mhcsj[k] = sum(nhas .* (vth[k,:] / vsth) .^j .* Mhcj[k,:])
#     end
# end

# # M̂a[t,j],
# function Mhcsj_fDM!(Mhcsjt::AbstractArray{T,N},na::AbstractVector{T},ua::AbstractArray{T,N},
#     vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcjt::AbstractArray{T,N3}) where{T,N,N3}

#     for k in 1:Nt
#         a = Mhcsjt[k,:]
#         Mhcsj_fDM!(a,na,vth[k,:],ua[k,:],njMs,Mhcjt[k,:,:])
#         Mhcsjt[k,:] = a
#     end
# end

# # M̂a[t], na[t,:]
# function Mhcsj_fDM!(Mhcsj::AbstractVector{T},na::AbstractArray{T,N},ua::AbstractArray{T,N},
#     vth::AbstractArray{T,N},Nt::Int64,j::Int64,Mhcj::AbstractArray{T,N}) where{T,N}

#     for k in 1:Nt
#         nhas = na[k,:] ./ sum(na[k,:])
#         Kρs32 = sum(nhas .* (vth[k,:].^2 + 1.5 * ua[k,:].^2))
#         vsth = (Kρs32 - 2/3 * (sum(nhas .* ua[k,:]))^2) .^ 0.5
#         Mhcsj[k] = sum(nhas .* (vth[k,:] / vsth) .^j .* Mhcj[k,:])
#     end
# end

# # M̂a[t,j], na[t,:]
# function Mhcsj_fDM!(Mhcsjt::AbstractArray{T,N},na::AbstractArray{T,N},ua::AbstractArray{T,N},
#     vth::AbstractArray{T,N},Nt::Int64,njMs::Int64,Mhcjt::AbstractArray{T,N3}) where{T,N,N3}

#     for k in 1:Nt
#         a = Mhcsjt[k,:]
#         Mhcsj_fDM!(a,na[k,:],vth[k,:],ua[k,:],njMs,Mhcjt[k,:,:])
#         Mhcsjt[k,:] = a
#     end
# end