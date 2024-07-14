
"""
 `n̂a`: is the normalized total number density of `spices[i]` which is unit (normalized by `na`).
      It's a function of the sub-component effective normalized number density `nai` as:
      `n̂a = 1 = ∑ₖ(n̂ₐₖ[k])`.
 `ûa`: is the normalized effective average velocity of `spices[i]` which is normalized by `vth`.
      It's a function of the sub-component effective average velocity `uai` when `nMod ≥ 2` as:
      `ûa = ûₐ = ∑ₖ(n̂ₐₖ[k] * ûₐₖ[k])`.
 `Êka`: is the normalized kinetic energy density of `spices[i] which is normalized by `na * Ta`.
      It's a function of `ûₐₖ` (but not `ûa` straightway when `nMod ≥ 2`):
      `Êka = ∑ₖ(n̂ₐₖ[k] * ûₐₖ[k]^2)`.
 `T̂a`: is the normalized effective (Experimental) thermal temperature of `spices[i]
      which is unit (normalized by `Ta`):
      `T̂a = 1 = ∑ₖ(n̂ₐₖ[k] * T̂ₐₖ[k])`.
 `K̂a`: is the normalized total energy density of `spices[i] which is normalized by `na * Ta`.
      `K̂a = 3/2 * T̂a + Êka`.

  n̂ₐ = ∑ᵢ(n̂ₐᵢ) = 1
  Îₐ = Iₐ / (mₐnₐvₜₕ) = ∑ᵢ(n̂ₐᵢûₐᵢ)
  K̂ₐ = Kₐ / (nₐTₐ) = 3/2 * ∑ᵢ(n̂ₐᵢv̂ₜₕᵢ²) + Êₖₐ
  Êₖₐ = Eₖₐ / (nₐTₐ) = ∑ᵢ(n̂ₐᵢûₐᵢ²)
"""

## The normalized moment by local moments
if 1 == 1
     n̂a0 = na ./ na            # .= 1
     ûa0 = ua ./ vth
     Êka0 = Eka ./ Ta
     Îa0 = deepcopy(ûa0)
     K̂a0 = Ka ./ (na .* Ta)
     nha = deepcopy(n̂a0)
     uha = deepcopy(ûa0)
     Iha = deepcopy(Îa0)
     Ehka = deepcopy(Êka0)
     Kha = deepcopy(K̂a0)
     Tha = 2/3 * (Kha - Ehka)
     vhth = Tha.^0.5
     # Initialization of the sub-moments
     
     println(".........................")
     nai0 = Vector{AbstractVector{datatype}}(undef,ns)     # `n̂a = naᵢ / na`
     uai0 = Vector{AbstractVector{datatype}}(undef,ns)     # `ûa = uaᵢ / vth`
     vthi0 = Vector{AbstractVector{datatype}}(undef,ns)    # `v̂th = vathᵢ / vth`
     # if is_nai_const
     #      for isp in 1:ns
     #           nai0[isp] = 0.1 * ones(datatype,nMod0[isp])
     #           uai0[isp] = zeros(datatype,nMod0[isp])
     #           vthi0[isp] = vhthInitial * ones(datatype,nMod0[isp])
     #      end
     # else
          for isp in 1:ns
               nai0[isp] = 0.1 * ones(datatype,NKmax0)
               uai0[isp] = zeros(datatype,NKmax0)
               vthi0[isp] = vhthInitial * ones(datatype,NKmax0)
          end
     # end
     LM = zeros(Int,ns)
     # LM = zeros(Int,nMod+1,ns)
     # LM = zeros(Int,nMod+1,LM1,ns)
     # vths = Vector((undef),ns)    # `v̂th = vathᵢ / vth`
end
@show -1,nai0

# nMod0 .= 1
                # The maximum number of sub-harmonic models of distribution function.
                # The maximum number of unknown is `3nMod`
is_uniform_nai = ones(Bool,ns) # (=false, default)
is_uniform_uai = ones(Bool,ns) # (=false, default) Whether there are the same temperature of
                # the sub-component distribution function, `f̂ₗₖ(v̂)`
is_uniform_Tai = ones(Bool,ns) # (=false, default) Whether there are the same temperature of
                # the sub-component distribution function, `f̂ₗₖ(v̂)`
if is_uniform_naiall == false
     is_uniform_nai .= false
end
is_uniform_uai .= false
if is_uniform_Taiall == false
     is_uniform_Tai .= false
end

# is_uniform_nai[1] = false
# is_uniform_nai[2] = false
# is_uniform_uai[1] = false
# is_uniform_uai[2] = false
# is_uniform_Tai[1] = false
# is_uniform_Tai[2] = false
if is_nai_const
     nai0,uai0,vthi0 = nuTsNorm(nai0,uai0,vthi0,nMod0,nha,uha,is_uniform_nai,
                              is_uniform_uai,is_uniform_Tai,ns;is_eliminate=is_eliminate)
     
     # nai0,uai0,vthi0,nMod0 = nuTsNormNK(nai0,uai0,vthi0,nMod0,nha,uha,is_uniform_nai,
     #                          is_uniform_uai,is_uniform_Tai,ns;is_eliminate=is_eliminate)
else
     nai0,uai0,vthi0,nMod0 = nuTsNormNK(nai0,uai0,vthi0,nMod0,nha,uha,is_uniform_nai,
                              is_uniform_uai,is_uniform_Tai,ns;is_eliminate=is_eliminate)
end
# vths0 = [vth[k] .* vthi0[k] for k in 1:ns]
@show 0,nai0, is_eliminate

####################################################### # Verifying the moments
n̂asum = [sum(nai0[k][1:nMod0[k]]) for k in 1:ns]
ûasum, v̂th0 = zeros(ns), zeros(ns)
nuTsNorm!(ûasum, v̂th0, nai0,uai0,vthi0)
# ûasum, v̂th0 = nuTsNorm(ûasum, v̂th0, nai0,uai0,vthi0)

Îasum = [sum(nai0[k][1:nMod0[k]] .* uai0[k][1:nMod0[k]]) for k in 1:ns]  # = `ûa`
Êkasum = ûasum.^2
K̂asum = [3/2 * sum(nai0[k][1:nMod0[k]] .* vthi0[k][1:nMod0[k]].^2) + sum(nai0[k][1:nMod0[k]] .* uai0[k][1:nMod0[k]].^2) for k in 1:ns]
T̂asum = [sum(nai0[k][1:nMod0[k]] .* vthi0[k][1:nMod0[k]].^2) + 2/3 * (sum(nai0[k][1:nMod0[k]] .* uai0[k][1:nMod0[k]].^2) - ûasum[k].^2) for k in 1:ns]
T̂asumK = 2/3 * [K̂asum[k] - Êkasum[k] for k in 1:ns]
K̂asumT = [3/2 * T̂asum[k] + Êkasum[k] for k in 1:ns]
KTEkh = [sum(nai0[k][1:nMod0[k]] .* vthi0[k][1:nMod0[k]].^2) + 2/3 * (sum(nai0[k][1:nMod0[k]] .* uai0[k][1:nMod0[k]].^2) - ûa0[k].^2) for k in 1:ns] .- 1

T̂a0 = deepcopy(T̂asum)
# v̂th0 = .√T̂a0

n̂a = na ./ na            # .= 1
ûa = ua ./ vth
Êka = Eka ./ Ta
K̂a = Ka ./ (na .* Ta)    # .= (3/2 .+ Êka)
T̂a = deepcopy(T̂a0)
v̂th = deepcopy(v̂th0)

nh = deepcopy(n̂a)
uh = deepcopy(ûa)
vhth = deepcopy(v̂th)
nMod = deepcopy(nMod0)
prod_nMod = prod(nMod)
################################
Kasum0 = (na .* Ta) .* K̂asum
Iasum0 = (ma .* na .* vth) .* Îasum
DIa0 = Iasum0 - Ia
RIa0 = Iasum0 ./ Ia .- 1
RKa0 = Kasum0 ./ Ka .- 1

IKk0 = IK_initial(ma, na, vth, nai0, uai0, vthi0, nMod0,ns)

@show K̂asumT - K̂a ;
@show K̂asum - K̂a ;
@show T̂asumK .- 1, T̂asum .- 1;
@show n̂asum .- 1;
@show KTEkh;
@show 21,nai0

