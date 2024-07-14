# Axis grids by adaptive Chebyshev notes of type 1: decide by the `orderVconst` moment that is computed by `Clenshaw-Curtis quadrature` .

vGmin = 0.0 |> datatype # (=0.0, default) which will affect the conservations owing to the lower boundary and
# #  the errors of values when v → 0 owing to CFL conditions, but now is overcome.
# # vGmax = 14.0       # When `û = 1.0, nMod = 3` and `j = ℓ + 6(nMod-1) + 2(3-1) = j = ℓ + 16`
# # vGmax = 11.7         # When `û = 1.0, nMod = 2,3`  → `nnv = 7 ~ 9`
# vGmax = 10.5       # When `û = 1.0, nMod = 1`
# # vGmax = 9.5        # When `û = 1e-2, nMod = 1`
# # The value mainly affects the accuracy of the higher-order moments `Ms` and `dtMs`;
# # Checking the value `vG0[end]^(j+2) * fLn[end] ≪ epsT` and `vG0[end]^(j+2) * δtfLn[end] ≪ epsT`
# # Maybe different `vGmax` for different spices when `vabth ≫ 1` or `vabth ≪ 1 ` is a best strategy.

nnv = zeros(Int64,ns)
nvG = zeros(Int64,ns)
nc0 = zeros(Int64,ns)
nck = zeros(Int64,ns)
ocp = zeros(Int64,ns)

nnv .= nnv0
nvG .= 2 .^nnv .+ 1
ocp .= ocp0
if gridv_type_initial == :uniform
    vhe = Vector{StepRangeLen}(undef,ns)
    ve = Vector{StepRangeLen}(undef,ns)          # ve = vGe * vth
else
    vhe = Vector{AbstractVector{datatype}}(undef,ns)
    ve = Vector{AbstractVector{datatype}}(undef,ns)          # ve = vGe * vth
end
vhk = Vector{AbstractVector{datatype}}(undef,ns)
nvlevele0 = Vector{Vector{Int64}}(undef,ns)
nvlevel0 = Vector{Vector{Int64}}(undef,ns)

vGdom = zeros(datatype,2,ns)
vGdom[1,:] .= vGmin

vHadapt1D!(vhe,vhk, vGdom, nvG, nc0, nck, ocp, 
    nvlevele0, nvlevel0, nai0, uai0, vthi0, nMod0, ns;
    eps_fup=eps_fup,eps_flow=eps_flow,
    maxiter_vGm=maxiter_vGm,vGm_limit=vGm_limit,
    abstol=abstol,reltol=reltol,
    vadaptlevels=vadaptlevels,gridv_type=gridv_type_initial,
    is_nvG_adapt=is_nvG_adapt,nvG_limit=nvG_limit)
for isp in 1:ns
    ve[isp] = vhe[isp] .* vth[isp]
end
# nvlevele = nvlevel0invGk(nvlevele,vGe,vGk,nvG,nck) # The position of the initial equally spaced points `vGe` among in `vGk`.
@show nvG, nc0, nck, vGdom[2,:],vadaptlevels

fnck(nnv0,ocp0) = (ocp0-1) * ((2^nnv0 + 1) -2) + ocp0
