
"""
  # Pre-mesh for velocity-space in (n,ℓ,m)-space based on spherical harmonics expansion
  when u/vth ~ 1, RErr_u ~ RErr_n_K

 Input
   û: the normalized group velocity
"""

"""
  nv, ℓM = nvLMnsp(ns,m0,na,ua,Ta,α,nv_limit0,L_limit0,R_dfLM,Rtol_nuT)

"""

# maximum of nᵥ and LM for all spice distribution coefficient functions, fvL.
function nvLMnsp(ns::Int64,û::AbstractVector;
    α::Real=0.0,nv_limit::Int64=69,L_limit::Int64=2,R_dfLM::Real=1e-1,Rtol_T::Real=1e-6,endptv=neither)

    nv = zeros(Int,ns)
    LM = zeros(Int,ns)
    for isp = 1:ns
        nv[isp], LM[isp] =  nvLM(û[isp];α=α,nv_limit=nv_limit,
             L_limit=L_limit,R_dfLM=R_dfLM,Rtol_T=Rtol_T,endptv=endptv)
    end
    return maximum(nv), max(1,maximum(LM))
end

# mesh grid of V for nᵥ and LM for single spice f
function nvLM(û::Real;α::Real=0.0,nv_limit::Int64=69,L_limit::Int64=2,
           R_dfLM::Real=1e-1,Rtol_T::Real=1e-6,endptv=neither)

    # parameters
    Ka = (1.5 + û^2)
    # dnv = 5          # for appropriating `nv` with the precision
    nv_vec = 10:2:nv_limit
    nnv = length(nv_vec)   # number of nv for best v
    nv = 0
    LM = 0
    Rerror = zeros(nnv)
    for iv =  1:nnv
        nv =  nv_vec[iv]
        nv1 = nv + 1     # i = 0:nv
        v, w1 = laguerre(nv1,α,endptv)
        fL0log = zeros(typeof(α),nv1,L_limit+1)
        LM,fl0log = flogInitial(fL0log,L_limit,û,v,R_dfLM)
        n̂a, ûa, K̂a = momentsGaussn(v,w1,exp.(fl0log))
        # error control
        if iv > 1.1
            num0 = 1    # =1 (Relative Error or else = 0 Absolut Error)
            if û > 0
                Δna = n̂a - 1
                R_dKa = (K̂a + ûa) - Ka
            else
                Δna = n̂a - 1
                R_dKa = K̂a - 1.5
            end
            Rerror[iv] = norm([Δna, R_dKa])
            if Rerror[iv] < Rtol_T
                nv_vec = nv_vec[1:iv]
                Rerror = Rerror[1:iv]
                println("nv=",nv, ",Relative Error of moments: Rel_dm =",Rerror[iv])
                break
            else
                if iv == nnv
                    println("Not convergence when nv = nvlimit0, Rel_dm =",fmtf4(Rerror[iv]))
                end
            end
        end
    end
    ivmin = findfirst(Rerror[2:end] .== minimum(Rerror[2:end]))
    xlabel = string("nG, Rtol_T=",Rtol_T)
    ylabel = "Relative errors of nuK in `log` coordinate"
    label = string("R_df=",R_dfLM)
    @show Rerror
    pp = plot(nv_vec,Rerror,yscale=:log10,label=label,xlabel=xlabel,ylabel=ylabel)
    display(pp)
    # @show DataFrame([nv_vec Rerror],:auto)
    return nv_vec[ivmin], LM
end
