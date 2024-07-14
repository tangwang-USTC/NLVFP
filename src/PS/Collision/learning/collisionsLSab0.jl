
"""
  Fokker-Planck collision
    δf̂/δt = CF * F̂(𝓿̂,μ) .* f̂(v̂,μ) + CH * ∇𝓿̂ Ĥ(𝓿̂) .* ∇v̂ f̂(v̂,μ) + CG * ∇𝓿̂∇𝓿̂ Ĝ(𝓿̂) .* ∇v̂∇v̂ f̂(v̂,μ)

          = ∑ᵢ[(SfLᵢ * Mun) .* (SFLᵢ * Mun)] * Mμ

    where
     CF = mM
     CH = (1 - mM) * vbath
     CG = 0.5 * vbath^2
     SF = Mvn * XLm * Mun , X = F, H ,G, X = X(v̂)

  Dierckx.jl: Spline1D
              derivative
              extropolate
  DataInterpolations.jl: QuadraticInterpolation
  SmoothingSpline.jl
    spl = fit(SmoothingSpline,v,dG[:,iu],1e-3)
    dG[:,iu] = predict(spl)

  Extrapolating for f(v̂ .≪ 1)

    fvLc = f(v) = cf * ...
      cf = (na./vth.^3 / π^(3/2))

"""


"""

  Inputs:
    is_v_const: (=true, default), where keep the value of `v` to be constant (or else `v̂`)
    vhk,vhe: = (v̂, v̂[nvlevele]) if `is_v_const = false`, or else
             = (v/vthk1, ve/vthk1)
    Rvthk1 = `vthk/vthk1` for `is_v_const = false`
    ma:
    na = na / n20
    vth = vth / Mms
    fvL0 = f̂(v̂,L), the normalized distribution function by cf,
              without cf = na / π^1.5 / vₜₕ³ due to fvu(v̂,μ) = fvL0(v̂,ℓ) * Mμ
    HvL = Ĥ(𝓋̂,L) , without cF due to fvL0 without cf
    GvL = Ĝ(𝓋̂,L) , without cF due to fvL0 without cf

  Outputs:
    δtf,fvL0, err_dtnIK, nIKTh = dtfvLSplineab(δtf,fvL0,vhk,vhe,Rvhk1,nvG,nc0,nck,ocp,
          nvlevele0,nvlevel0,mu,Mμ,Mun,Mun1,Mun2,CΓ,εᵣ,ma,Zq,na,vth,nai,uai,vthi,LM,ns,nMod;
          is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
          autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
          p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
          rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_full_fvL=is_full_fvL,
          is_δtfvLaa=is_δtfvLaa,is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0,
          is_check_conservation_dtM=is_check_conservation_dtM,is_update_nuTi=is_update_nuTi,
            is_LM_const=is_LM_const,is_v_const=is_v_const)
    δtf,fvL0 = dtfvLSplineab(δtf,fvL0,vhk,nvG,nc0,nck,ocp,nvlevele0,nvlevel0,
            mu,Mμ,Mun,Mun1,Mun2,CΓ,εᵣ,ma,Zq,na,vth,nai,uai,vthi,LM,ns,nMod;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
            rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_full_fvL=is_full_fvL,
            is_δtfvLaa=is_δtfvLaa,is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0,
            is_LM_const=is_LM_const,is_v_const=is_v_const)

"""

# 3.5D, [nMod,LM1,ns], `is_v_const = false`

function dtfvLSplineab(fvL0k::AbstractArray{T,N},
    vhk::AbstractArray{T,2},vhe::AbstractVector{StepRangeLen},
    nvG::Int64,nc0::Int64,nck::Int64,ocp::Int64,nvlevele0::Vector{Int},
    nvlevel0::Vector{Int},mu::AbstractArray{T,N2},Mμ::AbstractArray{T,N2},
    Mun::AbstractArray{T,N2},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int},na::AbstractVector{T},
    vth::AbstractVector{T},nai::AbstractVector{TA},uai::AbstractVector{TA},
    vthi::AbstractVector{TA},LM::Vector{Int},LM1::Int64,ns::Int64,nMod::Vector{Int};
    Rvth::AbstractVector{T}=ones(T,ns),is_nai_const::Bool=true,NL_solve::Symbol=:NLsolve,
    is_normal::Bool=true,restartfit::Vector{Int}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_full_fvL::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,p_noise_rel::T=0e-3,p_noise_abs::T=0e-15,
    is_δtfvLaa::Int=1,is_normδtf::Bool=true,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_update_nuTi::Bool=true,
    is_LM_const::Bool=true,is_v_const::Bool=true) where{T,TA,N2,N,NM1,NM2}
    
    # Normalization 
    if is_normδtf == false
        cf3 = na ./ vth.^3 / pi^1.5
        for isp in 1:ns
            fvL0k[:,:,isp] /= cf3[isp]
        end
    end

    # Updating the characteristic parameters of `f̂ₗᵐ(v̂)`: `nai`, `uai` and `vthi`
    if is_update_nuTi
        nModk = copy(nMod)
        naik, uaik, vthik = copy(nai), copy(uai), copy(vthi)
        if is_moments_out
            if prod(nModk) == 1
                nai, uai, vthi = submoment!(nai,uai,vthi,Mhcsd2l,ns)
            else
                nai, uai, vthi = submoment!(nai,uai,vthi,nMod,Mhcsd2l,ns;NL_solve=NL_solve,
                        optimizer=optimizer,factor=factor,autodiff=autodiff,
                        is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
                        p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,p_noise_rel=p_noise_rel,p_noise_abs=p_noise_abs)
            end
        else
            if prod(nModk) == 1
                nai, uai, vthi = submoment!(naik, uaik, vthik, fvL0k, vhe,ns)
            else
                nai, uai, vthi = submoment!(naik, uaik, vthik, nModk, fvL0k, vhe,ns;NL_solve=NL_solve,
                    optimizer=optimizer, factor=factor, autodiff=autodiff,
                    is_Jacobian=is_Jacobian, show_trace=show_trace, maxIterKing=maxIterKing,
                    p_tol=p_tol, f_tol=f_tol, g_tol=g_tol,
                    p_noise_rel=p_noise_rel, p_noise_abs=p_noise_abs)
            end
        end
        if is_v_const
        else
            dffffffff
        end
        
        # # Checking the quantities of the parameters `nai`, `uai` and `vthi`
        # if 1 == 1
        #     # Ihsum, vhthsum = zeros(ns), zeros(ns)
        #     # nuTsNorm!(Ihsum, vhthsum, nai, uai, vthi)
        #     # @show vhthsum .- 1
        # end

        # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi`
        LMk = 0LM
        if is_LM_const
            fvL0k = zeros(nvG, LM1, ns)
            LMk, fvL0k = fvLDMz(fvL0k, vhe, nvG, LMk, ns, naik, uaik, vthik, nModk; L_limit=LM1-1,
                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_full_fvL=is_full_fvL)
            LM1k = maximum(LMk) + 1
            if LM1k ≠ LM1
                ghjkk
            end
        else
            fvL0k = zeros(nvG, LM1 + 1, ns)    # `LM1 + 1` denotes an extra row is given which may be used.
            LMk, fvL0k = fvLDMz(fvL0k, vhe, nvG, LMk, ns, naik, uaik, vthik, nModk; L_limit=LM1,
                rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_full_fvL=is_full_fvL)
            LM1k = maximum(LMk) + 1
        end

        # Accepting the results according to the to the new parameters `nai`, `uai` and `vthi`
        # fvL0 = fvL0k
        nai, uai, vthi, nMod = naik, uaik, vthik, nModk
        if LM1k ≠ LM1
            LM = LMk
            LM1 = LM1k
            mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1 - 1)
        end
    end

    # Checking the conservation laws of the renormalized distribution function `fvL0k1`
    nIKTh = zeros(4, ns)
    nIKThs!(nIKTh, fvL0k[:, 1:2, :], vhe, ns; errnIKTh=errnIKTh, reltol_DMs=reltol_DMs)
    # printstyled(naik, uaik, vthik,color=:green)
    # println()
    
    δtf = zeros(nvG, LM1, ns)
    # if is_v_const 
    #     @show vhk[end,:]
    # else
    #     @show vhk[end,:] .* vth
    # end

    # # Updating the FP collision terms according to the `FPS` operators.
    dtfvLaa = zeros(T,nvG,LM1,ns)
    ddfvL = zeros(T,nvG,LM1,ns)
    dfvL = zeros(T,nvG,LM1,ns)
    fvL = zeros(T,nck,LM1,ns)
    FvL = zeros(T,nck,LM1,ns)
    # Verifying the mass, momentum and total energy conservation laws during the self-collisions.
    if is_δtfvLaa === -1
        δtf,ddfvL,dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,ncF = dtfvLSplineaa(δtf,
                ddfvL,dfvL,fvL,fvL0k,FvL,vhk,nvG,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,CΓ,εᵣ,ma,Zq,na,vth,nai,uai,vthi,LM,LM1,ns,nMod;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0)
        # Inverse-normalization 
        if is_normδtf == false
            if is_check_conservation_dtM
                dtnIKsc = zeros(3,ns)
                nIKsc!(dtnIKsc,δtf[:,1:2,:],vhe,ma,vth,ns;errnIKc=errnIKc)
                dtnIK = norm(dtnIKsc)
                if dtnIK > epsTe6
                    @warn("Caa: The mass, momentum or the total energy conservation laws doesn't be satisfied during the self-collisions processes!")
                end
            else
                dtnIK = 0.0
            end
            for isp in 1:ns
                fvL0k[:,:,isp] *= cf3[isp]
            end
        else
            if is_check_conservation_dtM
                dtn̂aEaa, dtIaEaa, dtKaEaa = nIKs(δtf[:,1:2,:],nhe,ma,na,vth,ns)
                dtnIK = norm([dtn̂aEaa, dtIaEaa, dtKaEaa])
                # @show fmtf2.([dtn̂aEaa; dtIaEaa; dtKaEaa])
                if dtnIK > epsTe6
                    @warn("Caa: The mass, momentum or the total energy conservation laws doesn't be satisfied during the self-collisions processes!")
                end
            else
                dtnIK = 0.0
            end
        end
        if is_update_nuTi
            return δtf, fvL0k, dtnIK, nIKTh, nai, uai, vthi, LM, LM1, nMod
        else
            return δtf, fvL0k, dtnIK, nIKTh
        end
    else
        if prod(nMod) == 1
        else
        end
        dtfvLaa,ddfvL,dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,ncF = dtfvLSplineaa(dtfvLaa,
                ddfvL,dfvL,fvL,fvL0k,FvL,vhk,nvG,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,CΓ,εᵣ,ma,Zq,na,vth,nai,uai,vthi,LM,LM1,ns,nMod;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0)
        nsp_vec = 1:ns
        nvlevele = nvlevel0[nvlevele0]
        for isp in nsp_vec
            nspF = nsp_vec[nsp_vec .≠ isp]
            iFv = nspF[1]
            mM = ma[isp] / ma[iFv]
            vabth = vth[isp] / vth[iFv]
            va0 = vhk[nvlevele,isp] * vabth     # 𝓋̂ = v̂a * vabth
            Zqab = Zq[isp] * Zq[iFv]
            # when `is_normδtf = 0`, `cf3[isp] = na[isp]/vth[isp] / π^(3/2)` is not included.
            lnAg = lnAgamma(ma[isp],Zqab,na[isp],na[iFv],vth[isp],vth[iFv],εᵣ;is_normδtf=is_normδtf)
            #####
            HvL,dHvL = zeros(T,nc0,LM1),zeros(T,nc0,LM1)
            GvL,dGvL = zeros(T,nc0,LM1),zeros(T,nc0,LM1)
            ddGvL = zeros(T,nc0,LM1)
            if ncF[isp] ≥ 2
                dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                            FvL[:,:,isp],vhk[:,isp]*vabth,nvlevel0,nc0,nck,ocp,LM[iFv], 
                            FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
            else
                dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
                            FvL[:,:,isp],vhk[:,isp]*vabth,nvlevel0,nc0,nck,ocp,LM[iFv])
            end
            if nMod[isp] == 1 && nMod[iFv] == 1
                δtf[:,:,isp] = dtfvLSplineab(ddfvL[:,:,isp],dfvL[:,:,isp],fvL0k[:,:,isp],
                        FvL[nvlevele,:,isp],dHvL[nvlevele0,:],HvL[nvlevele0,:],ddGvL[nvlevele0,:],
                        dGvL[nvlevele0,:],GvL[nvlevele0,:],vhk[nvlevele,isp],va0,mu,Mμ,Mun,Mun1,Mun2,
                        mM,vabth,uai[isp][1]/vthi[isp][1],uai[iFv][1]/vthi[iFv][1],LM1;is_boundaryv0=is_boundaryv0)
            else
                δtf[:,:,isp] = dtfvLSplineab(ddfvL[:,:,isp],dfvL[:,:,isp],fvL0k[:,:,isp],
                        FvL[nvlevele,:,isp],dHvL[nvlevele0,:],HvL[nvlevele0,:],ddGvL[nvlevele0,:],
                        dGvL[nvlevele0,:],GvL[nvlevele0,:],vhk[nvlevele,isp],va0,
                        mu,Mμ,Mun,Mun1,Mun2,mM,vabth,nai[isp],nai[iFv],uai[isp]./vthi[isp],uai[iFv]./vthi[iFv],
                        vthi[isp],vthi[iFv],LM1,nMod[isp],nMod[iFv];is_boundaryv0=is_boundaryv0)
            end
            if is_boundaryv0 == false
                for L1 in 1:LM1
                    if L1 == 1
                        δtf[1,L1,isp] = 2δtf[2,L1,isp] - δtf[3,L1,isp]
                    else
                        δtf[1,L1,isp] = 0.0
                    end
                end
            end
            δtf[:,:,isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
        end

        if is_normδtf == false

            # Verifying the mass, momentum and total energy conservation laws of `δtfab`.

            if is_check_conservation_dtM
                dtnIKsc = zeros(3,ns)
                nIKsc!(dtnIKsc,dtfvLaa[:,1:2,:],vhe,ma,vth,ns;errnIKc=errnIKc)
                if norm(dtnIKsc[1,:]) ≥ epsTe6
                    @warn("δₜn̂ab: The mass conservation laws doesn't be satisfied during the self-collisions processes!.",dtnIKsc[1,:])
                end
                if abs(sum(dtnIKsc[2,:])) > epsTe6
                    RDIab = abs(dtnIKsc[2,1] - dtnIKsc[2,2])
                    if RDIab ≠ 0.0
                        err_dtIab = sum(dtnIKsc[2,:]) / RDIab
                    else
                        err_dtIab = 0.0
                    end
                    if err_dtIab > epsTe6
                        @warn("Cab: The momentum conservation laws doesn't be satisfied during the self-collisions processes! Refining by increasing `nnv`.",err_dtIab)
                    end
                else
                    err_dtIab = 0.0
                end
                if abs(sum(dtnIKsc[3,:])) > epsTe6
                    RDKab = abs(dtnIKsc[3,1] - dtnIKsc[3,2])
                    if RDKab ≠ 0.0
                        err_dtKab = sum(dtKaEab) / RDKab
                    else
                        err_dtKab = 0.0
                    end
                    if err_dtKab > epsTe6
                        @warn("Cab: The the total energy conservation laws doesn't be satisfied during the self-collisions processes! Refining by increasing `nnv`.",err_dtKab)
                    end
                else
                    err_dtKab = 0.0
                end
                err_dtnIK = fmtf4(norm([dtnIKsc[1,:]; err_dtIab; err_dtKab]))
            end
            if is_δtfvLaa === 1
                if is_check_conservation_dtM
                    dtnIKsc = zeros(3,ns)
                    nIKsc!(dtnIKsc,dtfvLaa[:,1:2,:] + δtf[:,1:2,:],vhe,ma,vth,ns;errnIKc=errnIKc)
                    if norm(dtnIKsc[1,:])  ≥ epsTe6
                        @warn("δₜn̂a: The mass conservation laws doesn't be satisfied during the collisions processes!.",dtnIKsc[1,:])
                    end

                    if abs(sum(dtnIKsc[2,:])) > epsTe6
                        RDIab = abs(dtnIKsc[2,1] - dtnIKsc[2,2])
                        if RDIab ≠ 0.0
                            err_dtI = sum(dtnIKsc[2,:]) / RDIab
                        else
                            err_dtI = 0.0
                        end
                        if err_dtI > epsTe6
                            @warn("Ca: The momentum conservation laws doesn't be satisfied during the collisions processes!.",err_dtI)
                        end
                    else
                        err_dtI = 0.0
                    end
                    # @show dtnIKsc
                
                    if abs(sum(dtnIKsc[3,:])) > epsTe6
                        RDKab = abs(dtnIKsc[3,1] - dtnIKsc[3,2])
                        if RDKab ≠ 0.0
                            err_dtK = sum(dtnIKsc[3,:]) / RDKab
                        else
                            err_dtK = 0.0
                        end
                        if err_dtK > epsTe6
                            @warn("Ca: The the total energy conservation laws doesn't be satisfied during the collisions processes!.",err_dtK)
                            @show dtnIKsc
                        end
                    else
                        err_dtK = 0.0
                    end
                    err_dtnIK = fmtf4(norm([dtnIKsc[1,:]; err_dtI; err_dtK]))
                else
                    err_dtnIK = 0.0
                end
            end

            # Inverse-normalization 
            for isp in 1:ns
                fvL0k[:,:,isp] *= cf3[isp]
            end
        else
            # Verifying the mass, momentum and total energy conservation laws of `δtfab`.
            if is_check_conservation_dtM
                dtn̂aEab, dtIaEab, dtKaEab = nIKs(δtf[:,1:2,:],nhe,ma,na,vth,ns)
                @show fmtf2.([dtn̂aEab; dtIaEab; dtKaEab])
                if norm(dtn̂aEab) ≥ epsTe6
                    @warn("δₜn̂ab: The mass conservation laws doesn't be satisfied during the self-collisions processes!.",dtn̂aEab)
                end
                if sum(dtIaEab) > epsTe6
                    RDIab = abs(dtIaEab[1] - dtIaEab[2])
                    if RDIab ≠ 0.0
                        err_dtIab = sum(dtIaEab) / RDIab
                    else
                        err_dtIab = 0.0
                    end
                    if err_dtIab > epsTe6
                        @warn("Cab: The momentum conservation laws doesn't be satisfied during the self-collisions processes! Refining by increasing `nnv`.",err_dtIab)
                    end
                else
                    err_dtIab = 0.0
                end
                if sum(dtIaEab) > epsTe6
                    RDKab = abs(dtKaEab[1] - dtKaEab[2])
                    if RDKab ≠ 0.0
                        err_dtKab = sum(dtKaEab) / RDKab
                    else
                        err_dtKab = 0.0
                    end
                    if err_dtKab > epsTe6
                        @warn("Cab: The the total energy conservation laws doesn't be satisfied during the self-collisions processes! Refining by increasing `nnv`.",err_dtKab)
                    end
                else
                    err_dtKab = 0.0
                end
                err_dtnIK = fmtf4(norm([dtn̂aEab; err_dtIab; err_dtKab]))
            else
            end
            if is_δtfvLaa === 1
                if is_check_conservation_dtM
                    dtn̂aEa, dtIaEa, dtKaEa = nIKs(dtfvLaa[:,1:2,:] + δtf[:,1:2,:],nhe,ma,na,vth,ns)
                    if norm(dtn̂aEa) ≥ epsTe6
                        @warn("δₜn̂a: The mass conservation laws doesn't be satisfied during the collisions processes!.",dtn̂aEa)
                    end
                    if sum(dtIaEa) > epsTe6
                        RDIab = abs(dtIaEa[1] - dtIaEa[2])
                        if RDIab ≠ 0.0
                            err_dtI = sum(dtIaEa) / RDIab
                        else
                            err_dtI = 0.0
                        end
                        if err_dtI > epsTe6
                            @warn("Ca: The momentum conservation laws doesn't be satisfied during the collisions processes!.",err_dtI)
                        end
                    else
                        err_dtI = 0.0
                    end
                    if sum(dtKaEa) > epsTe6
                        RDKab = abs(dtKaEa[1] - dtKaEa[2])
                        if RDKab ≠ 0.0
                            err_dtK = sum(dtKaEa) / RDKab
                        else
                            err_dtK = 0.0
                        end
                        if err_dtK > epsTe6
                            @warn("Ca: The the total energy conservation laws doesn't be satisfied during the collisions processes!.",err_dtK)
                            @show dtn̂aEa, dtIaEa, dtKaEa
                        end
                    else
                        err_dtK = 0.0
                    end
                    err_dtnIK = fmtf4(norm([dtn̂aEa; err_dtI; err_dtK]))
                else
                    err_dtnIK = 0.0
                end
            end
        end

        # outputs
        if is_δtfvLaa === 0
            if is_update_nuTi
                return δtf, fvL0k, err_dtnIK, nIKTh, nai, uai, vthi, LM, LM1, nMod
            else
                return δtf, fvL0k, err_dtnIK, nIKTh
            end
        else is_δtfvLaa === 1
            if is_update_nuTi
                return δtf + dtfvLaa, fvL0k, err_dtnIK, nIKTh, nai, uai, vthi, LM, LM1, nMod
            else
                return δtf + dtfvLaa, fvL0k, err_dtnIK, nIKTh
            end
        end
    end
end



"""

  Inputs:
    fvL0k: = fvLc0k / cf3, where `cf3 = na ./ vth.^3 / pi^1.5`

  Outputs:
     dtfvLSplineab!(δtfc,fvL0,vhk,nvG,nc0,nck,ocp,nvlevele0,nvlevel0,
            mu,Mμ,Mun,Mun1,Mun2,CΓ,εᵣ,ma,Zq,na,vth,nai,uai,vthi,LM,ns,nMod;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
            is_δtfvLaa=is_δtfvLaa,is_boundaryv0=is_boundaryv0)

"""

# # Condensed version of the essence: is_update_nuTi=false, is_check_conservation_dtM = false

# function dtfvLSplineab!(δtfc::AbstractArray{T,N},fvL0k::AbstractArray{T,N},vhk::AbstractArray{T,2},
#     nvG::Int64,nc0::Int64,nck::Int64,ocp::Int64,nvlevele0::Vector{Int},
#     nvlevel0::Vector{Int},mu::AbstractArray{T,N2},Mμ::AbstractArray{T,N2},
#     Mun::AbstractArray{T,N2},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
#     CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int},na::AbstractVector{T},
#     vth::AbstractVector{T},nai::AbstractVector{TA},uai::AbstractVector{TA},
#     vthi::AbstractVector{TA},LM::Vector{Int},LM1::Int64,ns::Int64,nMod::Vector{Int};
#     is_normal::Bool=true,restartfit::Vector{Int}=[0,0,100],maxIterTR::Int64=100,
#     autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
#     p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
#     is_δtfvLaa::Int=1,is_boundaryv0::Bool=false) where{T,TA,N2,N,NM1,NM2}

#     # # Updating the FP collision terms according to the `FPS` operators.
#     ddfvL = zeros(T,nvG,LM1,ns)
#     dfvL = zeros(T,nvG,LM1,ns)
#     fvL = zeros(T,nck,LM1,ns)
#     FvL = zeros(T,nck,LM1,ns)
#     if is_δtfvLaa === -1
#         δtfc[:,:,:],~,~,~,~,~,~,~,~ = dtfvLSplineaa(δtfc,
#                 ddfvL,dfvL,fvL,fvL0k,FvL,vhk,nvG,nc0,nck,ocp,nvlevele0,nvlevel0,
#                 mu,Mμ,Mun,Mun1,Mun2,CΓ,εᵣ,ma,Zq,na,vth,nai,uai,vthi,LM,LM1,ns,nMod;
#                 is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
#                 autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
#                 p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
#                 is_normδtf=true,is_boundaryv0=is_boundaryv0)
#         # return δtfc
#     else
#         dtfvLaa = zeros(T,nvG,LM1,ns)
#         dtfvLaa,ddfvL,dfvL,fvL,FvL,FvLa,vaa,nvlevel0a,ncF = dtfvLSplineaa(dtfvLaa,
#                 ddfvL,dfvL,fvL,fvL0k,FvL,vhk,nvG,nc0,nck,ocp,nvlevele0,nvlevel0,
#                 mu,Mμ,Mun,Mun1,Mun2,CΓ,εᵣ,ma,Zq,na,vth,nai,uai,vthi,LM,LM1,ns,nMod;
#                 is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
#                 autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
#                 p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
#                 is_normδtf=true,is_boundaryv0=is_boundaryv0)
#         nsp_vec = 1:ns
#         nvlevele = nvlevel0[nvlevele0]
#         for isp in nsp_vec
#             nspF = nsp_vec[nsp_vec .≠ isp]
#             iFv = nspF[1]
#             mM = ma[isp] / ma[iFv]
#             vabth = vth[isp] / vth[iFv]
#             va0 = vhk[nvlevele,isp] * vabth     # 𝓋̂ = v̂a * vabth
#             Zqab = Zq[isp] * Zq[iFv]
#             lnAg = lnAgamma(ma[isp],Zqab,na[isp],na[iFv],vth[isp],vth[iFv],εᵣ;is_normδtf=true)
#             HvL,dHvL = zeros(T,nc0,LM1),zeros(T,nc0,LM1)
#             GvL,dGvL = zeros(T,nc0,LM1),zeros(T,nc0,LM1)
#             ddGvL = zeros(T,nc0,LM1)
#             if ncF[isp] ≥ 2
#                 dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
#                             FvL[:,:,isp],vhk[:,isp]*vabth,nvlevel0,nc0,nck,ocp,LM[iFv], 
#                             FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
#             else
#                 dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,
#                             FvL[:,:,isp],vhk[:,isp]*vabth,nvlevel0,nc0,nck,ocp,LM[iFv])
#             end
#             if nMod[isp] == 1 && nMod[iFv] == 1
#                 δtfc[:,:,isp] = dtfvLSplineab(ddfvL[:,:,isp],dfvL[:,:,isp],fvL0k[:,:,isp],
#                         FvL[nvlevele,:,isp],dHvL[nvlevele0,:],HvL[nvlevele0,:],ddGvL[nvlevele0,:],
#                         dGvL[nvlevele0,:],GvL[nvlevele0,:],vhk[nvlevele,isp],va0,mu,Mμ,Mun,Mun1,Mun2,
#                         mM,vabth,uai[isp][1]/vthi[isp][1],uai[iFv][1]/vthi[iFv][1],LM1;is_boundaryv0=is_boundaryv0)
#             else
#                 δtfc[:,:,isp] = dtfvLSplineab(ddfvL[:,:,isp],dfvL[:,:,isp],fvL0k[:,:,isp],
#                         FvL[nvlevele,:,isp],dHvL[nvlevele0,:],HvL[nvlevele0,:],ddGvL[nvlevele0,:],
#                         dGvL[nvlevele0,:],GvL[nvlevele0,:],vhk[nvlevele,isp],va0,
#                         mu,Mμ,Mun,Mun1,Mun2,mM,vabth,nai[isp],nai[iFv],uai[isp]./vthi[isp],uai[iFv]./vthi[iFv],
#                         vthi[isp],vthi[iFv],LM1,nMod[isp],nMod[iFv];is_boundaryv0=is_boundaryv0)
#             end
#             if is_boundaryv0 == false
#                 for L1 in 1:LM1
#                     if L1 == 1
#                         δtfc[1,L1,isp] = 2δtfc[2,L1,isp] - δtfc[3,L1,isp]
#                     else
#                         δtfc[1,L1,isp] = 0.0
#                     end
#                 end
#             end
#             δtfc[:,:,isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
#         end
#         if is_δtfvLaa === 1
#             δtfc += dtfvLaa
#         end
#     end
# end

"""
  Inputs:
    fvL: = fvL[nvlevele,:,isp]
    dfvL: = dfvLe
    ddfvL: = ddfvLe
    FvL: = FvLe
    uai = uai[isp] / vthi[isp]
    ubi = uai[iFv] / vthi[iFv]

  Outputs:
    δtf = dtfvLSplineaa(ddfvL,dfvL,fvL0k,FvL,dHvL,HvL,ddGvL,dGvL,GvL,vG0,va0,mu,Mμ,Mun,
                 Mun1,Mun2,mM,vabth,nai,nbi,uai,ubi,vathi,vbthi,LM1,nModa,nModb;is_boundaryv0=is_boundaryv0)
"""

# 2D, [nMod,LM1] , is_inner == 0.0, `4π` is not included in the following code but in `3D`.
function dtfvLSplineab(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},
    FvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},HvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    vG0::AbstractVector{T},va0::AbstractVector{T},mu::AbstractArray{T,N},Mμ::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    mM::T,vabth::T,nai::AbstractVector{T},nbi::AbstractVector{T},
    uai::AbstractVector{T},ubi::AbstractVector{T},vathi::AbstractVector{T},
    vbthi::AbstractVector{T},LM1::Int64,nModa::Int64,nModb::Int64;is_boundaryv0::Bool=false) where{T,N,NM1,NM2}

    CF = mM
    CH = (1.0 - mM) / vabth
    CG = 0.5 / vabth^2
    cotmu = mu ./ (1 .-mu.^2).^0.5
    # # #################### S = Mvn * XLm * Mun , X = F, H ,G, X = X(vG0)
    fvuP1 = fvL[:,2:end] * Mun1 ./ vG0   # owing to `Mun1[:,L=0] .≡ 0`
    ############ 1, Sf1 = CF * f * F
    Sf = CF * (fvL * Mun) .* (FvL * Mun) 
    Sf1 = copy(Sf[1:2,:])     # = Sf1[1,:]
    ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
    # ############ 2,  Sf2
    # ############ 3,  Sf3
    if CH ≠ 0.0
        GG = (dfvL * Mun) .* (dHvL * Mun)
        Sf1[1,:] += CH * GG[1,:]     # = Sf1[1,:] + Sf2[1,:]
        GG += fvuP1 .* (HvL[:,2:end] * Mun1 ./ va0)
        Sf += CH * GG
    end
    # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`

    ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    dX01 = GvL[:,2:end] ./ va0 - dGvL[:,2:end]
    GG = dX01 * Mun1
    dX01 = fvL[:,2:end] ./ vG0 - dfvL[:,2:end]
    GG .*= 2(dX01 * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = (dGvL * Mun) + (GvL[:,2:end] * Mun1 ./ va0) .* cotmu
    GG7 = copy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG += 2GG7
    else
        GG6 = dX01 + (GvL[:,3:end] * Mun2 ./ va0)
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvL[:,3:end] * Mun2 ./ vG0))
        GG += (GG7 + GG6)
    end
    GG ./= (vG0 .* va0)
    ############ 4, Sf5
    GG7 = (ddfvL * Mun) .* (ddGvL * Mun)    # GG5
    Sf1[1,:] += CG * GG7[1,:]                  # = Sf1[1,:] + Sf2[1,:] + Sf5[1,:]
    GG += GG7
    Sf += CG * GG
    if is_boundaryv0 && vG0[1] == 0.0
        if nModa == 1 && nModb == 1
            if mM == 1.0
                Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CG,nai[1],nbi[1],
                                uai[1],ubi[1],vathi[1],vbthi[1],LM1)
            else
                Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CH,CG,nai[1],nbi[1],
                                uai[1],ubi[1],vathi[1],vbthi[1],LM1)
            end
        else
            if mM == 1.0
                Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CG,nai,nbi,
                                uai,ubi,vathi,vbthi,LM1,nModa,nModb)
            else
                Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CH,CG,nai,nbi,
                                uai,ubi,vathi,vbthi,LM1,nModa,nModb)
            end
        end
        Sf[1,:] = Sf1[1,:]
    end
    return Sf * Mμ
end

"""
  Inputs:
    fvL0: = fvL[nvlevel0,:,isp]
    dfvL: = dfvL[nvlevel0,:,isp]
    ddfvL: = ddfvL[nvlevel0,:,isp]
    FvL: = FvL[nvlevel0,:,isp]
    uai = uai[isp] / vthi[isp]
    ubi = uai[iFv] / vthi[iFv]

  Outputs:
    δtf = dtfvLSplineab(ddfvL,dfvL,fvL0,FvL,dHvL,HvL,ddGvL,dGvL,GvL,vG0,va0,mu,Mμ,
                    Mun,Mun1,Mun2,mM,vabth,uai,ubi,LM1;is_boundaryv0=is_boundaryv0)
"""

# 2D, nc0 , is_inner == 0.0, `4π` is not included in the following code but in `3D`.
# nai = 1, vthi = 1
function dtfvLSplineab(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvL::AbstractArray{T,N},
    FvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},HvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    vG0::AbstractVector{T},va0::AbstractVector{T},mu::AbstractArray{T,N},Mμ::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    mM::T,vabth::T,uai::T,ubi::T,LM1::Int64;is_boundaryv0::Bool=false) where{T,N,NM1,NM2}

    CF = mM
    CH = (1.0 - mM) / vabth
    CG = 0.5 / vabth^2
    cotmu = mu ./ (1 .-mu.^2).^0.5
    # # #################### S = Mvn * XLm * Mun , X = F, H ,G, X = X(vG0)
    fvuP1 = fvL[:,2:end] * Mun1 ./ vG0   # owing to `Mun1[:,L=0] .≡ 0`
    ############ 1, Sf1 = CF * f * F
    Sf = CF * (fvL * Mun) .* (FvL * Mun)
    Sf1 = copy(Sf[1:2,:])     # = Sf1[1,:]
    ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
    # ############ 2,  Sf2
    # ############ 3,  Sf3
    if CH ≠ 0.0
        GG = (dfvL * Mun) .* (dHvL * Mun)
        Sf1[1,:] += CH * GG[1,:]     # = Sf1[1,:] + Sf2[1,:]
        GG += fvuP1 .* (HvL[:,2:end] * Mun1 ./ va0)
        Sf += CH * GG
    end
    # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`

    ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    dX01 = GvL[:,2:end] ./ va0 - dGvL[:,2:end]
    GG = dX01 * Mun1
    dX01 = fvL[:,2:end] ./ vG0 - dfvL[:,2:end]
    GG .*= 2(dX01 * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = (dGvL * Mun) + (GvL[:,2:end] * Mun1 ./ va0) .* cotmu
    GG7 = copy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG += 2GG7
    else
        GG6 = dX01 + (GvL[:,3:end] * Mun2 ./ va0)
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvL[:,3:end] * Mun2 ./ vG0))
        GG += (GG7 + GG6)
    end
    GG ./= (vG0 .* va0)
    ############ 4, Sf5
    GG7 = (ddfvL * Mun) .* (ddGvL * Mun)    # GG5
    Sf1[1,:] += CG * GG7[1,:]                  # = Sf1[1,:] + Sf2[1,:] + Sf5[1,:]
    GG += GG7
    Sf += CG * GG
    if is_boundaryv0 && vG0[1] == 0.0
        if mM == 1.0
            Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CG,uai[1],ubi[1],LM1)
        else
            Sf1 = dtfvLDMabv0(Sf1,mu,Mun,Mun1,Mun2,CH,CG,uai[1],ubi[1],LM1)
        end
    end
    Sf[1,:] = Sf1[1,:]
    return Sf * Mμ
end
