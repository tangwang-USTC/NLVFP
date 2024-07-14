
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
    vhk,vhe: = (v/vthk1, ve/vthk1)
    ma:
    na = na / n20
    vth = vth / Mms
    fvL0 = f̂(v̂,L), the normalized distribution function by cf,
              without cf = na / π^1.5 / vₜₕ³ due to fvu(v̂,μ) = fvL0(v̂,ℓ) * Mμ
    HvL = Ĥ(𝓋̂,L) , without cF due to fvL0 without cf
    GvL = Ĝ(𝓋̂,L) , without cF due to fvL0 without cf

  Outputs:
    δtf,fvL0, err_dtnIK, DThk = dtfvLSplineab2(Mhcsd2lk,
            vhk,vhe,nvG,nc0,nck,ocp,nvlevele0,nvlevel0,mu,Mμ,Mun,Mun1,Mun2,
            CΓ,εᵣ,ma,Zq,spices,na,vth,nai,uai,vthi,LM,LM1,ns,nMod;
            is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
            rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
            is_δtfvLaa=is_δtfvLaa,is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0,
            is_check_conservation_dtM=is_check_conservation_dtM)
    δtf,fvL0, err_dtnIK, DThk = dtfvLSplineab2(nai,uai,vthi,LM,Mhcsd2lk,
            vhk,vhe,nvG,nc0,nck,ocp,nvlevele0,nvlevel0,mu,Mμ,Mun,Mun1,Mun2,
            CΓ,εᵣ,ma,Zq,spices,na,vth,LM1,ns,nMod;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,is_normal=is_normal,
            restartfit=restartfit,maxIterTR=maxIterTR,maxIterKing=maxIterKing,
            autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,,
            Nspan_optim_nuTi=Nspan_optim_nuTi
            rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full,
            is_δtfvLaa=is_δtfvLaa,is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0,
            is_check_conservation_dtM=is_check_conservation_dtM,is_LM_const=is_LM_const)

""" 

# 3D, [nMod,LM1,ns], fvL0k(nai, uai, vthi)
function dtfvLSplineab2(Mhcsd2l::Vector{Matrix{T}},
    vhk::Vector{AbstractVector{T}},vhe::Vector{StepRangeLen},
    nvG::Vector{Int64},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},mu::AbstractArray{T,N2},Mμ::AbstractArray{T,N2},
    Mun::AbstractArray{T,N2},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},vth::AbstractVector{T},
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},
    LM::Vector{Int64},LM1::Int64,ns::Int64,nMod::Vector{Int64};
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    is_δtfvLaa::Int=1,is_normδtf::Bool=false,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false) where{T,N2,NM1,NM2}
    
    fvL0k = Vector{Matrix{T}}(undef,ns)
    nsp_vec = 1:ns
    for isp in nsp_vec
        fvL0k[isp] = zeros(nvG[isp],LM1)
    end
    LMk, fvL0k = fvLDMz(fvL0k, vhe, nvG, LM, ns, nai, uai, vthi, nMod; 
        L_limit=LM1-1, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full)
    LM1k = maximum(LMk) + 1
    if LM1k ≠ LM1
        skdkkkfk
        LM1 = LM1k
        LM = LMk
        mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1 - 1)
    end

    # Checking the conservation laws of the renormalized distribution function `fvL0k1`
    nIKTh = zeros(4, ns)
    nIKThs!(nIKTh, fvL0k, vhe, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)

    # # Updating the FP collision terms according to the `FPS` operators.
    δtf = Vector{Matrix{T}}(undef,ns)     # δtfvLa
    ddfvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    FvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        δtf[isp] = zeros(T,nvG[isp],LM1)
        ddfvL[isp] = zeros(T,nvG[isp],LM1)
        dfvL[isp] = zeros(T,nvG[isp],LM1)
        FvL[isp] = zeros(T,nck[isp],LM1)
    end
    # Verifying the mass, momentum and total energy conservation laws during the self-collisions.
    if is_δtfvLaa === -1
        δtf,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag2(δtf,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,mu,Mμ,Mun,Mun1,Mun2,CΓ,
                εᵣ,ma,Zq,spices,na,vth,nai, uai, vthi,LM,LM1,ns,nMod;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f)
        # Inverse-normalization 
        if is_check_conservation_dtM
            dtnIKs = zeros(3,ns)
            dtnIKsc!(dtnIKs,δtf,vhe,ma,vth,ns;atol_nIK=atol_nIK)
            dtnIK = norm(dtnIKs)
            if dtnIK > epsTe6
                @warn("Caa: The mass, momentum or the total energy conservation laws doesn't be satisfied during the self-collisions processes!")
            end
        else
            dtnIK = 0.0
        end
        if is_normδtf == false
            cf3 = na ./ vth.^3 / pi^1.5
            for isp in nsp_vec
                fvL0k[isp] *= cf3[isp]
            end
        end
        # DThk = nIKTh[4, :]
        return δtf, dtnIKs[4,:], fvL0k,  dtnIK, nIKTh[4, :]
    else
        dtfvLaa = Vector{Matrix{T}}(undef,ns)
        for isp in nsp_vec
            dtfvLaa[isp] = zeros(nvG[isp],LM1)
        end
        dtfvLaa,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LM,LM1,
                nai,uai,vthi,nMod,
                CΓ,εᵣ,ma,Zq,spices,na,vth,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f)
        for isp in nsp_vec
            nvlevele0a = nvlevele0[isp]
            nvlevele = nvlevel0[isp][nvlevele0a]
            nspF = nsp_vec[nsp_vec .≠ isp]
            iFv = nspF[1]
            mM = ma[isp] / ma[iFv]
            vabth = vth[isp] / vth[iFv]
            va0 = vhk[isp][nvlevele] * vabth     # 𝓋̂ = v̂a * vabth
            
            HvL,dHvL = zeros(T,nc0[isp],LM1),zeros(T,nc0[isp],LM1)
            GvL,dGvL = zeros(T,nc0[isp],LM1),zeros(T,nc0[isp],LM1)
            ddGvL = zeros(T,nc0[isp],LM1)
            if nMod[isp] == 1 && nMod[iFv] == 1
                uh = uai[iFv][1]
                if ncF[isp] ≥ 2
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv],uh,  
                                FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
                else
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv],uh)
                end
                δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                        FvL[isp][nvlevele,:],dHvL[nvlevele0a,:],HvL[nvlevele0a,:],ddGvL[nvlevele0a,:],
                        dGvL[nvlevele0a,:],GvL[nvlevele0a,:],vhk[isp][nvlevele],va0,
                        mu,Mμ,Mun,Mun1,Mun2,LM1,uai[isp][1],uai[iFv][1],
                        mM,vabth;is_boundaryv0=is_boundaryv0)
            else
                if ncF[isp] ≥ 2
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv], 
                                FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
                else
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv])
                end
                δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                        FvL[isp][nvlevele,:],dHvL[nvlevele0a,:],HvL[nvlevele0a,:],ddGvL[nvlevele0a,:],
                        dGvL[nvlevele0a,:],GvL[nvlevele0a,:],vhk[isp][nvlevele],va0,
                        mu,Mμ,Mun,Mun1,Mun2,LM1,
                        nai[isp],nai[iFv],uai[isp]./vthi[isp],uai[iFv]./vthi[iFv],
                        vthi[isp],vthi[iFv],nMod[isp],nMod[iFv],
                        mM,vabth;is_boundaryv0=is_boundaryv0)
            end
            if is_boundaryv0 == false
                for L1 in 1:LM1
                    if L1 == 1
                        δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                    else
                        δtf[isp][1,L1] = 0.0
                    end
                end
            end

            if is_lnA_const
                Zqab = Zq[isp] * Zq[iFv]
                lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],na[isp],na[iFv],vth[isp],vth[iFv];is_normδtf=true)
            else
                lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                                [na[isp],na[iFv]],[vth[isp],vth[iFv]],1,2;is_normδtf=is_normδtf)
            end
            δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
        end
        # Verifying the mass, momentum and total energy conservation laws of `δtfab`.
        Ihk = nIKTh[2,:]
        Ihk = [sum(nai[k][1:nMod[k]] .* uai[k][1:nMod[k]]) for k in nsp_vec]  # = `ûa`

        dtnIKs = zeros(4,ns)
        if is_δtfvLaa === 1
            for isp in nsp_vec
                δtf[isp] += dtfvLaa[isp]
            end
        end
        if is_normδtf
            RdtnIKTs!(dtnIKs,δtf,vhe,Ihk,ma,na,vth,ns;atol_nIK=atol_nIK,
                    is_out_errdt=true,is_enforce_errdtnIKab=is_enforce_errdtnIKab)
        else
            RdtnIKTcs!(dtnIKs,vhe,δtf,Ihk,na,vth,ns;atol_nIK=atol_nIK)
        end
        if is_check_conservation_dtM 
            if norm(dtnIKs[1,:])  ≥ epsTe6
                @warn("δₜn̂a: The mass conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",dtnIKs[1,:])
            end

            if abs(sum(dtnIKs[2,:])) > epsTe6
                RDIab = abs(dtnIKs[2,1] - dtnIKs[2,2])
                if RDIab ≠ 0.0
                    err_dtI = sum(dtnIKs[2,:]) / RDIab
                else
                    err_dtI = 0.0
                end
                if err_dtI > epsTe6
                    @warn("δₜÎa: The momentum conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_dtI)
                end
            else
                err_dtI = 0.0
            end
        
            if abs(sum(dtnIKs[3,:])) > epsTe6
                RDKab = abs(dtnIKs[3,1] - dtnIKs[3,2])
                if RDKab ≠ 0.0
                    err_dtK = sum(dtnIKs[3,:]) / RDKab
                else
                    err_dtK = 0.0
                end
                if err_dtK > epsTe6
                    @warn("δₜnK̂a: The the total energy conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_dtK)
                    @show dtnIKs
                end
            else
                err_dtK = 0.0
            end
            err_dtnIK = fmtf4(norm([dtnIKs[1,:]; err_dtI; err_dtK]))
        else
            err_dtnIK = 0.0
        end

        # outputs
        # DThk = nIKTh[4, :]
        # w3k = dtnIKs[4,:]           # `w3k = Rdtvth = 𝒲 / 3`
        if is_normδtf == false
            cf3 = na ./ vth.^3 / pi^1.5
            # Inverse-normalization 
            for isp in nsp_vec
                fvL0k[isp] *= cf3[isp]
            end
        end
        return δtf, dtnIKs[4,:], fvL0k,  err_dtnIK, nIKTh[4, :]
    end
end
function dtfvLSplineab2(Mhcsd2l::Vector{Matrix{T}},
    vhk::Vector{AbstractVector{T}},vhe::Vector{AbstractVector{T}},
    nvG::Vector{Int64},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},mu::AbstractArray{T,N2},Mμ::AbstractArray{T,N2},
    Mun::AbstractArray{T,N2},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},LM::Vector{Int64},LM1::Int64,
    nai::Vector{AbstractVector{T}},uai::Vector{AbstractVector{T}},vthi::Vector{AbstractVector{T}},nMod::Vector{Int64},
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},vth::AbstractVector{T},ns::Int64;
    is_normal::Bool=true, restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    is_δtfvLaa::Int=1,is_normδtf::Bool=false,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_fit_f::Bool=false) where{T,N2,NM1,NM2}
    
    fvL0k = Vector{Matrix{T}}(undef,ns)
    nsp_vec = 1:ns
    for isp in nsp_vec
        fvL0k[isp] = zeros(nvG[isp],LM1)
    end
    LMk, fvL0k = fvLDMz(fvL0k, vhe, nvG, LM, ns, nai, uai, vthi, nMod; 
        L_limit=LM1-1, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full)
    LM1k = maximum(LMk) + 1
    if LM1k ≠ LM1
        skdkkkfk
        LM1 = LM1k
        LM = LMk
        mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1 - 1)
    end

    # Checking the conservation laws of the renormalized distribution function `fvL0k1`
    nIKTh = zeros(4, ns)
    nIKThs!(nIKTh, fvL0k, vhe, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)

    # # Updating the FP collision terms according to the `FPS` operators.
    δtf = Vector{Matrix{T}}(undef,ns)     # δtfvLa
    ddfvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    FvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        δtf[isp] = zeros(T,nvG[isp],LM1)
        ddfvL[isp] = zeros(T,nvG[isp],LM1)
        dfvL[isp] = zeros(T,nvG[isp],LM1)
        FvL[isp] = zeros(T,nck[isp],LM1)
    end
    # Verifying the mass, momentum and total energy conservation laws during the self-collisions.
    if is_δtfvLaa === -1
        δtf,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag2(δtf,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LM,LM1,
                nai,uai,vthi,nMod,
                CΓ,εᵣ,ma,Zq,spices,na,vth,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f)
        # Inverse-normalization 
        if is_check_conservation_dtM
            dtnIKs = zeros(3,ns)
            dtnIKsc!(dtnIKs,δtf,vhe,ma,vth,ns;atol_nIK=atol_nIK)
            dtnIK = norm(dtnIKs)
            if dtnIK > epsTe6
                @warn("Caa: The mass, momentum or the total energy conservation laws doesn't be satisfied during the self-collisions processes!")
            end
        else
            dtnIK = 0.0
        end
        if is_normδtf == false
            cf3 = na ./ vth.^3 / pi^1.5
            for isp in nsp_vec
                fvL0k[isp] *= cf3[isp]
            end
        end
        # DThk = nIKTh[4, :]
        return δtf, dtnIKs[4,:], fvL0k,  dtnIK, nIKTh[4, :]
    else
        dtfvLaa = Vector{Matrix{T}}(undef,ns)
        for isp in nsp_vec
            dtfvLaa[isp] = zeros(nvG[isp],LM1)
        end
        dtfvLaa,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LM,LM1,
                nai,uai,vthi,nMod,
                CΓ,εᵣ,ma,Zq,spices,na,vth,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f)
        for isp in nsp_vec
            nvlevele0a = nvlevele0[isp]
            nvlevele = nvlevel0[isp][nvlevele0a]
            nspF = nsp_vec[nsp_vec .≠ isp]
            iFv = nspF[1]
            mM = ma[isp] / ma[iFv]
            vabth = vth[isp] / vth[iFv]
            va0 = vhk[isp][nvlevele] * vabth     # 𝓋̂ = v̂a * vabth
            
            HvL,dHvL = zeros(T,nc0[isp],LM1),zeros(T,nc0[isp],LM1)
            GvL,dGvL = zeros(T,nc0[isp],LM1),zeros(T,nc0[isp],LM1)
            ddGvL = zeros(T,nc0[isp],LM1)
            if nMod[isp] == 1 && nMod[iFv] == 1
                uh = uai[iFv][1]
                if ncF[isp] ≥ 2
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv],uh,  
                                FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
                else
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv],uh)
                end
                δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                        FvL[isp][nvlevele,:],dHvL[nvlevele0a,:],HvL[nvlevele0a,:],ddGvL[nvlevele0a,:],
                        dGvL[nvlevele0a,:],GvL[nvlevele0a,:],vhk[isp][nvlevele],va0,
                        mu,Mμ,Mun,Mun1,Mun2,LM1,uai[isp][1],uai[iFv][1],
                        mM,vabth;is_boundaryv0=is_boundaryv0)
            else
                if ncF[isp] ≥ 2
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv], 
                                FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
                else
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv])
                end
                δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                        FvL[isp][nvlevele,:],dHvL[nvlevele0a,:],HvL[nvlevele0a,:],ddGvL[nvlevele0a,:],
                        dGvL[nvlevele0a,:],GvL[nvlevele0a,:],vhk[isp][nvlevele],va0,
                        mu,Mμ,Mun,Mun1,Mun2,LM1,
                        nai[isp],nai[iFv],uai[isp]./vthi[isp],uai[iFv]./vthi[iFv],
                        vthi[isp],vthi[iFv],nMod[isp],nMod[iFv],
                        mM,vabth;is_boundaryv0=is_boundaryv0)
            end
            if is_boundaryv0 == false
                for L1 in 1:LM1
                    if L1 == 1
                        δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                    else
                        δtf[isp][1,L1] = 0.0
                    end
                end
            end

            if is_lnA_const
                Zqab = Zq[isp] * Zq[iFv]
                lnAg = lnAgamma(εᵣ,ma[isp],Zqab,na[isp],[spices[isp],spices[iFv]],na[iFv],vth[isp],vth[iFv];is_normδtf=true)
            else
                lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                                [na[isp],na[iFv]],[vth[isp],vth[iFv]],1,2;is_normδtf=is_normδtf)
            end
            δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
        end
        # Verifying the mass, momentum and total energy conservation laws of `δtfab`.
        Ihk = nIKTh[2,:]
        Ihk = [sum(nai[k][1:nMod[k]] .* uai[k][1:nMod[k]]) for k in nsp_vec]  # = `ûa`

        dtnIKs = zeros(4,ns)
        if is_δtfvLaa === 1
            for isp in nsp_vec
                δtf[isp] += dtfvLaa[isp]
            end
        end
        if is_normδtf
            RdtnIKTs!(dtnIKs,δtf,vhe,Ihk,ma,na,vth,ns;atol_nIK=atol_nIK,
                    is_out_errdt=true,is_enforce_errdtnIKab=is_enforce_errdtnIKab)
        else
            RdtnIKTcs!(dtnIKs,vhe,δtf,Ihk,na,vth,ns;atol_nIK=atol_nIK)
        end
        if is_check_conservation_dtM 
            if norm(dtnIKs[1,:])  ≥ epsTe6
                @warn("δₜn̂a: The mass conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",dtnIKs[1,:])
            end

            if abs(sum(dtnIKs[2,:])) > epsTe6
                RDIab = abs(dtnIKs[2,1] - dtnIKs[2,2])
                if RDIab ≠ 0.0
                    err_dtI = sum(dtnIKs[2,:]) / RDIab
                else
                    err_dtI = 0.0
                end
                if err_dtI > epsTe6
                    @warn("δₜÎa: The momentum conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_dtI)
                end
            else
                err_dtI = 0.0
            end
        
            if abs(sum(dtnIKs[3,:])) > epsTe6
                RDKab = abs(dtnIKs[3,1] - dtnIKs[3,2])
                if RDKab ≠ 0.0
                    err_dtK = sum(dtnIKs[3,:]) / RDKab
                else
                    err_dtK = 0.0
                end
                if err_dtK > epsTe6
                    @warn("δₜnK̂a: The the total energy conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_dtK)
                    @show dtnIKs
                end
            else
                err_dtK = 0.0
            end
            err_dtnIK = fmtf4(norm([dtnIKs[1,:]; err_dtI; err_dtK]))
        else
            err_dtnIK = 0.0
        end

        # outputs
        # DThk = nIKTh[4, :]
        # w3k = dtnIKs[4,:]           # `w3k = Rdtvth = 𝒲 / 3`
        if is_normδtf == false
            cf3 = na ./ vth.^3 / pi^1.5
            # Inverse-normalization 
            for isp in nsp_vec
                fvL0k[isp] *= cf3[isp]
            end
        end
        return δtf, dtnIKs[4,:], fvL0k,  err_dtnIK, nIKTh[4, :]
    end
end

# 3.5D, [nMod,LM1,ns], fvL0k(Mhcsd2lk), `is_update_nuTi = true`
function dtfvLSplineab2(naik::Vector{AbstractVector{T}},uaik::Vector{AbstractVector{T}},
    vthik::Vector{AbstractVector{T}},nModk::Vector{Int64},Mhcsd2l::AbstractArray{T,N},
    vhk::Vector{AbstractVector{T}},vhe::Vector{StepRangeLen},
    nvG::Vector{Int64},nc0::Vector{Int64},nck::Vector{Int64},ocp::Vector{Int64},
    nvlevele0::Vector{Vector{Int64}},nvlevel0::Vector{Vector{Int64}},mu::AbstractArray{T,N2},Mμ::AbstractArray{T,N2},
    Mun::AbstractArray{T,N2},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},LM::Vector{Int64},LM1::Int64,
    CΓ::T,εᵣ::T,ma::AbstractVector{T},Zq::AbstractVector{Int64},spices::Vector{Symbol},
    na::AbstractVector{T},vthk::AbstractVector{T},ns::Int64;
    NL_solve::Symbol=:NLsolve,rtol_DnuTi::T=1e-7,is_normal::Bool=true, 
    restartfit::Vector{Int64}=[0,0,100],maxIterTR::Int64=100,maxIterKing::Int64=100,
    autodiff::Symbol=:central,factorMethod::Symbol=:QR,show_trace::Bool=false,
    p_tol::Float64=epsT,f_tol::Float64=epsT,g_tol::Float64=epsT,n10::Int64=1,dnvs::Int64=1,
    Nspan_optim_nuTi::T=1.1,
    rel_dfLM::Real=eps0,abs_dfLM::Real=eps0,is_LM1_full::Bool=false,
    optimizer=LeastSquaresOptim.Dogleg,factor=LeastSquaresOptim.QR(),
    is_Jacobian::Bool=true,is_δtfvLaa::Int=1,is_normδtf::Bool=false,is_boundaryv0::Bool=false,
    is_check_conservation_dtM::Bool=true,is_LM_const::Bool=false,is_fit_f::Bool=false) where{T,N,N2,NM1,NM2}
    
    edfgbnmcsddf
    # Calculate the re-normalized moments `ℳ̂ⱼ,ₗ⁰` for parameters `nai, uai, vthi`
    submoment!(naik, uaik, vthik,nModk,Mhcsd2l,ns;
            NL_solve=NL_solve,rtol_DnuTi=rtol_DnuTi,
            optimizer=optimizer,factor=factor,autodiff=autodiff,
            is_Jacobian=is_Jacobian,show_trace=show_trace,maxIterKing=maxIterKing,
            p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,
            Nspan_optim_nuTi=Nspan_optim_nuTi)

    # Updating the normalized distribution function `fvL0k` according to the new parameters `nai`, `uai` and `vthi`
    LMk = 0LM
    fvL0k = Vector{Matrix{T}}(undef,ns)
    if is_LM_const
        for isp in nsp_vec
            fvL0k[isp] = zeros(nvG[isp],LM1)
        end
        LMk, fvL0k = fvLDMz(fvL0k, vhe, nvG, LMk, ns, naik, uaik, vthik, nModk; 
            L_limit=LM1-1,rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full)
        LM1k = maximum(LMk) + 1
        if LM1k ≠ LM1
            ghjkk
        end
    else
        # `LM1 + 1` denotes an extra row is given which may be used.
        for isp in nsp_vec
            fvL0k[isp] = zeros(nvG[isp],LM1+1)
        end
        LMk, fvL0k = fvLDMz(fvL0k, vhe, nvG, LMk, ns, naik, uaik, vthik, nModk; 
            L_limit=LM1, rel_dfLM=rel_dfLM, abs_dfLM=abs_dfLM, is_LM1_full=is_LM1_full)
        LM1k = maximum(LMk) + 1
    end

    # Accepting the results according to the to the new parameters `naik`, `uaik` and `vthik`
    if LM1k ≠ LM1
        LM = LMk
        LM1 = LM1k
        mu, Mμ, Mun, Mun1, Mun2 = LegendreMu012(LM1 - 1)
    end

    # Checking the conservation laws of the renormalized distribution function `fvL0k1`
    nIKTh = zeros(4, ns)
    nIKThs!(nIKTh, fvL0k, vhe, ns; atol_IKTh=atol_IKTh, rtol_IKTh=rtol_IKTh)
    # printstyled(naik, uaik, vthik,color=:green)
    # println()

    # # Updating the FP collision terms according to the `FPS` operators.
    δtf = Vector{Matrix{T}}(undef,ns)     # δtfvLa
    ddfvL = Vector{Matrix{T}}(undef,ns)
    dfvL = Vector{Matrix{T}}(undef,ns)
    FvL = Vector{Matrix{T}}(undef,ns)
    for isp in nsp_vec
        δtf[isp] = zeros(T,nvG[isp],LM1)
        ddfvL[isp] = zeros(T,nvG[isp],LM1)
        dfvL[isp] = zeros(T,nvG[isp],LM1)
        FvL[isp] = zeros(T,nck[isp],LM1)
    end
    # Verifying the mass, momentum and total energy conservation laws during the self-collisions.
    if is_δtfvLaa === -1
        δtf,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag2(δtf,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LM,LM1,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,na,vthk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f)
        # Inverse-normalization 
        if is_check_conservation_dtM
            dtnIKs = zeros(3,ns)
            dtnIKsc!(dtnIKs,δtf,vhe,ma,vthk,ns;atol_nIK=atol_nIK)
            dtnIK = norm(dtnIKs)
            if dtnIK > epsTe6
                @warn("Caa: The mass, momentum or the total energy conservation laws doesn't be satisfied during the self-collisions processes!")
            end
        else
            dtnIK = 0.0
        end
        # w3k = dtnIKs[4,:]           # `w3k = Rdtvth = 𝒲 / 3`
        if is_normδtf == false
            cf3 = na ./ vthk.^3 / pi^1.5
            for isp in nsp_vec
                fvL0k[isp] *= cf3[isp]
            end
        end
        # DThk = nIKTh[4, :]
        return δtf, dtnIKs[4,:], fvL0k,  dtnIK, nIKTh[4, :], naik, uaik, vthik, LM
    else
        dtfvLaa = similar(δtf)
        dtfvLaa,ddfvL,dfvL,FvL,FvLa,vaa,nvlevel0a,
                ncF = dtfvLSplineaaLag2(dtfvLaa,ddfvL,dfvL,fvL0k,FvL,
                vhk,nc0,nck,ocp,nvlevele0,nvlevel0,
                mu,Mμ,Mun,Mun1,Mun2,LM,LM1,
                naik,uaik,vthik,nModk,
                CΓ,εᵣ,ma,Zq,spices,na,vthk,ns;
                is_normal=is_normal,restartfit=restartfit,maxIterTR=maxIterTR,
                autodiff=autodiff,factorMethod=factorMethod,show_trace=show_trace,
                p_tol=p_tol,f_tol=f_tol,g_tol=g_tol,n10=n10,dnvs=dnvs,
                is_normδtf=is_normδtf,is_boundaryv0=is_boundaryv0,is_fit_f=is_fit_f)
        for isp in nsp_vec
            nvlevele0a = nvlevele0[isp]
            nvlevele = nvlevel0[isp][nvlevele0a]
            nspF = nsp_vec[nsp_vec .≠ isp]
            iFv = nspF[1]
            mM = ma[isp] / ma[iFv]
            vabth = vthk[isp] / vthk[iFv]
            va0 = vhk[isp][nvlevele] * vabth     # 𝓋̂ = v̂a * vabth
            
            HvL,dHvL = zeros(T,nc0[isp],LM1),zeros(T,nc0[isp],LM1)
            GvL,dGvL = zeros(T,nc0[isp],LM1),zeros(T,nc0[isp],LM1)
            ddGvL = zeros(T,nc0[isp],LM1)
            if nModk[isp] == 1 && nModk[iFv] == 1
                uh = uai[iFv][1]
                if ncF[isp] ≥ 2
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv],uh, 
                                FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
                else
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv],uh)
                end
                δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                        FvL[isp][nvlevele,:],dHvL[nvlevele0a,:],HvL[nvlevele0a,:],ddGvL[nvlevele0a,:],
                        dGvL[nvlevele0a,:],GvL[nvlevele0a,:],vhk[isp][nvlevele],va0,
                        mu,Mμ,Mun,Mun1,Mun2,LM1,uaik[isp][1],uaik[iFv][1],
                        mM,vabth;is_boundaryv0=is_boundaryv0)
            else
                if ncF[isp] ≥ 2
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv], 
                                FvLa[:,isp],vaa[isp],nvlevel0a[isp],ncF[isp])
                else
                    dHvL,HvL,ddGvL,dGvL,GvL = HGshkarofsky(dHvL,HvL,ddGvL,dGvL,GvL,FvL[isp],
                                vhk[isp]*vabth,nvlevel0[isp],nc0[isp],nck[isp],ocp[isp],LM[iFv])
                end
                δtf[isp] = dtfvLSplineab(ddfvL[isp],dfvL[isp],fvL0k[isp],
                        FvL[isp][nvlevele,:],dHvL[nvlevele0a,:],HvL[nvlevele0a,:],ddGvL[nvlevele0a,:],
                        dGvL[nvlevele0a,:],GvL[nvlevele0a,:],vhk[isp][nvlevele],va0,
                        mu,Mμ,Mun,Mun1,Mun2,LM1,
                        naik[isp],naik[iFv],uaik[isp]./vthik[isp],uaik[iFv]./vthik[iFv],
                        vthik[isp],vthik[iFv],nModk[isp],nModk[iFv],
                        mM,vabth;is_boundaryv0=is_boundaryv0)
            end
            if is_boundaryv0 == false
                for L1 in 1:LM1
                    if L1 == 1
                        δtf[isp][1,L1] = 2δtf[isp][2,L1] - δtf[isp][3,L1]
                    else
                        δtf[isp][1,L1] = 0.0
                    end
                end
            end

            if is_lnA_const
                Zqab = Zq[isp] * Zq[iFv]
                lnAg = lnAgamma(εᵣ,ma[isp],Zqab,[spices[isp],spices[iFv]],na[isp],na[iFv],vth[isp],vth[iFv];is_normδtf=true)
            else
                lnAg = lnAgamma_fM(εᵣ,[ma[isp],ma[iFv]],[Zq[isp],Zq[iFv]],[spices[isp],spices[iFv]],
                                [na[isp],na[iFv]],[vth[isp],vth[iFv]],1,2;is_normδtf=is_normδtf)
            end
            δtf[isp] *= (CΓ * lnAg)   # CΓ is owing to the dimensionless process
        end
        # Verifying the mass, momentum and total energy conservation laws of `δtfab`.
        Ihk = nIKTh[2,:]
        Ihk = [sum(naik[k] .* uaik[k]) for k in nsp_vec]  # = `ûa`
        dtnIKs = zeros(4,ns)
        if is_δtfvLaa === 1
            for isp in nsp_vec
                δtf[isp] += dtfvLaa[isp]
            end
        end
        if is_normδtf
            RdtnIKTs!(dtnIKs,δtf,vhe,Ihk,ma,na,vthk,ns;atol_nIK=atol_nIK,
                    is_out_errdt=true,is_enforce_errdtnIKab=is_enforce_errdtnIKab)
        else
            RdtnIKTcs!(dtnIKs,vhe,δtf,Ihk,na,vthk,ns;atol_nIK=atol_nIK)
        end
        if is_check_conservation_dtM 
            if norm(dtnIKs[1,:])  ≥ epsTe6
                @warn("δₜn̂a: The mass conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",dtnIKs[1,:])
            end

            if abs(sum(dtnIKs[2,:])) > epsTe6
                RDIab = abs(dtnIKs[2,1] - dtnIKs[2,2])
                if RDIab ≠ 0.0
                    err_dtI = sum(dtnIKs[2,:]) / RDIab
                else
                    err_dtI = 0.0
                end
                if err_dtI > epsTe6
                    @warn("δₜÎa: The momentum conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_dtI)
                end
            else
                err_dtI = 0.0
            end
        
            if abs(sum(dtnIKs[3,:])) > epsTe6
                RDKab = abs(dtnIKs[3,1] - dtnIKs[3,2])
                if RDKab ≠ 0.0
                    err_dtK = sum(dtnIKs[3,:]) / RDKab
                else
                    err_dtK = 0.0
                end
                if err_dtK > epsTe6
                    @warn("δₜnK̂a: The the total energy conservation laws doesn't be satisfied during the collisions processes! Refining by increasing `nnv`.",err_dtK)
                    @show dtnIKs
                end
            else
                err_dtK = 0.0
            end
            err_dtnIK = fmtf4(norm([dtnIKs[1,:]; err_dtI; err_dtK]))
        else
            err_dtnIK = 0.0
        end

        # Inverse-normalization 
        # w3k = dtnIKs[4,:]           # `w3k = Rdtvth = 𝒲 / 3`
        if is_normδtf == false
            cf3 = na ./ vthk.^3 / pi^1.5
            for isp in nsp_vec
                fvL0k[isp] *= cf3[isp]
            end
        end

        # outputs, DThk = nIKTh[4, :]
        return δtf, dtnIKs[4,:], fvL0k,  err_dtnIK, nIKTh[4, :], naik, uaik, vthik, LM
    end
end

"""
  Inputs:
    fvL0: = fvL0[isp][nvlevele[isp],:]
    dfvL: = dfvLe
    ddfvL: = ddfvLe
    FvL: = FvLe
    uai = uai[isp] / vthi[isp]
    ubi = uai[iFv] / vthi[iFv]

  Outputs:
    δtf = dtfvLSplineab(ddfvL,dfvL,fvL0k,FvL,
                dHvL,HvL,ddGvL,dGvL,GvL,vG0,va0,
                mu,Mμ,Mun,Mun1,Mun2,LM1,
                nai,nbi,uai,ubi,vathi,vbthi,nModa,nModb,
                mM,vabth;is_boundaryv0=is_boundaryv0)
"""

# 2D, [nMod,LM1] , is_inner == 0.0, `4π` is not included in the following code but in `3D`.
function dtfvLSplineab(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvL0::AbstractArray{T,N},
    FvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},HvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    vG0::AbstractVector{T},va0::AbstractVector{T},mu::AbstractArray{T,N},Mμ::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},LM1::Int64,
    nai::AbstractVector{T},nbi::AbstractVector{T},uai::AbstractVector{T},ubi::AbstractVector{T},
    vathi::AbstractVector{T},vbthi::AbstractVector{T},nModa::Int64,nModb::Int64,
    mM::T,vabth::T;is_boundaryv0::Bool=false) where{T,N,NM1,NM2}

    CF = mM
    CH = (1.0 - mM) / vabth
    CG = 0.5 / vabth^2
    cotmu = mu ./ (1 .-mu.^2).^0.5
    # # #################### S = Mvn * XLm * Mun , X = F, H ,G, X = X(vG0)
    fvuP1 = fvL0[:,2:end] * Mun1 ./ vG0   # owing to `Mun1[:,L=0] .≡ 0`
    ############ 1, Sf1 = CF * f * F
    Sf = CF * (fvL0 * Mun) .* (FvL * Mun) 
    Sf1 = deepcopy(Sf[1:2,:])     # = Sf1[1,:]
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
    dX01 = fvL0[:,2:end] ./ vG0 - dfvL[:,2:end]
    GG .*= 2(dX01 * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = (dGvL * Mun) + (GvL[:,2:end] * Mun1 ./ va0) .* cotmu
    GG7 = deepcopy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG += 2GG7
    else
        GG6 = dX01 + (GvL[:,3:end] * Mun2 ./ va0)
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvL0[:,3:end] * Mun2 ./ vG0))
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

# dtCFL
function dtfvLSplineab(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvL0::AbstractArray{T,N},
    FvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},HvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    vG0::AbstractVector{T},va0::AbstractVector{T},mu::AbstractArray{T,N},Mμ::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},LM1::Int64,
    nai::AbstractVector{T},nbi::AbstractVector{T},uai::AbstractVector{T},ubi::AbstractVector{T},
    vathi::AbstractVector{T},vbthi::AbstractVector{T},nModa::Int64,nModb::Int64,
    mM::T,vabth::T,dtkCFL::T;is_boundaryv0::Bool=false) where{T,N,NM1,NM2}

    CF = mM
    CH = (1.0 - mM) / vabth
    CG = 0.5 / vabth^2
    cotmu = mu ./ (1 .-mu.^2).^0.5
    dvh = vG0[3] - vG0[2]

    # # #################### S = Mvn * XLm * Mun , X = F, H ,G, X = X(vG0)
    fvuP1 = fvL0[:,2:end] * Mun1 ./ vG0   # owing to `Mun1[:,L=0] .≡ 0`
    ############ 1, Sf1 = CF * f * F
    Sf = CF * (fvL0 * Mun) .* (FvL * Mun) 
    Sf1 = deepcopy(Sf[1:2,:])     # = Sf1[1,:]
    ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
    # ############ 2,  Sf2
    # ############ 3,  Sf3
    if CH ≠ 0.0
        dtkCFL = dvh / abs(CH * maximum(abs.(dHvL[2:end,1])))
        GG = (dfvL * Mun) .* (dHvL * Mun)
        Sf1[1,:] += CH * GG[1,:]     # = Sf1[1,:] + Sf2[1,:]
        GG += fvuP1 .* (HvL[:,2:end] * Mun1 ./ va0)
        Sf += CH * GG
    else
        dtkCFL = 1e8
    end
    # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`

    ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    dX01 = GvL[:,2:end] ./ va0 - dGvL[:,2:end]
    dtkCFLG5 = maximum(abs.(dX01[2:end,1]))
    GG = dX01 * Mun1
    dX01 = fvL0[:,2:end] ./ vG0 - dfvL[:,2:end]
    GG .*= 2(dX01 * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = (dGvL * Mun) + (GvL[:,2:end] * Mun1 ./ va0) .* cotmu
    dtkCFLG6 = dvh / (CG * max(maximum(abs.(dX01[2:end,1]) ./ (vG0 .* va0)[2:end]), dtkCFLG5))
    # @show dtkCFLG6
    GG7 = deepcopy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG += 2GG7
    else
        GG6 = dX01 + (GvL[:,3:end] * Mun2 ./ va0)
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvL0[:,3:end] * Mun2 ./ vG0))
        GG += (GG7 + GG6)
    end
    GG ./= (vG0 .* va0)
    ############ 4, Sf5
    GG7 = (ddfvL * Mun) .* (ddGvL * Mun)    # GG5
    dtkCFLG4 = dvh^2 / (CG * maximum(abs.(ddGvL[2:end,1])))
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
    return Sf * Mμ, min(min(dtkCFL, dtkCFLG4),dtkCFLG6)
end

"""
  Inputs:
    fvL0: = fvL0[isp]
    dfvL: = dfvL[isp][nvlevel0,:]
    ddfvL: = ddfvL[isp][nvlevel0,:]
    FvL: = FvL[isp][nvlevel0,:]
    uai = uai[isp] / vthi[isp]
    ubi = uai[iFv] / vthi[iFv]

  Outputs:
    δtf = dtfvLSplineab(ddfvL,dfvL,fvL0,FvL,
                    dHvL,HvL,ddGvL,dGvL,GvL,vG0,va0,
                    mu,Mμ,Mun,Mun1,Mun2,LM1,uai,ubi,
                    mM,vabth;is_boundaryv0=is_boundaryv0)
"""

# 2D, [LM1], nc0 , is_inner == 0.0, `4π` is not included in the following code but in `3D`.
# nai = 1, vthi = 1
function dtfvLSplineab(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvL0::AbstractArray{T,N},
    FvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},HvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    vG0::AbstractVector{T},va0::AbstractVector{T},
    mu::AbstractArray{T,N},Mμ::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},LM1::Int64,
    uai::T,ubi::T,mM::T,vabth::T;is_boundaryv0::Bool=false) where{T,N,NM1,NM2}

    CF = mM
    CH = (1.0 - mM) / vabth
    CG = 0.5 / vabth^2
    cotmu = mu ./ (1 .-mu.^2).^0.5
    # # #################### S = Mvn * XLm * Mun , X = F, H ,G, X = X(vG0)
    fvuP1 = fvL0[:,2:end] * Mun1 ./ vG0   # owing to `Mun1[:,L=0] .≡ 0`
    ############ 1, Sf1 = CF * f * F
    Sf = CF * (fvL0 * Mun) .* (FvL * Mun)
    Sf1 = deepcopy(Sf[1:2,:])     # = Sf1[1,:]
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
    dX01 = fvL0[:,2:end] ./ vG0 - dfvL[:,2:end]
    GG .*= 2(dX01 * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = (dGvL * Mun) + (GvL[:,2:end] * Mun1 ./ va0) .* cotmu
    GG7 = deepcopy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG += 2GG7
    else
        GG6 = dX01 + (GvL[:,3:end] * Mun2 ./ va0)
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvL0[:,3:end] * Mun2 ./ vG0))
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

# dtCFL
function dtfvLSplineab(ddfvL::AbstractArray{T,N},dfvL::AbstractArray{T,N},fvL0::AbstractArray{T,N},
    FvL::AbstractArray{T,N},dHvL::AbstractArray{T,N},HvL::AbstractArray{T,N},
    ddGvL::AbstractArray{T,N},dGvL::AbstractArray{T,N},GvL::AbstractArray{T,N},
    vG0::AbstractVector{T},va0::AbstractVector{T},
    mu::AbstractArray{T,N},Mμ::AbstractArray{T,N},
    Mun::AbstractArray{T,N},Mun1::AbstractArray{T,NM1},Mun2::AbstractArray{T,NM2},LM1::Int64,
    uai::T,ubi::T,mM::T,vabth::T,dtkCFL::T;is_boundaryv0::Bool=false) where{T,N,NM1,NM2}

    CF = mM
    CH = (1.0 - mM) / vabth
    CG = 0.5 / vabth^2
    cotmu = mu ./ (1 .-mu.^2).^0.5
    dvh = vG0[3] - vG0[2]

    # # #################### S = Mvn * XLm * Mun , X = F, H ,G, X = X(vG0)
    fvuP1 = fvL0[:,2:end] * Mun1 ./ vG0   # owing to `Mun1[:,L=0] .≡ 0`
    ############ 1, Sf1 = CF * f * F
    Sf = CF * (fvL0 * Mun) .* (FvL * Mun)
    Sf1 = deepcopy(Sf[1:2,:])     # = Sf1[1,:]
    ############ SH = S2 + S3 + (S4) = CH * ∇f : ∇H
    # ############ 2,  Sf2
    # ############ 3,  Sf3
    if CH ≠ 0.0
        dtkCFL = dvh / abs(CH * maximum(abs.(dHvL[2:end,1])))
        GG = (dfvL * Mun) .* (dHvL * Mun)
        Sf1[1,:] += CH * GG[1,:]     # = Sf1[1,:] + Sf2[1,:]
        GG += fvuP1 .* (HvL[:,2:end] * Mun1 ./ va0)
        Sf += CH * GG
    else
        dtkCFL = 1e8
    end
    # @show dtkCFL
    # ############ 4,  Sf4 = 0 owing to `∂/∂ϕ(f) = 0`

    ############ SG = S5 + S6 + S7 + S8 + S9 + S10 = CG * ∇∇f : ∇∇G
    # ############ 5,  Sf6
    dX01 = GvL[:,2:end] ./ va0 - dGvL[:,2:end]
    dtkCFLG5 = maximum(abs.(dX01[2:end,1]))
    GG = dX01 * Mun1
    dX01 = fvL0[:,2:end] ./ vG0 - dfvL[:,2:end]
    GG .*= 2(dX01 * Mun1)
    ############ (6,7), (Sf8, Sf10)
    dX01 = (dGvL * Mun) + (GvL[:,2:end] * Mun1 ./ va0) .* cotmu
    dtkCFLG6 = dvh / (CG * max(maximum(abs.(dX01[2:end,1]) ./ (vG0 .* va0)[2:end]), dtkCFLG5))
    # @show dtkCFLG6
    GG7 = deepcopy(dX01)
    if NM2 == 0
        # GG6 = GG7 = Sf8 = Sf10
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG += 2GG7
    else
        GG6 = dX01 + (GvL[:,3:end] * Mun2 ./ va0)
        dX01 = dfvL * Mun + fvuP1 .* cotmu
        GG7 .*= dX01
        GG6 .*= (dX01 + (fvL0[:,3:end] * Mun2 ./ vG0))
        GG += (GG7 + GG6)
    end
    GG ./= (vG0 .* va0)
    ############ 4, Sf5
    GG7 = (ddfvL * Mun) .* (ddGvL * Mun)    # GG5
    dtkCFLG4 = dvh^2 / (CG * maximum(abs.(ddGvL[2:end,1])))
    # @show dtkCFLG4
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
    # @show dtkCFL, dtkCFLG4,dtkCFLG6
    # dtkCFL = min(min(dtkCFL, dtkCFLG4),dtkCFLG6)
    Sf[1,:] = Sf1[1,:]
    return Sf * Mμ, min(min(dtkCFL, dtkCFLG4),dtkCFLG6)
end
