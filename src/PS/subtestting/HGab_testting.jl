#### Computing `Ĥₗ(𝓋̂)` and `Ĝₗ(𝓋̂)` and in theory
if nMod0[iFv3] == 1
    if va[1] ≠ 0.0
        HLnt, dHLnt, ddHLnt, GLnt, dGLnt, ddGLnt = HGL0DMabz(vb,uai[iFv3][1];L1=L1)
        HLn0t, dHLn0t, ddHLn0t = HLnt[nvlevel0], dHLnt[nvlevel0], ddHLnt[nvlevel0]
        GLn0t, dGLn0t, ddGLn0t = GLnt[nvlevel0], dGLnt[nvlevel0], ddGLnt[nvlevel0]
        HLnbt, dHLnbt, ddHLnbt, GLnbt, dGLnbt, ddGLnbt = HGL0DMabz(va,uai[isp3];L1=L1)
    else
        HLnt, dHLnt, ddHLnt = zero.(vGk),zero.(vGk),zero.(vGk)
        GLnt, dGLnt, ddGLnt = zero.(vGk),zero.(vGk),zero.(vGk)
        HLnbt, dHLnbt, ddHLnbt = zero.(vGk),zero.(vGk),zero.(vGk)
        GLnbt, dGLnbt, ddGLnbt = zero.(vGk),zero.(vGk),zero.(vGk)
        if L1 ≤ 2
            HLnt[2:end], dHLnt[2:end], ddHLnt[2:end],GLnt[2:end], dGLnt[2:end], ddGLnt[2:end] =
                                                       HGL0DMabz(vb[2:end],uai[iFv3][1];L1=L1)
            # HLnt[1], dHLnt[1], ddHLnt[1],GLnt[1], dGLnt[1], ddGLnt[1] = HGL0DMabz(uai[iFv3][1];L1=L1)
    
            HLnbt[2:end], dHLnbt[2:end], ddHLnbt[2:end],GLnbt[2:end], dGLnbt[2:end], 
                                                ddGLnbt[2:end] = HGL0DMabz(vb[2:end],uai[iFv3][1];L1=L1)
            # HLnbt[1], dHLnbt[1], ddHLnbt[1],GLnbt[1], dGLnbt[1], ddGLnbt[1] = HGL0DMabz(uai[iFv3][1];L1=L1)
        end
        HLn0t, dHLn0t, ddHLn0t = HLnt[nvlevel0], dHLnt[nvlevel0], ddHLnt[nvlevel0]
        GLn0t, dGLn0t, ddGLn0t = GLnt[nvlevel0], dGLnt[nvlevel0], ddGLnt[nvlevel0]
    end
else
    rfgh
end

# # boundary conditions of `Ĥ(𝓋̂)` and `Ĝ(𝓋̂)`he Rosenbluth potentials `Ĥₗ(𝓋̂)` and `Ĝₗ(𝓋̂)`
## `H(v)`, `dH = ∂ᵥH(𝓋̂)` and `ddH = # ∂ᵥ²H(𝓋̂)`
## `G(v)`, `dG = ∂ᵥG(𝓋̂)` and `ddG = # ∂ᵥ²G(𝓋̂)`
# isRel ∈ [:unit, :Max, :Maxd0, :Maxd1, :Maxd2]
isRel = :unit
isRel = :Max
# 1D 
ddHLn0,dHLn0,HLn0, ddGLn0,dGLn0,GLn0 = zeros(nc0), zeros(nc0), zeros(nc0), zeros(nc0), zeros(nc0), zeros(nc0)
ddHLn0,dHLn0,HLn0, ddGLn0,dGLn0,GLn0 = HGshkarofsky(ddHLn0,dHLn0,HLn0,
               ddGLn0,dGLn0,GLn0,FLn,vb,nvlevel,nvlevel0,nc0,nck,ocp,L1)
RddH0 = RddHLp(ddHLn0,dHLn0,HLn0,FLn0,vb0,L1;isRel=isRel)  *neps
RddG0 = RddGLp(ddGLn0,dGLn0,GLn0,HLn0,vb0,L1;isRel=isRel)  *neps
RddH0t = RddHL(ddHLn0t,dHLn0t,HLn0t,FLnb0,vb0,L1)  *neps
RddG0t = RddGL(ddGLn0t,dGLn0t,GLn0t,HLn0t,vb0,L1)  *neps
dataRddHG = [RddH0 RddG0 RddH0t RddG0t]
if is_plotdH == 1
    label = [string("RddH0,L=",L1-1)  string("RddG0,isp=",isp3)]
    pRH= plot(vb0,dataRddHG[:,1:2],label=label,line=(3,:auto))
    label = ["RddH0t"  "RddG0t"]
    pRG= plot(vb0,dataRddHG[:,3:4],label=label,line=(3,:auto))
    display(plot(pRH,pRG,layout=(2,1)))
end
# # 2D
# ddHvL2,dHvL2,HvL2, ddGvL2,dGvL2,GvL2 = zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1)
# ddHvL2,dHvL2,HvL2, ddGvL2,dGvL2,GvL2 = HGshkarofsky(ddHvL2,dHvL2,HvL2,
#                ddGvL2,dGvL2,GvL2,FvLb2,vb,nvlevel,nvlevel0,nc0,nck,ocp,LM[iFv3])
# dHvL22,HvL22, ddGvL22,dGvL22,GvL22 = zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1)
# dHvL22,HvL22, ddGvL22,dGvL22,GvL22 = HGshkarofsky(dHvL22,HvL22,
#                ddGvL22,dGvL22,GvL22,FvLb2,vb,nvlevel,nvlevel0,nc0,nck,ocp,LM[iFv3])
# # 3D

dataHLn0 = [vG0 HLn0t  (HLn0t - HLn0) *neps  (HLn0t./HLn0.-1) *neps]
datadHLn0 = [vG0 dHLn0t  (dHLn0t - dHLn0) *neps  (dHLn0t./dHLn0.-1) *neps]
dataddHLn0 = [vG0 ddHLn0t  (ddHLn0t - ddHLn0) *neps  (ddHLn0t./ddHLn0.-1) *neps]

dataGLn0 = [vG0 GLn0t  (GLn0t - GLn0) *neps  (GLn0t./GLn0.-1) *neps]
datadGLn0 = [vG0 dGLn0t  (dGLn0t - dGLn0) *neps  (dGLn0t./dGLn0.-1) *neps]
dataddGLn0 = [vG0 ddGLn0t  (ddGLn0t - ddGLn0) *neps  (ddGLn0t./ddGLn0.-1) *neps]

va0log = log.(va0)
nvplot = 0 .< va0
# Plotting
if is_plotdH == 1
    xlabel = string("log10(va0)")
    label = string("HLn0")
    pH = plot(va0log[nvplot],HLn0[nvplot],label=label,line=(3,:auto),legend=legendbL)
    label = string("HLn0t")
    pH = plot!(va0log[nvplot],HLn0t[nvplot],label=label,line=(3,:auto),legend=legendbL)
    #
    label = string("dH,L=",L1-1)
    pdH = plot(va0log[nvplot],dHLn0[nvplot],label=label,legend=legendbL,line=(3,:auto))
    #
    label = string("ddH")
    pddH = plot(va0log[nvplot],ddHLn0[nvplot],label=label,legend=legendtR,xlabel=xlabel,line=(1,:auto))
    if L1 == 2
        label = string("dH")
        pddH = plot!(va0log[nvplot],dHLn0[nvplot],label=label,line=(3,:auto))
        label = string("dHt")
        pddH = plot!(va0log[nvplot],dHLn0t[nvplot],label=label,line=(3,:auto))
    else
        label = string("dH/v")
        dHv = dHLn0[nvplot] ./ va0[nvplot]
        pddH = plot!(va0log[nvplot],dHv,label=label,line=(3,:auto))
        label = string("dHt/v")
        dHvt = dHLn0t[nvplot] ./ va0[nvplot]
        pddH = plot!(va0log[nvplot],dHvt,label=label,line=(3,:auto))
    end
    # display(pddH)
    label = string("FLn0")
    pddH = plot!(va0log[nvplot],FLn0[nvplot],label=label,line=(3,:solid))
    #
    label = string("RddH0")
    pRddHs = plot(va0log[nvplot],RddH0[nvplot],label=label,xlabel=xlabel,line=(3,:auto))
    # label = string("RddHt")
    # pRddHS = plot!(va0log[nvplot], RddH0t[nvplot],label=label,legend=legendtR,line=(3,:auto))
    display(plot(pH,pdH,pddH,pRddHs,layout=(2,2)))
end
if is_plotdG == 1
    xlabel = string("log10(v)")
    label = string("GLn0")
    pG = plot(va0log[nvplot],GLn0[nvplot],label=label,legend=legendtL,line=(3,:auto))
    label = string("GLn0t")
    pG = plot!(va0log[nvplot],GLn0t[nvplot],label=label,legend=legendtL,line=(3,:auto))
    #
    label = string("dG,L=",L1-1)
    pdG = plot(va0log[nvplot],dGLn0[nvplot],label=label,legend=legendtL,line=(3,:auto))
    #
    label = string("ddG")
    pddG = plot(va0log[nvplot],ddGLn0[nvplot],label=label,legend=legendtR,xlabel=xlabel,line=(1,:auto))
    if L1 == 2
        label = string("dG")
        pddG = plot!(va0log[nvplot],dGLn0[nvplot],label=label,line=(3,:auto))
        label = string("dGt")
        pddG = plot!(va0log[nvplot],dGLn0t[nvplot],label=label,line=(3,:auto))
    else
        label = string("dG/v")
        dG2v = dGLn0[nvplot] ./ va0[nvplot]
        pddG = plot!(va0log[nvplot],dG2v,label=label,line=(3,:auto))
        label = string("dGt/v")
        dG2vt = dGLn0t[nvplot] ./ va0[nvplot]
        pddG = plot!(va0log[nvplot],dG2vt,label=label,line=(3,:auto))
    end
    # display(pddG)
    label = string("HLn0")
    pddG = plot!(va0log[nvplot],HLn0[nvplot],label=label,line=(3,:solid))
    #
    label = string("RddG0")
    pRddGs = plot(va0log[nvplot],RddG0[nvplot],label=label,xlabel=xlabel,line=(3,:auto))
    # label = string("RddGt")
    # pRddG = plot!(va0log[nvplot], RddG0t[nvplot],label=label,legend=legendtR,line=(3,:auto))
    display(plot(pG,pdG,pddG,pRddGs,layout=(2,2)))
end
println()
@show norm((HLn0 - HLn0t)[2:end])*neps
@show norm((dHLn0 - dHLn0t)[2:end]) *neps
@show norm((ddHLn0 - ddHLn0t)[2:end]) *neps
println()
@show norm((GLn0 - GLn0t)[2:end]) *neps
@show norm(dGLn0 - dGLn0t) *neps
@show norm((ddGLn0 - ddGLn0t)[2:end]) *neps
# Plotting the first three-order derivatives of `f̂(v̂)`, `Ĥₗ(𝓋̂)` and `Ĝₗ(𝓋̂)`
if is_plotdfHG == 1
    label = string("f,L=",L1-1)
    ppf = plot(va0log,(fLn0 - fLn0t) * neps,label=label,legend=legendbL,line=(3,:auto))
    label = string("F")
    ppF = plot!(va0log,(FLn0 - FLn0t) * neps,label=label,legend=legendbL,line=(3,:auto))
    label = string("df")
    ppdf = plot(va0log,(dfLn0 - dfLn0t) * neps,label=label,legend=legendbL,line=(3,:auto))
    label = string("ddf")
    ppddf = plot(va0log,(ddfLn0 - ddfLn0t) * neps,label=label,legend=legendbL,line=(3,:auto))

    label = string("H")
    ppH = plot(va0log,(HLn0 - HLn0t) * neps,label=label,legend=legendbL,line=(3,:auto))
        ylabel!("Error of value normalized by `eps(1.0)`")
    label = string("dH")
    ppdH = plot(va0log,(dHLn0 - dHLn0t) * neps,label=label,legend=legendbL,line=(3,:auto))
    label = string("ddH")
    ppddH = plot(va0log,(ddHLn0 - ddHLn0t) * neps,label=label,legend=legendbL,line=(3,:auto))

    label = string("G")
    ppG = plot(va0log,(GLn0 - GLn0t) * neps,label=label,legend=legendbL,line=(3,:auto))
    label = string("dG")
    ppdG = plot(va0log,(dGLn0 - dGLn0t) * neps,label=label,legend=legendbL,line=(3,:auto))
    label = string("ddG")
    ppddG = plot(va0log,(ddGLn0 - ddGLn0t) * neps,label=label,line=(3,:auto))
    display(plot(ppf,ppdf,ppddf,ppH,ppdH,ppddH,ppG,ppdG,ppddG,layout=(3,3)))
end
