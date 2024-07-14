
# Computing the Rosenbluth potentials `Ĥₗ(𝓋̂)` and `Ĝₗ(𝓋̂)`
## `H(v)`, `dH = ∂ᵥH(𝓋̂)` and `ddH = # ∂ᵥ²H(𝓋̂)`
## `G(v)`, `dG = ∂ᵥG(𝓋̂)` and `ddG = # ∂ᵥ²G(𝓋̂)`
# isRel ∈ [:unit, :Max, :Maxd0, :Maxd1, :Maxd2]
isRel = :unit
isRel = :Max
# # 2D
ddHvL2,dHvL2,HvL2, ddGvL2,dGvL2,GvL2 = zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1), zeros(nc0,LM1)
ddHvL2,dHvL2,HvL2, ddGvL2,dGvL2,GvL2 = HGshkarofsky(ddHvL2,dHvL2,HvL2,
               ddGvL2,dGvL2,GvL2,FvL3[:,:,isp3],va,nvlevel0,nc0,nck,ocp,LM[isp3])

RddH0a(L1) = RddHLp(ddHvL2[:,L1],dHvL2[:,L1],HvL2[:,L1],FvL3[nvlevel0,L1,isp3],va0,L1;isRel=isRel)  *neps
RddG0a(L1) = RddGLp(ddGvL2[:,L1],dGvL2[:,L1],GvL2[:,L1],HvL2[:,L1],va0,L1;isRel=isRel)  *neps

Hvu = HvL2 * Mun
HvL22 = Hvu * Mμ
# # 3D

# 1D
ddHLn0,dHLn0,HLn0, ddGLn0,dGLn0,GLn0 = ddHvL2[:,L1],dHvL2[:,L1],HvL2[:,L1],ddGvL2[:,L1],dGvL2[:,L1],GvL2[:,L1]
dataRddHG = [RddH0a(L1) RddG0a(L1)]
if is_plotdH == 1
    label = [string("RddH0,L=",L1-1)  string("RddG0,isp=",isp3)]
    pRH= plot(vb0,dataRddHG[:,1:2],label=label,line=(3,:auto))
    display(plot(pRH))
end
va0log = log.(va0)
nvplot = 0 .< va0
# Plotting
if is_plotdH == 1
    xlabel = string("log10(va0)")
    label = string("HLn0")
    pH = plot(va0log[nvplot],HLn0[nvplot],label=label,line=(3,:auto),legend=legendbL)
    #
    label = string("dH,L=",L1-1)
    pdH = plot(va0log[nvplot],dHLn0[nvplot],label=label,legend=legendbL,line=(3,:auto))
    #
    label = string("ddH")
    pddH = plot(va0log[nvplot],ddHLn0[nvplot],label=label,legend=legendtR,xlabel=xlabel,line=(1,:auto))
    if L1 == 2
        label = string("dH")
        pddH = plot!(va0log[nvplot],dHLn0[nvplot],label=label,line=(3,:auto))
    else
        label = string("dH/v")
        dHv = dHLn0[nvplot] ./ va0[nvplot]
        pddH = plot!(va0log[nvplot],dHv,label=label,line=(3,:auto))
    end
    # display(pddH)
    label = string("FLn0")
    pddH = plot!(va0log[nvplot],FLn0[nvplot],label=label,line=(3,:solid))
    #
    label = string("RddH0")
    pRddHs = plot(va0log[nvplot],dataRddHG[nvplot,1],label=label,xlabel=xlabel,line=(3,:auto))
    display(plot(pH,pdH,pddH,pRddHs,layout=(2,2)))
end
if is_plotdG == 1
    xlabel = string("log10(v)")
    label = string("GLn0")
    pG = plot(va0log[nvplot],GLn0[nvplot],label=label,legend=legendtL,line=(3,:auto))
    #
    label = string("dG,L=",L1-1)
    pdG = plot(va0log[nvplot],dGLn0[nvplot],label=label,legend=legendtL,line=(3,:auto))
    #
    label = string("ddG")
    pddG = plot(va0log[nvplot],ddGLn0[nvplot],label=label,legend=legendtR,xlabel=xlabel,line=(1,:auto))
    if L1 == 2
        label = string("dG")
        pddG = plot!(va0log[nvplot],dGLn0[nvplot],label=label,line=(3,:auto))
    else
        label = string("dG/v")
        dG2v = dGLn0[nvplot] ./ va0[nvplot]
        pddG = plot!(va0log[nvplot],dG2v,label=label,line=(3,:auto))
    end
    # display(pddG)
    label = string("HLn0")
    pddG = plot!(va0log[nvplot],HLn0[nvplot],label=label,line=(3,:solid))
    #
    label = string("RddG0")
    pRddGs = plot(va0log[nvplot],dataRddHG[nvplot,2],label=label,xlabel=xlabel,line=(3,:auto))
    display(plot(pG,pdG,pddG,pRddGs,layout=(2,2)))
end
# Plotting the first three-order derivatives of `f̂(v̂)`, `Ĥₗ(𝓋̂)` and `Ĝₗ(𝓋̂)`
if is_plotdfHG == 1
    label = string("f,L=",L1-1)
    ppf = plot(va0log,(fLn0),label=label,legend=legendbL,line=(3,:auto))
    label = string("F")
    ppF = plot!(va0log,(FLn0),label=label,legend=legendbL,line=(3,:auto))
    label = string("df")
    ppdf = plot(va0log,(dfLn0),label=label,legend=legendbL,line=(3,:auto))
    label = string("ddf")
    ppddf = plot(va0log,(ddfLn0),label=label,legend=legendbL,line=(3,:auto))

    label = string("H")
    ppH = plot(va0log,(HLn0),label=label,legend=legendbL,line=(3,:auto))
        ylabel!("Error of value normalized by `epsT`")
    label = string("dH")
    ppdH = plot(va0log,(dHLn0),label=label,legend=legendbL,line=(3,:auto))
    label = string("ddH")
    ppddH = plot(va0log,(ddHLn0),label=label,legend=legendbL,line=(3,:auto))

    label = string("G")
    ppG = plot(va0log,(GLn0),label=label,legend=legendbL,line=(3,:auto))
    label = string("dG")
    ppdG = plot(va0log,(dGLn0),label=label,legend=legendbL,line=(3,:auto))
    label = string("ddG")
    ppddG = plot(va0log,(ddGLn0),label=label,line=(3,:auto))
    display(plot(ppf,ppdf,ppddf,ppH,ppdH,ppddH,ppG,ppdG,ppddG,layout=(3,3)))
end
