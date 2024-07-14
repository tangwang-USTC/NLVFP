# Computing the Shkarofsky integrals
ILFLmt,IL2FLmt = zeros(nc0), zeros(nc0)
JL1FLmt,JLn1FLmt = zeros(nc0), zeros(nc0)
if L1 ≤ 3
    ILFLmt,IL2FLmt,JL1FLmt,JLn1FLmt = shkarofskyIJt(ILFLmt,IL2FLmt,JL1FLmt,JLn1FLmt,vb0,uai[isp3][1],L1)
end
# if vb0[1] == 0.0
#     ILFLmt[1],IL2FLmt[1],JL1FLmt[1] = 0.0, 0.0, 0.0
#     if L1 ≥ 2
#         JLn1FLmt[1] = 0.0 
#     end
# end

ILFLm,IL2FLm = zeros(nc0),zeros(nc0)
ILFLm,IL2FLm = shkarofskyI(ILFLm,IL2FLm,FLn,vb,nvlevel0,nc0,ocp,L1)

JL1FLm,JLn1FLm = zeros(nc0), zeros(nc0)
JL1FLm,JLn1FLm = shkarofskyJ(JL1FLm,JLn1FLm,FLn,vb,nvlevel0,nc0,nck,ocp,L1)
if L1 == 1
    ILn1FL0,IL1FL0 = shkarofskyIL0(zeros(nc0),zeros(nc0),FLn,vb,nvlevel0,nc0,ocp)
    ILFLm2,IL2FLm2 = vb0 .* ILn1FL0,vb0 .* IL1FL0
    JLFL0,JL0FL0 = shkarofskyJL0(zeros(nc0),zeros(nc0),FLn,vb,nc0,nck,ocp)
    JL1FLm2,JLn1FLm2 = vb0 .* JLFL0, JL0FL0
elseif L1 == 2
    ILn2FL0,IL2FL0 = shkarofskyIL1(zeros(nc0),zeros(nc0),FLn,vb,nvlevel0,nc0,ocp)
    ILFLm2,IL2FLm2 = vb0.^2 .* ILn2FL0, IL2FL0
    JL1n2FL0,JLn1FLm2 = shkarofskyJL1(zeros(nc0),zeros(nc0),FLn,vb,nc0,nck,ocp)
    JL1FLm2 = vb0.^2 .* JL1n2FL0
elseif L1 == 3
    ILn1FL0,IL1FL0 = shkarofskyIL2(zeros(nc0),zeros(nc0),FLn,vb,nvlevel0,nc0,ocp)
    ILFLm2,IL2FLm2 = vb0 .* ILn1FL0,vb0 .* IL1FL0
    JLFL0,JLn2FL0 = shkarofskyJL2(zeros(nc0),zeros(nc0),FLn,vb,nvlevel0,nc0,nck,ocp)
    JL1FLm2,JLn1FLm2 = vb0 .* JLFL0, vb0 .* JLn2FL0
else
    ILFLm2,IL2FLm2,JL1FLm2,JLn1FLm2 = zeros(nc0),zeros(nc0),zeros(nc0),zeros(nc0)
    ILFLm2,IL2FLm2 = shkarofskyI(ILFLm2,IL2FLm2,FLn,vb,nvlevel0,nc0,ocp,L1)
    JL1FLm2,JLn1FLm2 = shkarofskyJ(JL1FLm2,JLn1FLm2,FLn,vb,nvlevel0,nc0,nck,ocp,L1)
end
dataIL0 = [vG0 ILFLmt ILFLm2  (ILFLmt -ILFLm2) *neps   (ILFLmt./ILFLm.-1) *neps]
dataIL2 = [vG0 IL2FLmt IL2FLm2  (IL2FLmt -IL2FLm2) *neps  (IL2FLmt./IL2FLm.-1) *neps]
dataJL1 = [vG0 JL1FLmt JL1FLm2 (JL1FLmt -JL1FLm) *neps  (JL1FLmt./JL1FLm.-1) *neps]
dataJLn1 = [vG0 JLn1FLmt JLn1FLm2  (JLn1FLmt -JLn1FLm) *neps  (JLn1FLmt./JLn1FLm.-1) *neps]
if is_plotIJ == 1
    label = string("IL0,L=",L1-1)
    pL0 = plot(vG0,ILFLm,label=label,line=(3,:auto))
    label = string("IL02")
    pL0 = plot!(vG0,ILFLm2,label=label,line=(3,:auto))
    label = string("IL0t")
    pL0 = plot!(vG0,ILFLmt,label=label,line=(3,:auto))

    label = string("IL2")
    pL2 = plot(vG0,IL2FLm,label=label,line=(3,:auto))
    label = string("IL22")
    pL2 = plot!(vG0,IL2FLm2,label=label,line=(3,:auto))
    label = string("IL2t")
    pL2 = plot!(vG0,IL2FLmt,label=label,line=(3,:auto))

    label = string("errIL0")
    pReL0 = plot(vG0[2:end],(ILFLm-ILFLmt)[2:end]*neps,label=label,line=(3,:auto))
    label = string("errIL02")
    pReL0 = plot!(vG0[2:end],(ILFLm2-ILFLmt)[2:end]*neps,label=label,line=(3,:auto))

    label = string("errIL2")
    pReL2 = plot(vG0[2:end],(IL2FLm-IL2FLmt)[2:end]*neps,label=label,line=(3,:auto))
    label = string("errIL22")
    pReL2 = plot!(vG0[2:end],(IL2FLm2-IL2FLmt)[2:end]*neps,label=label,line=(3,:auto))

    display(plot(pL0,pL2,pReL0,pReL2,layout=(2,2)))
end
if is_plotIJ == 1
   label = string("JL1,L=",L1-1)
   pL1 = plot(vG0,JL1FLm2,label=label,line=(3,:auto))
   label = string("JL1m,L=",L1-1)
   pL1 = plot!(vG0,JL1FLm,label=label,line=(3,:auto))
   label = string("JL1t,L=",L1-1)
   pL1 = plot!(vG0,JL1FLmt,label=label,line=(3,:auto))

   label = string("JLn1")
   pLn1 = plot(vG0,JLn1FLm2,label=label,line=(3,:auto))
   label = string("JLn1m")
   pLn1 = plot!(vG0,JLn1FLm,label=label,line=(3,:auto))
   label = string("JLn1mt")
   pLn1 = plot!(vG0,JLn1FLmt,label=label,line=(3,:auto))

   label = string("errJL1")
   peL1 = plot(vG0[2:end],(JL1FLm2-JL1FLmt)[2:end]*neps,label=label,line=(3,:auto))
   label = string("errJL1m")
   peL1 = plot!(vG0[2:end],(JL1FLm-JL1FLmt)[2:end]*neps,label=label,line=(3,:auto))
   label = string("errJ1")
   peL1 = plot!(vG0[2:end],(JL1FLm-JL1FLm2)[2:end]*neps,label=label,line=(3,:auto))

   label = string("errJLn1")
   peLn1 = plot(vG0[2:end],(JLn1FLm2-JLn1FLmt)[2:end]*neps,label=label,line=(3,:auto))
   label = string("errJLn1m")
   peLn1 = plot!(vG0[2:end],(JLn1FLm-JLn1FLmt)[2:end]*neps,label=label,line=(3,:auto))
   label = string("errJn1")
   peLn1 = plot!(vG0[2:end],(JLn1FLm-JLn1FLm2)[2:end]*neps,label=label,line=(3,:auto))
   display(plot(pL1,pLn1,peL1,peLn1,layout=(2,2)))
end
