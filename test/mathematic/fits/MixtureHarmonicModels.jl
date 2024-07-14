
ℓ = 1
u1 = 0
u2 = 0.2
u2 = 1.0
u3 = 6.0

fLn10,fLn20,fLn30 = zero.(vG0),zero.(vG0),zero.(vG0)

fLn10 = fDM(fLn10,vG0,u1,ℓ)
fLn20 = fDM(fLn20,vG0,u2,ℓ)
fLn30 = fDM(fLn30,vG0,u3,ℓ)

fLn120 = fLn10 + fLn20
fLn130 = fLn10 + fLn30
fLn230 = fLn30 + fLn20

fLn1202 = 2fLn10 + fLn20
fLn1302 = 2fLn10 + fLn30
fLn2302 = 2fLn30 + fLn20

fLn1203 = fLn10 + 3fLn20
fLn1303 = fLn10 + 3fLn30
fLn2303 = fLn30 + 3fLn20

fLn1230 = fLn10 + fLn20 + fLn30
fLn12301 = 2fLn10 + fLn20 + fLn30
fLn12302 = fLn10 + 2fLn20 + fLn30
fLn12303 = fLn10 + fLn20 + 2fLn30
# plotting
if 1 == 1
    label = string("u=",u1,",p2=",1)
    pf1 = plot(vG0,fLn10,label=label,line=(wline,:auto))
    label = string("u=",u2,",p2=",0.1)
    pf1 = plot!(vG0,fLn20,label=label,line=(wline,:auto))
    label = string("u=",u3,",p2=",10)
    pf1 = plot!(vG0,fLn30,label=label,line=(wline,:auto))

    label = string("p2=",[1,0.1])
    pf12 = plot(vG0,fLn120,label=label,line=(wline,:auto))
    label = string("p2=",[1,10])
    pf12 = plot!(vG0,fLn130,label=label,line=(wline,:auto))
    label = string("p2=",[0.1,10])
    pf12 = plot!(vG0,fLn230,label=label,line=(wline,:auto))

    label = string("p2=",[1,0.1])
    pf122 = plot(vG0,fLn1202,label=label,line=(wline,:auto))
    label = string("p2=",[1,10])
    pf122 = plot!(vG0,fLn1302,label=label,line=(wline,:auto))
    label = string("p2=",[0.1,10])
    pf122 = plot!(vG0,fLn2302,label=label,line=(wline,:auto))


    label = string("p2=",[1,0.1])
    pf123 = plot(vG0,fLn1203,label=label,line=(wline,:auto))
    label = string("p2=",[1,10])
    pf123 = plot!(vG0,fLn1303,label=label,line=(wline,:auto))
    label = string("p2=",[0.1,10])
    pf123 = plot!(vG0,fLn2303,label=label,line=(wline,:auto))

    label = string("p2=",[1,0.1])
    pf1230 = plot(vG0,fLn1230,label=label,line=(wline,:auto))
    label = string("p2=",[1,10])
    pf1230 = plot!(vG0,fLn12301,label=label,line=(wline,:auto))
    label = string("p2=",[1,10])
    pf1230 = plot!(vG0,fLn12302,label=label,line=(wline,:auto))
    label = string("p2=",[0.1,10])
    pf1230 = plot!(vG0,fLn12303,label=label,line=(wline,:auto))
end
display(pf1)
display(pf12)
display(pf122)
display(pf123)
display(pf1230)
