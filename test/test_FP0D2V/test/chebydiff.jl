

nn = 20
vccc = vccn(nn;datatype=Float64)
vv = vCmapping(vccc,vGdom[1],vGdom[2];isinv=true)
Dcc = Dc1n(nn;datatype = BigFloat)
isrel = 0
 fff(x) = sin.(x)
 dfff = Dcc * fff(vv) * dxdv
 dff = zero.(vv)
 dflog0  = cos(vv[1])
 dff = diffMDcprecond(fff(vv),Dcc,dff,dflog0,dxdv;issolve=false)

 if isrel == 1
     label = string("Ddf,nn=",nn)
     pdfff = plot(vv,cos.(vv)./ dfff .-1,label=label,line=(1,:auto))
     label = string("sin")
     psc = plot(vv,sin.(vv),label=label,line=(1,:auto))
     label = string("cos")
     psc = plot!(vv,cos.(vv),label=label,line=(1,:auto))
 else
     label = string("Ddf,nn=",nn)
     pdfff = plot(vv,cos.(vv) - dff,label=label,line=(1,:auto))
     label = string("Ddfff")
     pdfff = plot!(vv,cos.(vv) - dfff,label=label,line=(1,:auto))
     label = string("sin")
     psc = plot(vv,sin.(vv),label=label,line=(1,:auto))
     label = string("cos")
     psc = plot!(vv,cos.(vv),label=label,line=(1,:auto))
 end
 display(plot(pdfff,psc,layout=(2,1)))
