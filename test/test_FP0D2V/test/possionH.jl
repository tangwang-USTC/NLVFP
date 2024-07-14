
# using MatrixEquations
# using SylvesterEquations
using IterativeSolvers

"""
  Possion equation for Rosenbluth potential Ĥₗ(𝓋̂)

  -`L == 0`
   `(D² + 2/𝓋̂ * D¹)Ĥₗ(𝓋̂) = - 4πF̂ₗ(𝓋̂)`

   -`L ≥ 1`
    `(D² + 2/𝓋̂ * D¹ - L(L+1) / 𝓋̂²)Ĥₗ(𝓋̂) = - 4πF̂ₗ(𝓋̂)`

"""

L1 = 1
isp3 = 1
nspF = nsp_vec[nsp_vec .≠ isp3]
iFv3 = nspF[1]
vabth = vth[isp3] / vth[iFv3]
va = vG * vabth
##
x0 = HvL0[nvgauss,L1,isp3]
# b = 1     # vG = vG / b
Dv = laguerrediff(vG; M = 2, b = 1)
df1 = Dv[:,:,1] * fvL[nvgauss,L1,isp3]
df2 = dfvL[:,L1,isp3]
label = string("df,L=",L1-1)
pdf1 = plot(log.(vremesh*vabth),df2,line=(2,:solid))
label = string("Df,L=",L1-1)
pdf2 = plot!(log.(va),df1,line=(2,:dot))
display(plot(pdf1))
##
b = - 4π * FvL[nvgauss,L1,isp3]
if L1 == 1
    A = Dv[:,:,2] + 2 ./ va .* Dv[:,:,1]
    rhs = Dv[:,:,2] * x0 + 2Dv[:,:,1] * x0 ./ va - b
else
    A = Dv[:,:,2] + 2 ./ va .* Dv[:,:,1] - L1 * (L1 - 1) ./ va.^2 .* ones(1,nv1)
    rhs = Dv[:,:,2] * x0 + 2Dv[:,:,1] * x0 ./ va - L1 * (L1 - 1) ./ va.^2 .* x0 - b
end
prhs = plot(log.(va),rhs)
xx = HvL0[nvgauss,L1,isp3] * 1
# x1, hstr1 = gmres!(xx,A,b;log=true,maxiter=500)
x1, hstr1 = idrs!(xx,A,b;log=true,maxiter=500)
# x1, hist1 = bicgstabl!(xx,A,b,4;log=true)
plot(log.(va),xx)
plot!(log.(va),HvL0[nvgauss,L1,isp3])
plot!(log.(vremesh*vabth),HvL0[:,L1,isp3])
