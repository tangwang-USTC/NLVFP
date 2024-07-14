
ntest = 17
vGab = [0.1, 1]
T = Float64
vcc17 = clenshawcurtisnodes(T,ntest)
# vcc9 = vcc17[1:2:end]
# vcc5 = vcc17[1:4:end]
# vcc3 = vcc17[1:8:end]
v17 = vCmapping(vcc17,vGab[1],vGab[2];isinv=true)
# v9 = vCmapping(vcc9,vGab[1],vGab[2];isinv=true)
# v5 = vCmapping(vcc5,vGab[1],vGab[2];isinv=true)
# v3 = vCmapping(vcc3,vGab[1],vGab[2];isinv=true)
v9 = v17[1:2:end]
v5 = v17[1:4:end]
v3 = v17[1:8:end]
vcc33 = clenshawcurtisnodes(T,33)
v33 = vCmapping(vcc33,vGab[1],vGab[2];isinv=true)
vcc65 = clenshawcurtisnodes(T,65)
v65 = vCmapping(vcc65,vGab[1],vGab[2];isinv=true)

wc17 = clenshawcurtisweights(μccn17)
wc9 = clenshawcurtisweights(μccn9)
wc5 = clenshawcurtisweights(μccn5)
wc3 = clenshawcurtisweights(μccn3)
wc33 = clenshawcurtisweights(μccn33)
wc65 = clenshawcurtisweights(chebyshevmoments1(Float64,65))
cI = - 2/sqrtpi * (v17[1] - v17[end])

ftest(v) = exp.(-v^2)

I17 = zeros(6)
I17[1] = cI * dot(wc3, (v3.*2 .* ftest.(v3)))
I17[2] = cI * dot(wc5, (v5.*2 .* ftest.(v5)))
I17[3] = cI * dot(wc9, (v9.*2 .* ftest.(v9)))
I17[4] = cI * dot(wc17, (v17.*2 .* ftest.(v17)))
I17[5] = cI * dot(wc33, (v33.*2 .* ftest.(v33)))
I17[6] = cI * dot(wc65, (v65.*2 .* ftest.(v65)))



Dc65 = Dc12n65[:,:,1]
Dc33 = Dc12n33[:,:,1]
Dc17 = Dc12n17[:,:,1]
Dc9 = Dc12n9[:,:,1]
Dc5 = Dc12n5[:,:,1]
Dc3 = Dc12n3[:,:,1]
dxdv = 2 / (v[1] - v[end])

df3 =  dxdv * (Dc3 * ftest.(v3))
df5 =  dxdv * (Dc5 * ftest.(v5))
df9 =  dxdv * (Dc9 * ftest.(v9))
df17 =  dxdv * (Dc17 * ftest.(v17))
df33 =  dxdv * (Dc33 * ftest.(v33))
df65 =  dxdv * (Dc65 * ftest.(v65))

i = 1
dfi = [df3[i], df5[i], df9[i],df17[i],df33[i],df65[i]]
df0 = [df3[2], df5[3], df9[5],df17[9],df33[17],df65[33]]
dfend = [df3[end], df5[end], df9[end],df17[end],df33[end],df65[end]]
