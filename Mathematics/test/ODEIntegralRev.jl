using SpecialFunctions

"""
du/dt = f(u,p,t)

"""

plotly()

t0 = 0
tend = 4.2 + t0
dt = 1.e-3
x = t0:dt:tend |> Vector
# tend += dt
# x = [x; tend]

f(x) = exp.(-x.^2)
F(x) = @. - √π / 2 * erfc(x) # F = ∫f(t)dt where t ∈[∞,x]

##
legendlocal = :bottomright
FT = F(x)

I0 = 0.0

FTrap = reverse(cumul_integrate(reverse(x),reverse(f(x))))


####  Errors
xplot = x .< 3
lable = string("M=","Trapz")
err = (FTrap - FT) ./ FT
pe = plot(x[xplot],err[xplot],line=(2,:auto),label=lable,legend=legendlocal)

# k = 1
# Fs1 = cIsimpson1Dv(f(x),x; k = k, inv = true,I0=I0)
# lable = string("M=","Simp",k)
# err = (Fs1 - FT) ./ FT
# pe = plot!(x[xplot],err[xplot],line=(2,:auto),label=lable,legend=legendlocal)

k = 2
Fs2 = cIsimpson1Dv(f(x),x; k = k, inv = true,I0=I0)
lable = string("M=","Simp",k)
err = (Fs2 - FT) ./ FT
pe = plot!(x[xplot],err[xplot],line=(2,:auto),label=lable,legend=legendlocal)

k = 3
Fs3 = cIsimpson1Dv(f(x),x; k = k, inv = true,I0=I0)
lable = string("M=","Simp",k)
err = (Fs3 - FT) ./ FT
pe = plot!(x[xplot],err[xplot],line=(2,:auto),label=lable,legend=legendlocal)

k = 4
lable = string("M=","Simp",k)
Fs4 = cIsimpson1Dv(f(x),x; k = k, inv = true,I0=I0)
err = (Fs4 - FT) ./ FT
pe = plot!(x[xplot],err[xplot],line=(2,:auto),label=lable,legend=legendlocal)

k = 5
lable = string("M=","Simp",k)
Fs5 = cIsimpson1Dv(f(x),x; k = k, inv = true,I0=I0)
err = (Fs5 - FT) ./ FT
pe = plot!(x[xplot],err[xplot],line=(2,:auto),label=lable,legend=legendlocal)

data = [x FT FTrap Fs2 Fs3 Fs4 Fs5]

lable = string("M=","Theory")
plot(x,FT,line=(2,:auto),label=lable,legend=legendlocal)
lable = string("M=","Trap")
pF = plot!(x,FTrap,line=(2,:auto),label=lable,legend=legendlocal)

# k = 1
# lable = string("M=","Simp",k)
# pF = plot!(x,Fs1,line=(2,:auto),label=lable,legend=legendlocal)
k = 2
lable = string("M=","Simp",k)
pF = plot!(x,Fs2,line=(2,:auto),label=lable,legend=legendlocal)
k = 3
lable = string("M=","Simp",k)
pF = plot!(x,Fs3,line=(2,:auto),label=lable,legend=legendlocal)
k = 4
lable = string("M=","Simp",k)
pF = plot!(x,Fs4,line=(2,:auto),label=lable,legend=legendlocal)
k = 5
lable = string("M=","Simp",k)
pF = plot!(x,Fs5,line=(2,:auto),label=lable,legend=legendlocal)
pe
