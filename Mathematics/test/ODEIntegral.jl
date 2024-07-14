using SpecialFunctions

"""
du/dt = f(u,p,t)

"""

plotly()

t0 = 0.0
tend = 6.0 + t0
dt = 1e-1
x = t0:dt:tend |> Vector
tend += dt
x = [x; tend]

f(x) = exp.(-x.^2)
F(x) = @. √π / 2 * erf(x) # F = ∫f(t)dt where t ∈[0,x]
u0 = 0.0

# f(x) = x
# F(x) = (x.^2) /2 # F = ∫f(t)dt where t ∈[0,x]
# u0 = 0.0

# f(x) = x.^2
# F(x) = (x.^3) /3 # F = ∫f(t)dt where t ∈[0,x]
# u0 = 0.0

fun(x,p,t) = f(x)  # ∂F/∂x = f(x)
tspan = [t0, tend]
prob = ODEProblem(fun,u0,tspan)
##
FT = F(x)
FTrap = cumul_integrate(x,f(x))
FTrap .-= FTrap[1]
####  Errors

k = 1
Fs1 = cIsimpson1Dv(f(x),x; k = k, inv = false,I0 = 0.0)
lable = string("M=","Simp",k)
pe = plot(x,(Fs1 - FT) ./ FT,line=(2,:auto),label=lable)

k = 2
Fs2 = cIsimpson1Dv(f(x),x; k = k, inv = false,I0 = 0.0)
lable = string("M=","Simp",k)
pe = plot!(x,(Fs2 - FT) ./ FT,line=(2,:auto),label=lable)

k = 3
Fs3 = cIsimpson1Dv(f(x),x; k = k, inv = false,I0 = 0.0)
lable = string("M=","Simp",k)
pe = plot!(x,(Fs3 - FT) ./ FT,line=(2,:auto),label=lable)

k = 4
Fs4 = cIsimpson1Dv(f(x),x; k = k, inv = false,I0 = 0.0)
lable = string("M=","Simp",k)
pe = plot!(x,(Fs4 - FT) ./ FT,line=(2,:auto),label=lable)

k = 5
Fs5 = cIsimpson1Dv(f(x),x; k = k, inv = false,I0 = 0.0)
lable = string("M=","Simp",k)
pe = plot!(x,(Fs5 - FT) ./ FT,line=(2,:auto),label=lable)

lable = string("M=","Trapz")
pe = plot!(x,(FTrap - FT) ./ FT,line=(2,:auto),label=lable)


data = [x FT FTrap Fs1 Fs2 Fs3 Fs4 Fs5]

legendlocal = :topright

lable = string("M=","Theory")
pF = plot(x,FT,line=(2,:auto),label=lable,legend=legendlocal)
k = 1
lable = string("M=","Simp",k)
pF = plot!(x,Fs1,line=(2,:auto),label=lable,legend=legendlocal)
k = 2
lable = string("M=","Simp",k)
pF = plot!(x,Fs2,line=(2,:auto),label=lable,legend=legendlocal)
lable = string("M=","Trapz")
pF = plot!(x,FTrap,line=(2,:auto),label=lable,legend=legendlocal)

println("nx=",length(x))
pe

# methods = Euler()
# sol = solve(prob,methods,tstops=x)
# lable = string("M=",methods)
# plot!(x,sol,line=(2,:auto),label=lable)
# #
# # methods = SSPRK22()
# # sol = solve(prob,methods,tstops=x)
# # lable = string("M=",methods)
# # plot!(x,sol,label=lable,line=(2,:auto))
#
# methods = RosenbrockW6S4OS()
# sol = solve(prob,methods,tstops=x)
# lable = string("M=",methods)
# plot!(x,sol,label=lable,line=(2,:auto),legend=false)
#
# methods = DGLDDRK73_C()
# sol = solve(prob,methods,tstops=x)
# lable = string("M=",methods)
# plot!(x,sol,label=lable,line=(2,:auto),legend=false)
