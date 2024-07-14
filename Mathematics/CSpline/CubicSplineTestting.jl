


vG0 = vGk[nvlevel0]
testing = :triangulars
# telse = :polynomials

n = -1    # n ∈ [0, N⁺]

if testing == :polynomials
    vG0[1] = vGk[1] = 2e-16
    if n == 0
        yy = - ones(nc0)     # ` = fLn[nvlevel0]`
        d0yy = - ones(nck)     # ` = fLn[nvlevel0]`
        d1yy = zeros(nck)
        d2yy = zeros(nck)
    elseif n > 0
        if n == 1
            yy = - vG0
            d0yy = - vGk     # ` = fLn[nvlevel0]`
            d1yy = - ones(nck)
            d2yy = zeros(nck)
        elseif n == 2
            yy = - vG0.^n     # ` = fLn[nvlevel0]`
            d0yy = - vGk.^n     # ` = fLn[nvlevel0]`
            d1yy = - n * vGk
            d2yy = - n * (n-1) * ones(nck)
        else
            yy = - vG0.^n     # ` = fLn[nvlevel0]`
            d0yy = - vGk.^n     # ` = fLn[nvlevel0]`
            d1yy = - n * vGk.^(n - 1)
            d2yy = - n * (n-1) * vGk.^(n-2)
        end
    else
        if n == -1
            yy = - 1 ./ vG0
            d0yy = - 1 ./ vGk     # ` = fLn[nvlevel0]`
            d1yy = 1 ./ vGk.^2
            d2yy = - 2 ./ vGk.^3
        else
            sgdrf
        end
    end
elseif testing == :triangulars
    yy = - cos.(vG0)     # ` = fLn[nvlevel0]`
    d0yy = - cos.(vGk)     # ` = fLn[nvlevel0]`
    d1yy = sin.(vGk)
    d2yy = cos.(vGk)
else
    rjh
end

bcyy = (:Neumann,[d1yy[1],d1yy[end]])

ddfLn,dfLn,fLn = zeros(nck), zeros(nck), zeros(nck)
ddfLn,dfLn,fLn = dfvLCS3p4(ddfLn,dfLn,fLn,yy,va,nc0,ocp,nvlevel0,bcyy;method3M=method3M)

title = string("Cubic spline interpolation of `f(v)=v^n`, n=",n)
xlabel = string("vGk")
ylabel = string("Error normalized by `eps(1.0)`")
label = string("δf")
pyy = plot(vGk,(fLn - d0yy)*neps,label=label,title=title)
label = string("δdf")
pdyy = plot(vGk,(dfLn - d1yy)*neps,label=label,ylabel=ylabel)
label = string("δddf")
pddyy = plot(vGk,(ddfLn - d2yy)*neps,label=label,xlabel=xlabel)
display(plot(pyy,pdyy,pddyy,layout=(3,1)))
