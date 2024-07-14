using Format

fmtf = generate_formatter( "%1.1e" )   # "%e", "%'e'", "%1.1e", [d,f,s,e]
fmtf2 = generate_formatter( "%1.2e" )
fmtf4 = generate_formatter( "%1.4e" )

nx = 150
t0 = 0.0
A0 = 1
T = 20
ω = 1/8
ϕ₀ = 0.0
ctx = 0.5
dt = 0.002
isource = 30
nt_max = 272
Ebound = zeros(4)
Bbound = zeros(4)
# ABC is needed for LHS of the problem space
function FDTD1d()
    E = zeros(nx)
    B = zeros(nx)
    t = t0
    for k in 1: nt_max
        t += k * dt
        E[isource] += A0 * sin(2π * ω * t + ϕ₀)
        # Update_E
        E[2:nx] += - ctx * (B[2:nx] - B[1:nx-1])
        # Boundary of Ey
        E[1] = Ebound[2]
        Ebound[2] = Ebound[1]
        Ebound[1] = E[2]
        E[nx] = Ebound[4]
        Ebound[4] = Ebound[3]
        Ebound[3] = E[nx - 1]
        # Update_B
        B[1:nx-1] += - ctx * (E[2:nx] - E[1:nx-1])
        # Boundary of Bz
        B[1] = Bbound[2]
        Bbound[2] = Bbound[1]
        Bbound[1] = B[2]
        B[nx] = Bbound[4]
        Bbound[4] = Bbound[3]
        Bbound[3] = B[nx - 1]
    end
    return E, B
end
E, B = FDTD1d()
# fmtf2.(E[isource-3:isource+3])
pE = plot(B)
pB = plot(E)
plot(pE,pB,layout=(2,1))
