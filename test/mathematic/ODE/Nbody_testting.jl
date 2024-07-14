using IRKGaussLegendre, DifferentialEquations
# using GeometricIntegrators
using Plots, LinearAlgebra, LaTeXStrings


function NbodyODE!(du,u,Gm,t)
    N = length(Gm)
    du[1,:,:] .= 0
    for i in 1:N
       qi = u[2,:,i]
       Gmi = Gm[i]
       du[2,:,i] = u[1,:,i]
       for j in (i+1):N
          qj = u[2,:,j]
          Gmj = Gm[j]
          qij = qi - qj
          auxij = (qij[1]*qij[1]+qij[2]*qij[2]+qij[3]*qij[3])^(-3/2)
          du[1,:,i] -= Gmj*auxij*qij
          du[1,:,j] += Gmi*auxij*qij
       end
    end

   return
end

# Energy error
function NbodyEnergy(u,Gm)
    N = length(Gm)
    zerouel = zero(eltype(u))
    T = zerouel
    U = zerouel
    for i in 1:N
       qi = u[2,:,i]
       vi = u[1,:,i]
       Gmi = Gm[i]
       T += Gmi*(vi[1]*vi[1]+vi[2]*vi[2]+vi[3]*vi[3])
       for j in (i+1):N
          qj = u[2,:,j]  
          Gmj = Gm[j]
          qij = qi - qj
          U -= Gmi*Gmj/norm(qij)
       end
    end
   1/2*T + U
end


Gm = [5, 4, 3]
N=length(Gm)
q=[1,-1,0,-2,-1,0,1,3,0]
v=zeros(size(q))
q0 = reshape(q,3,:)
v0 = reshape(v,3,:)
u0 = Array{Float64}(undef,2,3,N)
u0[1,:,:] = v0
u0[2,:,:] = q0
E0 = NbodyEnergy(u0,Gm)            # The initial total energy
tspan = (0.0,83.0)
Nstep = 1e4

is_geo = false
Algtype = :explicit               # (=:explicit, default)
Algtype = :implicit
dtmax = 5e-1
is_cb = false

if is_geo
    prob = GeometricIntegrators.ODEProblem(NbodyODE!,tspan,Nstep,u0)
    if Algtype == :explicit
        algn, algname = ExplicitEuler(), ExplicitEuler     # (Δt, RΔE, Nt, sol_t) ~ (3.04e-1,-9.25, 1.03k, 0.23s)
        # algn, algname = RK41(), RK41     # (Δt, RΔE, Nt, sol_t) ~ (3.04e-1,-9.25, 1.03k, 0.23s)
    elseif Algtype == :implicit
        algn, algname = IRKGL16(), IRKGL16     # (Δt, RΔE, Nt, sol_t) ~ (3.04e-1,-9.25, 1.03k, 0.23s)
    end
    alg = Integrator(prob, algn)
    if is_cb 
        function energyC(resid,u,p,t)
    
            resid = map(x->NbodyEnergy(x,Gm), sol1.u) / E0 - 1
            @show 1,resid
            resid
        end
    
        cb = ManifoldProjection(energyC)
        @time sol1 = solve(prob,alg,dtmax = dtmax,adaptive=true, reltol=1e-12, abstol=1e-12,callback=cb)
    else
        @time sol1 = solve(prob,alg,dtmax = dtmax,adaptive=true, reltol=1e-12, abstol=1e-12)
        @time sol1 = GeometricIntegrators.integrate(prob,alg)
    end
else
    if Algtype == :implicit
        #### `dtmax = 1e-2` default
        alg, algname = IRKGL16(), IRKGL16             # (Δt, RΔE, Nt, sol_t) ~ (3.04e-1,-9.25, 1.03k, 0.23s)
    
        # alg, algname = Rodas4(),Rodas4                # (Δt, RΔE, Nt, sol_t) ~ (1.70e-3,-7.05, 405.37k, 33.32s),Rosenbrock
        # alg, algname = Velds4(),Velds4                # (Δt, RΔE, Nt, sol_t) ~ (1.11e-3,-7.68, 286.40k, 23.53s),Rosenbrock
        # alg, algname = Ros4LStab(),Ros4LStab          # (Δt, RΔE, Nt, sol_t) ~ (1.14e-3,-7.63, 233.54k, 18.62s),Rosenbrock
        # alg, algname = RosShamp4(),RosShamp4          # (Δt, RΔE, Nt, sol_t) ~ (1.26e-3,-7.43, 251.91k, 20.01s),Rosenbrock
        # alg, algname = GRK4T(),GRK4T                  # (Δt, RΔE, Nt, sol_t) ~ (1.56e-3,-7.47, 247.15k, 20.76s),Rosenbrock
        # alg, algname = Veldd4(),Veldd4                # (Δt, RΔE, Nt, sol_t) ~ (1.56e-3,-7.68, 260.15k, 20.13s),Rosenbrock
        # alg, algname = Rodas42(),Rodas42              # (Δt, RΔE, Nt, sol_t) ~ (1.59e-3,-6.88, 207.92k, 17.46s),Rosenbrock
        # alg, algname = GRK4A(),GRK4A                  # (Δt, RΔE, Nt, sol_t) ~ (1.53e-3,-7.79, 163.37k, 12.68s),Rosenbrock
        # alg, algname = Rodas4P(),Rodas4P              # (Δt, RΔE, Nt, sol_t) ~ (1.78e-3,-7.33, 155.42k, 11.99s),Rosenbrock
        # alg, algname = ESDIRK436L2SA2(),ESDIRK436L2SA2# (Δt, RΔE, Nt, sol_t) ~ (3.00e-3,-7.60, 124.54k, 12.43s)
        # alg, algname = ESDIRK437L2SA(),ESDIRK437L2SA  # (Δt, RΔE, Nt, sol_t) ~ (3.33e-3,-7.56, 64.07k, 8.02s)
        # alg, algname = Rodas5P(),Rodas5P              # (Δt, RΔE, Nt, sol_t) ~ (3.33e-3,-7.70, 43.00k, 4.20s),Rosenbrock
        # alg, algname = ESDIRK54I8L2SA(),ESDIRK54I8L2SA# (Δt, RΔE, Nt, sol_t) ~ (3.33e-3,-7.47, 38.76k, 5.87s)
        # alg, algname = ESDIRK547L2SA2(),ESDIRK547L2SA2# (Δt, RΔE, Nt, sol_t) ~ (3.33e-3,-7.55, 33.74k, 4.36s)
        # alg, algname = Rodas5(),Rodas5                # (Δt, RΔE, Nt, sol_t) ~ (3.33e-3,-7.43, 30.97k, 2.94s),Rosenbrock
        # alg, algname = RadauIIA5(),RadauIIA5          # (Δt, RΔE, Nt, sol_t) ~ (3.33e-3,-6.49, 27.29k, 1.03s), FIRK
    
        # #### Pre-Built Stiffness Detecting and Auto-Switching Algorithms
        # alg, algname = AutoTsit5(Rosenbrock23()),AutoTsit5  
                                                        # (Δt, RΔE, Nt, sol_t) ~ (3.33e-3,-8.39, 35.47k, 0.95s)
        # alg, algname = ROS34PW1a(),ROS34PW1a          # (Δt, RΔE, Nt, sol_t) ~ (3.33e-3,-8.39, 35.47k, 0.95s)
    

        # ################################# not convergent algorithm

        # alg, algname = ImplicitEuler(), ImplicitEuler # (Δt, RΔE, Nt, sol_t) ~ (1.23e-6,-6.44, 666.67k, 17.0s, small `dt`)
        # alg, algname = SDIRK2(),SDIRK2                # (Δt, RΔE, Nt, sol_t) ~ (8.08e-5,-1.22, 889.16k, 37.35s, not correct)
        # alg, algname = RadauIIA3(),RadauIIA3          # (Δt, RΔE, Nt, sol_t) ~ (8.07e-5,-7.46, 833.50k, 40.11s, small `dt`), FIRK
        # alg, algname = Trapezoid(), Trapezoid         # (Δt, RΔE, Nt, sol_t) ~ (1.77e-4,-5.56,202.80k,7.26s, wrong picture)
        # alg, algname = Rodas3(),Rodas3                # (Δt, RΔE, Nt, sol_t) ~ (1.80e-4,-5.91, 999.00k, 80.25s, wrong picture),Rosenbrock
        # alg, algname = TRBDF2(),TRBDF2                # (Δt, RΔE, Nt, sol_t) ~ (1.70e-3,+0.19, 125.29k, 5.93s, big error)
        # alg, algname = Kvaerno3(),Kvaerno3            # (Δt, RΔE, Nt, sol_t) ~ (1.08e-3,+0.48, 474.16k, 30.20s, not correct)
        # alg, algname = ROS3P(),ROS3P                  # (Δt, RΔE, Nt, sol_t) ~ (1.20e-3,-7.06, 999.20k, 82.58s, wrong picture),Rosenbrock
        # alg, algname = KenCarp3(),KenCarp3            # (Δt, RΔE, Nt, sol_t) ~ (1.90e-3,-0.24, 57.81k, 5.67s, not correct)
        # alg, algname = Cash4(),Cash4                  # (Δt, RΔE, Nt, sol_t) ~ (3.33e-3,-0.14, 42.10k, 3.92s, not good)
        # alg, algname = Kvaerno4(),Kvaerno4            # (Δt, RΔE, Nt, sol_t) ~ (3.33e-3,-2.12, 22.44k, 1.75s, worse)
        # alg, algname = Hairer42(),Hairer42            # (Δt, RΔE, Nt, sol_t) ~ (5.84e-3,-0.95, 19.64k, 2.18s, not good), dtmax = 1e-1
        # alg, algname = Hairer4(),Hairer4              # (Δt, RΔE, Nt, sol_t) ~ (3.33e-3,-2.02, 17.94k, 1.98s, not good)
        # alg, algname = KenCarp4(),KenCarp4            # (Δt, RΔE, Nt, sol_t) ~ (6.37e-2,+2.80, 13.34k, 1.54s, worse), dtmax = 1e-1
        # alg, algname = Kvaerno5(),Kvaerno5            # (Δt, RΔE, Nt, sol_t) ~ (1.55e-2,-0.58, 286.02k, 84.00s, not good)
        # alg, algname = KenCarp5(),KenCarp5            # (Δt, RΔE, Nt, sol_t) ~ (2.40e-2,-0.11, 368.80k, 145.54s, not good)
        # alg, algname = KenCarp47(),KenCarp47          # (Δt, RΔE, Nt, sol_t) ~ (2.40e-2,-0.11, 368.80k, 145.54s, not good)
        # alg, algname = KenCarp58(),KenCarp58          # (Δt, RΔE, Nt, sol_t) ~ (2.40e-2,-0.11, 368.80k, 145.54s, not good)
    elseif Algtype == :explicit
        # # # alg, algname = Ralston(), Ralston # (Δt, RΔE) ~ (2e-6, ~, wrong picture)   # The lower order algorithms such as `Ralston`, `Midpoint` and `Euler` need so small time steps that are not suitable.
        # alg, algname = RK4(), RK4             # (Δt, RΔE, Nt, sol_t) ~ (1.13e-3,-7.76, 295.78k, 3.66s) 
        # alg, algname = OwrenZen4(), OwrenZen4 # (Δt, RΔE, Nt, sol_t) ~ (1.44e-3,-8.02, 205.42k, 2.51s)
        # alg, algname = OwrenZen5(), OwrenZen5 # (Δt, RΔE, Nt, sol_t) ~ (4.66e-3,-8.01, 58.91k, 1.07s)
        # alg, algname = ExplicitRK(tableau = constructDormandPrince()),ExplicitRK    
        #                                       # (Δt, RΔE, Nt, sol_t) ~ (7.35e-3,-8.27, 32.48k, 0.67s)
        # alg, algname = Stepanov5(),Stepanov5  # (Δt, RΔE, Nt, sol_t) ~ (7.97e-3,-8.09, 40.26k, 1.19s)
        # alg, algname = Tsit5(), Tsit5         # (Δt, RΔE, Nt, sol_t) ~ (8.86e-3,-8.11, 34.66k, 0.56s)
        # alg, algname = DP5(), DP5             # (Δt, RΔE, Nt, sol_t) ~ (8.91e-3,-8.24, 26.84k, 0.50s)
        # alg, algname = BS5(), BS5             # (Δt, RΔE, Nt, sol_t) ~ (1.34e-2,-8.69, 22.28k, 0.47s)
        # alg, algname = Vern6(),Vern6          # (Δt, RΔE, Nt, sol_t) ~ (1.85e-2,-7.76, 18.18k, 0.53s)
        # alg, algname = Vern7(),Vern7          # (Δt, RΔE, Nt, sol_t) ~ (2.80e-2,-8.67, 9.89k, 0.51s)
        # alg, algname = TanYam7(),TanYam7      # (Δt, RΔE, Nt, sol_t) ~ (3.41e-2,-7.57, 7.93k, 0.45s)
        # alg, algname = TsitPap8(),TsitPap8    # (Δt, RΔE, Nt, sol_t) ~ (5.58e-2,-8.99, 4.58k, 0.17s)
        # alg, algname = Vern8(),Vern8          # (Δt, RΔE, Nt, sol_t) ~ (5.69e-2,-8.42, 4.56k, 0.36s)
        # alg, algname = DP8(),DP8              # (Δt, RΔE, Nt, sol_t) ~ (6.51e-2,-8.40, 4.58k, 0.23s)
        # alg, algname = Vern9(),Vern9          # (Δt, RΔE, Nt, sol_t) ~ (6.83e-2,-9.14, 3.73k, 0.18s)
        alg, algname = Feagin10(),Feagin10    # (Δt, RΔE, Nt, sol_t) ~ (6.56e-2,-6.59, 3.43k, 0.17s, fewer difference)
        alg, algname = Feagin12(),Feagin12    # (Δt, RΔE, Nt, sol_t) ~ (8.89e-2,-8.34, 2.36k, 0.17s)
        alg, algname = Feagin14(),Feagin14    # (Δt, RΔE, Nt, sol_t) ~ (9.33e-2,-7.57, 2.04k, 0.20s)
        # # The following are adaptive order, adaptive step size extrapolation methods:
        # alg, algname = ExtrapolationMidpointHairerWanner(),ExtrapolationMidpointHairerWanner  
        #                                       # (Δt, RΔE, Nt, sol_t) ~ (4.22e-2,-7.81, 5.88k, 0.35s)
        # alg, algname = AitkenNeville(),AitkenNeville      # Euler extrapolation using Aitken-Neville with the Romberg Sequence.
        #                                       # (Δt, RΔE, Nt, sol_t) ~ (2.26e-1,-7.24, 3.36k, 2.81s)
        # alg, algname = ExtrapolationMidpointDeuflhard(),ExtrapolationMidpointDeuflhard    
        #                                       # (Δt, RΔE, Nt, sol_t) ~ (1.79e-1,-7.61, 3.91k, 0.34s)
        alg, algname = IRKGL16(), IRKGL16     # (Δt, RΔE, Nt, sol_t) ~ (3.04e-1,-9.25, 1.03k, 0.23s)
    
        alg, algname = JVODE_Adams(),JVODE_Adams          # (Δt, RΔE, Nt, sol_t) ~ (8.44e-5,-9.39, 30.95k, 0.21s)           , AB_adap
        # alg, algname = VCAB4(),VCAB4                      # (Δt, RΔE, Nt, sol_t) ~ (1.59e-4,-7.65, 21.68k, 2.50s, not good) , AB_adap
        # alg, algname = RDPK3Sp35(), RDPK3Sp35             # (Δt, RΔE, Nt, sol_t) ~ (5.17e-4,-7.32, 436.91k, 6.41s)
        # alg, algname = RDPK3SpFSAL510(), RDPK3SpFSAL510   # (Δt, RΔE, Nt, sol_t) ~ (6.25e-4,-6.55, 554.50k, 2.50s, not good)
        # alg, algname = VCAB5(),VCAB5                      # (Δt, RΔE, Nt, sol_t) ~ (1.50e-3,-7.00, 338.10k, 1.56s, not good) , AB_adap
        # alg, algname = CKLLSRK54_3N_4R(), CKLLSRK54_3N_4R # (Δt, RΔE, Nt, sol_t) ~ (1.77e-3,-7.75, 157.77k, 2.77s)
        # alg, algname = CKLLSRK54_3C_3R(), CKLLSRK54_3C_3R # (Δt, RΔE, Nt, sol_t) ~ (1.87e-3,-7.70, 129.63k, 2.33s)
        # alg, algname = CKLLSRK54_3M_3R(), CKLLSRK54_3M_3R # (Δt, RΔE, Nt, sol_t) ~ (2.05e-3,-7.62, 113.35k, 2.26s)
        # alg, algname = CKLLSRK54_3M_4R(), CKLLSRK54_3M_4R # (Δt, RΔE, Nt, sol_t) ~ (2.17e-3,-7.50, 110.02k, 1.94s)
        # alg, algname = VCABM4(),VCABM4                    # (Δt, RΔE, Nt, sol_t) ~ (2.74e-3,-5.94, 235.43k, 2.18s, not good) , AB_adap
        # alg, algname = RDPK3Sp49(), RDPK3Sp49             # (Δt, RΔE, Nt, sol_t) ~ (3.30e-3,-6.57, 137.81k, 3.92s)
        # alg, algname = RDPK3SpFSAL49(),RDPK3SpFSAL49      # (Δt, RΔE, Nt, sol_t) ~ (4.95e-3,-7.21, 101.80k, 3.00s)
        # alg, algname = VCABM5(),VCABM5                    # (Δt, RΔE, Nt, sol_t) ~ (5.86e-3,-6.61, 52.00k, 0.34s, fewer difference) , AB_adap
        # alg, algname = AN5(),AN5                          # (Δt, RΔE, Nt, sol_t) ~ (6.04e-3,-6.38, 47.74k, 0.23s) , AB_adap
        # alg, algname = ESERK5(), ESERK5                   # (Δt, RΔE, Nt, sol_t) ~ (6.00e-3,-6.58, 32.50k, 9.39s), Stiffness
        # alg, algname = CKLLSRK95_4S(), CKLLSRK95_4S       # (Δt, RΔE, Nt, sol_t) ~ (6.22e-3,-7.78, 33.71k, 1.10s)
        # alg, algname = CKLLSRK85_4M_3R(), CKLLSRK85_4M_3R # (Δt, RΔE, Nt, sol_t) ~ (6.33e-3,-7.83, 50.63k, 1.36s)
        # alg, algname = CKLLSRK85_4P_3R(), CKLLSRK85_4P_3R # (Δt, RΔE, Nt, sol_t) ~ (6.64e-3,-7.95, 37.03k, 1.10s)
        # alg, algname = CKLLSRK85_4C_3R(), CKLLSRK85_4C_3R # (Δt, RΔE, Nt, sol_t) ~ (6.93e-3,-8.25, 36.50k, 1.00s)
        # alg, algname = CKLLSRK65_4M_4R(), CKLLSRK65_4M_4R # (Δt, RΔE, Nt, sol_t) ~ (7.80e-3,-7.76, 44.89k, 0.92s)
        # alg, algname = CKLLSRK75_4M_5R(), CKLLSRK75_4M_5R # (Δt, RΔE, Nt, sol_t) ~ (8.88e-3,-8.04, 28.16k, 0.77s)
        # alg, algname = RDPK3Sp510(), RDPK3Sp510           # (Δt, RΔE, Nt, sol_t) ~ (1.51e-2,-7.68, 19.14k, 0.65s)
        # alg, algname = VCABM(),VCABM                      # (Δt, RΔE, Nt, sol_t) ~ (2.68e-2,-7.80, 15.44k, 0.14s), ode113 , AB_adap
    end
    prob = DifferentialEquations.ODEProblem(NbodyODE!,u0,tspan,Gm)
    if is_cb 
        function energyC(resid,u,p,t)
    
            resid = map(x->NbodyEnergy(x,Gm), sol1.u) / E0 - 1
            @show 1,resid
            resid
        end
    
        cb = ManifoldProjection(energyC)
        @time sol1 = solve(prob,alg,dtmax = dtmax,adaptive=true, reltol=1e-12, abstol=1e-12,callback=cb)
    else
        @time sol1 = solve(prob,alg,dtmax = dtmax,adaptive=true, reltol=1e-12, abstol=1e-12)
    end
end

NNt = length(sol1.t)

xlabel=string(algname)
bodylist = ["Body-1", "Body-2", "Body-3"]
pl = plot(xlabel=xlabel,title="Burrau problem (Adaptive)",aspect_ratio=1)

ulist1 = sol1.u[1:end]
tlist1 = sol1.t[1:end]

for j = 1:3
    xlist  = map(u->u[2,1,j], ulist1)
    ylist  = map(u->u[2,2,j], ulist1)
    plot!(xlist,ylist, label = bodylist[j])   
end  
p1 = plot(pl)

xlabel=string("t")
p2 = plot(ylabel="step size",title="Adaptive step size")
steps1 =sol1.t[2:end]-sol1.t[1:end-1]
plot!(sol1.t[2:end],steps1)


# energy error
setprecision(BigFloat, 256)
u0Big=BigFloat.(u0)
GmBig=BigFloat.(Gm)

E0Big = NbodyEnergy(u0Big,GmBig)            # The initial total energy
ΔE1 = map(x->NbodyEnergy(BigFloat.(x),GmBig), sol1.u)./E0Big.-1
p3 = plot(title="Relative Energy error", xlabel=xlabel, ylabel=L"\Delta E(log10)")
plot!(sol1.t,log10.(abs.(ΔE1)), label="")

p23 = plot(p2,p3,layout=(2,1))

display(plot(p1,p23,layout=(1,2)))

@show (algname, fmtf2(maximum(steps1) / 3), fmtf2(maximum(log10.(abs.(ΔE1[2:end])))), NNt / 1000), fmtf2(dtmax)

# a = map(x->NbodyEnergy(BigFloat.(x),GmBig), sol1.u)
# a1 = map(x->NbodyEnergy(x,Gm), sol1.u)