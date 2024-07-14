
"""
  Eq.: mu + x - x^3 = 0

"""

using BifurcationKit, Parameters, Plots

function Fsl(X, p)
    @unpack r, μ, ν, c3 = p
    u, v = X

    ua = u^2 + v^2

    [
        r * u - ν * v - ua * (c3 * u - μ * v)
        r * v + ν * u - ua * (c3 * v + μ * u)
    ]
end

par_sl = (r = 0.1, μ = 0., ν = 1.0, c3 = 1.0)
u0 = zeros(2)
prob = BifurcationProblem(Fsl, u0, par_sl, (@lens _.r))

opts = ContinuationPar()
br = continuation(prob, PALC(), opts, bothside = true)

br_po = continuation(br, 2, opts,
        PeriodicOrbitOCollProblem(20, 5)
        )

br_po[1]
ps = plot(br, br_po, line=(2,:auto), branchlabel = ["equilibria", "periodic orbits"])
display(ps)

sol = get_periodic_orbit(br_po, 10)
pu = plot(sol.t, sol[1,:], line=(2,:auto), label = "u", xlabel = "time")
pv = plot!(sol.t, sol[2,:], line=(2,:auto), label = "v", xlabel = "time")

br_po = continuation(br, 2, opts,
        PeriodicOrbitOCollProblem(20, 5);
        plot = true,
        plot_solution = (x, par; k...) -> begin
                # par is a Named tuple which contains
                # the problem for computing periodic orbits
                # and the value of the parameter at the current step
                sol = get_periodic_orbit(par.prob, x, par.p)
                plot!(sol.t, sol.u'; xlabel = "time", label="", k...)
        end
        )

# br_po = continuation(br, 2, opts,
#         PeriodicOrbitOCollProblem(20, 5);
#         plot = true,
#         )

