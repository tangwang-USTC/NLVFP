
"""
  Eq.: mu + x - x^3 = 0
"""

using BifurcationKit, Plots

# F(x, p) = @. p.μ + x - x^3/3

function F(x,p)

  @. p.μ + x - x^3/3
end

x0 = -0.1
μ0 = 0.4
pM = [-1.0, 1.0]
prob = BifurcationProblem(F, [x0], (μ = μ0,), (@lens _.μ);
        record_from_solution = (x,p) -> (x = x[1]))
opts = ContinuationPar(p_min = pM[1], p_max = pM[2])
br = continuation(prob, PALC(), opts)
plot(br)

