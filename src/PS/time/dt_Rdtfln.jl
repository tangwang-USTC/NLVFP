"""
  Limit the timestep to satisfy the perturbation assumption of 
  the variables `∂ₜfLn` of all spices by `dt * (∂ₜfLn / fLn) = ratio_Rdtfln ≪ 1`.

  Inputs:
    dtk: = τ₀ / nτ
    Rdtfln3: = ∂ₜfLn[3] / fLn[3], The value of `Rdtfln` at the third grid from the left endpoint.
    dtIK: dtIa, dtKa

  Outputs:
    dtk = dt_dtfln(dtk,Rdtfln3;ratio_Rdtfln=ratio_Rdtfln)
"""

function dt_dtfln(dtk,Rdtfln3;ratio_Rdtfln=ratio_Rdtfln)

    Rdtfln = maximum(abs.(Rdtfln3))
    if Rdtfln ≥ epsT1000
        dtk1 = 1dtk
        dtk = min(dtk, ratio_Rdtfln / Rdtfln)
        dtk ≥ dtk1 || @warn("The timestep is decided by `dt_fln`!")
        sdefghjnm
    end
    return dtk
end
