"""
  Wave: A = A0 * sin(ω * t - k * x)
          = A0 * sin(2π(t/T - x/λ)), A0 = [E0, B0]
       ω: = 2π/T = 2π * freq, circle frequency
       k: = 2π/λ, wave vector
       A0: Initial Amplitude
       velocity: λ / T = λ * freq = ω / k = c₀.
     Parameters (t, x, B, Jq) is normalized by (T, λ , 1/c₀, n₀c₀e), so the wave equation will be:
       E = E * sin(2π * (t - x))
       B = B * sin(2π * (t - x))

    The Maxwell equations are:
      ∂t/∂B = - ∇ × E(x,y)
      ∂t/∂E = + ∇ × B(x,y,) - Jq
"""

## initial and boundrary conditions
# Example202: Gaussian Laser pusle
function laser_pulse_gauss(x_min, width, duration)
  function f(t, xs...)
    dims = length(xs)
    if dims == 1
      x = xs[1]
      # return exp(-((t - (x - x_min) - duration)/duration)^2) * sin(2.0*pi*(t - x))
      # return exp(-((t - (x - x_min) - duration)/duration)^2)
      return exp(-0.5 * ((t - duration)/duration)^2)
    elseif dims == 2
      x = xs[1]
      y = xs[2]
      return exp(-(y/width)^2)*exp(-((t - (x - x_min) - duration)/duration)^2) * sin(2.0*pi*(t - x))
    else
    end
  end
  return f
end
## Plane Electromegnetic wave
  # # Example201: Plane electromegnetic wave, TEM
"""
 Wave: (T, λ) with λ / T = λ * freq = ω / k = c₀. With (t, x, B) is normalized by (T, λ , 1/c₀)
   ω / k = 1

   Models:
    2D
    1D: ∂ₜE2 = - ∂x₁B3, E2 = A0 * sin(t - x) where A0 = E0
        ∂ₜB3 = + ∂x₁E2, B3 = A0 * sin(t - x) where A0 = B0

"""
function EMWplane(ω, A0,ϕ₀)
  function f(t, xs...)
    dims = length(xs)
    if dims == 1
      x = xs[1]
      # return A0 * sin(2π * ω * (t - x) + ϕ₀)
      return A0 * cos(2π * ω * (t - x) + ϕ₀)
      # return A0 * sin(2π * ω * t + ϕ₀)
    elseif dims == 2
      x = xs[1]
      y = xs[2]
      return A0 * sin(2π * ω * (t - x) + ϕ₀)
    end
  end
  return f
end
