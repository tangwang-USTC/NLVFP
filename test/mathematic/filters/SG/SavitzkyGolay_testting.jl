
using SavitzkyGolay

t = LinRange(-4, 4, 500) |> Vector{Float64}
# y = exp.(-t.^2) .+ 0.05 .* (1.0 .+ randn(length(t)))
# y = sg.y
sg = savitzky_golay(y, 9, 2)
plot(t, [y sg.y], label=["Original signal" "Filtered signal"])
