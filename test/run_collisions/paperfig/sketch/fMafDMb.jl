
xx = 0:0.1:14

xlabel = "v"

label = "species a"
nna = 1.0
uua = 0e-3
vaath = 7
flna = nna * exp.(- (xx .- uua).^2 / vaath^2)
plot(xx, flna, fill = (0, :gray),xlabel=xlabel,label=label,ticks=nothing)


label = "species b"
nnb = 1.2
uub = 2.3
vbbth = 1
flnb = nnb * exp.(- (xx .- uub).^2 / vbbth^2)
pfab = plot!(xx, flnb, fill = (0, :green),label=label,ticks=nothing)
display(pfab)

label = "species a"
plot(xx, flna, fill = (0, :gray),xlabel=xlabel,label=label,ticks=nothing)
label = "species b"
plot!(xx, flnb, fill = (0, :green),label=label,ticks=nothing)
savefig(string(file_Ms_fold,"_sketchfab.png"))
