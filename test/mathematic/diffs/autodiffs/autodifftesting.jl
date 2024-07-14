
using ForwardDiff, ReverseDiff
using DiffTests

f(v) = exp.(-v.^2)
df(v) = -2v .* exp.(-v.^2)
ddf(v) = (-2 .+ 4v.^2) .* exp.(-v.^2)

vs = 0:0.1:10
dvf_model(v) = ForwardDiff.derivative.(f,v)
dvf = dvf_model(vs)
ddvf = ForwardDiff.derivative.(dvf_model,vs)


errdf = (df(vs) - dvf) / eps(Float64)
errddf = (ddf(vs) - ddvf) / eps(Float64)

label = string("err_df")
pdf = plot(vs,errdf,label=label)
label = string("err_ddf")
pddf = plot!(vs,errddf,label=label)
