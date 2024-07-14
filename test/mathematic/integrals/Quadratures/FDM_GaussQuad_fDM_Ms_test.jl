using Plots
using GaussQuadrature, ChebyshevApprox, FastTransforms

gr()
# plotlyjs()
# pyplot()
L = 0
u = 8
vGdom = [1e-15,10]
if u == 0
    f(x) = exp.(-x.^2)
else
    f(v) = @. (exp(-u^2 - v^2) * (1 + 2L) * √(2Pi) * besseli(1/2+L,2u * v)) / √(2u * v)
end
dxdv = 4 / Pi^0.5
fn0(x) = dxdv * x.^(2+0) .* f.(x)
fK2(x) = dxdv * x.^(2+2) .* f.(x)

"""
  gqconvergence(f,vGdom,nc0)
"""

function gqconvergence(f::Function,vGdom::Vector{T}, nc0::Int) where{T}

    na = zeros(nc0-4)
    Ka = zeros(nc0-4)
    na1 = zeros(nc0-4)
    Ka1 = zeros(nc0-4)
    na2 = zeros(nc0-4)
    Ka2 = zeros(nc0-4)
    for i in 5:nc0
        na[i-4], Ka[i-4] = gaussquad(vGdom,i)
        na1[i-4], Ka1[i-4] = FDM12(vGdom,i;order=1)
        na2[i-4], Ka2[i-4] = FDM12(vGdom,i;order=2)
    end
    nagk = quadgk(fn0,vGdom[1],vGdom[2])[1]
    Kagk = quadgk(fK2,vGdom[1],vGdom[2])[1]
    @show na - na1
    title = string("M(s) = ∫₀^∞ (v²⁺ˢ fDM(v)) dv")
    xlabel = ("nᵥ,û = u/vₜₕ=",u)
    ylabel = "Log(Abs(M(s) - Ms_theory))"
    label=string("GaussQuad,na(s=0)")
    pp = plot(5:nc0,abs.(na .- nagk).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    label=string("GaussQuad,Ka(s=2)")
    pp = plot!(5:nc0,abs.(Ka .- Kagk).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    label=string("trapez,na(s=0)")
    pp = plot!(5:nc0,abs.(na1 .- nagk).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    label=string("trapez,Ka(s=2)")
    pp = plot!(5:nc0,abs.(Ka1 .- Kagk).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    label=string("Simpson,na(s=0)")
    pp = plot!(5:nc0,abs.(na2 .- nagk).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    label=string("Simpson,Ka(s=2)")
    pp = plot!(5:nc0,abs.(Ka2 .- Kagk).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    yaxis!(:log)
    display(pp)
    na, Ka
end

function gaussquad(f::Function,vGdom::Vector{T}, nc0::Int) where{T}

    # vcc = clenshawcurtisnodes(T,nc0)
    v = vCmapping(clenshawcurtisnodes(T,nc0),vGdom[1],vGdom[2];isinv=true)
    wc = clenshawcurtisweights(chebyshevmoments1(T, nc0))
    dxdv = - 2/sqrtpi * (v[1] - v[end])
    na = dxdv * dot(wc, (v .^2 .* f(v)))
    Ka = dxdv * dot(wc, (v .^4 .* f(v)))
    return na, Ka
end

function FDM12(vGdom::Vector{T}, nc0::Int;order::Int=1) where{T}

    v = range(vGdom[1],vGdom[2],nc0) |> Vector{T}
    fv = fn0(v)
    if order == 1
        na = trapezoidal1D(v, fv)
        fv = fK2(v)
        Ka = trapezoidal1D(v, fv)
        return na, Ka
    elseif order == 2
        na = simpson1Dv(v, fv)
        fv = fK2(v)
        Ka = simpson1Dv(v, fv)
        return na, Ka
    end
end
