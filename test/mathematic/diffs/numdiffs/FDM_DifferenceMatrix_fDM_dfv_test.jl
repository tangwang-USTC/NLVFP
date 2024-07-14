using Plots
using GaussQuadrature, ChebyshevApprox, FastTransforms
using SpecialPolynomials

include(joinpath(pathroot,"mathematics\\derivationCD.jl"))

L = 0
u = 1e-0
if u == 0
    f(x) = exp.(-x.^2)
    dft(x) = -2x .* f(x)
elseif u < 1e-7
    hi
else
    f(v) = @. (exp(-u^2 - v^2) * (1 + 2L) / 2 * √(2Pi) * besseli(1/2+L,2u * v)) / √(2u * v)
    dft(v) = @. (exp(-u^2 - v^2) * (1 + 2L) / 2 * √Pi * u *
         ((L .- 2v^2) * besseli(1/2+L,2u * v) + 2u * v * besseli(3/2+L,2u * v))) / (u * v) ^ 1.5
end
dxdv = 4 / Pi^0.5
fn0(x) = dxdv * x.^(2+0) .* f.(x)
fK2(x) = dxdv * x.^(2+2) .* f.(x)
# df0 = zero.(vGk)
# ddf0 = zero.(vGk)
# f0 = f.(vGk)
# f0log = log.(f0)
# ddf0,df0 = dfvLg(ddf0,df0,f0,f0log,vGk,nck,dxdv,L1;isvboundary=false)

"""
 gqconvergence(f,dft,vGdom,nc0)
"""

function gqconvergence(f::Function,dft::Function,vGdom::Vector{T}, nc0::Int;isrelative::Bool=true) where{T}

    # df = zeros(nc0,nc0-4)
    # df1 = zeros(nc0,nc0-4)
    # df2 = zeros(nc0,nc0-4)
    order2 = 3
    # for i in 5:nc0
    #     df[:,i-4] = diffMatrixf(f,vGdom,i)
    #     df1[:,i-4] = diffFDM12(f,vGdom,i;order=order1)
    #     df2[:,i-4] = diffFDM12(f,vGdom,i;order=order2)
    # end
    vG, df = diffMatrixf(f,vGdom,nc0)
    v, dflog, dfCDS,dfSpline = diffFDM12(f,vGdom,nc0;order=order2)
    title = string("∂ᵥf(v)")
    xlabel = string("v̂ = v/vth,û=",u)
    ylabel = "Log(Abs(norm(Dc * f(v) - ∂ᵥf_theory)))"
    if isrelative == 0
        ylabel = "Log(Abs(∂ᵥf - ∂ᵥf_theory))"
        label=string("DiffMatrix,∂ᵥf(v)")
        a = df .- dft(vG)
        pp = plot(vG[2:end-1],abs.(a[2:end-1]),label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
        label=string("CDS(f)")
        a = dfCDS .- dft(v)
        pp = plot!(v[2:end-1],abs.(a[2:end-1]),label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
        label=string("CDS(log(f))")
        a = dflog .- dft(v)
        pp = plot!(v[2:end-1],abs.(a[2:end-1]),label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
        label=string("Spline3")
        a = dfSpline.- dft(v)
        pp = plot!(v[2:end-1],abs.(a[2:end-1]),label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    else
        ylabel = "Log(Abs(∂ᵥf / ∂ᵥf_theory - 1))"
            label=string("DiffMatrix,∂ᵥf(v)")
            a = df ./ dft(vG) .- 1
            pp = plot(vG[2:end-1],abs.(a[2:end-1]).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
            # label=string("CDS(log(f))")
            # a = dflog ./ dft(v) .- 1
            # pp = plot!(v[2:end-1],abs.(a[2:end-1]).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))

            # label=string("CDS(f)")
            # a = dfCDS ./ dft(v) .- 1
            # pp = plot!(v[2:end-1],abs.(a[2:end-1]).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
            # label=string("Spline3")
            # a = dfSpline./ dft(v) .- 1
            # pp = plot!(v[2:end-1],abs.(a[2:end-1]).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    end
    # yaxis!(:log)
    display(pp)
end

"""
  v,dfg,Rdfg = diffMatrixf(f,vGdom,15)
  dfcd = diffFDM12(f,vGdom,15;order=2)
"""

function diffMatrixf(f::Function,vGdom::Vector{T}, nc0::Int;isrelative::Bool=true) where{T}

    # vcc = clenshawcurtisnodes(T,nc0)
    df = zeros(T,nc0)
    v = vCmapping(clenshawcurtisnodes(T,nc0),vGdom[1],vGdom[2];isinv=true)
    dxdv = 2 / (v[1] - v[end])
    # ddfLn,dfLn = dfvLg(ddfLn,df,f.(v),fLnlog,vG,nc0,dxdv,L1
    # df = Dc1n(nc0) * f.(v) * dxdv
    #
    dlogf = Dc1n(nc0) * log.(f.(v)) * dxdv
    df[:] = f.(v) .* dlogf
    df[1] = - 2v[1] * (1 - v[1]^2)
    title = string("∂ᵥf(v) = Dc * f(v)")
    xlabel = "nᵥ"
    if isrelative == 0
        ylabel = "Log(Abs(Dc * f(v) - ∂ᵥf_theory))"
        label=string("DiffMatrix,∂ᵥf(v)")
        a = df .- dft(v)
        pp = plot(v[2:end],abs.(a[2:end]).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    else
        ylabel = "Log(Abs(Dc * f(v) / ∂ᵥf_theory - 1))"
        label=string("DiffMatrix,∂ᵥf(v)")
        a = df ./ dft(v) .- 1
        pp = plot(v[2:end],abs.(a[2:end]).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    end
    yaxis!(:log)
    display(pp)
    return v, df, df ./ dft(v) .- 1
end

function diffFDM12(f::Function,vGdom::Vector{T}, nc0::Int;order::Int=3,isrelative::Bool=true) where{T}

    v = range(vGdom[1],vGdom[2],nc0) |> Vector{T}
    df_CDS = derivationCDS(f(v),v)
    #
    itp = Spline1D(v,f(v);k=order)
    df_Spline3 = Dierckx.derivative(itp,v)
    #
    fv = f.(v)
    dlogf = derivationCDS(log.(fv),v)
    poly = Lagrange(v[2:end-1],dlogf[2:end-1])
    dlogf[[1,end]] = poly.(v[[1,end]])
    df = f.(v) .* dlogf
    df[1] = 0.0
    title = string("∂ᵥf(v) = derivative(f(v))")
    xlabel = "nᵥ"
    if isrelative == 0
        ylabel = "Log(Abs(Dc * f(v) - ∂ᵥf_theory))"
        label=string("CDS(log(f)),∂ᵥf(v)")
        pp = plot(v,abs.(df .- dft(v)).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
        label=string("CDS")
        pp = plot!(v,abs.(df_CDS .- dft(v)).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
        label=string("Spline3")
        pp = plot!(v,abs.(df_CDS .- dft(v)).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    elseif isrelative == 1
        ylabel = "Log(Abs(Dc * f(v) / ∂ᵥf_theory - 1))"
        label=string("derivative,∂ᵥf(v)")
        pp = plot(v,abs.(df ./ dft(v) .- 1).+1e-16,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    end
    # ylabel = "∂ᵥf"
    # label=string("df_Spline3")
    # pp = plot(v,df_Spline3,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    # label=string("df_CDS")
    # pp = plot!(v,df_Spline3,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    # label=string("df_Chebyshev")
    # pp = plot!(v,df,label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    # label=string("df_theory")
    # pp = plot!(v,dft.(v),label=label,xlabel=xlabel,ylabel=ylabel,title=title,line=(3,:auto))
    # yaxis!(:log)
    display(pp)
    return v, df, df_CDS, df_Spline3
end
