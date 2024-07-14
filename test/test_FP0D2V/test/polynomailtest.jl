
using Polynomials, LaurentPolynomials, PuiseuxPolynomials
using SpecialPolynomials

xs = copy(vGk[1:5:end])
ys = copy(fLn[1:5:end])
# Polynomials.jl
fLnpoly = fit(xs,ys) |> p -> round.(coeffs(p),digits=10) |> Polynomial

fLnpoly = fit(ChebyshevT,xs,ys) |> p -> round.(coeffs(p),digits=10) |> ChebyshevT

fLnpoly = Lagrange(xs,ys)  # SpecialPolynomials.jl

fLnpoly = Pol(xs,ys)       # LaurentPolynomials.jl
xn = xs / 2
fLnn = fLnpoly.(xn)
fLnnt = exp.(-xn.^2)

@testset "Fitting" begin
    for P in Ps
        P <: FactoredPolynomial && continue
        xs = range(0, stop = π, length = 10)
        ys = sin.(xs)

        p = fit(P, xs, ys)
        y_fit = p.(xs)
        abs_error = abs.(y_fit .- ys)
        @test maximum(abs_error) <= 0.03

        p = fit(P, xs, ys, 2)
        y_fit = p.(xs)
        abs_error = abs.(y_fit .- ys)
        @test maximum(abs_error) <= 0.03

        # Test weighted
        for W in [1, ones(size(xs)), diagm(0 => ones(size(xs)))]
            p = fit(P, xs, ys, 2, weights = W)
            @test p.(xs) ≈ y_fit
        end


        # Getting error on passing Real arrays to polyfit #146
        xx = Real[20.0, 30.0, 40.0]
        yy = Real[15.7696, 21.4851, 28.2463]
        fit(P, xx, yy, 2)

        # issue #214 --  should error
        @test_throws ArgumentError fit(Polynomial, rand(2,2), rand(2,2))

        # issue #268 -- inexacterror
        @test fit(P, 1:4, 1:4, var=:x) ≈ variable(P{Float64}, :x)
        @test fit(P, 1:4, 1:4, 1, var=:x) ≈ variable(P{Float64}, :x)

    end

    f(x) = 1/(1 + 25x^2)
    N = 250
    xs = [cos(j*pi/N) for j in N:-1:0]
    q = fit(ArnoldiFit, xs, f.(xs))
    @test maximum(abs, q(x) - f(x) for x ∈ range(-1,stop=1,length=500)) < 10eps()
    q = fit(ArnoldiFit, xs, f.(xs), 100);
    @test maximum(abs, q(x) - f(x) for x ∈ range(-1,stop=1,length=500)) < sqrt(eps())


    f(x) = exp(-x)
    nvG = 150
    vc0 = vccn(nvG)
    xs = vCmapping(vc0,vGdom[1],vGdom[2];isinv=true)
    xs = vGk
    q = fit(ArnoldiFit,xs,dfLn)
    q = fit(LaurentPolynomial,xs,dfLn)
    q = fit(SparsePolynomial,xs,dfLn)
    q = fit(RationalFunction,xs,dfLn,91,91)
    # degreef = 5
    # q = fit(xs,dfLn,degreef )
    label = string("Dfcheby")
    pf1 = plot(xs,dfLn - dfLnt,label=label,line=(2,:auto))
    label = string("Dfpoly")
    pf1 = plot!(xs,q.(xs) - dfLnt,label=label,line=(2,:auto))
    display(pf1)
    @show norm((q.(xs) - dfLnt))


    # test default   (issue  #228)
    fit(1:3,  rand(3))

    # weight test (PR #291)
    # we specify w^2.
    x = range(0, stop=pi, length=30)
    y = sin.(x)
    wts = 1 ./ sqrt.(1 .+ x)
    # cs = numpy.polynomial.polynomial.polyfit(x, y, 4, w=wts)
    cs = [0.0006441172319036863, 0.985961582190304, 0.04999233434370933, -0.23162369757680354, 0.036864056406570644]
    @test maximum(abs, coeffs(fit(x, y, 4, weights=wts.^2)) - cs) <= sqrt(eps())
end


xs = range(0, 10, length=40)
ys = @. exp(-xs)
ff = fit(xs, ys) # degree = length(xs) - 1
f2 = fit(xs, ys, 9) # degree = 2

scatter(xs, ys, markerstrokewidth=0, label="Data")
plot!(ff, extrema(xs)..., label="Fit")
plot!(f2, extrema(xs)..., label="Quadratic Fit")
