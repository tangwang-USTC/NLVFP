using FastGaussQuadrature: gausslegendre

@testset "Independent quadrature of biased King mixtures" begin
    scales, atoms = (0.87, 1.21, 1.3), (a=[0.3,0.7], s=[0.65,1.2], q=[0.0,0.42])
    vth, N, vref = scales
    a,s,q = atoms
    x,g = gausslegendre(48)
    bins = (0.0, 0.2, 0.7, 1.5, 3.0, 6.0, 11.0, 17.0)
    r, w = Float64[], Float64[]
    for (lo,hi) in zip(bins[1:end-1],bins[2:end])
        append!(r,(lo+hi)/2 .+ ((hi-lo)/2).*x)
        append!(w,((hi-lo)/2).*g)
    end
    measure = radial_measure(r,w;convention=:dr,coordinate="v / v_ref_case")
    function profile(radius)
        sum(zip(a,s,q)) do (weight,squared_width,squared_shift)
            width, shift = vth*sqrt(squared_width), vth*sqrt(squared_shift)
            z = 2radius*shift/width^2
            angular = iszero(z) ? 1.0 : -expm1(-2z)/(2z)
            N*weight*exp(-((radius-shift)/width)^2)*angular/(π^1.5*width^3)
        end
    end
    f = profile.(r)
    for mode in (:even,:odd,:integer), normalization in (:raw,:maxwell)
        space = power_space(mode,9;conserved=true,velocity_scale=vref,normalization)
        numerical = observe(space,measure,f)
        analytical = king_observations(space,a,s,q;thermal_speed=vth,number=N)
        @test maximum(abs.(numerical./analytical.-1)) <= 3e-12
        @test numerical[1] ≈ N rtol=3e-12
    end
end
