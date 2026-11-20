# This evaluation uses the original King density and its derivative in v².
# The per-component factors are prepared once per collision-source evaluation.
function prepared_model(model,bulk,number)
    width=bulk.*model.widths
    shift=bulk.*model.shifts
    invs2=inv.(width.^2)
    (;shift,invs2,
      log_amplitude=log.(number.*model.weights).-1.5log(pi).-3log.(width),
      slope=2 .*shift.^2 .*invs2.^2, zscale=2 .*shift.*invs2)
end

function prepared_profile(c,r)
    peak=-Inf
    weight=0.
    gradient=0.
    @inbounds for i in eachindex(c.shift)
        z=r*c.zscale[i]
        v=c.shift[i]
        invs2=c.invs2[i]
        if z<1e-3
            zz=z*z
            L=c.log_amplitude[i]-(r*r+v*v)*invs2+zz/6-zz^2/180+zz^3/2835-zz^4/37800
            G=-invs2+c.slope[i]*(1/3-zz/45+2zz^2/945-zz^3/4725+2zz^4/93555)
        else
            decay=exp(-2z)
            L=c.log_amplitude[i]-(r-v)^2*invs2+log(-expm1(-2z))-log(2z)
            G=-invs2+(c.zscale[i]/(2r))*(1+2decay/(1-decay)-inv(z))
        end
        if L>peak
            scale=exp(peak-L)
            weight=weight*scale+1.
            gradient=gradient*scale+G
            peak=L
        else
            scale=exp(L-peak)
            weight+=scale
            gradient+=scale*G
        end
    end
    exp(peak)*weight,gradient/weight
end

function KD.rates(models::Vector{<:NamedTuple{(:weights,:widths,:shifts)}},
                  bulk_widths::Vector{Float64},numbers::Vector{Float64},geo::KD.Geometry)
    prepared=prepared_model.(models,bulk_widths,numbers)
    density=similar(geo.points);gradient=similar(density)
    @inbounds for j in eachindex(density)
        density[j],gradient[j]=prepared_profile(prepared[geo.species[j]],geo.points[j])
    end
    output=zeros(last(geo.offsets))
    for p in geo.pairs
        KD.ordered_source!(output,p,p.a,p.b,p.cells_a,p.cells_b,density,gradient,geo)
        p.a!=p.b && KD.ordered_source!(output,p,p.b,p.a,p.cells_b,p.cells_a,density,gradient,geo)
        flux=similar(p.triangle_weight)
        @inbounds for j in eachindex(flux)
            l,r=p.left[j],p.right[j]
            flux[j]=p.triangle_weight[j]*density[l]*density[r]*
                (gradient[r]/geo.masses[geo.species[r]]-gradient[l]/geo.masses[geo.species[l]])
        end
        mul!(output,transpose(p.triangle_gradient),flux,-1.,1.)
    end
    output
end
