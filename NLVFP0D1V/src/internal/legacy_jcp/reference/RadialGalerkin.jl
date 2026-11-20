module RadialGalerkin
using LinearAlgebra,FastGaussQuadrature,ForwardDiff
include("RadialLandau.jl")

struct Space
    knots::Vector{Float64}
    squared_speed::Vector{Float64}
    weights::Vector{Float64}
    basis::Matrix{Float64}
    derivative::Matrix{Float64}
end

function radial_space(order,width;quadrature_order=8)
    laguerre,_=gausslaguerre(order,0.5)
    knots=vcat(0.0,width^2 .* laguerre)
    points,weights=gausslegendre(quadrature_order)
    nq=order*quadrature_order
    x=zeros(nq);volume=zeros(nq);basis=zeros(nq,order+1);derivative=zero(basis)
    for cell in 1:order
        left,right=sqrt(knots[cell]),sqrt(knots[cell+1])
        dx=knots[cell+1]-knots[cell]
        for j in 1:quadrature_order
            i=(cell-1)*quadrature_order+j
            radius=(left+right)/2+(right-left)*points[j]/2
            x[i]=radius^2
            volume[i]=4pi*radius^2*(right-left)*weights[j]/2
            basis[i,cell]=(knots[cell+1]-x[i])/dx
            basis[i,cell+1]=(x[i]-knots[cell])/dx
            derivative[i,cell]=-inv(dx)
            derivative[i,cell+1]=inv(dx)
        end
    end
    Space(knots,x,volume,basis,derivative)
end

function evaluate(y,spaces,offsets)
    [exp.(spaces[a].basis*y[offsets[a]+1:offsets[a+1]]) for a in eachindex(spaces)]
end

function moments(y,spaces,offsets)
    values=evaluate(y,spaces,offsets)
    reduce(vcat,[transpose(spaces[a].basis)*(spaces[a].weights.*values[a]) for a in eachindex(spaces)])
end

function source(y,spaces,masses,pairs,offsets;with_jacobian=false)
    values=evaluate(y,spaces,offsets)
    gradients=[spaces[a].derivative*y[offsets[a]+1:offsets[a+1]]/masses[a] for a in eachindex(spaces)]
    forces=[zero(value) for value in values]
    jacobian=with_jacobian ? zeros(length(y),length(y)) : nothing
    for pair in pairs
        a,b=pair.a,pair.b
        flux=pair.geometry .* (values[a]*transpose(values[b])) .* (transpose(gradients[b]).-gradients[a])
        forces[a].+=vec(sum(flux;dims=2))/masses[a]
        a==b || (forces[b].-=vec(sum(flux;dims=1))/masses[b])
        if with_jacobian
            pa,pb=spaces[a].basis,spaces[b].basis
            da,db=spaces[a].derivative,spaces[b].derivative
            mobility=pair.geometry .* (values[a]*transpose(values[b]))
            ia=offsets[a]+1:offsets[a+1];ib=offsets[b]+1:offsets[b+1]
            rows=vec(sum(flux;dims=2));columns=vec(sum(flux;dims=1))
            rows_mobility=vec(sum(mobility;dims=2));columns_mobility=vec(sum(mobility;dims=1))
            if a==b
                variation=rows.*pa+flux*pa+(mobility*da-rows_mobility.*da)/masses[a]
                jacobian[ia,ia].+=transpose(da)*variation/masses[a]
            else
                jacobian[ia,ia].+=transpose(da)*(rows.*pa-rows_mobility.*da/masses[a])/masses[a]
                jacobian[ia,ib].+=transpose(da)*(flux*pb+mobility*db/masses[b])/masses[a]
                jacobian[ib,ia].-=transpose(db)*(transpose(flux)*pa-transpose(mobility)*da/masses[a])/masses[b]
                jacobian[ib,ib].-=transpose(db)*(columns.*pb+columns_mobility.*db/masses[b])/masses[b]
            end
        end
    end
    value=reduce(vcat,[transpose(spaces[a].derivative)*forces[a] for a in eachindex(spaces)])
    with_jacobian ? (value,jacobian) : value
end

function moment_jacobian(y,spaces,offsets)
    values=evaluate(y,spaces,offsets)
    result=zeros(length(y),length(y))
    for a in eachindex(spaces)
        indices=offsets[a]+1:offsets[a+1]
        basis=spaces[a].basis
        result[indices,indices]=transpose(basis)*((spaces[a].weights.*values[a]).*basis)
    end
    result
end

function stage(seed,base,spaces,masses,pairs,offsets,step,tolerance)
    y=copy(seed)
    reference=moments(seed,spaces,offsets)
    scale=max.(reference,maximum(reference)*1e-12)
    residual(candidate)=(moments(candidate,spaces,offsets).-base.-step.*source(candidate,spaces,masses,pairs,offsets))./scale
    for iteration in 1:40
        defect=residual(y)
        particle_L1=sum(abs.(defect).*scale)/sum(reference)
        if particle_L1<tolerance
            return (;y,residual=particle_L1,iterations=iteration)
        end
        _,source_jacobian=source(y,spaces,masses,pairs,offsets;with_jacobian=true)
        jacobian=(moment_jacobian(y,spaces,offsets).-step.*source_jacobian)./scale
        factor=svd(jacobian)
        inverse_values=[value>maximum(factor.S)*1e-14 ? inv(value) : 0.0 for value in factor.S]
        direction=clamp.(-(factor.V*(inverse_values.*(transpose(factor.U)*defect))),-5.0,5.0)
        accepted=false
        for power in 0:20
            candidate=y+2.0^(-power)*direction
            all(isfinite,candidate)||continue
            trial=residual(candidate)
            if all(isfinite,trial) && sum(abs2,trial.*scale)<sum(abs2,defect.*scale)
                y=candidate;accepted=true;break
            end
        end
        accepted || error("Galerkin stage line search failed; particle L1 ",particle_L1)
    end
    error("Galerkin stage iteration limit")
end

function advance(y,spaces,masses,pairs,offsets,step;tolerance=1e-12)
    diagonal=1-inv(sqrt(2.0))
    previous=moments(y,spaces,offsets)
    first=stage(y,previous,spaces,masses,pairs,offsets,diagonal*step,tolerance)
    base=previous+(1-diagonal)*step.*source(first.y,spaces,masses,pairs,offsets)
    second=stage(first.y,base,spaces,masses,pairs,offsets,diagonal*step,tolerance)
    (;y=second.y,residual=max(first.residual,second.residual),iterations=max(first.iterations,second.iterations))
end

function invariants(y,spaces,masses,offsets)
    values=evaluate(y,spaces,offsets)
    number=[sum(spaces[a].weights.*values[a]) for a in eachindex(spaces)]
    energy=[masses[a]/2*dot(spaces[a].squared_speed,spaces[a].weights.*values[a]) for a in eachindex(spaces)]
    entropy=[dot(spaces[a].weights.*values[a],spaces[a].basis*y[offsets[a]+1:offsets[a+1]]) for a in eachindex(spaces)]
    (;number,energy,entropy)
end
end
