module RadialHighOrder
using LinearAlgebra,SparseArrays,FastGaussQuadrature,ForwardDiff
include("RadialGalerkin.jl")

struct Space
    knots::Vector{Float64}
    squared_speed::Vector{Float64}
    weights::Vector{Float64}
    basis::Matrix{Float64}
    derivative::Matrix{Float64}
    cells::Vector{Float64}
    degree::Int
end

function local_basis(degree,t)
    nodes=[-cospi(j/degree) for j in 0:degree]
    bary=[(-1.0)^j*(j in (0,degree) ? 0.5 : 1.0) for j in 0:degree]
    match=findfirst(x->abs(t-x)<8eps(Float64),nodes)
    values=zeros(degree+1);derivative=zeros(degree+1)
    if match!==nothing
        values[match]=1
        for j in eachindex(nodes)
            j==match&&continue
            derivative[j]=bary[j]/(bary[match]*(nodes[match]-nodes[j]))
        end
        derivative[match]=-sum(derivative)
    else
        terms=bary./(t.-nodes)
        values=terms/sum(terms)
        derivative=values.*(sum(bary./(t.-nodes).^2)/sum(terms).-inv.(t.-nodes))
    end
    values,derivative
end

function evaluate_basis(cells,degree,x)
    count=(length(cells)-1)*degree+1
    basis=zeros(length(x),count);derivative=zero(basis)
    for i in eachindex(x)
        cell=clamp(searchsortedlast(cells,x[i]),1,length(cells)-1)
        span=cells[cell+1]-cells[cell]
        t=(2x[i]-cells[cell]-cells[cell+1])/span
        values,slopes=local_basis(degree,t)
        columns=(cell-1)*degree+1:cell*degree+1
        basis[i,columns]=values
        derivative[i,columns]=slopes.*(2/span)
    end
    basis,derivative
end

evaluate_space_basis(space,x) = evaluate_basis(space.cells,space.degree,x)

function radial_space(order,width;quadrature_order=8,degree=1,boundaries=nothing)
    roots,_=gausslaguerre(order,0.5)
    cells=boundaries===nothing ? vcat(0.0,width^2 .* roots) : width^2 .* Float64.(boundaries)
    first(cells)==0 && all(diff(cells).>0)||error("Radial cells must increase from zero")
    nodes=Float64[first(cells)]
    for cell in 1:length(cells)-1,j in 1:degree
        t=(1-cospi(j/degree))/2
        push!(nodes,cells[cell]+t*(cells[cell+1]-cells[cell]))
    end
    points,weights=gausslegendre(quadrature_order)
    x=Float64[];volume=Float64[]
    for cell in 1:length(cells)-1
        left,right=sqrt(cells[cell]),sqrt(cells[cell+1])
        for j in eachindex(points)
            r=(left+right)/2+(right-left)*points[j]/2
            push!(x,r^2);push!(volume,4pi*r^2*(right-left)*weights[j]/2)
        end
    end
    basis,derivative=evaluate_basis(cells,degree,x)
    Space(nodes,x,volume,basis,derivative,cells,degree)
end

struct CellEvaluation
    basis::SparseMatrixCSC{Float64,Int}
    gradient::SparseMatrixCSC{Float64,Int}
    weight1::Vector{Float64}
    weight4::Vector{Float64}
end
struct PairGeometry
    a::Int
    b::Int
    coefficient::Float64
    cells_a::Vector{CellEvaluation}
    cells_b::Vector{CellEvaluation}
    triangle_gradient::SparseMatrixCSC{Float64,Int}
    triangle_density::SparseMatrixCSC{Float64,Int}
    triangle_weight::Vector{Float64}
    triangle_left_density::SparseMatrixCSC{Float64,Int}
    triangle_right_density::SparseMatrixCSC{Float64,Int}
end

function cell_evaluation(space,mass,left,right,order)
    points,weights=gausslegendre(order)
    r=(left+right)/2 .+ (right-left)/2 .* points
    if (left+right)/2>=sqrt(last(space.cells))
        z=spzeros(order,length(space.knots))
        return CellEvaluation(z,copy(z),zeros(order),zeros(order))
    end
    basis,derivative=evaluate_space_basis(space,r.^2)
    dr=weights.*((right-left)/2)
    CellEvaluation(sparse(basis),sparse(derivative/mass),dr.*r,dr.*r.^4)
end

function pair_geometry(spaces,gamma,masses,offsets;quadrature_order=8)
    result=PairGeometry[]
    total=last(offsets)
    nodes,weights=gausslegendre(quadrature_order)
    for a in eachindex(spaces),b in a:length(spaces)
        boundaries=sort(unique(vcat(sqrt.(spaces[a].cells),sqrt.(spaces[b].cells))))
        cells_a=[cell_evaluation(spaces[a],masses[a],boundaries[i],boundaries[i+1],quadrature_order) for i in 1:length(boundaries)-1]
        cells_b=[cell_evaluation(spaces[b],masses[b],boundaries[i],boundaries[i+1],quadrature_order) for i in 1:length(boundaries)-1]
        rows=Int[];columns=Int[];gradient_values=Float64[];density_values=Float64[]
        left_density_values=Float64[];right_density_values=Float64[]
        triangle_weight=Float64[]
        orientations=a==b ? [(a,b)] : [(a,b),(b,a)]
        coefficient=8gamma[a,b]/3*(4pi)^2
        for (outer,inner) in orientations,cell in 1:length(boundaries)-1
            left,right=boundaries[cell:cell+1]
            right<=min(sqrt(last(spaces[a].cells)),sqrt(last(spaces[b].cells)))||continue
            for i in eachindex(nodes),j in eachindex(nodes)
                r=(left+right)/2+(right-left)*nodes[i]/2
                s=(left+r)/2+(r-left)*nodes[j]/2
                row=length(triangle_weight)+1
                push!(triangle_weight,coefficient*r*s^4*(right-left)*(r-left)*weights[i]*weights[j]/4)
                for (species,x,sign) in ((outer,r^2,-1.0),(inner,s^2,1.0))
                    basis,derivative=evaluate_space_basis(spaces[species],[x])
                    for k in eachindex(spaces[species].knots)
                        (basis[1,k]==0 && derivative[1,k]==0)&&continue
                        push!(rows,row);push!(columns,offsets[species]+k)
                        push!(gradient_values,sign*derivative[1,k]/masses[species])
                        push!(density_values,basis[1,k])
                        push!(left_density_values,sign<0 ? basis[1,k] : 0.)
                        push!(right_density_values,sign>0 ? basis[1,k] : 0.)
                    end
                end
            end
        end
        z=sparse(rows,columns,gradient_values,length(triangle_weight),total)
        p=sparse(rows,columns,density_values,length(triangle_weight),total)
        left_density=sparse(rows,columns,left_density_values,length(triangle_weight),total)
        right_density=sparse(rows,columns,right_density_values,length(triangle_weight),total)
        push!(result,PairGeometry(a,b,coefficient,cells_a,cells_b,z,p,triangle_weight,left_density,right_density))
    end
    result
end

function cell_moment(cell,y,power,jacobian;gradient_y=y,density_pair=nothing)
    weights=power==1 ? cell.weight1 : cell.weight4
    f=density_pair===nothing ? exp.(cell.basis*y) :
        (exp.(cell.basis*density_pair[1])+exp.(cell.basis*density_pair[2]))/2
    g=cell.gradient*gradient_y
    weighted=weights.*f
    i=sum(weighted);j=dot(weighted,g)
    # Sparse arithmetic can discard a zero primal value with nonzero Dual
    # derivatives, which occurs in self-collision cancellation at a Maxwellian.
    v=transpose(cell.gradient)*weighted
    u=transpose(cell.gradient)*(weighted.*g)
    if !(eltype(y)<:ForwardDiff.Dual)
        v=sparse(v)
        u=sparse(u)
    end
    if jacobian
        di=sparse(transpose(cell.basis)*weighted)
        dj=sparse(transpose(cell.basis)*(weighted.*g))+v
        dv=transpose(cell.gradient)*(Diagonal(weighted)*cell.basis)
        du=transpose(cell.gradient)*(Diagonal(weighted.*g)*cell.basis+Diagonal(weighted)*cell.gradient)
        return (;i,j,v,u,di,dj,dv,du)
    end
    (;i,j,v,u)
end

function add_moments(a,b)
    NamedTuple{keys(a)}(map(+,values(a),values(b)))
end

function add_sparse_block!(target,rows,columns,block,coefficient)
    for column in axes(block,2),position in nzrange(block,column)
        row=rowvals(block)[position]
        target[first(rows)+row-1,first(columns)+column-1]+=coefficient*nonzeros(block)[position]
    end
end

function add_outer!(target,rows,columns,left,right,coefficient)
    for j in eachindex(nonzeros(right)),i in eachindex(nonzeros(left))
        target[first(rows)+SparseArrays.nonzeroinds(left)[i]-1,first(columns)+SparseArrays.nonzeroinds(right)[j]-1]+=coefficient*nonzeros(left)[i]*nonzeros(right)[j]
    end
end

function ordered_region!(output,jacobian,y,outer,inner,cells_outer,cells_inner,coefficient,offsets,with_jacobian;gradient_state=y,density_pair=nothing)
    ia=offsets[outer]+1:offsets[outer+1];ib=offsets[inner]+1:offsets[inner+1]
    ya=y[ia];yb=y[ib]
    ga=gradient_state[ia];gb=gradient_state[ib]
    density_a=density_pair===nothing ? nothing : (density_pair[1][ia],density_pair[2][ia])
    density_b=density_pair===nothing ? nothing : (density_pair[1][ib],density_pair[2][ib])
    first_moment=cell_moment(first(cells_inner),yb,4,with_jacobian;gradient_y=gb,density_pair=density_b)
    prefix=map(zero,first_moment)
    for cell in eachindex(cells_outer)
        target=cell_moment(cells_outer[cell],ya,1,with_jacobian;gradient_y=ga,density_pair=density_a)
        output[ia].-=coefficient.*(target.u.*prefix.i-target.v.*prefix.j)
        output[ib].+=coefficient.*(prefix.v.*target.j-prefix.u.*target.i)
        if with_jacobian
            add_sparse_block!(jacobian,ia,ia,target.du,-coefficient*prefix.i)
            add_sparse_block!(jacobian,ia,ia,target.dv,coefficient*prefix.j)
            add_outer!(jacobian,ia,ib,target.u,prefix.di,-coefficient)
            add_outer!(jacobian,ia,ib,target.v,prefix.dj,coefficient)
            add_outer!(jacobian,ib,ia,prefix.v,target.dj,coefficient)
            add_outer!(jacobian,ib,ia,prefix.u,target.di,-coefficient)
            add_sparse_block!(jacobian,ib,ib,prefix.dv,coefficient*target.j)
            add_sparse_block!(jacobian,ib,ib,prefix.du,-coefficient*target.i)
        end
        prefix=add_moments(prefix,cell_moment(cells_inner[cell],yb,4,with_jacobian;gradient_y=gb,density_pair=density_b))
    end
end

function source(y,spaces,masses,pairs,offsets;with_jacobian=false,gradient_state=y,density_pair=nothing,coupled_gradient_variation=false)
    # With an independent gradient, this Jacobian varies the density log and
    # the gradient coefficients by the same increment. DG applies its chain rule.
    with_jacobian&&gradient_state!==y&&!coupled_gradient_variation&&error("Independent gradient requires explicit coupled variation convention")
    with_jacobian&&density_pair!==nothing&&error("Two-density mobility uses numerical differentiation")
    output=zeros(eltype(y),length(y))
    jacobian=with_jacobian ? zeros(length(y),length(y)) : nothing
    for pair in pairs
        ordered_region!(output,jacobian,y,pair.a,pair.b,pair.cells_a,pair.cells_b,pair.coefficient,offsets,with_jacobian;gradient_state,density_pair)
        if pair.a!=pair.b
            ordered_region!(output,jacobian,y,pair.b,pair.a,pair.cells_b,pair.cells_a,pair.coefficient,offsets,with_jacobian;gradient_state,density_pair)
        end
        q=pair.triangle_gradient*gradient_state
        weight=if density_pair===nothing
            pair.triangle_weight.*exp.(pair.triangle_density*y)
        else
            left=(exp.(pair.triangle_left_density*density_pair[1])+exp.(pair.triangle_left_density*density_pair[2]))/2
            right=(exp.(pair.triangle_right_density*density_pair[1])+exp.(pair.triangle_right_density*density_pair[2]))/2
            pair.triangle_weight.*left.*right
        end
        output.-=transpose(pair.triangle_gradient)*(weight.*q)
        if with_jacobian
            variation=copy(pair.triangle_gradient)
            rowvals(variation)==rowvals(pair.triangle_density)&&variation.colptr==pair.triangle_density.colptr||error("Triangle sparsity mismatch")
            for i in eachindex(nonzeros(variation))
                row=rowvals(variation)[i]
                nonzeros(variation)[i]=weight[row]*(q[row]*nonzeros(pair.triangle_density)[i]+nonzeros(pair.triangle_gradient)[i])
            end
            jacobian.-=transpose(pair.triangle_gradient)*variation
        end
    end
    with_jacobian ? (output,jacobian) : output
end

const moments=RadialGalerkin.moments
const invariants=RadialGalerkin.invariants
const moment_jacobian=RadialGalerkin.moment_jacobian
particle_number(values,spaces,offsets) = sum(values)

function positive_basis_masses(y,spaces,offsets)
    reduce(vcat,[transpose(abs.(spaces[a].basis))*(spaces[a].weights.*exp.(spaces[a].basis*y[offsets[a]+1:offsets[a+1]])) for a in eachindex(spaces)])
end

mutable struct NewtonCache
    factor::Any
    upper::Matrix{Float64}
    rank::Int
    scale::Vector{Float64}
    uses::Int
    factorizations::Int
    fallbacks::Int
    maximum_uses::Int
end
NewtonCache(;maximum_uses=32)=NewtonCache(nothing,zeros(0,0),0,Float64[],0,0,0,maximum_uses)

function damped_newton_trial(y,defect,jacobian,residual;admissible=nothing,column_penalties=nothing)
    column_norm=sqrt.(vec(sum(abs2,jacobian;dims=1)))
    column_scale=max.(column_norm,maximum(column_norm)*1e-6)
    column_penalties===nothing || (column_scale .*= column_penalties)
    factor=svd(jacobian./transpose(column_scale))
    projected=transpose(factor.U)*defect
    for damping_power in -28:2:0
        damping=maximum(factor.S)^2*10.0^damping_power
        direction=-(factor.V*((factor.S./(factor.S.^2 .+damping)).*projected))./column_scale
        # A coefficient cap depends on the basis coordinates. With a physical
        # representation guard, line search accepts the represented density.
        admissible===nothing && maximum(abs,direction)>20 && continue
        for power in 0:20
            trial=y+2.0^(-power)*direction
            admissible===nothing || admissible(trial) || continue
            trial_defect=residual(trial)
            if all(isfinite,trial_defect)&&sum(abs2,trial_defect)<sum(abs2,defect)
                return trial
            end
        end
    end
    error("High-order stage damped Newton line search failed")
end

function stage(seed,base,spaces,masses,pairs,offsets,step,tolerance;cache=nothing,equilibrate=false,damped_newton=false,positive_basis_scale=false,admissible=nothing,column_penalties=nothing)
    y=copy(seed)
    reference=moments(y,spaces,offsets)
    particles=particle_number(reference,spaces,offsets)
    local_mass=positive_basis_scale ? positive_basis_masses(y,spaces,offsets) : abs.(reference)
    scale=max.(local_mass,maximum(local_mass)*1e-12)
    residual(candidate)=(moments(candidate,spaces,offsets).-base.-step.*source(candidate,spaces,masses,pairs,offsets))./scale
    previous_defect=Inf
    for iteration in 1:(damped_newton ? 60 : 40)
        defect=residual(y)
        particle_L1=sum(abs.(defect).*scale)/particles
        if particle_L1<tolerance
            admissible===nothing || admissible(y) || error("High-order stage continuous representation guard failed")
            return (;y,residual=particle_L1,iterations=iteration)
        end
        if damped_newton
            cache===nothing||error("Damped Newton requires fresh stage Jacobians")
            _,jsource=source(y,spaces,masses,pairs,offsets;with_jacobian=true)
            jacobian=(moment_jacobian(y,spaces,offsets).-step.*jsource)./scale
            y=damped_newton_trial(y,defect,jacobian,residual;admissible,column_penalties)
            continue
        end
        refresh=cache===nothing || cache.factor===nothing || cache.uses>=cache.maximum_uses || particle_L1>0.3previous_defect
        if refresh
            _,jsource=source(y,spaces,masses,pairs,offsets;with_jacobian=true)
            jacobian=(moment_jacobian(y,spaces,offsets).-step.*jsource)./scale
            column_norm=sqrt.(vec(sum(abs2,jacobian;dims=1)))
            column_scale=equilibrate ? max.(column_norm,maximum(column_norm)*1e-6) : ones(length(y))
            factor=qr(jacobian./transpose(column_scale),ColumnNorm())
            upper=Matrix(factor.R)
            diagonal=diag(upper)
            threshold=maximum(abs,diagonal)*1e-14
            rank=count(value->abs(value)>threshold,diagonal)
            if cache!==nothing
                cache.factor=factor;cache.upper=upper;cache.rank=rank;cache.scale=copy(scale)
                cache.uses=0;cache.factorizations+=1
            end
        else
            factor=cache.factor;upper=cache.upper;rank=cache.rank
        end
        right_hand_side=cache===nothing ? defect : defect.*scale./cache.scale
        pivoted=zeros(length(y))
        pivoted[1:rank]=-(upper[1:rank,1:rank]\(transpose(factor.Q)*right_hand_side)[1:rank])
        direction=zeros(length(y))
        direction[factor.p]=pivoted
        if equilibrate
            # The column-scaled factor is refreshed for every iteration.
            direction ./= column_scale
            direction .*= min(1.,5/max(maximum(abs,direction),1e-300))
        else
            direction=clamp.(direction,-5.0,5.0)
        end
        previous_defect=particle_L1
        cache===nothing||(cache.uses+=1)
        accepted=false
        for power in 0:20
            candidate=y+2.0^(-power)*direction
            all(isfinite,candidate)||continue
            admissible===nothing || admissible(candidate) || continue
            trial=residual(candidate)
            improves=equilibrate ? sum(abs2,trial)<sum(abs2,defect) : sum(abs2,trial.*scale)<sum(abs2,defect.*scale)
            if all(isfinite,trial) && improves
                y=candidate;accepted=true;break
            end
        end
        accepted||error("High-order stage line search failed at L1 ",particle_L1)
    end
    error("High-order stage iteration limit")
end

function advance(y,spaces,masses,pairs,offsets,step;tolerance=1e-12,cache=nothing,equilibrate=false,damped_newton=false,positive_basis_scale=false,admissible=nothing,column_penalties=nothing)
    diagonal=1-inv(sqrt(2.0))
    previous=moments(y,spaces,offsets)
    try
        equilibrate && cache!==nothing && error("Column-equilibrated Newton currently requires fresh factorizations")
        first=stage(y,previous,spaces,masses,pairs,offsets,diagonal*step,tolerance;cache,equilibrate,damped_newton,positive_basis_scale,admissible,column_penalties)
        base=previous+(1-diagonal)*step.*source(first.y,spaces,masses,pairs,offsets)
        second=stage(first.y,base,spaces,masses,pairs,offsets,diagonal*step,tolerance;cache,equilibrate,damped_newton,positive_basis_scale,admissible,column_penalties)
        return (;y=second.y,residual=max(first.residual,second.residual),iterations=max(first.iterations,second.iterations),first_stage=first.y)
    catch failure
        cache===nothing&&rethrow()
        cache.factor=nothing;cache.fallbacks+=1
        return advance(y,spaces,masses,pairs,offsets,step;tolerance,cache=nothing,equilibrate,damped_newton,positive_basis_scale,admissible,column_penalties)
    end
end
end
