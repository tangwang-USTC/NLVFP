module IsotropicKingDynamics

using LinearAlgebra
include("RadialHighOrder.jl")
include("IsotropicKingFit.jl")

struct MomentSpace
    cells::Vector{Float64}
    knots::Vector{Float64}
    degree::Int
    velocity_scale::Float64
end

function RadialHighOrder.evaluate_space_basis(space::MomentSpace,x)
    basis = ones(length(x),space.degree+1)
    derivative = zeros(size(basis))
    for power in 1:space.degree
        coefficient = prod(j+.5 for j in 1:power;init=1.)*space.velocity_scale^(2power)
        basis[:,power+1] = x.^power./coefficient
        derivative[:,power+1] = power.*x.^(power-1)./coefficient
    end
    basis,derivative
end

struct ProfileCell
    evaluation::RadialHighOrder.CellEvaluation
    points::Vector{Int}
end

struct ProfilePair
    a::Int
    b::Int
    coefficient::Float64
    cells_a::Vector{ProfileCell}
    cells_b::Vector{ProfileCell}
    triangle_gradient::Matrix{Float64}
    triangle_weight::Vector{Float64}
    left::Vector{Int}
    right::Vector{Int}
end

struct Geometry
    points::Vector{Float64}
    species::Vector{Int}
    pairs::Vector{ProfilePair}
    offsets::Vector{Int}
    masses::Vector{Float64}
    maximum_power::Int
    velocity_scale::Float64
end

function Geometry(masses,gamma,velocity_scale,edges,maximum_power;order=12)
    cells = (velocity_scale.*edges).^2
    spaces = [MomentSpace(cells,collect(0.:maximum_power),maximum_power,velocity_scale) for mass in masses]
    offsets = vcat(0,cumsum(length(space.knots) for space in spaces))
    original = RadialHighOrder.pair_geometry(spaces,gamma,masses,offsets;quadrature_order=order)
    points = Float64[]
    species = Int[]
    lookup = Dict{Tuple{Int,Float64},Int}()
    function point_index(a,radius)
        key = (a,radius)
        get!(lookup,key) do
            push!(points,radius)
            push!(species,a)
            length(points)
        end
    end
    profile_cell(cell,a) = ProfileCell(cell,[point_index(a,sqrt(max(0.,1.5velocity_scale^2*x))) for x in cell.basis[:,2]])
    xvector = zeros(last(offsets))
    for a in eachindex(masses)
        xvector[offsets[a]+2] = 1.5velocity_scale^2
    end
    pairs = ProfilePair[]
    for pair in original
        left_x = pair.triangle_left_density*xvector
        right_x = pair.triangle_right_density*xvector
        left_a = pair.triangle_left_density[:,offsets[pair.a]+1]
        right_a = pair.triangle_right_density[:,offsets[pair.a]+1]
        left = [point_index(left_a[row]!=0 ? pair.a : pair.b,sqrt(max(0.,left_x[row]))) for row in eachindex(left_x)]
        right = [point_index(right_a[row]!=0 ? pair.a : pair.b,sqrt(max(0.,right_x[row]))) for row in eachindex(right_x)]
        push!(pairs,ProfilePair(pair.a,pair.b,pair.coefficient,profile_cell.(pair.cells_a,pair.a),
            profile_cell.(pair.cells_b,pair.b),Matrix(pair.triangle_gradient),pair.triangle_weight,left,right))
    end
    Geometry(points,species,pairs,offsets,Float64.(masses),maximum_power,velocity_scale)
end

function component_source_edges(base,velocity_scale,states...)
    edges=copy(base)
    for (models,bulk_widths) in states
        for (model,bulk) in zip(models,bulk_widths)
            shells=hasproperty(model,:squared_shifts) ? sqrt.(model.squared_shifts) : model.shifts
            for (shell,width) in zip(shells,model.widths), multiplier in (-12,-6,-3,-1,0,1,3,6,12)
                point=bulk/velocity_scale*(shell+multiplier*width)
                if 0<point<last(base)
                    push!(edges,point)
                end
            end
        end
    end
    sort!(edges)
    # Coalesce nearly identical edges from coincident or overlapping shells.
    selected=[first(edges)]
    for point in edges[2:end]
        point-last(selected)>8eps(max(point,1.)) && push!(selected,point)
    end
    selected[end]=last(base)
    selected
end

function kernel_gradient(radius,shell,width)
    z = 2radius*shell/width^2
    if abs(z)<1e-3
        q = z^2
        return -inv(width^2)+2shell^2/width^4*(1/3-q/45+2q^2/945-q^3/4725+2q^4/93555)
    end
    decay = exp(-2z)
    -inv(width^2)+shell/(radius*width^2)*(1+2decay/(1-decay)-inv(z))
end

function profile(model,bulk_width,number,radius)
    if hasproperty(model,:squared_shifts)
        parts = [squared_kernel(radius,bulk_width^2*q,bulk_width*s) for (q,s) in zip(model.squared_shifts,model.widths)]
        terms = [log(model.weights[j])+parts[j][1] for j in eachindex(parts)]
        peak=maximum(terms);probabilities=exp.(terms.-peak);normalization=sum(probabilities)
        gradient=sum(probabilities[j]*parts[j][2] for j in eachindex(parts))/normalization
        return number*exp(peak)*normalization,gradient
    end
    terms = [log(w)+IsotropicKingFit.log_kernel(radius,bulk_width*u,bulk_width*s)
        for (w,u,s) in zip(model.weights,model.shifts,model.widths)]
    maximum_log = maximum(terms)
    probabilities = exp.(terms.-maximum_log)
    probabilities ./= sum(probabilities)
    gradient = sum(probabilities[j]*kernel_gradient(radius,bulk_width*model.shifts[j],bulk_width*model.widths[j])
        for j in eachindex(probabilities))
    density = number*exp(maximum_log)*sum(exp.(terms.-maximum_log))
    density,gradient
end

# q=u^2 is differentiable at the Gaussian shell radius u=0. The small-z
# branch uses z^2 directly, so no derivative of sqrt(q) enters this limit.
function squared_kernel(radius,q,width)
    z2=4radius^2*q/width^4
    if z2<1e-6
        log_sinhc=z2/6-z2^2/180+z2^3/2835-z2^4/37800
        logvalue=-(radius^2+q)/width^2+log_sinhc-1.5log(pi)-3log(width)
        gradient=-inv(width^2)+2q/width^4*(1/3-z2/45+2z2^2/945-z2^3/4725+2z2^4/93555)
        return logvalue,gradient
    end
    u=sqrt(q)
    IsotropicKingFit.log_kernel(radius,u,width),kernel_gradient(radius,u,width)
end

function cell_data(cell,density,gradient,mass,power)
    data = cell.evaluation
    weights = power==1 ? data.weight1 : data.weight4
    weighted = weights.*density[cell.points]
    g = gradient[cell.points]./mass
    (;i=sum(weighted),j=dot(weighted,g),
        v=transpose(data.gradient)*weighted,u=transpose(data.gradient)*(weighted.*g))
end

function ordered_source!(output,pair,outer,inner,cells_outer,cells_inner,density,gradient,geometry)
    ia = geometry.offsets[outer]+1:geometry.offsets[outer+1]
    ib = geometry.offsets[inner]+1:geometry.offsets[inner+1]
    initial_data = cell_data(first(cells_inner),density,gradient,geometry.masses[inner],4)
    prefix = map(zero,initial_data)
    for cell in eachindex(cells_outer)
        target = cell_data(cells_outer[cell],density,gradient,geometry.masses[outer],1)
        output[ia] .+= pair.coefficient.*(-target.u.*prefix.i+target.v.*prefix.j)
        output[ib] .+= pair.coefficient.*(prefix.v.*target.j-prefix.u.*target.i)
        next = cell_data(cells_inner[cell],density,gradient,geometry.masses[inner],4)
        prefix = NamedTuple{keys(prefix)}(map(+,values(prefix),values(next)))
    end
end

function rates(models,bulk_widths,numbers,geometry::Geometry)
    T = eltype(first(models).weights)
    density = Vector{T}(undef,length(geometry.points))
    gradient = similar(density)
    for j in eachindex(density)
        a = geometry.species[j]
        density[j],gradient[j] = profile(models[a],bulk_widths[a],numbers[a],geometry.points[j])
    end
    output = zeros(T,last(geometry.offsets))
    for pair in geometry.pairs
        ordered_source!(output,pair,pair.a,pair.b,pair.cells_a,pair.cells_b,density,gradient,geometry)
        if pair.a!=pair.b
            ordered_source!(output,pair,pair.b,pair.a,pair.cells_b,pair.cells_a,density,gradient,geometry)
        end
        grad_log = [gradient[pair.right[j]]/geometry.masses[geometry.species[pair.right[j]]] -
            gradient[pair.left[j]]/geometry.masses[geometry.species[pair.left[j]]] for j in eachindex(pair.left)]
        weight = pair.triangle_weight.*density[pair.left].*density[pair.right]
        output .-= transpose(pair.triangle_gradient)*(weight.*grad_log)
    end
    output
end

function decode(parameters,count,masses,numbers,total_energy;squared_shells=false)
    dimension = 3count-2
    function model(a)
        p=parameters[(a-1)*dimension+1:a*dimension]
        squared_shells||return IsotropicKingFit.decode(p,count)
        logits=vcat(p[1:count-1],0.);weights=exp.(logits.-maximum(logits));weights./=sum(weights)
        relative_widths=vcat(exp.(p[count:2count-2]),1.)
        q=p[2count-1:3count-2].*relative_widths.^2
        scale2=sum(weights.*(relative_widths.^2 .+ 2q/3))
        (;weights,widths=relative_widths./sqrt(scale2),shifts=sqrt.(q./scale2),squared_shifts=q./scale2)
    end
    models = [model(a) for a in 1:2]
    fraction = inv(1+exp(-last(parameters)))
    widths = [sqrt(4total_energy*fraction/(3numbers[1]*masses[1])),
        sqrt(4total_energy*(1-fraction)/(3numbers[2]*masses[2]))]
    models,widths
end

function moments(models,widths,numbers,geometry)
    reduce(vcat,[[numbers[a]*(widths[a]/geometry.velocity_scale)^(2power)*
        normalized_moment(models[a],power) for power in 0:geometry.maximum_power] for a in eachindex(models)])
end

function normalized_moment(model,power)
    hasproperty(model,:squared_shifts)||return IsotropicKingFit.moment(model,power)
    sum(model.weights[atom]*sum(binomial(power,j)*2.0^j/
        prod(2k+1 for k in 1:j;init=1.)*model.squared_shifts[atom]^j*model.widths[atom]^(2(power-j))
        for j in 0:power) for atom in eachindex(model.weights))
end

end
