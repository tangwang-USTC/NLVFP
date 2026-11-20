module EDElectronObservables

using LinearAlgebra, FastGaussQuadrature, SpecialFunctions
import ..IsotropicKingDynamics
const KD=IsotropicKingDynamics

struct Theory
    thermal::Float64
    scale_factors::Vector{Float64}
    mode_counts::Vector{Int}
    exchange_scales::Vector{Float64}
end

function rate_features(x,scale)
    s=x/scale^2
    g=exp(-s)
    if s<1e-3
        h=(4/(scale*sqrt(pi)))*s*(1/3-s/5+s^2/14-s^3/54+s^4/264)
        dh=(4/(scale^3*sqrt(pi)))*(1/3-2s/5+3s^2/14-4s^3/54+5s^4/264)
    else
        r=sqrt(x)
        h=erf(sqrt(s))/r-2g/(scale*sqrt(pi))
        dh=2g/(scale^3*sqrt(pi))-h/(2x)
    end
    g,h,-g/scale^2,dh
end

function raw(theory,x)
    q=x./theory.thermal^2
    n=2+sum(theory.mode_counts)+2length(theory.exchange_scales)
    A=zeros(length(q),n); D=similar(A)
    A[:,1].=1.; A[:,2].=q; D[:,1].=0.; D[:,2].=inv(theory.thermal^2)
    col=2
    for (scale,nk) in zip(theory.scale_factors,theory.mode_counts)
        alpha=.5
        for j in 0:nk-1
            col+=1
            for (i,x) in enumerate(q)
                p=x/scale^2
                envelope=exp(-p/2)
                a=laguerre(j,p,alpha)
                da=j==0 ? 0. : -laguerre(j-1,p,alpha+1)
                A[i,col]=envelope*a
                D[i,col]=envelope*(da-a/2)/(scale*theory.thermal)^2
            end
        end
    end
    for radius in theory.exchange_scales
        for (i,x) in enumerate(q)
            g,h,dg,dh=rate_features(x,radius)
            A[i,col+1]=g; A[i,col+2]=h
            D[i,col+1]=dg/theory.thermal^2; D[i,col+2]=dh/theory.thermal^2
        end
        col+=2
    end
    A,D
end

function laguerre(n,x,a)
    n==0 && return 1.
    p=1.; q=1+a-x
    for k in 2:n
        p,q=q,((2k-1+a-x)*q-(k-1+a)*p)/k
    end
    q
end

struct Space
    theory::Theory
    cells::Vector{Float64}
    knots::Vector{Float64}
    transform::Matrix{Float64}
end
basis(s::Space,x)=map(M->M*s.transform,raw(s.theory,x))
KD.RadialHighOrder.evaluate_space_basis(s::Space,x)=basis(s,x)

function radial_grid(edges,order)
    nodes,weights=gausslegendre(order)
    x=Float64[]; w=Float64[]
    for (lo,hi) in zip(edges[1:end-1],edges[2:end])
        r=(lo+hi)/2 .+ (hi-lo)/2 .* nodes
        append!(x,r);append!(w,2pi*(hi-lo).*r .^ 2 .* weights)
    end
    x,w
end

function make_space(w,number,edges,modes,scales,exchange;rank_threshold=2e-9)
    theory=Theory(w,Float64.(scales),Int.(modes),Float64.(exchange))
    r,v=radial_grid(edges,36)
    A,_=raw(theory,r.^2)
    norm_weights=sqrt.(v.*exp.(-(r./w).^2).*number/(pi^1.5*w^3))
    core=norm_weights.*A[:,1:2]
    features=norm_weights.*A[:,3:end]
    projection=core\features
    projected=features-core*projection
    # Equalize candidate norms before the rank check, including core
    # features whose physical support is small relative to the electron.
    norms=vec(sqrt.(sum(abs2,projected;dims=1)))
    Q=svd(projected./transpose(norms))
    k=count(>(first(Q.S)*rank_threshold),Q.S)
    B=(Q.V[:,1:k]./transpose(Q.S[1:k]))./norms
    transform=vcat(hcat(Matrix{Float64}(I,2,2),-projection*B),hcat(zeros(size(B,1),2),B))
    Space(theory,edges.^2,collect(1.:k+2),transform),Q.S
end

function geometry(s::Space,dmax,espeed,mass,gamma,edges,order)
    d=KD.MomentSpace(edges[2].^2,collect(0.:dmax),dmax,espeed)
    offsets=[0,length(s.knots),length(s.knots)+length(d.knots)]
    raw=KD.RadialHighOrder.pair_geometry([s,d],gamma,mass,offsets;quadrature_order=order)
    points=Float64[];species=Int[]
    index=Dict{Tuple{Int,Float64},Int}()
    function point(a,x)
        get!(index,(a,x)) do
            push!(points,x);push!(species,a)
            length(points)
        end
    end
    square=[s.theory.thermal^2,1.5espeed^2]
    function cell(c,a)
        x=c.basis[:,2]*square[a]
        KD.ProfileCell(c,[point(a,v) for v in x])
    end
    selector=zeros(last(offsets))
    selector[2]=square[1]
    selector[offsets[2]+2]=square[2]
    pairs=map(raw) do p
        i,j=p.triangle_left_density*selector,p.triangle_right_density*selector
        left=p.triangle_left_density[:,offsets[p.a]+1];right=p.triangle_right_density[:,offsets[p.a]+1]
        a,b=p.a,p.b
        KD.ProfilePair(a,b,p.coefficient,cell.(p.cells_a,a),cell.(p.cells_b,b),
            Matrix(p.triangle_gradient),p.triangle_weight,
            [point(left[k]!=0 ? a : b,i[k]) for k in eachindex(i)],
            [point(right[k]!=0 ? a : b,j[k]) for k in eachindex(j)])
    end
    ids=findall(==(1),species)
    d_ids=findall(==(2),species)
    A,D=basis(s,points[ids])
    (;geometry=KD.Geometry(sqrt.(max.(0.,points)),species,pairs,offsets,mass,dmax,espeed),
       A,D,ids,d_ids)
end

function weak(g,density,gradient)
    geo=g.geometry
    result=zeros(last(geo.offsets))
    for p in geo.pairs
        KD.ordered_source!(result,p,p.a,p.b,p.cells_a,p.cells_b,density,gradient,geo)
        p.a!=p.b && KD.ordered_source!(result,p,p.b,p.a,p.cells_b,p.cells_a,density,gradient,geo)
        flux=[p.triangle_weight[j]*density[l]*density[r]*
            (gradient[r]/geo.masses[geo.species[r]]-gradient[l]/geo.masses[geo.species[l]])
            for (j,(l,r)) in enumerate(zip(p.left,p.right))]
        mul!(result,transpose(p.triangle_gradient),flux,-1.,1.)
    end
    result
end

function rates(g,α,deuteron,wD,nD)
    density=zeros(length(g.geometry.points))
    gradient=similar(density)
    density[g.ids]=exp.(g.A*α)
    gradient[g.ids]=g.D*α
    for i in g.d_ids
        density[i],gradient[i]=KD.profile(deuteron,wD,nD,g.geometry.points[i])
    end
    weak(g,density,gradient)
end

# Read-back of a positive piecewise-polylog Landau state and its log slope.
function reference_profile(ref,t,a,r)
    xx=r^2
    boundaries=ref.cells[a]
    xx>last(boundaries) && return 0.,0.
    cell=clamp(searchsortedlast(boundaries,xx),1,length(boundaries)-1)
    width=2/(boundaries[cell+1]-boundaries[cell])
    x=width*(xx-boundaries[cell])-1
    c=ref.coefficients[(t,a,cell)]
    d=ref.degree[a]
    v=last(c); g=0.
    for j in d:-1:1
        gap=x+cospi((j-1)/d)
        g=v+gap*g
        v=c[j]+gap*v
    end
    exp(v),width*g
end

function ref_rates(g,ref,t;electron=nothing)
    density=zeros(length(g.geometry.points)); gradient=similar(density)
    for (i,(a,r)) in enumerate(zip(g.geometry.species,g.geometry.points))
        density[i],gradient[i]=reference_profile(ref,t,a,r)
    end
    if electron!==nothing
        density[g.ids]=exp.(g.A*electron)
        gradient[g.ids]=g.D*electron
    end
    weak(g,density,gradient)
end

end
