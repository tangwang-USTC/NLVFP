
module EDKingObservations

using LinearAlgebra
using SpecialFunctions: loggamma

export speed_orders, king_moments, maxwell_moment_factor, generalized_observations,
       king_parameter_count, zero_offset_inverse, affine_widths, affine_shell_moments,
       lift_log_density, collision_tangent_guess

"""
    speed_orders(mode, highest; conserved=false)

Return the radial speed orders for :even, :odd or :integer. The optional
conserved flag includes orders 0 and 2. An empty selection has no coordinates.
"""
function speed_orders(mode::Symbol, highest::Integer; conserved=false)
    highest>=0||throw(ArgumentError("Expected a nonnegative highest speed order"))
    orders=if mode===:even
        collect(0:2:highest)
    elseif mode===:odd
        collect(1:2:highest)
    elseif mode===:integer
        collect(0:highest)
    else
        throw(ArgumentError("Expected :even, :odd or :integer speed powers"))
    end
    conserved ? sort!(union!(orders,[0,2])) : orders
end

"""Coefficient in the physical Maxwell moment N*vth^p*C_p."""
function maxwell_moment_factor(p::Integer, ::Type{T}=Float64) where {T<:AbstractFloat}
    p>=0||throw(ArgumentError("Expected a nonnegative speed order"))
    odd=isodd(p)
    start=Int(odd)
    c=odd ? 2/sqrt(T(pi)) : one(T)
    for j in start:2:p-2
        c*=T(j+3)/2
    end
    c
end

function check_atoms(a,s,q)
    length(a)==length(s)==length(q) && !isempty(a)||throw(DimensionMismatch("One weight, squared width and squared offset per King component"))
    all(x->isfinite(x)&&x>0,a)&&all(x->isfinite(x)&&x>0,s)&&all(x->isfinite(x)&&x>=0,q) ||
        throw(DomainError((a,s,q),"Expected positive weights and squared widths, and nonnegative squared offsets"))
end

function even_kernel(s,q,k)
    value=s^k
    # Sum the polynomial with only positive terms. Start with the widths
    # and offsets themselves so that q=0 is an exact central moment.
    coefficient=one(s)
    for j in 1:k
        coefficient*=oftype(s,k-j+1)/(oftype(s,j)*(oftype(s,j)+oftype(s,0.5)))
        value+=coefficient*s^(k-j)*q^j
    end
    value
end

# Relative Poisson weights from the mode. The Gamma-ratio recurrence
# tilts those same weights for the odd fractional moment.
function odd_kernel(s,q,p)
    alpha=oftype(s,1.5)
    power=oftype(s,p)/2
    q==0 && return s^power
    mu=q/s
    isfinite(mu) && mu<typemax(Int) || throw(DomainError(mu,"Poisson mode is outside integer indexing range"))
    mode=floor(Int,mu)
    number=one(s)
    moment=one(s)
    tol=4eps(typeof(s))
    plain=weighted=one(s)
    for j in mode:-1:1
        ratio=oftype(s,j)/mu
        tilt=(alpha+j-1)/(alpha+j-1+power)
        plain*=ratio
        weighted*=ratio*tilt
        number+=plain
        moment+=weighted
        next_ratio=oftype(s,j-1)/mu
        next_ratio<1 && plain*next_ratio<=tol*number*(1-next_ratio) &&
            weighted*next_ratio<=tol*moment*(1-next_ratio) && break
    end
    plain=weighted=one(s)
    j=mode
    while true
        prob_ratio=mu/(j+1)
        tilted_ratio=prob_ratio*(1+power/(alpha+j))
        if tilted_ratio<1 && plain*prob_ratio<=tol*number*(1-prob_ratio) &&
                              weighted*tilted_ratio<=tol*moment*(1-tilted_ratio)
            break
        end
        plain*=prob_ratio
        weighted*=tilted_ratio
        number+=plain
        moment+=weighted
        j+=1
    end
    logratio=loggamma(alpha+mode+power)-loggamma(alpha+mode)-
        (loggamma(alpha+power)-loggamma(alpha))
    exp(power*log(s)+logratio)*(moment/number)
end

"""
    king_moments(weights, squared_widths, squared_offsets, orders)

Return Maxwell-normalized radial moments in the order requested. Width and
offset use the same thermal speed as the reference Maxwellian. Weighted sums
retain the supplied total mass fraction.
"""
function king_moments(weights, squared_widths, squared_offsets, orders)
    check_atoms(weights,squared_widths,squared_offsets)
    all(p->p isa Integer && p>=0,orders)||throw(ArgumentError("Moment orders must be nonnegative integers"))
    T=mapreduce(x->typeof(float(x)),promote_type,Iterators.flatten((weights,squared_widths,squared_offsets)))
    values=[sum(T(ai)*(iseven(p) ? even_kernel(T(si),T(qi),p÷2) :
        odd_kernel(T(si),T(qi),p)) for (ai,si,qi) in zip(weights,squared_widths,squared_offsets)) for p in orders]
    all(isfinite,values)||throw(OverflowError("King moment exceeds numerical range"))
    values
end

"""Count raw slots and normalized shape coordinates for independently fitted widths."""
function king_parameter_count(count::Integer; variable_weights=true, variable_offsets=true)
    count>=1 || throw(ArgumentError("Expected at least one King component"))
    raw=count*(1+Int(variable_weights)+Int(variable_offsets))
    shape=raw-1-Int(variable_weights)
    first=variable_weights ? 0 : 2
    (;raw,shape,first_even_order=first,highest_even_order=first+2(raw-1))
end

"""
    zero_offset_inverse(orders, target, count; precision_bits=384, collapse_maxwell=false)

Positive Jacobi inversion for free weights and zero offsets. Supply exactly
2*count moments with orders 0:2*count-1 (width nodes) or 0:2:4*count-2
(squared-width nodes). Return effective rank and normalized fit residual.
"""
function zero_offset_inverse(orders, target, count::Integer; precision_bits=384, collapse_maxwell=false)
    count>=1||throw(ArgumentError("Expected at least one King component"))
    length(target)==2count||throw(DimensionMismatch("Expected 2*count target moments"))
    length(orders)==length(target)||throw(DimensionMismatch("Target order and value counts differ"))
    even=collect(orders)==collect(0:2:4count-2)
    integer=collect(orders)==collect(0:2count-1)
    even||integer||throw(ArgumentError("Expected a full even or consecutive integer sequence starting at 0"))
    all(x->isfinite(x)&&x>0,target)||throw(DomainError(target,"Expected finite positive radial moments"))
    if collapse_maxwell && all(==(first(target)),target)
        return (;weights=[Float64(first(target))], widths=[1.], effective_count=1,
            requested_count=count, relative_residual=0., measure=even ? :squared_width : :width)
    end
    setprecision(BigFloat,precision_bits) do
        m=BigFloat.(target)
        gram=[m[i+j-1] for i in 1:count,j in 1:count]
        shifted=[m[i+j] for i in 1:count,j in 1:count]
        L=cholesky(Symmetric(gram)).L
        isposdef(Symmetric(shifted))||throw(DomainError(target,"Width moment pencil has nonpositive support"))
        eigenpair=eigen(Symmetric(Matrix{Float64}(Symmetric(L\shifted/L'))))
        weights=Float64(first(m)).*abs2.(eigenpair.vectors[1,:])
        nodes=eigenpair.values
        all(>(0.),weights) && all(>(0.),nodes)||throw(DomainError(target,"Positive width rule was lost on readback"))
        widths=even ? sqrt.(nodes) : nodes
        residual=maximum(abs(sum(BigFloat.(weights).*BigFloat.(widths).^p)/t-1)
            for (p,t) in zip(orders,m))
        (;weights,widths,effective_count=count,requested_count=count,
            relative_residual=Float64(residual),measure=even ? :squared_width : :width)
    end
end

"""Compute temperature-normalized squared widths and the shared d for affine King."""
function affine_widths(weights,squared_offsets,lambda)
    length(weights)==length(squared_offsets) && !isempty(weights)||throw(DimensionMismatch("Affine width inputs"))
    all(x->isfinite(x)&&x>0,weights) && all(x->isfinite(x)&&x>=0,squared_offsets) &&
        isfinite(lambda)||throw(DomainError((weights,squared_offsets,lambda),"Invalid affine King parameters"))
    isapprox(sum(weights),1;rtol=2e-12,atol=0)||throw(DomainError(weights,"Expected unit mass fraction"))
    mean_offset=dot(weights,squared_offsets)
    d=(lambda+oftype(mean_offset,2)/3)*mean_offset
    squared_widths=1 .- d .+ lambda.*squared_offsets
    all(>(0.),squared_widths)||throw(DomainError(squared_widths,"Nonpositive affine widths"))
    (;squared_widths,d)
end

"""Convert even Maxwell-normalized physical targets to signed shell power moments for admissibility checks."""
function affine_shell_moments(target,d,lambda; precision_bits=256)
    !isempty(target) && all(x->isfinite(x)&&x>0,target) && isfinite(d) && isfinite(lambda) ||
        throw(DomainError(target,"Expected positive even radial targets and a finite affine parameter"))
    setprecision(BigFloat,precision_bits) do
        b=1-BigFloat(d)
        slope=BigFloat(lambda)
        m=BigFloat.(target)
        values=BigFloat[]
        denominators=ones(BigFloat,length(m))
        for j in 1:length(m)-1
            denominators[j+1]=denominators[j]*(BigFloat(j)+.5)
        end
        for j in 0:length(m)-1
            a=sum(binomial(big(j),l)*slope^(j-l)/denominators[l+1] for l in 0:j)
            !iszero(a)||throw(DomainError(lambda,"Affine moment transform is singular at order $(2j)"))
            push!(values,sum(binomial(big(j),k)*(-b)^(j-k)*m[k+1] for k in 0:j)/a)
        end
        values
    end
end

"""Compute generalized velocity observables from a sampled basis, radial measure and density."""
function generalized_observations(basis::AbstractMatrix, measure, density)
    size(basis,1)==length(measure)==length(density)||throw(DimensionMismatch("Generalized observation grid"))
    all(x->isfinite(x)&&x>=0,measure) && all(x->isfinite(x)&&x>=0,density) ||
        throw(DomainError((measure,density),"Generalized observation requires a finite nonnegative measure and density"))
    basis'*(measure.*density)
end

"""
    lift_log_density(coefficients, dimension; trailing=...)

Append zero log-density coefficients. The caller supplies a verified exact
embedding of the inherited basis. Coupled coordinates follow the new block.
"""
function lift_log_density(coefficients::AbstractVector, dimension::Integer; trailing=similar(coefficients,0))
    dimension>=length(coefficients)||throw(DimensionMismatch("Exact lift requires enough inherited directions"))
    vcat(coefficients,zeros(eltype(coefficients),dimension-length(coefficients)),trailing)
end

"""Collision-source Newton increment using a fixed basis and G*dalpha/dt=S."""
function collision_tangent_guess(basis::AbstractMatrix, measure, density, source, step; retained_prefix=0)
    size(basis,1)==length(measure)==length(density) && size(basis,2)==length(source) ||
        throw(DimensionMismatch("Tangent predictor grid or source"))
    isfinite(step) && step>=0 && 0<=retained_prefix<=length(source) ||
        throw(ArgumentError("Invalid tangent predictor controls"))
    gram=Symmetric(basis'*(basis.*(measure.*density)))
    isempty(source)&&throw(ArgumentError("A tangent predictor requires a mass direction"))
    all(==(1),view(basis,:,1))||throw(ArgumentError("The first basis direction must be the mass weight 1"))
    rhs=collect(source)
    # The standard electron basis keeps 1 as its first direction.
    rhs[1]=zero(eltype(rhs))
    change=step.*(cholesky(gram)\rhs)
    fill!(view(change,1:retained_prefix),0)
    change
end

end
