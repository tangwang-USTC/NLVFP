# Reuse frozen numerical kernels as libraries. Time control is in Evolution.jl.
include(joinpath(LEGACY,"reference/IsotropicKingDynamics.jl"))
include(joinpath(LEGACY,"reference/EDElectronObservables.jl"))
include(joinpath(LEGACY,"reference/C1PositiveShellClosure.jl"))
include(joinpath(LEGACY,"reference/EDContinuousShell.jl"))
const KD = IsotropicKingDynamics
const EO = EDElectronObservables
const MATRIX_RADIUS = 0.85
include(joinpath(LEGACY,"scripts/ED_spectral_shell_rule.jl"))
include(joinpath(LEGACY,"scripts/ED_prepared_King_source.jl"))

struct AffinePencils
    ratio::Vector{BigFloat}
    gram::Vector{Matrix{Float64}}
    shifted::Vector{Matrix{Float64}}
end

mutable struct PencilCache
    slope::Float64
    pencils::Union{Nothing,AffinePencils}
end

# Same whitened pencil as the frozen K10 rule. Allow an explicit shared slope.
function affine_pencils(n, slope)
    setprecision(BigFloat,256) do
        p = C1PositiveShellClosure.gamma_moments(n,BigFloat)
        a(k) = sum(binomial(k,j)*BigFloat(slope)^(k-j)/p[j+1] for j in 0:k)
        ratio = [inv(p[k+1]*a(k)) for k in 0:n]
        prior = p.*ratio
        H = C1PositiveShellClosure.hankel
        function white(rows, shift)
            L = cholesky(Symmetric(H(prior,rows,shift))).L
            [Matrix{Float64}(Symmetric(L \ H([k<j ? big"0.0" : binomial(k,j)*prior[k+1]
                for k in 0:n],rows,shift)/L')) for j in 2:n]
        end
        AffinePencils(ratio,white(n÷2+1,0),white((n+1)÷2,1))
    end
end

function positive_shell(target,n,d)
    try
        C1PositiveShellClosure.invert(target,n,n;fixed_d=d)
    catch err
        err isa ErrorException && startswith(err.msg,"Nonpositive shell Jacobi completion") || rethrow()
        throw(StepRejected(err.msg))
    end
end

function reconstruct_deuteron(c,common,cache,λ)
    n = length(c)+1
    all(iszero,c) && return (;weights=fill(inv(n),n),widths=ones(n),shifts=zeros(n))
    coeff = vcat(big"1.0",big"0.0",BigFloat.(c))
    if iszero(λ)
        d = spectral_contraction(common,c)
        targets = [sum(binomial(k,j)*coeff[j+1] for j in 0:k) for k in 0:n]
        return positive_shell(targets,n,d)
    end
    if cache.pencils === nothing || cache.slope != λ
        cache.pencils = affine_pencils(n,λ)
        cache.slope = λ
    end
    selector = cache.pencils
    d = spectral_contraction(selector,c)
    base = 1-BigFloat(d)
    shell = [selector.ratio[k+1]*sum(binomial(k,j)*BigFloat(d)^(k-j)*coeff[j+1] for j in 0:k) for k in 0:n]
    targets = [sum(binomial(k,j)*base^(k-j)*shell[j+1] for j in 0:k) for k in 0:n]
    p = positive_shell(targets,n,d)
    s = 1-d .+ λ .* p.shifts.^2
    all(x -> isfinite(x) && x>0,s) || throw(StepRejected("Affine closure produced a nonpositive width"))
    (;p.weights,widths=sqrt.(s),p.shifts)
end

function case1_physics(d)
    rows = [d[k] for k in ("mass_kg","densities_m3","temperature_keV","charge_e")]
    all(v -> length(v)==2,rows) || throw(ArgumentError("Case 1 requires two species"))
    all(v -> all(x -> is_real_finite(x) && x>0,v),rows[1:3]) ||
        throw(ArgumentError("Mass, particle number and temperature must be finite and positive"))
    me,mD = rows[1]
    m = [1.0,mD/me]
    n = rows[2]./first(rows[2])
    T0 = Float64.(rows[3])
    tunit = d["temperature_unit_keV"]
    isfinite(tunit) && tunit>0 || throw(ArgumentError("Positive temperature unit required"))
    T = T0./tunit
    w = sqrt.(T./m)
    logΛ = reduce(vcat,permutedims.(d["coulomb_log"]))
    size(logΛ)==(2,2) || throw(DimensionMismatch("Coulomb logarithm matrix"))
    charge = rows[4]
    L = logΛ.*(charge*charge').^2
    issymmetric(L) && all(x -> is_real_finite(x) && x>0,L) ||
        throw(ArgumentError("Expected positive symmetric collision scaling"))
    equilibrium = dot(n,T)/sum(n)
    gamma_ref = 3sqrt(π)*prod(m)*(equilibrium*sum(inv.(m)))^1.5/(16sum(n))
    gamma = gamma_ref.*L./L[1,2]
    (;m,n,w,T0,energy=0.75dot(n,T),gamma)
end

function case1_edges(d,w)
    keys = ("electron_edges_in_initial_thermal_speed","deuteron_edges_in_initial_thermal_speed")
    map(1:2) do a
        edges = Float64.(d[keys[a]])
        length(edges)>=2 && iszero(first(edges)) && all(isfinite,edges) &&
            all(i -> edges[i-1]<edges[i],2:length(edges)) ||
            throw(ArgumentError("Radial boundaries must start at zero and strictly increase"))
        edges.*w[a]
    end
end

function k10_source(geo,α,model,w,n)
    f = zeros(length(geo.geometry.points))
    gradient = similar(f)
    f[geo.ids] = exp.(geo.A*α)
    gradient[geo.ids] = geo.D*α
    prepared = prepared_model(model,w,n)
    for i in geo.d_ids
        f[i],gradient[i] = prepared_profile(prepared,geo.geometry.points[i])
    end
    EO.weak(geo,f,gradient)
end

"""
    king_case1(input; width_rule=:affine, slope=constant_slope(), quadrature_order=12)

Build an isotropic two-species Case 1 model from physical input and radial
boundaries. Both routes evolve the same electron log observations and ten
deuteron components with hard even moments through M20. Each model owns its
selector cache and is intended for one evolution worker.
"""
function king_case1(input::AbstractDict; width_rule=:affine, slope::SlopePolicy=constant_slope(),
                    quadrature_order::Integer=12)
    width_rule in (:common,:affine) || throw(ArgumentError("Unknown King width rule $width_rule"))
    quadrature_order >= 2 || throw(ArgumentError("Collision quadrature needs at least two nodes"))
    p = case1_physics(input)
    w,N = p.w,p.n
    edges = case1_edges(input,w)
    ratio = w[2]/w[1]
    space,_ = EO.make_space(w[1],N[1],edges[1],[20,8,4],[1.0,0.13,ratio],
                           [0.94,1.08,1.22,3.0,5.0,8.0].*ratio)
    n = 10
    common = EDContinuousShell.Selector(n)
    cache = PencilCache(NaN,nothing)
    policy = width_rule===:common ? constant_slope(0.0) : slope
    policy.mode in (:cpe,:evolved) &&
        throw(ArgumentError("An independently supplied slope needs an augmented evolution model"))
    geo = EO.geometry(space,n,w[1],p.m,p.gamma,edges,quadrature_order)
    ne = length(space.knots)
    r,W = EO.radial_grid(edges[1],48)
    B,_ = EO.basis(space,r.^2)
    Ce = p.m[1]*w[1]^2/2
    D0 = 0.75*p.m[2]*N[2]*w[2]^2
    z0 = zeros(ne+n-1)
    z0[1] = log(N[1]/(π^1.5*w[1]^3))
    z0[2] = -1.0
    coefficients = [k<j ? 0.0 : binomial(k,j) for k in 0:n,j in 0:n]
    center = [(-1.0)^(k+j)*coefficients[k+1,j+1] for k in 0:n,j in 0:n]
    normal = [N[2]*ratio^(2j) for j in 0:n]
    scaled = 1e5
    function evaluate(z,t,left)
        α = view(z,1:ne)
        α[2] < -0.2 || throw(DomainError(α[2],"Nondecaying electron log density"))
        lp = B*α
        f = exp.(lp)
        all(isfinite,f) || throw(DomainError(α,"Electron density overflow"))
        obs = B'*(W.*f)
        ue = obs[2]/(1.5N[1])
        ud = (p.energy-Ce*obs[2])/D0
        ud>0 || throw(DomainError(ud,"Nonpositive deuteron thermal scale"))
        ws = w[2]*sqrt(ud)
        c = view(z,ne+1:length(z))./scaled
        time = left && policy.mode===:checkpoint ? prevfloat(t) : t
        context = (;electron_observations=obs, deuteron_centered=c,
            electron_temperature_keV=p.T0[1]*ue, deuteron_temperature_keV=p.T0[2]*ud)
        λ = Float64(resolve_slope(policy;time_tau=time,state=context).value)
        isfinite(λ) || throw(DomainError(λ,"Stage slope not representable in Float64"))
        # The positive Stieltjes spectral bound uses a nonnegative reference slope.
        λ>=0 || throw(DomainError(λ,"Time evolution requires a nonnegative affine slope"))
        model = reconstruct_deuteron(c,common,cache,λ)
        raw = k10_source(geo,α,model,ws,N[2])
        source = raw[ne+1:end]./normal
        udot = -Ce*raw[2]/D0
        exchange_error = abs(udot-source[2])
        exchange_error <= 3e-10 || throw(StepRejected("Weak electron/deuteron exchange imbalance $exchange_error"))
        moments = coefficients*vcat(1.0,0.0,c)
        rates = [source[j+1]/ud^j-j*moments[j+1]*udot/ud for j in 0:n]
        dc = scaled.*(center*rates)[3:end]
        cpe = maximum(abs(KD.normalized_moment(model,j)-moments[j+1]) for j in 0:n)
        cpe<=2e-12 || throw(StepRejected("Hard King M20 residual $cpe"))
        (;z=copy(z),lp,f,obs,dc,e_dot=raw[1:ne],model,ws,ue,ud,udot,exchange_error,cpe,slope_value=λ)
    end
    function diagnostics(s)
        prepared = prepared_model(s.model,s.ws,N[2])
        integrand(x) = begin
            f = first(prepared_profile(prepared,x*s.ws))
            f>0 ? 4π*s.ws^3*x^2*f*log(f) : 0.0
        end
        Hd = first(quadgk(integrand,0.0,1.0,2.0,4.0,8.0,16.0,Inf;rtol=2e-11,atol=1e-12))
        H = dot(W.*s.f,s.lp)+Hd
        number = max(abs(s.obs[1]/N[1]-1),abs(sum(s.model.weights)-1))
        energy = abs((s.obs[2]*Ce+D0*s.ud*KD.normalized_moment(s.model,1))/p.energy-1)
        (;H,number,momentum=0.0,energy,s.cpe,electron_temperature_keV=s.ue*p.T0[1],
          deuteron_temperature_keV=s.ud*p.T0[2],heat_keV_per_tau=p.T0[2]*s.udot)
    end
    function jump(before,after)
        A = prepared_model(before.model,before.ws,N[2])
        D = prepared_model(after.model,after.ws,N[2])
        integrand(x) = 4π*before.ws^3*x^2*abs(first(prepared_profile(A,x*before.ws))-
            first(prepared_profile(D,x*before.ws)))/N[2]
        L1 = first(quadgk(integrand,0.0,1.0,2.0,4.0,8.0,16.0,Inf;rtol=1e-7,atol=1e-13))
        (;L1,heat=p.T0[2]*abs(before.udot-after.udot))
    end
    events = policy.mode in (:checkpoint,:linear_time) ? policy.parameters.times_tau : ()
    WeakEvolution(B,W,N[1],z0,evaluate,diagnostics,jump,events)
end

king_case1(input::AbstractString; kwargs...) = king_case1(TOML.parsefile(input);kwargs...)
