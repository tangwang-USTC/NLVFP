"""Explicit weak-form problem. The evaluator takes (coefficient vector, time_tau, left_limit)."""
struct WeakEvolution{F,G,J,E}
    basis::Matrix{Float64}
    weights::Vector{Float64}
    particle_number::Float64
    initial::Vector{Float64}
    evaluate::F
    diagnostics::G
    jump::J
    events::E
end

Base.@kwdef struct StepOptions
    residual_limit::Float64 = 1e-12
    mass_residual_limit::Float64 = 1e-13
    invariant_limit::Float64 = 2e-12
    cpe_limit::Float64 = 2e-12
    H_increase_limit::Float64 = 1e-11
    switch_L1_limit::Float64 = 2e-10
    switch_heat_limit::Float64 = 2e-9
    switch_H_limit::Float64 = 1e-10
    iterations::Int = 20
end

struct StepRejected <: Exception
    reason::String
end
Base.showerror(io::IO, e::StepRejected) = print(io,e.reason)

mutable struct StepWorkspace
    factor::Union{Nothing,LU{Float64,Matrix{Float64},Vector{Int}}}
    dt::Float64
    stage_slope::Float64
    stage_time::Float64
    evaluations::Int
end

StepWorkspace() = StepWorkspace(nothing,NaN,NaN,NaN,0)

function evaluate_stage(problem, z, t, left, work)
    work.evaluations += 1
    problem.evaluate(z,t,left)
end

function trial_stage(problem, z, t, left, work)
    try
        evaluate_stage(problem,z,t,left,work)
    catch err
        err isa Union{DomainError,PosDefException,SingularException,StepRejected} || rethrow()
        nothing
    end
end

function jacobian(problem, y, r, residual, t, left, ne, work)
    J = zeros(length(y),length(y))
    u = similar(y)
    for j in eachindex(y)
        deltas = j<=ne ? (2e-8,2e-9,2e-10) : (1e-6,1e-7,1e-8,1e-9,1e-10,1e-11)
        found = false
        for δ in deltas, sign in (1.0,-1.0)
            found && break
            copyto!(u,y)
            eps_j = sign*δ*max(1.0,abs(y[j]))
            u[j] += eps_j
            probe = trial_stage(problem,u,t,left,work)
            probe === nothing && continue
            J[:,j] = (residual(probe)-r)/eps_j
            found = true
        end
        found || throw(StepRejected("No admissible finite difference for Newton column $j"))
    end
    lu(J)
end

function implicit_stage(problem, old, time_tau, left, h, e_explicit, d_explicit, seed, work, options)
    ne = size(problem.basis,2)
    dcols = ne+1:length(old.z)
    function residual(state)
        increment = problem.basis'*(problem.weights.*old.f.*expm1.(state.lp-old.lp))
        increment .-= e_explicit + h*state.e_dot
        increment[1] = state.obs[1]-problem.particle_number
        d = view(state.z,dcols)-view(old.z,dcols)-d_explicit-h*state.dc
        vcat(increment,1e-4.*d)
    end
    y = copy(old.z)
    view(y,dcols) .+= seed
    current = trial_stage(problem,y,time_tau,left,work)
    if current === nothing
        y = copy(old.z)
        # Always re-evaluate the state at this stage's time and side.
        current = trial_stage(problem,y,time_tau,left,work)
        current === nothing && throw(StepRejected("Neither stage seed nor parent is admissible"))
    end
    r = residual(current)
    for it in 1:options.iterations
        defect = norm(r,Inf)
        if defect<=options.residual_limit && abs(r[1])<=options.mass_residual_limit
            return (;current,it,defect)
        end
        if work.factor === nothing || work.dt != h || work.stage_time != time_tau ||
                work.stage_slope != current.slope_value || it in (6,12,17)
            work.factor = jacobian(problem,y,r,residual,time_tau,left,ne,work)
            work.dt = h
            work.stage_time = time_tau
            work.stage_slope = current.slope_value
        end
        delta = -(work.factor\r)
        moved = false
        for j in 0:15
            next = y + 2.0^-j*delta
            candidate = trial_stage(problem,next,time_tau,left,work)
            candidate === nothing && continue
            v = residual(candidate)
            if norm(v,Inf) < defect
                y,current,r,moved = next,candidate,v,true
                break
            end
        end
        moved || (work.factor = nothing)
    end
    throw(StepRejected("Implicit stage residual $(norm(r,Inf)) exceeded its gate"))
end

function physical_gate(value, H0, options)
    all(isfinite,(value.number,value.momentum,value.energy,value.cpe,value.H)) ||
        throw(StepRejected("Nonfinite physical diagnostic"))
    max(abs(value.number),abs(value.momentum),abs(value.energy)) <= options.invariant_limit ||
        throw(StepRejected("Theoretical invariant defect exceeded its gate"))
    abs(value.cpe) <= options.cpe_limit || throw(StepRejected("CPE residual exceeded its gate"))
    value.H-H0 <= options.H_increase_limit || throw(StepRejected("Total H increased beyond its gate"))
end

function validate_step_options(options)
    all(p -> isfinite(p) && p>0, (options.residual_limit,options.mass_residual_limit,
        options.invariant_limit,options.cpe_limit,options.H_increase_limit,options.switch_L1_limit,
        options.switch_heat_limit,options.switch_H_limit)) && options.iterations>0 ||
        throw(ArgumentError("Expected finite positive tolerances and a positive Newton iteration cap"))
end

"""
    advance_step!(work, problem, z, t0, t1; method=:sdirk2, options=StepOptions())

Advance one step, keeping caller-owned z intact. Scheduled knots must be step
endpoints. The old slope closes the step at a discontinuity, then the candidate
right limit is checked separately against distribution/source/H switch gates.
"""
function advance_step!(work::StepWorkspace, problem::WeakEvolution, z, t0::Real, t1::Real;
                       method=:sdirk2, options=StepOptions())
    validate_step_options(options)
    a,b = Float64(t0),Float64(t1)
    isfinite(a) && isfinite(b) && 0<=a<b || throw(ArgumentError("Expected finite increasing step times"))
    length(z)==length(problem.initial) && all(is_real_finite,z) ||
        throw(ArgumentError("Wrong evolution state dimension or values"))
    any(t -> a<t<b,problem.events) && throw(ArgumentError("Step crosses a scheduled slope knot"))
    method in (:sdirk2,:trapezoid) || throw(ArgumentError("Expected :sdirk2 or :trapezoid"))
    h = b-a
    left = b in problem.events
    work.factor = nothing
    start_evaluations = work.evaluations
    old = evaluate_stage(problem,z,a,false,work)
    previous = problem.diagnostics(old)
    physical_gate(previous,previous.H,options)
    if method === :sdirk2
        γ = 1-inv(sqrt(2.0))
        first_stage = implicit_stage(problem,old,a+γ*h,false,γ*h,zero(old.e_dot),
            zero(old.dc),γ*h*old.dc,work,options)
        mid = first_stage.current
        terminal = implicit_stage(problem,old,b,left,γ*h,(1-γ)*h*mid.e_dot,
            (1-γ)*h*mid.dc,h*mid.dc,work,options)
        iterations = first_stage.it + terminal.it
        residual = max(first_stage.defect,terminal.defect)
    else
        terminal = implicit_stage(problem,old,b,left,h/2,h/2*old.e_dot,h/2*old.dc,
            h*old.dc,work,options)
        iterations,residual = terminal.it,terminal.defect
    end
    final = terminal.current
    diagnostic = problem.diagnostics(final)
    physical_gate(diagnostic,previous.H,options)
    switch = nothing
    if left
        right = evaluate_stage(problem,final.z,b,false,work)
        if right.slope_value != final.slope_value
            right_diagnostic = problem.diagnostics(right)
            physical_gate(right_diagnostic,diagnostic.H,options)
            gap = problem.jump(final,right)
            switch = (; gap..., H=abs(right_diagnostic.H-diagnostic.H),
                        before=final.slope_value,after=right.slope_value)
            all(isfinite,(switch.L1,switch.heat,switch.H)) &&
                0<=switch.L1<=options.switch_L1_limit && 0<=switch.heat<=options.switch_heat_limit &&
                switch.H<=options.switch_H_limit ||
                throw(StepRejected("Slope switch at t/τ=$b failed continuity gates: $switch"))
            final,diagnostic = right,right_diagnostic
        end
    end
    physical_gate(diagnostic,previous.H,options)
    (;z=copy(final.z), time_tau=b, slope=final.slope_value, diagnostic,
       H_increment=diagnostic.H-previous.H, residual, iterations,
       evaluations=work.evaluations-start_evaluations, switch, method)
end

"""
    evolve!(work, problem, z, times; method=:case1, on_accept=identity)

Compute on the prescribed time grid with interior knots inserted. :case1 uses
SDIRK2 before t/τ=0.125 and trapezoid thereafter. The callback receives each
accepted step, and the function returns the last step. Rejections propagate.
"""
function evolve!(work::StepWorkspace, problem::WeakEvolution, z, times;
                 method=:case1, options=StepOptions(), on_accept=identity)
    grid = collect(Float64.(times))
    length(grid)>=2 && all(t -> isfinite(t) && t>=0,grid) &&
        all(i -> grid[i-1]<grid[i],2:length(grid)) ||
        throw(ArgumentError("Time grid must be finite, nonnegative and strictly increasing"))
    knots = method===:case1 ? (problem.events...,0.125) : problem.events
    complete = sort!(unique!(vcat(grid,filter(t -> first(grid)<t<last(grid),collect(knots)))))
    start = z
    result = nothing
    for i in 2:length(complete)
        a,b = complete[i-1],complete[i]
        scheme = method===:case1 ? (a<0.125 ? :sdirk2 : :trapezoid) : method
        result = advance_step!(work,problem,start,a,b;method=scheme,options)
        start = copy(result.z)
        on_accept===identity || on_accept(merge(result,(z=copy(result.z),)))
    end
    result
end
