
function manufactured(law; events=(), discontinuity=0.0, jump_L1=0.0, calls=nothing)
    evaluator(z,t,left) = begin
        λ = law(left ? prevfloat(t) : t)
        calls === nothing || push!(calls,(t,left,λ))
        (;z=copy(z),lp=[z[1]],f=[exp(z[1])],obs=[exp(z[1])],
         e_dot=[0.0],dc=[λ],slope_value=λ)
    end
    diagnostics(s) = (;number=s.obs[1]-1,momentum=0.0,energy=0.0,cpe=0.0,H=-s.z[2])
    jump(x,y) = (;L1=jump_L1,heat=discontinuity)
    WeakEvolution(ones(1,1),[1.0],1.0,[0.0,0.0],evaluator,diagnostics,jump,events)
end

@testset "Explicit stages and second-order manufactured evolution" begin
    q = manufactured(t -> 1+t*t)
    exact = 0.25+0.25^3/3
    for scheme in (:sdirk2,:trapezoid)
        errors = map((4,8,16)) do n
            end_state = evolve!(StepWorkspace(),q,q.initial,range(0,0.25;length=n+1);method=scheme)
            @test end_state.residual <= 1e-12
            @test end_state.H_increment < 0
            abs(end_state.z[2]-exact)
        end
        @test all(x -> 3.95<x<4.05, [errors[1]/errors[2],errors[2]/errors[3]])
    end
    calls = Tuple{Float64,Bool,Float64}[]
    q = manufactured(t -> 1+t*t;calls)
    start = copy(q.initial)
    step = advance_step!(StepWorkspace(),q,start,0.0,0.25)
    @test start == q.initial
    @test any(x -> x[1] == (1-inv(sqrt(2.0)))*0.25, calls)
    @test step.time_tau == 0.25 && step.slope == 1.0625
    @test_throws ArgumentError advance_step!(StepWorkspace(),q,start,1.0,0.0)
    @test_throws ArgumentError evolve!(StepWorkspace(),q,start,[0.0,0.25,0.25])
    @test_throws ArgumentError advance_step!(StepWorkspace(),q,start,0.0,0.25;options=StepOptions(iterations=0))
    a = Tuple{Float64,Symbol}[]
    evolve!(StepWorkspace(),q,start,[0.0,0.25];on_accept=s->push!(a,(s.time_tau,s.method)))
    @test a == [(0.125,:sdirk2),(0.25,:trapezoid)]
end

@testset "Checkpoint alignment and continuity gates" begin
    law(t) = t < 0.125 ? 1.0 : 2.0
    calls = Tuple{Float64,Bool,Float64}[]
    q = manufactured(law;events=(0.125,),calls)
    @test_throws ArgumentError advance_step!(StepWorkspace(),q,q.initial,0.0,0.25)
    accepted = NamedTuple[]
    final = evolve!(StepWorkspace(),q,q.initial,[0.0,0.25];method=:trapezoid,on_accept=x->push!(accepted,x))
    @test length(accepted)==2
    @test accepted[1].z[2] ≈ 0.125
    @test accepted[1].switch.before==1 && accepted[1].switch.after==2
    @test any(==((0.125,true,1.0)),calls) && any(==((0.125,false,2.0)),calls)
    @test final.z[2] ≈ 0.375
    bad = manufactured(law;events=(0.125,),jump_L1=1e-3)
    @test_throws StepRejected advance_step!(StepWorkspace(),bad,bad.initial,0.0,0.125)
    @test bad.initial == [0.0,0.0]
end
