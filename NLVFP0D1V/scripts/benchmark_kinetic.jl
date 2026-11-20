using NLVFP0D1V, LinearAlgebra
include(joinpath(@__DIR__,"../test/analytic_case.jl"))

function measure(operation, repetitions)
    first_call = @elapsed result = operation()
    stable = operation()
    @assert stable == result
    bytes = minimum(@allocated(operation()) for _ in 1:repetitions)
    times = sort!([@elapsed(operation()) for _ in 1:repetitions])
    (;first_call_seconds=first_call, steady_seconds=times[cld(repetitions,2)], steady_bytes=bytes, result)
end

function main()
    BLAS.set_num_threads(1)
    h = 2.0^-31
    model = king_case1(case1_smoke_input())
    parent = advance_step!(StepWorkspace(),model,model.initial,0.0,h).z
    source() = begin
        stage = model.evaluate(parent,h,false)
        (;stage.udot,stage.cpe)
    end
    workspace = StepWorkspace()
    step() = begin
        result = advance_step!(workspace,model,parent,h,2h)
        (;result.residual,result.evaluations,result.iterations, result.H_increment)
    end
    println("Affine K10 at the same analytic-start parent state, OpenBLAS threads=",BLAS.get_num_threads())
    println("Collision and reconstruction: ",measure(source,7))
    println("Implicit SDIRK2 step: ",measure(step,7))
end

main()
