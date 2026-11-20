using NLVFP0D1V, LinearAlgebra
BLAS.set_num_threads(1)

function benchmark(sample,f,repetitions)
    first_time = @elapsed reference = observe(sample,f)
    scratch = ObservationWorkspace(sample)
    result = similar(reference)
    @assert observe!(result,scratch,sample,f) == reference
    allocating() = begin
        a = 0.0
        for _ in 1:repetitions
            a += first(observe(sample,f))
        end
        a
    end
    in_place() = begin
        a = 0.0
        for _ in 1:repetitions
            observe!(result,scratch,sample,f)
            a += first(result)
        end
        a
    end
    @assert allocating()==in_place()
    function measure(operation)
        bytes = @allocated operation()
        times = [@elapsed operation() for _ in 1:5]
        (;bytes_per_projection=bytes/repetitions, seconds_per_projection=sort(times)[3]/repetitions)
    end
    println("First allocating call (s): ",first_time)
    println("Steady allocating: ",measure(allocating))
    println("Steady in-place: ",measure(in_place))
    println("Maximum output difference: ",maximum(abs,result-reference))
end

r = collect(range(1/4096,12.0;length=4096))
rule = radial_measure(r,fill(12.0/4096,4096);convention=:dr,coordinate="r / v_ref")
shape = radial_space("manufactured-42",["psi$j" for j in 0:41],
    points -> [exp(-x*x/2)*cos(j*x/4) for x in points,j in 0:41])
benchmark(sample_basis(shape,rule),exp.(-r.^2),200)
