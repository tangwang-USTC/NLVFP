"""
    execution_metadata(; options=StepOptions())

Report the interpreter, threading, active environment, and step tolerances
for an execution. The driver can attach the values to future run records.
"""
function execution_metadata(;options=StepOptions())
    validate_step_options(options)
    project = Base.active_project()
    manifest = project===nothing ? nothing : joinpath(dirname(project),"Manifest.toml")
    filehash(path) = path!==nothing && isfile(path) ? open(io -> bytes2hex(sha256(io)),path) : nothing
    tolerances = NamedTuple{fieldnames(typeof(options))}(Tuple(getfield(options,i) for i in 1:fieldcount(typeof(options))))
    (;library_version=string(Base.pkgversion(@__MODULE__)), formulation=FORMULATION,
       julia_version=string(VERSION), platform=string(Sys.KERNEL,"-",Sys.ARCH),
       julia_threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
       blas_configuration=sprint(show,BLAS.get_config()),
       project_sha256=filehash(project),manifest_sha256=filehash(manifest),
       tolerances, evolution_precision=Float64,rank_diagnostic_precision=Float64,
       affine_pencil_bits=256,shell_cholesky_bits=160,shell_eigen_precision=Float64)
end

"""
    invert_zero_offset(target, count; precision_bits=384, collapse_maxwell=false)

Positive zero-offset rule on an ordered SpeedMomentTarget. Report the
working precision of each decomposition separately from the fit residual.
"""
function invert_zero_offset(target::SpeedMomentTarget, count::Integer; precision_bits::Integer=384,
                            collapse_maxwell::Bool=false)
    target.normalization === :thermal_maxwell && first(target.orders)==0 &&
        abs(first(target.values)-1)<=2e-12 ||
        throw(ArgumentError("Expected same-temperature normalized zero-offset moments with M0=1"))
    energy = findfirst(==(2),target.orders)
    energy === nothing || abs(target.values[energy]-1)<=2e-12 ||
        throw(ArgumentError("Same-temperature M2 must be 1"))
    result = EDCase1Interfaces.EDKingObservations.zero_offset_inverse(target.orders,target.values,count;
        precision_bits,collapse_maxwell)
    exact_collapse = result.effective_count < result.requested_count
    (;result..., precision=(;moment_pencil_bits=exact_collapse ? nothing : precision_bits,
       eigen=exact_collapse ? nothing : Float64, output=Float64))
end
