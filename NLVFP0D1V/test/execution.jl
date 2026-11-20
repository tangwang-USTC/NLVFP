using TOML

@testset "Execution and inversion precision records" begin
    m = speed_moment_target(0:2:6,[1.0,1.0,1.25,1.75];
        normalization=:thermal_maxwell,velocity_scale=1.0,coordinate="normalized speed")
    result = invert_zero_offset(m,2;precision_bits=256)
    @test result.effective_count==2
    @test result.precision==(moment_pencil_bits=256,eigen=Float64,output=Float64)
    @test result.relative_residual<1e-13
    maxwell = speed_moment_target(0:2:6,ones(4);
        normalization=:thermal_maxwell,velocity_scale=1.0,coordinate="normalized speed")
    collapsed = invert_zero_offset(maxwell,2;collapse_maxwell=true)
    @test collapsed.effective_count==1 && collapsed.precision.eigen===nothing
    meta = execution_metadata()
    @test meta.library_version == TOML.parsefile(joinpath(@__DIR__,"../Project.toml"))["version"]
    @test meta.formulation == "0D1V-isotropic-3V-radial"
    @test meta.julia_version == string(VERSION)
    @test meta.blas_threads == BLAS.get_num_threads()
    @test meta.shell_cholesky_bits==160 && meta.shell_eigen_precision==Float64
end
