function test_case(name, file)
    joinpath(@__DIR__,"cases",name,file)
end

function load_test_case(name)
    load_case(test_case(name,"physical-input.toml"),test_case(name,"dimensionless-input.toml"))
end

case1_smoke_input() = load_test_case("ed_relaxation")
