## Mathematic constants
Pi = π |> datatype
const sqrt2 = 2^0.5
const sqrtpi = 1.772453850905516027298
const pi3 = π^3      # = π^3)
const sqrtpi3 = π^(3/2)      # = √(π^3) = π^(3/2)
const sqrt2pi = (2π)^0.5
const sqrt2invpi = (2 / π) ^ 0.5

const lnpi = log(pi)
const lnsqrtpi3 = log(sqrtpi3)

const epsT = eps(Float64)  # for the limit of the absolute error when `Float64` is used
const epsT1 = eps(Float64)  # for the limit of the absolute error when `Float64` is used
const epsT2 = 2eps(Float64)  # for the limit of the absolute error when `Float64` is used
const epsT5 = 5eps(Float64)  # for the limit of the absolute error when `Float64` is used
const epsT10 = 10eps(Float64)  # for the limit of the absolute error when `Float64` is used
const epsT100 = 100eps(Float64)  # for the limit of the absolute error when `Float64` is used
const epsT1000 = 1000eps(Float64)  # for the limit of the absolute error when `Float64` is used
const epsTe4 = 1e4 * epsT
const epsTe5 = 1e5 * epsT
const epsTe6 = 1e6 * epsT
const epsn3 = 1e-3
const epsn4 = 1e-4
const epsn6 = 1e-6
const epsn8 = 1e-8
const epsn10 = 1e-10
const epsn12 = 1e-12
const neps = 1 / epsT

const eps0 = epsT5
const epsT01 = epsT / 10
const epsT001 = epsT / 100
fmtf1(x) = parse(Float64,(generate_formatter( "%1.1e" )(x)))   # "%e", "%'e'", "%1.1e", [d,f,s,e]
fmtf2(x) = parse(Float64,(generate_formatter( "%1.2e" )(x)))
fmtf4(x) = parse(Float64,(generate_formatter( "%1.4e" )(x)))
fmtf6(x) = parse(Float64,(generate_formatter( "%1.6e" )(x)))
fmtf8(x) = parse(Float64,(generate_formatter( "%1.8e" )(x)))
fmtf10(x) = parse(Float64,(generate_formatter( "%1.10e" )(x)))
fmtf12(x) = parse(Float64,(generate_formatter( "%1.12e" )(x)))
fmtf14(x) = parse(Float64,(generate_formatter( "%1.14e" )(x)))
fmtf16(x) = parse(Float64,(generate_formatter( "%1.16e" )(x)))
fmt2(x) = generate_formatter( "%1.2e" )(x)
fmt4(x) = generate_formatter( "%1.4e" )(x)
fmt6(x) = generate_formatter( "%1.6e" )(x)
fmt7(x) = generate_formatter( "%1.7e" )(x)
fmt8(x) = generate_formatter( "%1.8e" )(x)
fmt10(x) = generate_formatter( "%1.10e" )(x)
fmt12(x) = generate_formatter( "%1.12e" )(x)
fmt14(x) = generate_formatter( "%1.14e" )(x)
fmt16(x) = generate_formatter( "%1.16e" )(x)
fmt18(x) = generate_formatter( "%1.18e" )(x)
