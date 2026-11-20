"""
Validated input deck for the isotropic 0D1V physical-authority mode.

Physical values are given in SI and keV. The accompanying independent
dimensionless file fixes the grid and checks the normalization.
"""
struct CaseDeck
    id::String
    _input::Dict{String,Any}
    scales::NamedTuple
    times_tau::Vector{Float64}
    quadrature_order::Int
end

const FORMULATION = "0D1V-isotropic-3V-radial"

function check_case_identity(p, q)
    contract = ("schema", "authority", "formulation", "case_id", "species")
    for k in contract
        p[k] == q[k] || throw(ArgumentError("Input decks disagree on $k"))
    end
    p["schema"]==1 && p["authority"]=="physical" && p["formulation"]==FORMULATION &&
        p["species"]==["e","D"] && p["case_id"] isa String && !isempty(strip(p["case_id"])) ||
        throw(ArgumentError("Expected the schema 1, physical-authority e–D radial input deck"))
    for deck in (p,q)
        deck["source"] isa String && !isempty(strip(deck["source"])) ||
            throw(ArgumentError("Each deck requires a source description"))
    end
end

function check_case_assertion(field, value, expected; rtol)
    if value isa AbstractVector
        expected isa AbstractVector && length(expected)==length(value) ||
            throw(ArgumentError("Wrong assertion dimension for $field"))
        all(is_real_finite,expected) && all(isapprox.(value,expected;rtol,atol=0.0)) ||
            throw(ArgumentError("Dimensionless assertion $field: computed $value; expected $expected"))
    else
        is_real_finite(expected) && isapprox(value,expected;rtol,atol=0.0) ||
            throw(ArgumentError("Dimensionless assertion $field: computed $value; expected $expected"))
    end
end

function case_scales(physical, p)
    ec = 1.602176634e-19
    ε0 = 8.8541878188e-12
    n_ref = first(physical["densities_m3"])
    mass_ref = first(physical["mass_kg"])
    T_ref = physical["temperature_unit_keV"]
    v_ref = sqrt(2T_ref*1000ec/mass_ref)
    charge = physical["charge_e"]
    coefficient = physical["coulomb_log"][1][2]*(charge[1]*charge[2])^2
    Γ_si = 2π*(ec^2/(4π*ε0))^2*coefficient
    τ = p.gamma[1,2]*mass_ref^2*v_ref^3/(Γ_si*n_ref)
    (;density_m3=n_ref,mass_kg=mass_ref,temperature_keV=T_ref,velocity_m_per_s=v_ref,time_s=τ)
end

"""
    load_case(physical_path, dimensionless_path)

Validate the two independent TOML input files. Resolve scale assertions
without replacing physical values. Throws ArgumentError on a conflict.
"""
function load_case(physical_path::AbstractString, dimensionless_path::AbstractString)
    isfile(physical_path) && isfile(dimensionless_path) ||
        throw(ArgumentError("Supply both required input files"))
    realpath(physical_path) != realpath(dimensionless_path) ||
        throw(ArgumentError("The physical and dimensionless inputs must be separate files"))
    physical = TOML.parsefile(physical_path)
    dimensionless = TOML.parsefile(dimensionless_path)
    check_case_identity(physical,dimensionless)
    input = copy(physical["plasma"])
    Set(keys(input)) == Set(("mass_kg","densities_m3","temperature_keV","charge_e",
                             "coulomb_log","temperature_unit_keV")) ||
        throw(ArgumentError("The physical block must contain exactly the documented physical fields"))
    p = case1_physics(input)
    scales = case_scales(input,p)
    checks = (;
        mass_ratios=p.m, density_ratios=p.n, thermal_widths=p.w, total_energy=p.energy,
        cross_collision_coefficient=p.gamma[1,2], velocity_unit_m_per_s=scales.velocity_m_per_s,
        time_unit_s=scales.time_s)
    values = dimensionless["assertions"]
    rtol = dimensionless["relative_assertion_tolerance"]
    is_real_finite(rtol) && 0<rtol<=1e-10 || throw(ArgumentError("Invalid scale assertion tolerance"))
    Set(keys(values)) == Set(string.(keys(checks))) ||
        throw(ArgumentError("Expected exactly the documented scale assertions"))
    for (key,value) in pairs(checks)
        field = String(key)
        check_case_assertion(field,value,values[field];rtol)
    end

    options = dimensionless["discretization"]
    radial_keys = ("electron_edges_in_initial_thermal_speed","deuteron_edges_in_initial_thermal_speed")
    Set(keys(options)) == Set((radial_keys...,"quadrature_order","time_nodes_tau")) ||
        throw(ArgumentError("The discretization block must contain only grid, time and quadrature fields"))
    for k in radial_keys
        input[k] = options[k]
    end
    case1_edges(input,p.w)
    q = options["quadrature_order"]
    q isa Integer && q>=2 || throw(ArgumentError("Expected an integer quadrature order of at least two"))
    t = Float64.(options["time_nodes_tau"])
    length(t)>=2 && iszero(first(t)) && all(isfinite,t) &&
        all(i -> t[i-1]<t[i], 2:length(t)) ||
        throw(ArgumentError("Time nodes must start from zero and increase strictly"))
    CaseDeck(physical["case_id"],input,scales,t,q)
end

"""Build the problem from a checked two-file deck."""
king_case1(case::CaseDeck;quadrature_order=case.quadrature_order,kwargs...) =
    king_case1(case._input;quadrature_order,kwargs...)
