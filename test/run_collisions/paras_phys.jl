
 
## Physical Parameters: Moments        
nimps = [0, 0] * 0
is_Ek0 = true       # [Ta, Tb], when 
                    # `ua ~ ub, ∀T` are ok!
                    # `Ta ~ Tb, ∀u` are ok!
                    # `Ta ≫ Tb, ua ≫ ub` are ok!
                    # `Ta ≫ Tb, ua ≪ ub` are challenge now, which need a smart integral for `Rhcj` !

spices0 = [:e,   :e,    :H,      :D,       :Tri,    :α, :impurity1, :impurity2]  # hcat() = cat(v,dims=2)
# variables: moments                                                                       # Normalized by
n0 = [ne1,       ne2,   np0,    nD0,     nT0,    nα0, nimps[1], nimps[1]] |> Vector{datatype} # `n20`
T0 = [Te1,       Te2,   Tp0,    TD0,     TT0,    Tα0,   1.0,         1.0] |> Vector{datatype}   # `keV`
if is_Ek0
    μEk=[sign(Eke1),sign(Eke2),sign(Ekp0),sign(EkD0),sign(EkT0),sign(Ekα0), 1, 1]   |> Vector{Int}
    Ek0=[abs(Eke1), abs(Eke2), abs(Ekp0), abs(EkD0),abs(EkT0),abs(Ekα0), 0, 0.0] |> Vector{datatype}  # `keV`
else
    #     ues = [1.1, 0.9] * 1e-0
    #     μEk = Int.(sign(ues))
    rhn
    u0 = [ues[1], ues[2], up0,    uD0,     uT0, 0, 0, 0.0] |> Vector{datatype}   # `Mms`
end
mD0 = [me0[1],me0[2],mp0[1],mp0[2],mp0[3],mα0, 1, 15] |> Vector{datatype}   # `Dₐ`, except for `spices==:e`, instead by `me`
Zq = [-1, -1, 1, 1, 1, 2, 1, 7] |> Vector{Int}        # `e`

