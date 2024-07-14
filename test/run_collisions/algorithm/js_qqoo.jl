
# jvec = j1:dj:jMsmax |> Vector{Int}
# njMs = length(jvec)   
# That is means that the number total observative moments will be `njMs01 = 2njMs - 1` when `is_nai_constant = true`.

nMjMs = zeros(Int64, ns)
Mhinitial_fDM!(nMjMs, ns, nMod, uai0, njMs_fix; is_MjMs_max=is_MjMs_max)
if is_MjMs_max
    njMs = maximum(nMjMs)
else
    dfghbjnm
end

