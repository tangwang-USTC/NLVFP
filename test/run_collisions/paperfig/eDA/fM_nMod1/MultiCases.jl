
# Multi-cases to show the results

tplotM = Vector{AbstractVector{datatype}}(undef,NCase)
CRDnM = Vector{AbstractVector{datatype}}(undef,NCase)
# if ns == 2
#     CRDn2M = Vector{AbstractVector{datatype}}(undef,NCase)
# else
#     CRDn3M = Vector{AbstractVector{datatype}}(undef,NCase)
#     if ns ≥ 4
#         CRDn4M = Vector{AbstractVector{datatype}}(undef,NCase)
#     end
# end
CRDIM = Vector{AbstractVector{datatype}}(undef,NCase)
CRDKM = Vector{AbstractVector{datatype}}(undef,NCase)
CRDsM = Vector{AbstractVector{datatype}}(undef,NCase)

eRdtnaM = Vector{AbstractVector{datatype}}(undef,NCase)
eRdtIaM = Vector{AbstractVector{datatype}}(undef,NCase)
eRdtKaM = Vector{AbstractVector{datatype}}(undef,NCase)
eRdtTaM = Vector{AbstractVector{datatype}}(undef,NCase)

eRdtnbM = Vector{AbstractVector{datatype}}(undef,NCase)
eRdtIbM = Vector{AbstractVector{datatype}}(undef,NCase)
eRdtKbM = Vector{AbstractVector{datatype}}(undef,NCase)
eRdtTbM = Vector{AbstractVector{datatype}}(undef,NCase)

@show ns
if ns ≥ 3
    @show ns + 1
    eRdtncM = Vector{AbstractVector{datatype}}(undef,NCase)
    eRdtIcM = Vector{AbstractVector{datatype}}(undef,NCase)
    eRdtKcM = Vector{AbstractVector{datatype}}(undef,NCase)
    eRdtTcM = Vector{AbstractVector{datatype}}(undef,NCase)
end
# eRdtnM = Vector{AbstractVector{datatype}}(undef,NCase)
# eRdtIM = Vector{AbstractVector{datatype}}(undef,NCase)
# eRdtKM = Vector{AbstractVector{datatype}}(undef,NCase)
# eRdtTM = Vector{AbstractVector{datatype}}(undef,NCase)
