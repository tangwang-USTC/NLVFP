A1 = (vG * dxdv^2 .* (Dc * Dc) + 2dxdv .* Dc)
b1 = - 4Pi * vabth^2 * vG .* FLn
HLn = pinv(A1) * b1
# HLn = pinv(Float64.(A1)) * b1
if HLnv0 > 0.0
    HLn .+= HLnv0 - HLn[1]
end

# x0 = zero(vG)
# x0 = IterativeSolvers.gmres!(x0,A1,b1)
#
# x0 = zero(vG)
# x0 = IterativeSolvers.idrs!(x0,A1,b1)
#
# x0 = zero(vG)
# x0 = IterativeSolvers.lsmr!(x0,A1,b1)
# if HLnv0 > 0.0
#     x0 .+= HLnv0 - x0[1]
# end
#
# x0 = zero(vG)
# x0 = IterativeSolvers.lsqr!(x0,A1,b1)
# if HLnv0 > 0.0
#     x0 .+= HLnv0 - x0[1]
# end


# x0 = zero(vG)
# # solve(A1, b1, RugeStubenAMG(), maxiter = 1, abstol = 1e-6)
# x0, it = mg(A1,b1)
# if HLnv0 > 0.0
#     x0 .+= HLnv0 - x0[1]
# end
