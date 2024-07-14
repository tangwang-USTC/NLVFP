"""
  Parameters in the International System of units (SI)          
  Non-relativistic models:

    uₑ < 1.87e8 (Ke ~ 100 keV)
    vₑₜₕ < 1e8 (Te ~ 30 keV)
    Kα = 3 MeV -- (uα ~ 1.2e7 ~ ue)  -- Ke = 0.4 keV
"""


ε = deepcopy(ε₀)
μ = deepcopy(μ₀)

# Parameters in SI system from practical units
maSI = zeros(datatype,ns)
for isp3 in 1:ns
    if spices0[isp3] == :e
        maSI[isp3] = me * mD0[isp3]
    else
        maSI[isp3] = Dₐ * mD0[isp3]
    end
end
maSI = massNorm.(maSI)
qZ = Zq * e
##
naSI0 = n0 * n20
TaSI0 = T0 * Tk
vthSI0 = zeros(datatype,ns)
if is_Ek0
    EkaSI0 = Ek0 * Tk
    uaSI0 = zeros(datatype,ns)
    for isp3 in 1:ns
        if spices0[isp3] == :e
            vthSI0[isp3] = (2 * TaSI0[isp3] ./ maSI[isp3]).^0.5
            uaSI0[isp3] = (2 * EkaSI0[isp3] ./ maSI[isp3]).^0.5
        else
            vthSI0[isp3] = (2 * TaSI0[isp3] ./ maSI[isp3]).^0.5
            uaSI0[isp3] = (2 * EkaSI0[isp3] ./ maSI[isp3]).^0.5
        end
        if μEk[isp3] < 0
            uaSI0[isp3] *= μEk[isp3]
        end
    end
    # The normalized effective average velocity by `Mms`
    u0 = uaSI0 ./ Mms 
else
    for isp3 in 1:ns
        if spices0[isp3] == :e
            vthSI0[isp3] = (2 * TaSI0[isp3] ./ maSI[isp3]).^0.5
        else
            vthSI0[isp3] = (2 * TaSI0[isp3] ./ maSI[isp3]).^0.5
        end
    end
    uaSI0 = u0 .* Mms               #
    EkaSI0 = 0.5 * (maSI .* uaSI0.^2)
    Ek0 = EkaSI0 / Tk
end
KaSI0 = naSI0 .* (1.5TaSI0 + EkaSI0)
IaSI0 = maSI .* naSI0 .* uaSI0
