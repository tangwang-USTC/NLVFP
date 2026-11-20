module KingDefinition
using SpecialFunctions: besselix
include("../../king_fpr/src/KingFPR.jl")

export physical, scaled, legacy_normalized

"""Physical Legendre coefficient of a unit-density drifting Gaussian.

The signed axial drift gives the odd-mode parity. The component thermal
width is the square-root-of-two standard deviation convention.
"""
function physical(l::Integer,v::T,u::T,width::T) where {T<:AbstractFloat}
    l>=0||throw(DomainError(l))
    isfinite(v)&&v>=0||throw(DomainError(v))
    isfinite(u)||throw(DomainError(u))
    isfinite(width)&&width>0||throw(DomainError(width))
    drift=abs(u)
    argument=2drift*v/width^2
    parity=u<0 && isodd(l) ? -one(T) : one(T)
    if argument<=80
        return parity*KingFPR.king_physical(l,v,drift,width)
    end
    parity*T(2l+1)/(T(pi)^T(1.5)*width^3)*exp(-((v-drift)/width)^2)*sqrt(T(pi)/(2argument))*besselix(l+T(0.5),argument)
end

"""V010 variable-width definition with r=v/reference_width and k=2u/component_width."""
function scaled(l::Integer,r::T,k::T,reference_width::T,rho::T) where {T<:AbstractFloat}
    isfinite(reference_width)&&reference_width>0||throw(DomainError(reference_width))
    isfinite(rho)&&rho>0||throw(DomainError(rho))
    physical(l,r*reference_width,k*rho*reference_width/2,rho*reference_width)
end

"""Legacy dimensionless f-hat coefficient, with density divided by its reference.

Physical f_l = reference_density/(pi^(3/2)*reference_width^3) * fhat_l.
"""
legacy_normalized(l,r,k,reference_width,rho,density_ratio)=density_ratio*pi^1.5*reference_width^3*scaled(l,r,k,reference_width,rho)
end
