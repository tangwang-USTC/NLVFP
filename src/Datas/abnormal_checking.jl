
"""

  Intputs:

  Outputs:
    is_abnormal_abs(y,scalea09)
"""

function is_abnormal_abs(y::AbstractVector{T},scalea09::Real) where{T}

    
    a = abs.(y) .+ epsT
    Na = length(a)
    a09 = sum(a) / Na 
    return a .< (a09 * scalea09), a
end