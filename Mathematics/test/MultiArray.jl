"""
   Multiplication of Arrays
   C = ∑ₙ₌₀ᴺ∑ₗ₌₀ˡᴹ(Lₙ(vᵢ) * fₙₗ)
"""

function sumAB(A::AbstractArray{T,N},B::AbstractArray{T,N},nv1::Int64,nL1::Int64) where {T,N}
    C = 0 * B
    for iv = 1:nv1
        for iL = 1:nL1
            # C[iv,iL] = sum(A[iv,:] .* B[:,iL])
            # C[iv,iL] = sum(A[iv,:] .* B[iv,iL])
            # C[iv,iL] = sum(A[:,iv] .* B[:,iL])
            # C[iv,iL] = sum(A[:,iv] .* B[iv,iL])
            # C[iv,iL] = transpose(A[iv,:]) * B[:,iL]
        end
    end
    return C
end
function sumAB(v::AbstractVector,B::AbstractArray{T,N},nv1::Int64,nL1::Int64) where {T,N}
    C = 0 * B
    for n = 0:nv1-1
        for iL = 1:nL1
            C[n+1,iL] = transpose(LaguerreP(n,v,alpha = α)) * B[:,iL]
        end
    end
    return C
end
