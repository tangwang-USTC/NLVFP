
"""
  Solving the linear algebraic problem:

    𝔸𝒙 = 𝒃

  with boundary conditions

    𝒙[1] = x₀

  Using `qr` decomposition as the preconditioner which can apply
  The boundary information to reduce the condition number of matrix `𝔸`

    ℚ, ℝ = qr(𝔸).

  The equivalence equations are:

    ℝ𝒙 = 𝒃* = ℚ⁻¹𝒃

  where `𝒃*[end] ≈ 0` and `ℝ[end,end] ≈ 0`.
  So the linear algebraic equation could be simplified as:

    ℝ[1:end-1,2:end] 𝒙[2:end] = 𝒃1*

  where

    𝒃1* = 𝒃*[1:end-1]
    𝒃1*[1] -= (ℝ[1,1] * 𝒙[1])

  --------------------------------------------------------
   When `issolve == true` where `𝒃` is known,
   solving above euqation can give the initial solution:

    𝒙[2:end] = inv(ℝ[1:end-1,2:end]) 𝒃1*

  --------------------------------------------------
  When `issolve == false`, computing vector:

    𝒃1* = ℝ[1:end-1,2:end] * 𝒙[2:end]

  and then calculating

    𝒃*[1:end-1] = 𝒃1*
    𝒃*[1] += (ℝ[1,1] * 𝒙[1])
    𝒃*[end] = 0

  Solving equation:

    𝒃* = ℚ⁻¹𝒃

  gives

    𝒃 = ℚ𝒃*

  Inputs:
    x0: the boundary conditions, `x[1] = x0`
    issolve: == true gives the solution `𝒙` of the linear algebraic eq. `𝔸𝒙 = 𝒃`, or else
             == false, computing the value `𝒃 = 𝔸𝒙`

  Outputs:
    x = diffMDcprecond(x,A1,b1,x0,dxdv;issolve=true)
    b1 = diffMDcprecond(x,A1,b1,x0,dxdv;issolve=false)

"""

function diffMDcprecond(x::AbstractVector{T},A::AbstractArray{T2,N},
    b1::AbstractVector,x0::T,dxdv::Real;issolve::Bool=true) where {T,T2,N}

    # solving the linear algebraic  equation for `x`
    if issolve == 1
        QA, RA = qr(A)
        # boundary conditions
        x[1] = x0
        b1 = inv(QA) * b1
        # b1 = QA \ b1 # Update the right vector `b1 → b2` by left-multiplying
                          # the precondition matrix `inv(QA)` and matrix `A1 → R2 = RA`
        # eliminating the degeneration (boudnary conditions) raw owing to the degeneration of matrix `𝔸`
        # Checking out whether the elements `b2[end] = b1[end] ≈ 0`
        @show b1[end]
        if abs(b1[end]) < 1e-14
            b1[1] -= RA[1,1] * x[1] # Update the vector `𝒃*[1] = 𝒃[1] - r₁₁ * x[1]`
            if rank(RA[1:end-1,2:end]) == length(b1[2:end])
                x[2:end] = inv(RA[1:end-1,2:end]) * b1[1:end-1] / dxdv
                if cond(RA[1:end-1,2:end]) > 1e10
                    @warn("`cond(RA[1:nc-1,2:nc] > 1e10` which maybe lead to instability!)")
                end
            else
                @show cond(RA[1:end-1,2:end])
                @show length(b1[2:end])
                ehrth
            end
        else
            paraM(A)
            @show RA[end,end]
            rdjy
        end
        return x
    else # compute the value `b1` = A * x = QA * (RA * x)
        QA, RA = qr(A)
        b1[:] = Dc * x * dxdv
        b1[1] = x0
        return b1
        #
        b1[1:end-1] = RA[1:end-1,2:end] * x[2:end]  # compute `𝒃1*` and then `𝒃*`
        b1[1] += RA[1,1] * x[1] # Update the vector `𝒃[1] = 𝒃[*1] + r₁₁ * x[1]`
        b1[end] = 0.0
        b1[:] = QA * b1 * dxdv
        b1[1] = x0
        return b1
        # b1 = ℚ * b1               # Calculating: 𝒃 = ℚ 𝒃*
    end
end

# Dc = Dc1n(nc0;datatype = BigFloat)
# # Dc = Dc1n(nc0;datatype = Float64)
# fLog = copy(fLnlog)
# fLog[4] *= 1.0 + 0e-8
# dflog = zero.(vGk)
# dflog0 = - 2vGk[1]
# dflog = diffMDcprecond(fLog,Dc,dflog,dflog0,dxdv;issolve=false)
# dflog2 = zero.(vGk)
# dflog2[2:end] = Dc[2:end,2:end]*fLog[2:end]*dxdv  #
# dflog2[1] = dflog0
# label = string("Ddflog,nc=",nc0)
# rDdf = dflog./dfLnlogt.-1
# pflog = plot(vlog,rDdf,label=label)
# rDdf2 = dflog2./dfLnlogt.-1
# label = string("Ddflog2")
# pflog = plot!(vlog,rDdf2,label=label,line=(1,:auto),legend=:topleft)
#
# # dflog = copy(dfLnlogt)
# flog = zero.(vGk)
# flog0 = - vGk[1].^2
# i = 5
# dflog[i] = dfLnlogt[i]
# flog = diffMDcprecond(flog,Dc,dflog,flog0,dxdv;issolve=true)
# rDf = flog./fLnlogt.-1
# rDft = flog./fLog.-1
# label = string("Dflog,nc=",nc0)
# pdflog = plot(vlog,rDf,label=label,line=(1,:auto))
# label = string("DfLog")
# pdflog = plot!(vlog,rDft,label=label,line=(1,:auto),legend=:topleft)
# display(plot(pflog,pdflog,layout=(2,1)))
