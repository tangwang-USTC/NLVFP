using AlgebraicMultigrid, SparseArrays
using Preconditioners
using IncompleteLU
# using RandomizedPreconditioners
using Krylov

k = 1   # ∈ [1:nc0-1]
        # When `k ≥ 2`, the linear algebraic equation `𝔸𝒙 = 𝒃` is underdetermined equation
        # owing to matrix `𝔸` is degenerated (number of Eq.s ≤ number of x).
        # Least-squares algorithms (lsqr, lsmr, lslq, cgls, crls), gradient projection metho and
        # inner point method are are commonly used to solve the underdetermined equations.
ismulva = true
# ismulva = false
ishermitian(Dc)
issymmetric(Dc)
issparse(sparse(Dc))
issparse(diagm([1,2,3]))
diagm([1,2,3])
Diagonal([1,2,3])
pM2(A1) = [size(A1),isposdef(A1),cond(A1), rank(A1), det(A1),tr(A1)]
pM(A1) = @show [isposdef(A1),cond(A1), rank(A1), det(A1),tr(A1)]

function dHLnChebyk(k::Int;ismulva::Bool=true,errdAR::Float64=1.0)

    # k = 1
    nk = nvlevel[k]
    nk1 = sum(nvlevel[1:(k-1)]) - k + 2
    nk9 = nk1 + nk - 1
    veck = nk1:nk9 |> Vector{Int}
    vk = vGk[veck]
    HLnvk, dHLnvk = copy(HLnt[nk1]), copy(dHLnt[nk1])
    dxdvk = 2 / (vGk[nk1] - vGk[nk9])
    Dck = Dc1n(nk)
    # compute the first derivative: `dHLn`
    if ismulva == 0
        A1 = dxdvk * Dck + diagm(2 ./ vk)
        b1 = - FLn[veck]
    else
        # A2 = dxdvk * vk .* Dck + Matrix(2.0I,nk,nk) # 2diagm(one.(vk))) #
        # A2 = sparse(dxdvk * vk .* Dck + Matrix(2.0I,nk,nk)) # 2diagm(one.(vk))) #
        A1 = dxdvk * vk .* Dck + Diagonal(2.0I,nk) # Matrix(2.0I,nk,nk) # 2diagm(one.(vk))) #
        b1 = - vk * vabth^2 .* FLn[veck]
    end
    dA = qr(A1)
    @show k, dA.R[nk,nk]
    # if rank(A1) == nk
    if dA.R[nk,nk] > errdAR
        dHLnk = inv(A1) * b1
        # dHLnk = GenericLinearAlgebra.pinv(A1) * b1
        # dHLnk = IterativeSolvers.idrs(A1,b1;s = 8,abstol=1e-15,verbose=true)
        # orth_meth = IterativeSolvers.ModifiedGramSchmidt()
        # orth_meth = IterativeSolvers.ClassicalGramSchmidt()
        # orth_meth = IterativeSolvers.DGKS()
        # dHLnk = IterativeSolvers.gmres!(dHLnk,A1,b1;abstol=1e-15,orth_meth=orth_meth,verbose=true)
        # dHLnk = IterativeSolvers.bicgstabl!(dHLnk,A1,b1,3;verbose=true)
        # dHLnk = IterativeSolvers.lsmr!(dHLnk,A1,b1;atol=1e-15,conlim=1e15,verbose=true)
        # dHLnk = IterativeSolvers.lsqr!(dHLnk,A1,b1;atol=1e-15,conlim=1e15,verbose=true)
        # dHLnk = IterativeSolvers.jacobi!(dHLnk,A1,b1;)
        # dHLnk = IterativeSolvers.gauss_seidel!(dHLnk,A1,b1;)
        # dHLnk = IterativeSolvers.sor!(dHLnk,A1,b1,0;)
        # dHLnk = IterativeSolvers.ssor!(dHLnk,A1,b1,1.1;)
        if dHLnvk ≠ 0.0
            dHLnk .+= (dHLnvk - dHLnk[1])
        end
    else
        dHLnk = zeros(nk)
        dHLnk[1] = dHLnvk
        # Decompositing the original matrix `A1` by the `QR` factorization
        # dA = qr(A1)
        b2 = inv(dA.Q) * b1 # Update the right vector by left-multiplying the precondition matrix `inv(dA.Q)`
                            # and matrix `A1 → R2 = dA.R`
        # Checking out whether the elements `dA.Q[end,end] ≈ 0` and `b2[end] ≈ 0`

        # Solving the new algebraic eq. `ℝ𝒙 = 𝒃* = ℚ⁻¹𝒃` by
        # eliminating the degeneration raw owing to the degeneration of matrix `𝔸`
        # according to the boudnary conditions
        b3 = b2[1:end-1]
        b3[1] -= dA.R[1,1] * dHLnk[1]
        R = zeros(nk-1,nk-1)
        [R[i,j] = dA.R[i,j+1] for i in 1:nk-1 for j in 1:nk-1]
        dHLnk[2:end] = inv(R) * b3
    end
    if ismulva == 0
        RdHLn = A1 * dHLnk + FLn[veck]
    else
        RdHLn = A1 * dHLnk + vk * vabth .* FLn[veck]
    end
    xlabel = string("va^=",ismulva,",k=",k,",p,c,r,d,t=",fmtf1.(pM(A1)))
    # compute the solution of `H` Poisson equation: `HLn`
    A1 = dxdvk * Dck
    b1 = dHLnk
    if rank(A1) == nk
        HLnk = inv(A1) * b1
        if HLnvk ≠ 0.0
            HLnk .+= (HLnvk - HLnk[1])
        end
    elseif rank(A1) < nk
        HLnk = zeros(nk)
        HLnk[1] = HLnvk
        # Decompositing the original matrix `A1` by the `QR` factorization
        dA = qr(A1)
        b2 = inv(dA.Q) * b1 # Update the right vector by left-multiplying the precondition matrix `inv(dA.Q)`
                            # and matrix `A1 → R2 = dA.R`
        # Checking out whether the elements `dA.Q[end,end] ≈ 0` and `b2[end] ≈ 0`

        # Solving the new algebraic eq. `ℝ𝒙 = 𝒃* = ℚ⁻¹𝒃` by
        # eliminating the degeneration raw owing to the degeneration of matrix `𝔸`
        # according to the boudnary conditions
        b3 = b2[1:end-1]
        b3[1] -= dA.R[1,1] * HLnk[1]
        R = zeros(nk-1,nk-1)
        [R[i,j] = dA.R[i,j+1] for i in 1:nk-1 for j in 1:nk-1]
        HLnk[2:end] = inv(R) * b3
    end
    if ismulva == 0
        RddHLn = A1 * HLnk + FLn[veck]
    else
        RddHLn = A1 * HLnk + vk * vabth .* FLn[veck]
    end
    @show norm(dHLnk - dHLnt[veck]), norm(HLnk - HLnt[veck])
    label = string("δdH//epsT5")
    pdH = plot((dHLnk - dHLnt[veck])/epsT5,label=label)
    label = string("RdHLn")
    pRdH = plot(vk,RdHLn/epsT5,label=label,xlabel=xlabel)
    display(plot(pdH,pRdH,layout=(2,1)))
    return dHLnk,HLnk
end

function HLnChebyk(k;ismulva::Bool=true)

    nk = nvlevel[k]
    nk1 = sum(nvlevel[1:(k-1)]) - k + 2
    nk9 = nk1 + nk - 1
    veck = nk1:nk9 |> Vector{Int}
    vk = vGk[veck]
    HLnvk, dHLnvk = copy(HLnt[nk1]), copy(dHLnt[nk1])
    dxdvk = 2 / (vGk[nk1] - vGk[nk9])
    Dck = Dc12n(nk)[:,:,1]
    Dc2k = Dc12n(nk)[:,:,2]
    # compute the solution of `H` Poisson equation directively.
    if ismulva == 0
        A1 = dxdvk^2 * Dc2k + 2dxdvk ./ vk .* Dck
        b1 = - vabth^2 .* FLn[veck]
    else
        A1 = dxdvk^2 * vk .* Dc2k + 2dxdvk * Dck
        b1 = - vk * vabth^2 .* FLn[veck]
    end
    if rank(A1) == nk
        HLnk = inv(A1) * b1
        if HLnvk ≠ 0.0
            HLnk .+= (HLnvk - HLnk[1])
        end
    elseif rank(A1) < nk
        HLnk = zeros(nk)
        HLnk[1] = HLnvk
        # Decompositing the original matrix `A1` by the `QR` factorization
        dA = qr(A1)
        b2 = inv(dA.Q) * b1 # Update the right vector by left-multiplying the precondition matrix `inv(dA.Q)`
                            # and matrix `A1 → R2 = dA.R`
        # Checking out whether the elements `dA.Q[end,end] ≈ 0` and `b2[end] ≈ 0`

        # Solving the new algebraic eq. `ℝ𝒙 = 𝒃* = ℚ⁻¹𝒃` by
        # eliminating the degeneration raw owing to the degeneration of matrix `𝔸`
        # according to the boudnary conditions
        b3 = b2[1:end-1]
        b3[1] -= dA.R[1,1] * HLnk[1]
        R = zeros(nk-1,nk-1)
        [R[i,j] = dA.R[i,j+1] for i in 1:nk-1 for j in 1:nk-1]
        HLnk[2:end] = inv(R) * b3
    end
    @show norm(dHLnk - dHLnt[veck]), norm(HLnk - HLnt[veck])
    if ismulva == 0
        RddHLn = A1 * HLnk + FLn[veck]
    else
        RddHLn = A1 * HLnk + vk * vabth .* FLn[veck]
    end
    xlabel = string("m=",ismulva,",k=",k,",p,c,r,d,t=",fmtf1.(pM(A1)))
    label = string("δH")
    pdH = plot((HLnk - HLnt[veck])/epsT5,label=label)
    label = string("RddHLn")
    pRdH = plot(vk,RddHLn/epsT5,label=label,xlabel=xlabel)
    display(plot(pdH,pRdH,layout=(2,1)))
    return HLnk
end

function RdHLnk(dHLn,k;ismulva::Bool=true)

    nk = nvlevel[k]
    nk1 = sum(nvlevel[1:(k-1)]) - k + 2
    nk9 = nk1 + nk - 1
    veck = nk1:nk9 |> Vector{Int}
    vk = vGk[veck]
    dxdvk = 2 / (vGk[nk1] - vGk[nk9])
    Dck = Dc12n(nk)[:,:,1]
    if ismulva == 0
        A1 = dxdvk * Dck + diagm(2 ./ vk)
        RdHLn = A1 * dHLn[veck] + FLn[veck]
    else
        A1 = dxdvk * vk .* Dck + 2diagm(one.(vk))
        RdHLn = A1 * dHLn[veck] + vk .* FLn[veck]
    end
    @show cond(A1), norm(RdHLn)
    label = string("ismulva=",ismulva,",k=",k)
    ylabel = string("RdHLn")
    display(plot(vk,RdHLn,label=label,ylabel=ylabel))
    RdHLn
end

function RddHLnk(HLn,k;ismulva::Bool=true)

    nk = nvlevel[k]
    nk1 = sum(nvlevel[1:(k-1)]) - k + 2
    nk9 = nk1 + nk - 1
    veck = nk1:nk9 |> Vector{Int}
    vk = vGk[veck]
    dxdvk = 2 / (vGk[nk1] - vGk[nk9])
    Dck = Dc12n(nk)[:,:,1]
    Dc2k = Dc12n(nk)[:,:,2]
    if ismulva == 0
        A2 = dxdvk^2 * Dc2k + 2dxdvk ./ vk .* Dck
        RddHLn = A2 * HLn[veck] + FLn[veck]
    else
        A2 = dxdvk^2 * vk .* Dc2k + 2dxdvk .* Dck
        RddHLn = A2 * HLn[veck] + vk .* FLn[veck]
    end
    @show cond(A2), norm(RddHLn)
    label = string("RddHLn,k=",k)
    display(plot(vk,RddHLn,label=label))
    RddHLn
end
