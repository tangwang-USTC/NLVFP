
"""
  Boundarys for `fLn = f̂ₗ(v̂)` and its first derivative `dfLn = ∂ᵥf̂ₗ(v̂)`
  and similar to `FLn = F̂ₗ(𝓋̂)`

  Inputs:
    [v0,v9]: default = vGdom = [eps,10] for `fLn`, which is the two end points of velocity domain.
                     = vGdom * vabth for `FLn`
    fLnbc: = zeros(6,LM1) = [f(0), ∂ᵥf(0), ∂ᵥ²f(0), f(10), ∂ᵥf(10), ∂ᵥ²f(10)]

  Outputs:
    fLnbc = fLnbcRR(fLnbc,v0,v9,L1)
"""

# isp = isp3
function fLnbcRR(fLnbc::AbstractVector{T},v0::T2,v9::T2,L1::Int) where{T,T2,N}

    # when `f(v)` is Maxwellian where `u = 0`
    if L1 == 1
        fLnbc[1] = 1 - v0^2 + 0.5v0^4
        fLnbc[2] = - 2v0 * fLnbc[1]
        fLnbc[3] = - 2 + 6v0^2 - 5v0^4
        fLnbc[4:6] = 0.0
        return fLnbc
    end
end

"""
  The first pᵗʰ-order derivatives at the left boundary , `v̂ = 0`, where `p ∈ 0:n1-1`

  Defining `∂ᵖv̂ f(v̂) = f(v̂) when p = 0`.

  ∂ᵖv̂ f(v̂=0) ≡ 0

  Inputs:
    np: The maximum order of derivatives gives `p ∈0:np`
  Outputs:
    dvpfLnbc = zeros(np+1)
    dvpfLnbc = dvfvLbcv0(dvpfLnbc,np,L1)

"""

function dvfvLbcv0(fLnbc::AbstractVector{T},np::Int,L1::Int) where{T}

    if L1 == 1
        # isodd(p1) == 1,  p = 0:2:np
        for p1 in 1:2:np+1
            if p1 == 1
                fLnbc[p1] = 1
            elseif p1 == 3
                fLnbc[p1] = - 2
            elseif p1 == 5
                fLnbc[p1] = 12
            elseif p1 == 7
                fLnbc[p1] = - 120
            elseif p1 == 9
                fLnbc[p1] = 1680
            elseif p1 == 11
                fLnbc[p1] = - 30240
            elseif p1 == 13
                fLnbc[p1] = 665280
            elseif p1 == 15
                fLnbc[p1] = -17297280
            elseif p1 == 17
                fLnbc[p1] = 518918400
            elseif p1 == 19
                fLnbc[p1] = -17643225600
            elseif p1 == 21
                fLnbc[p1] = 670442572800
            elseif p1 == 23
                fLnbc[p1] = -28158588057600
            else
                sshngtn
            end
        end
        return fLnbc
    else
        fdrhtrjn
    end
end

"""

  Outputs:
    FLnbc = FLnbcRR(FLnbc,v0,v9,L1)
"""

# FLn(v=0) and FLn(v=∞)
function FLnbcRR(FLnbc::AbstractVector{T},v0::T2,v9::T2,L1::Int) where{T,T2,N}

    # when `f(v)` is Maxwellian where `u = 0`
    if L1 == 1
        FLnbc[1] = 1 - v0^2 + 0.5v0^4  # F00(v=0)
        FLnbc[2] = 0.0                 # F00(v=∞)
        return FLnbc
    end
end

"""
  Boundarys for `HLn = Ĥₗ(𝓋̂)` and its first derivative `dHLn = ∂ᵥĤₗ(𝓋̂)`

  Inputs:
    [v0,v9]: = [va[1],va[end]]
             = vdom * vabth = [eps,10] * vabth default, which is the two end points of velocity domain.
    HLnbc: = zeros(6,LM1) = [H(0), ∂ᵥH(0), ∂ᵥ²H(0), H(10), ∂ᵥH(10), ∂ᵥ²H(10)]

  Outputs:
    HLnbc = HLnbcRR(HLnbc,va0,va9,L1)
"""


# isp = isp3,
function HLnbcRR(HLnbc::AbstractVector{T},v0::T2,v9::T2,L1::Int) where{T,T2,N}

    # when `f(v)` is Maxwellian where `u = 0`
    if L1 == 1
        HLnbc[1] = 0.5(1 - 1//3 * v0^2 + 0.1v0^4)
        HLnbc[2] = v0 * (- 1//3 + 0.2v0^2 - 1//14 * v0^4)
        HLnbc[3] = - 1//3 + 0.6v0^2 - 5//14 * v0^4
        HLnbc[4] = sqrtpi / 4v9
        HLnbc[5] = - sqrtpi / 4v9^2
        HLnbc[6] = sqrtpi / 2v9^3
        return HLnbc
    end
end

"""
  Boundarys for `GLn = Ĥₗ(𝓋̂)` and its first derivative `dGLn = ∂ᵥĜₗ(𝓋̂)`

  Inputs:
    [v0,v9]: = [va[1],va[end]]
             = vdom * vabth = [eps,10] * vabth default, which is the two end points of velocity domain.
    GLnbc: = zeros(6) = [G(0), ∂ᵥG(0), ∂ᵥ²G(0), G(10), ∂ᵥG(10), ∂ᵥ²G(10)]

  Outputs:
    GLnbc = GLnbcRR(GLnbc,va0,va9,L1)
"""

# isp = isp3,
function GLnbcRR(GLnbc::AbstractVector{T},v0::T2,v9::T2,L1::Int) where{T,T2,N}

    # when `f(v)` is Maxwellian where `u = 0`
    if L1 == 1
        GLnbc[1] = 0.5(1 + 1//3 * v0^2 - 1 // 30 * v0^4)
        GLnbc[2] = v0 * ( 1//3 - 1 // 15 * v0^2 + 1//70 * v0^4)
        GLnbc[3] = 1 // 3 - 0.2v0^2 + 1 // 14 * v0^4
        GLnbc[4] =   sqrtpi / 4 * (v9 + 1 / 2v9)
        GLnbc[5] = - sqrtpi / 4 * (1 - 1 / 2v9^2)
        GLnbc[6] = sqrtpi / 4v9^3
        return GLnbc
    end
end
