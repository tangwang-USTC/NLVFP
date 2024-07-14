
"""
  isinv = true: `vc` → `v`
      meaning mapping `x = vc ∈ [-1,1]` to velocity space, `vv = v1`, with domain as `vdom = [A, B] = [a,b]` * vth
  isinv = false: `v` → `vc`
      mapping velocity `v ∈ vdom = [a,b]*vth` on Chebyshev domian  `vc = x ∈ domain = [-1,1]`.

  Inputs:
    x: the Chebyshev grids , `vc ∈ [[-1.0,1.0]]` if `isinv = true`
       the velocity space grids , `vc ∈ [A,B]` if `isinv = false`

  Outputs:
    vc: (default, when isinv = false), outputs is the chebyshev grids;

      vc = vCmapping(x,A,B;isinv=isinv)
      vc = vCmapping(x,a,b,vth;isinv=isinv)

    v1: (if isinv = true), outputs is the velocity space grids;

      v1 = vCmapping(x,A,B;isinv=true)
      v1 = vCmapping(x,a,b,vth;isinv=true)
"""
# subsection
function vCmapping(x::AbstractVector{T},A::T,B::T;isinv::Bool=false) where{T}

    # vc = x → vv = v
    if isinv == 1
        vv = 0.5(A - B) * (x .+ 1) .+ B
        vv[1] ≠ A ? vv[1] = 1A : 1
        vv[end] ≠ B ? vv[end] = 1B : 1
        return vv
    else
        vc = 2 / (A - B) * x .- (A + B)/(A - B)
        x[1] == A ? vc[1] = 1.0 : 1
        x[end] == B ? vc[end] = -1.0 : 1
        return vc
    end
end

# subsection
function vCmapping(x::AbstractVector{T},a::T,b::T,vth::T2;isinv::Bool=false) where{T,T2}

    # vc = x → vv = v
    if isinv == 1
        A = a * vth
        B = b * vth
        vv = 0.5(A - B) * (x .+ 1) .+ B
        vv[1] ≠ A ? vv[1] = 1A : 1
        vv[end] ≠ B ? vv[end] = 1B : 1
        return vv
    else
        A = a * vth
        B = b * vth
        vc = 2 / (A - B) * x .- (A + B)/(A - B)
        x[1] == A ? vc[1] = 1.0 : 1
        x[end] == B ? vc[end] = -1.0 : 1
        return vc
    end
end


"""
  Outputs:
      vGk,nvlevel0 = vCmapping(vGk,nvlevel0,vG0,nc0,ocp)
      vGk,nvlevel0 = vCmapping(vGk,nvlevel0,vG0,nc0,nvlevel)
"""

# 1D, [ocp], level = 1
function vCmapping(vGk::AbstractVector{T},nvlevel0::Vector{Int},
    vG0::AbstractVector{T},nc0::Int,ocp::Int) where{T<:Real}

    k = 1
    nk1 = 1
    nk9 = 0 + ocp
    vGk[nk1:nk9] = vCmapping(vccn(ocp;datatype=T),vG0[k],vG0[k+1];isinv=true)
    nvlevel0[k] = 1
    for k in 2:nc0-1
        nk1 = nk9
        nk9 = nk1 + ocp - 1
        vGk[nk1:nk9] = vCmapping(vccn(ocp;datatype=T),vG0[k],vG0[k+1];isinv=true)
        nvlevel0[k] = nk1
    end
    k = nc0
    nvlevel0[k] = nk9
    return vGk,nvlevel0
end

# 1D, [nk[k]], level = 1
function vCmapping(vGk::AbstractVector{T},nvlevel0::Vector{Int},
    vG0::AbstractVector{T},nc0::Int,nvlevel::Vector{Int}) where{T<:Real}

    k = 1
    nk = nvlevel[k]
    nk1 = 1
    nk9 = 0 + nk
    vGk[nk1:nk9] = vCmapping(vccn(nk;datatype=T),vG0[k],vG0[k+1];isinv=true)
    nvlevel0[k] = 1
    for k in 2:nc0-1
        nk = nvlevel[k]
        nk1 = nk9
        nk9 = nk1 + nk - 1
        vGk[nk1:nk9] = vCmapping(vccn(nk;datatype=T),vG0[k],vG0[k+1];isinv=true)
        nvlevel0[k] = nk1
    end
    k = nc0
    nvlevel0[k] = nk9
    return vGk,nvlevel0
end

"""

   `vc` → `v`
       meaning mapping `x = vc ∈ [-1,1]` to velocity space, `vv = v1`, with domain as `vdom = [A, B] = [a,b]` * vth

  Inputs:
    x: the Chebyshev grids , `vc ∈ [[-1.0,1.0]]`

  Outputs:
    v1: outputs is the velocity space grids;

    v1 = vCmapping(x,a,b,vth,ns,nc)
"""
# [ns]
function vCmapping(x::AbstractVector{T},a::T,b::T,vth::AbstractVector{T2},ns::Int,nc::Int) where{T,T2}
    # vc = x → vv = v
    vv = zeros(T,nc,ns)
    for isp in 1:ns
        A = a * vth[isp]
        B = b * vth[isp]
        vv[:,isp] = 0.5(A - B) * (x .+ 1) .+ B
        vv[1,isp] ≠ A ? vv[1,isp] = 1A : 1
        vv[end,isp] ≠ B ? vv[end,isp] = 1B : 1
    end
    return vv
end

"""

   `v` → `vc`
       mapping velocity `v ∈ vdom = [a,b]*vth` on Chebyshev domian  `vc = x ∈ domain = [-1,1]`.

  Inputs:
    x: the velocity space grids , `vc ∈ [A,B]` if `isinv = false`

  Outputs:
    vc: outputs is the chebyshev grids;

      vc = vCmapping(x,a,b,vth;isinv=isinv)
"""
# [ns]
function vCmapping(x::AbstractArray{T,N},a::T,b::T,vth::AbstractVector{T2},ns::Int,nc::Int) where{T,T2,N}

    vc = zeros(T,nc,ns)
    for isp in 1:ns
        A = a * vth[isp]
        B = b * vth[isp]
        vc[:,isp] = 2 / (A - B) * x[:,isp] .- (A + B)/(A - B)
        x[1,isp] == A ? vc[1,isp] = 1.0 : 1
        x[end,isp] == B ? vc[end,isp] = - 1.0 : 1
    end
    return vc
end
