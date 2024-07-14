


mutable struct vGssubup{T2,T <: Real}

  issub        :: Vector{Bool}
  nck0sub      :: Vector{Int}
  nvlevelsub   :: Vector{Vector{Int}}
  vlevelsub    :: Vector{Vector{Vector{T}}}
end

function vGssubup!{vGsub <: vGssubup, T <: Real}

  append!()
end

  iss = [0,1] |> Vector{Bool}
  nck0s = [1,2] |> Vector{Int}
  nvlevels = [[4], [ 3,5]]
  vlevels = [[[1,2,3,4]], [[5,6,7],[8,9,3,2,1]]]

vGssubup(iss,nck0s,nvlevels,vlevels)
