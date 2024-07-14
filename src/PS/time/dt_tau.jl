

"""
  dtk = dt_tau(dtk,tauk;ratio_dtk1=ratio_dtk1,dt_ratio=dt_ratio)
"""

function dt_tau(dtk::T,tauk::T;ratio_dtk1::T=ratio_dtk1,dt_ratio::T=dt_ratio) where{T}

    dtk1 = ratio_dtk1 * dtk
    dtk = min(dt_ratio * tauk, dtk1)
    # @show dt_ratio * tauk - dtk
    dtk == dtk1 || printstyled("The timestep is decided by `dt_tau`!",color=:purple,"\n")
    return dtk
end


function dt_tau_warn(dtk::T,tauk::T;ratio_dtk1::T=ratio_dtk1,dt_ratio::T=dt_ratio) where{T}

  dtk1 = ratio_dtk1 * dtk
  dtk = min(dt_ratio * tauk, dtk1)
  # @show dt_ratio * tauk - dtk
  dtk == dtk1 || printstyled("Warnning: The timestep should be decided by `dt_tau`!",color=:yellow,"\n")
  return dtk1
end
