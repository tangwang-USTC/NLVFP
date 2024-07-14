

"""
  Matrices of `ℂ𝕏uⱼⁿᵐ` for analytical results of Shkarofsky integrals:

    Cⱼᵖ = ℂ𝕏uⱼⁿᵐ [v̂⁰, v̂¹, v̂², ⋯, v̂ᵐ]ᵀ

  where

    ℂ𝕏uⱼⁿᵐ = [ûᵦ⁰, ûᵦ¹, ûᵦ², ⋯, ûᵦⁿ] (ℂⱼᵖ × 𝕏ⱼⁿᵐ)

  Inputs:

  Outputs:
    M = 
"""

"""
  Intputs:
    L:

  Outputs:
    mCM, CXuL0::Matrix = CXuL0nm(uhbs,vhbths,L)
    mCM, CXuL2::Matrix = CXuL2nm(uhbs,vhbths,L)
    mCM, CXuL1::Matrix = CXuL1nm(uhbs,vhbths,L)
    mCM, CXuLn1::Matrix = CXuLn1nm(uhbs,vhbths,L)

    a = CXuLn1nm(0.1,0.5,3)

"""

function CXuL0nm(uhbs::T,vhbths::T,L::Int64) where{T}

    if abs(uhbs) ≤ 2epsT
    else
      if L == 0
      # elseif L == 1
      else 
        (nCM, mCM), CM = CjpIL0(L)

        uhnM = ones(T,1,nCM)
        uhnM[1,1] = 1.0
        for i in 2:nCM
          uhnM[1,i] = uhbs ^ (i - 1)
        end

        if vhbths == 1.0
          # CMu = uhnM * T.(CM)
          return mCM, uhnM * T.(CM)
        else
          CM = CM .* vhbths .^ orderXL0nm(L)    # CXM
          CM[isnan.(CM)] .= 0.0
          
          # CMu = uhnM * CM
          return mCM, uhnM * CM
        end
      end
    end
end

function CXuL2nm(uhbs::T,vhbths::T,L::Int64) where{T}

  if abs(uhbs) ≤ 2epsT
    sdfgvb
  else
    if L == 0
    # elseif L == 1
    else 
      (nCM, mCM), CM = CjpIL2(L)

      uhnM = ones(T,1,nCM)
      uhnM[1,1] = 1.0
      for i in 2:nCM
        uhnM[1,i] = uhbs ^ (i - 1)
      end

      if vhbths == 1.0
        # CMu = uhnM * T.(CM)
        return mCM, uhnM * T.(CM)
      else
        CM = CM .* vhbths .^ orderXL2nm(L)    # CXM
        CM[isnan.(CM)] .= 0.0
        
        # CMu = uhnM * CM
        return mCM, uhnM * CM
      end
    end
  end
end

function CXuL1nm(uhbs::T,vhbths::T,L::Int64) where{T}

  if abs(uhbs) ≤ 2epsT
    sdfgvb
  else
    if L == 0
    # elseif L == 1
    else 
      (nCM, mCM), CM = CjpJL1(L)
      @show nCM, mCM

      uhnM = ones(T,1,nCM)
      uhnM[1,1] = 1.0
      for i in 2:nCM
        uhnM[1,i] = uhbs ^ (i - 1)
      end

      if vhbths == 1.0
        # CMu = uhnM * T.(CM)
        return mCM, uhnM * T.(CM)
      else
        CM = CM .* vhbths .^ orderXL1nm(L)    # CXM
        CM[isnan.(CM)] .= 0.0
        
        # CMu = uhnM * CM
        return mCM, uhnM * CM
      end
    end
  end
end

function CXuLn1nm(uhbs::T,vhbths::T,L::Int64) where{T}

  if abs(uhbs) ≤ 2epsT
    sdfgvb
  else
    if L == 0
    # elseif L == 1
    else 
      (nCM, mCM), CM = CjpJLn1(L)
      @show nCM, mCM

      uhnM = ones(T,1,nCM)
      uhnM[1,1] = 1.0
      for i in 2:nCM
        uhnM[1,i] = uhbs ^ (i - 1)
      end

      if vhbths == 1.0
        # CMu = uhnM * T.(CM)
        return mCM, uhnM * T.(CM)
      else
        CM = CM .* vhbths .^ orderXLn1nm(L)    # CXM
        CM[isnan.(CM)] .= 0.0
        
        # CMu = uhnM * CM
        return mCM, uhnM * CM
      end
    end
  end
end



