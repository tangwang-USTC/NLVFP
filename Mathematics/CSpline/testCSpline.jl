
"""
  Multi-conservations scheme where

    ' jvec = -2:1:jMax, jMax = oss - 4`.

  Inputs:
    nj: The number of conservations

  Outputs:
    M2, M1 = dfvLCS3dp1n1(M2,M1,fLn0,vG0,nc0,oss,Ij,bc;method3M=method3M)
"""

function dfvLCS3dp1n2(M2::AbstractVector{T},M1::AbstractVector{T},fLn0::AbstractVector{T},
    vG0::AbstractVector{Tb},nc0::Int,oss::Int,Ij::AbstractArray{T},bc::Tuple{Symbol,Vector{T}};
    method3M::Symbol=:chasing,ja::Int=-2,dj::Int=1) where{T,Tb,N}

    if oss == 4
        return dfvLCS3p4(M2,M1,fLn0,vG0,nc0,bc;method3M=method3M)
    else
        if oss == 5
            nc2 = 24
            cu  = 1
            cv1 = 3
            cv2 = - 3
            cw  = 1
            cd1 = - 7/2
            cd2 = - 3/2
            cD1 = 3/2
            cD2 = 7/2
            cIf = [5]
            cIv2 = [[1]
                   ]
            cIvp = [[1]
                   ]
        elseif oss == 6
            nc2 = 60
            cu  = - 1
            cv1 = 4
            cv2 = - 4
            cw  = - 1
            cd1 = - 7
            cd2 = 2
            cD1 = - 2
            cD2 = 7
            cIf = [2, -42]
            cIv2 = [[13, 21],
                    [1]
                   ]
            cIvp = [[8, 21],
                    [1]
                   ]
        elseif oss == 7
        elseif oss == 8
        elseif oss == 9
        elseif oss == 10
        else
            jfrhuyjk
        end
        if oss == 5
        else
            nj = oss - 4
            if bc[1] == :Neumann
                u = zeros(T,nc0)
                # v = zeros(T,nc0)
                w = zeros(T,nc0)
                d = zeros(T,nc0)
                h = diff(vG0)       # nodes
                # Computing the elements of matrix `𝔸` in the master equations: The basic three-moment equations
                # Applying the Neumman boundary conditions at the left endpoint
                Idf2 = zeros(T,nc0-1)
                k = 1
                hk = h[k]
                DI = zeros(T,nc0-1)
                cwVh = zeros(T,nc0-1)
                cv2Vh = zeros(T,nc0-1)
                if vG0[1] == 0.0
                    df2k = (cd2 * fLn0[k+1] + cd1 * fLn0[k]) / hk
                    ij = 1
                    If2k = cIf[ij] * cIv2[ij][1] * Ij[k,ij]
                    for ij in 2:nj
                        If2k += cIf[ij] * cIv2[ij][1] * Ij[k,ij]
                    end
                    If2k /= hk ^2
                    Idf2[k] = df2k + If2k
                    #
                    v = copy(cv2)
                    w[k] = cw / cv2
                    d[k] = - nc2 * (Idf2[k] - bc[2][1]) / hk / v
                    k = 2
                    hkn1 = copy(hk)
                    hk = h[k]
                    hxk = hk / vG0[k]
                    if oss == 6
                        IVhk = [[1.0, 1.0 / hxk],
                                 [1.0 /hxk]
                                ]
                    elseif oss == 7
                    else
                        sgndjhn
                    end
                    ij = 1
                    If2k = cIf[ij] * dot(cIv2[ij], IVhk[ij]) * Ij[k,ij]
                    for ij in 2:nj
                        If2k += cIf[ij] * dot(cIv2[ij], IVhk[ij]) * Ij[k,ij]
                    end
                    If2k /= hk ^2
                    ij = 1
                    Ifk = cIf[ij] * cIvp[ij][1] * Ij[k,ij]
                    for ij in 2:nj
                        If2k += cIf[ij] * cIvp[ij][1] * Ij[k,ij]
                    end
                    Ifk /= hkn1 ^2
                    IVhkn1 = copy(IVhk)
                    Dfk = (cD2 * fLn0[k] + cD1 * fLn0[k-1]) / hkn1
                    df2k = (cd2 * fLn0[k+1] + cd1 * fLn0[k]) / hk
                    Idf2[k] = df2k + If2k
                    #
                    v = hk * cv2 - hkn1 * cv1
                    u[k] = hkn1 * cu / v
                    w[k] = hk * cw / v
                    d[k] = - nc2 * (Idf2[k] - Ifk - Dfk) / v
                else
                    @warn(" The left endpoint `v[1] → 0` will cause larger numerical errors in the local-normalized shceme.
                    `v[1] = 0` or `v[1] ≫ 0`, i.e. `v[1] > 1e-6` is proposed.")
                    hxk = hk / vG0[k]
                    if oss == 6
                        IVhk = [[1.0, 1.0 / hxk],
                                 [1.0 /hxk]
                                ]
                    elseif oss == 7
                    else
                        sgndjhn
                    end
                    ij = 1
                    If2k = cIf[ij] * dot(cIv2[ij], IVhk[ij]) * Ij[k,ij]
                    for ij in 2:nj
                        If2k += cIf[ij] * dot(cIv2[ij], IVhk[ij]) * Ij[k,ij]
                    end
                    If2k /= hk ^2
                    ij = 1
                    Ifk = cIf[ij] * dot(cIvp[ij], IVhkn1[ij]) * Ij[k-1,ij]
                    for ij in 2:nj
                        Ifk += cIf[ij] * dot(cIvp[ij], IVhkn1[ij]) * Ij[k-1,ij]
                    end
                    Ifk /= hkn1 ^2
                    IVhkn1 = copy(IVhk)
                    df2k = (cd2 * fLn0[k+1] + cd1 * fLn0[k]) / hk
                    Idf2[k] = df2k + If2k
                    #
                    v = cv2
                    w[k] = cw / v
                    d[k] = - nc2 * (Idf2[k] - bc[2][1]) / hk / v

                    k = 2
                    hkn1 = copy(hk)
                    hk = h[k]
                    hxk = hk / vG0[k]
                    if oss == 6
                        IVhk = [[1.0, 1.0 / hxk],
                                 [1.0 /hxk]
                                ]
                    elseif oss == 7
                    else
                        sgndjhn
                    end
                    ij = 1
                    If2k = cIf[ij] * dot(cIv2[ij], IVhk[ij]) * Ij[k,ij]
                    for ij in 2:nj
                        If2k += cIf[ij] * dot(cIv2[ij], IVhk[ij]) * Ij[k,ij]
                    end
                    If2k /= hk ^2
                    ij = 1
                    Ifk = cIf[ij] * dot(cIvp[ij], IVhkn1[ij]) * Ij[k-1,ij]
                    for ij in 2:nj
                        Ifk += cIf[ij] * dot(cIvp[ij], IVhkn1[ij]) * Ij[k-1,ij]
                    end
                    Ifk /= hkn1 ^2
                    IVhkn1 = copy(IVhk)
                    Dfk = (cD2 * fLn0[k] + cD1 * fLn0[k-1]) / hkn1
                    df2k = (cd2 * fLn0[k+1] + cd1 * fLn0[k]) / hk
                    Idf2[k] = df2k + If2k
                    #
                    v = hk * cv2 - hkn1 * cv1
                    u[k] = hkn1 * cu / v
                    w[k] = hk * cw  / v
                    d[k] = - nc2 * (Idf2[k] - Ifk - Dfk) / v
                end
                for k in 3:nc0-1
                    hkn1 = copy(hk)
                    hk = h[k]
                    hxk = hk / vG0[k]
                    if oss == 6
                        IVhk = [[1.0, 1.0 / hxk],
                                 [1.0 /hxk]
                                ]
                    elseif oss == 7
                    else
                        sgndjhn
                    end
                    ij = 1
                    If2k = cIf[ij] * dot(cIv2[ij], IVhk[ij]) * Ij[k,ij]
                    for ij in 2:nj
                        If2k += cIf[ij] * dot(cIv2[ij], IVhk[ij]) * Ij[k,ij]
                    end
                    If2k /= hk ^2
                    ij = 1
                    Ifk = cIf[ij] * dot(cIvp[ij], IVhkn1[ij]) * Ij[k-1,ij]
                    for ij in 2:nj
                        Ifk += cIf[ij] * dot(cIvp[ij], IVhkn1[ij]) * Ij[k-1,ij]
                    end
                    Ifk /= hkn1 ^2
                    IVhkn1 = copy(IVhk)
                    Dfk = (cD2 * fLn0[k] + cD1 * fLn0[k-1]) / hkn1
                    df2k = (cd2 * fLn0[k+1] + cd1 * fLn0[k]) / hk
                    Idf2[k] = df2k + If2k
                    #
                    v = hk * cv2 - hkn1 * cv1
                    u[k] = hkn1 * cu / v
                    w[k] = hk * cw  / v
                    d[k] = - nc2 * (Idf2[k] - Ifk - Dfk) / v
                end
                # Applying the Neumman boundary conditions at the right endpoint
                k = nc0
                hkn1 = copy(hk)
                Dfk = (cD2 * fLn0[k] + cD1 * fLn0[k-1]) / hkn1
                ij = 1
                Ifk = cIf[ij] * dot(cIvp[ij], IVhkn1[ij]) * Ij[k-1,ij]
                for ij in 2:nj
                    Ifk += cIf[ij] * dot(cIvp[ij], IVhkn1[ij]) * Ij[k-1,ij]
                end
                Ifk /= hkn1 ^2
                v = cv1
                u[k] = - cu / v
                d[k] = - nc2 * (Ifk + Dfk - bc[2][2]) / hkn1 / v
                # Solving the linear equations including the boundary conditions to give the second derivatives.
                if method3M == :chasing
                    M2 = chasingM1(M2,u[2:end],ones(T,nc0),w[1:end-1],d)
                else
                    𝔸 = diagm(-1 => u[2:end], 0 => ones(T,nc0), 1 => w[1:end-1])
                    M2 = solver3M(M2,𝔸,d;method3M=method3M)
                end
                ##### Computing the first derivatives and the function values.
                M1[1] = bc[2][1]        # M1[k+1] = ck2
                vec1 = 2:nc0-1
                for k in vec1
                    M1[k] = (cw * M2[k+1] + cv2 * M2[k]) * h[k] / nc2 + Idf2[k]
                end
                M1[nc0] = copy(bc[2][2])
                return M2,M1
            else
                rdj
            end
        end
    end
end

    nc2 = 210
    cu  = - 1
    cv1 = 6
    cv2 = - 6
    cw  = - 1
    cd1 = - 17
    cd2 = 3
    cD1 = - 3
    cD2 = 17
    cIf = [8, -72, 72, -1320]
    cIv2 = [[22, 144, 279, 165],
            [16, 62, 55],
            [31, 55],
            [1]
           ]
    cIvp = [[8, 81, 216, 165],
            [9, 48, 55],
            [24, 55],
            [1]
           ]
