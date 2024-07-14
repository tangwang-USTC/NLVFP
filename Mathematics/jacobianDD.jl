
function jacobianDD(Jf::AbstractArray,f::AbstractVector,v::AbstractVector)

    nv = length(v)
    df = zero(f)
    dv = zero(v)
    df[1] = f[2] - f[1]
    df[nv] = f[nv] - f[nv-1]
    dv[1] = v[2] - v[1]
    dv[nv] = v[nv] - v[nv-1]
    for i in 2:nv-1
        df[i] = f[i+1] - f[i-1]
        dv[i] = v[i+1] - v[i-1]
    end
    for i in 1:nv
        for j in 1:nv
            Jf[i,j] = df[i] / dv[j]
        end
    end
    return Jf
end
