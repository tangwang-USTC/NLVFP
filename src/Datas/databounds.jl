
function bitbounds(T::DataType)

    if T == Float64
        boundln = 64
    elseif T == Int64 || T == Int
        boundln = 64
    elseif T == Int32 || T == Float32
        boundln = 32
    elseif T == Int16 || T == Float16
        boundln = 16
    else
        dgrefrg
    end
    return boundln
end
