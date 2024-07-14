using AbstractFFTs

"""
    bandpass(signal, fs, fmin, fmax)

带通滤波

## 参数

- `signal`: 原信号
- `fs`: 采样频率
- `fmin`: 通带下限
- `fmax`: 通带上限
"""

function bandpass(signal, fs::Real, fmin::Real, fmax::Real)
    spectrum = fft(signal)[:]
    n = length(spectrum)

    pmin = round(Int32, n * fmin / fs) + 1
    pmax = round(Int32, n * fmax / fs) + 1

    if fmin ≥ eps(fmin) spectrum[1] = 0 end
    pmin = max(pmin, 1)
    clear!(spectrum, 2:pmin)         # pmin - 2           + 1 = pmin - 1
    clear!(spectrum, n - pmin + 2:n) # n - (n - pmin + 2) + 1 = pmin - 1

    if fmax < fs spectrum[n ÷ 2] = 0 end
    pmax = min(pmax, n ÷ 2 - 1)
    clear!(spectrum, pmax:n ÷ 2 - 1)     # n ÷ 2 - 1 - pmax       + 1 = n ÷ 2 - pmax
    clear!(spectrum, n ÷ 2 + 1:n - pmax) # n - pmax - (n ÷ 2 + 1) + 1 = n ÷ 2 - pmax

    ifft(spectrum)
end



function clear!(vector::Vector, range)
    vector[range] .=0
    nothing
end
