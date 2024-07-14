using Plots, FFTW

freq = 1000
T = 1/ freq
nt =1500
tvec = (0:nt-1) * T
f50 = 50  # freq = 50 [Hz]
f120 = 120  # freq = 120 [Hz]
Signs(t) = 0.7 * sin.(2π * f50 * t) + sin.(2π * f120 * t)
Signs_sinrandn = Signs(tvec) + 2 * randn(length(tvec))
plot(1000 * tvec[1:50], Signs_sinrandn[1:50])
#######  #####  FFT
Y = fft(Signs_sinrandn)
P2 = abs.(Y / nt)
P1 = P2[1:(nt/2 + 1 |>Int)]
P1[2:end-1] = 2 * P1[2:end-1]
f = freq * (0:nt/2) / nt
plot(f,P1)
dims = 1
Y = fft(Y,dims)
## Gaussian pusle
freq_g = 99
t = -0.5:1/freq_g:0.5
nt = length(t)
Sign_g = 1/(4 * √(2π * 0.01)) * (exp.(- t.^2 /(2 * 0.01)))
plot(t, Sign_g)
n = nextpow(2,nt)
Y = fft(Sign_g)
f = freq_g * (0:(nt/2)) / nt |> collect
P = abs.(Y / nt).^2

plot(f, P[1:(nt/2 + 1|>Int)])
###################################
freq = 1000
T = 1/freq
L = 1000
t = (0:L-1) * T
Ei(t,freq) = cos.(2π * freq * t)
E = [Ei(t,50) Ei(t,150)  Ei(t,300)]
p1= plot(t[1:100],E[1:100,1])
p2 = plot(t[1:100],E[1:100,2])
p3 = plot(t[1:100],E[1:100,3])
plot(p1,p2,p3,layout=(3,1))
n = nextpow(2,L)
n = L
dims = 1
Y = fft(E, dims)
P2 = abs.(Y / L)
P1 = P2[1:(n/2+1|>Int),:]
P1[2:end-1,:] = 2 * P1[2:end-1,:]
p31 = plot(0:(freq/n |>Int):((freq/2 - freq/n)|>Int),P1[1:(n/2 |>Int),1])
p32 = plot(0:(freq/n |>Int):((freq/2 - freq/n)|>Int),P1[1:(n/2 |>Int),2])
p33 = plot(0:(freq/n |>Int):((freq/2 - freq/n)|>Int),P1[1:(n/2 |>Int),3])
plot(p31,p32,p33,layout=(3,1))
###################################
