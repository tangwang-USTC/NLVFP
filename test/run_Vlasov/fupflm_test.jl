p_dict = init_params(params_correct)
dimsv = p_dict["dims"][2]
nv = p_dict["nG"] + 2
vG, w1 = laguerre(nG, 0.0)
vnGL = 2vG[nG] - vG[nG-1]
v = [0.0; vG; vnGL]
nL = p_dict["nL"]
LM = nL - 1
nϕ = 2nL
#################33
ma = p_dict["ma"]
Zma = p_dict["Za"] / ma
dimsx = p_dict["dims"][1]
nx = p_dict["grid_size"]
t0 = 0.0
Momt = initial_Mom(dimsx,t0,nuT0,xb,nx)
####################
nt_max = p_dict["time_steps"]         # Maximum time steps,
dt = p_dict["time_step"]             # Initial time step
nEBt = p_dict["EB_probe_time"][1]
dataEB, dataEBt = init_dataEB(p_dict)
##########
if dimsx == 1
  xb = p_dict["xb"]
  if dimsv == 2
    datatype = Float64
    μ, w2, Mμ, Mun = LegendreMμ0(LM,datatype = Float64)
    fup = initial_fup2V(datatype,dimsx,nv,nL,nx)
    flm = SPTLmnD2V(datatype,dimsx,fup,Mμ, nx)
  elseif dimsv == 3
    datatype = ComplexF64
    μ, w2, Mμ, Mun = LegendreMμ(LM, datatype = Float64)
    fup = initial_fup3V(datatype,dimsx,nv,nL,nϕ,nx)
    Mom = Momt.Mom[1]
    initial_fup3V!(fup,dimsx,nL,v,μ,ma,Mom,nx)
    flm = SPTLmnD3V(datatype,dimsx,fup, μ, w2, nx)
  end
end
momentsN!(Mom,nv,v,w1,flm,nx)
vth = (2Mom[3] ./ ma).^0.5
f0 = real(flm[4][:,1,1])
f10 = real(flm[4][:,2,1])
f11 = real(flm[4][:,2,2])
f1n1 = imag((flm[4][:,2,2]))
ua = Mom[2]
  nv30 = v .< 10
  pf0 = plot(v[nv30],real(flm[4][nv30,1,1]),ylabel="f₀⁰")
  pf10 = plot(v[nv30],real(flm[4][nv30,2,1]),ylabel="f₁⁰",xlabel="v̂")
  pf11 = plot(v[nv30],real(flm[4][nv30,2,2]),ylabel="f₁¹")
  pf1n1 = plot(v[nv30],imag(flm[4][nv30,2,2]),ylabel="f₁⁻¹",xlabel="v̂")
  display(plot(pf0,pf10,pf11,pf1n1,layout=(4,1)))
  it = 1
  plot(xb, [Momt.Mom[it][1][ix] for ix in 1:nx],xlabel="x",ylabel="na" )      # na(nt=4,x)
  plot(xb, [Momt.Mom[it][3][ix] for ix in 1:nx],xlabel="x",ylabel="Ta" )      # Ta(nt=4,x)
  plot(xb, [Momt.Mom[it][2][1][ix] for ix in 1:nx],xlabel="x",ylabel="ux" )    # ux[nt=4,x]
  plot(xb, [Momt.Mom[it][2][2][ix] for ix in 1:nx],xlabel="x",ylabel="uy" )    # uy[nt=4,x]
  plot(xb, [Momt.Mom[it][2][3][ix] for ix in 1:nx],xlabel="x",ylabel="uz" )    # uz[nt=4,x]
