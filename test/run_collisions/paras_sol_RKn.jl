
## Parameters: Time integration (ODE solver)
t0 = 0.0 
tmax = t_unit * nτ              # The maximal time moment.
tspan = (0.0, tmax)
if is_fixed_timestep == false
    count_tau_update = 20                    # for updating the parameter `nMod`
    tau0 = deepcopy(tau)
else
    count_tau_update = 20                    # for updating the parameter `nMod`
    tau0 = [NaN, NaN]
    # ratio_tau_min = tau[1] / τ₀
    # ratio_tau_max = tau[2] / τ₀
    # Nstep_max0 = ceil(Int64,Nτ_fix * nτ * ratio_tau_max) 
    # Nstep_max = min(Nstep_max0, Nstep_max)
    # Nstep_max = 5                 # For the testting of the fixed time step algorithm.
end

if is_fixed_timestep
    if is_dt_unit == true
        dt_initial = 1 / Nτ_fix
    else
        dt_initial = (tspan[2] - tspan[1]) / Nτ_fix
    end
    @show tspan, nτ, dt_initial
else
    dt_initial = t_unit / Nτ_fix    # (~ τ₀) initial time step when given and time step for fixed_timestep algorithms.
    dt_ratio = dt_initial / τ₀      # which gives the arbitrary time step to be `dtk = dt_ratio * tauk[1]`
    tau_min_0 = deepcopy(tau0[1])
    tau_max_0 = deepcopy(tau0[2])
    dt_initial_k = dt_ratio * tau_min_0    # = dt_initial * ratio_tau_min # (~ tau_min_0) 
    dt_initial_max = dt_ratio * tau_max_0  # = dt_initial * ratio_tau_max # (~ tau_max_0)
    dt_initial = min(dt_initial_min, dt_initial)
    Nt_predic = nτ  / dt_initial_k * Nτ_fix
    @show tspan, (nτ, Nstep_max), dt_initial, dt_initial_k
end
contert5 = 0       # (==0), for show when solving during ODE

# [:Euler(1,:,N), :Trapz(2,:,N⁺), :RadauI3(3,2,N), :GL4(4,2,N), :LobattoIIIA4(4,3,N), :LobattoIIIA6(6,4,N), ⋯]
# [:Euler(1,:,N), :Trapz(2,:,N⁺), :Kutta3(3,3,N⁺), :Heun3(3,4,N⁺), :Runge416(4,3,N⁺), :Runge438(4,4,N⁺), :Runge42(4,4(5),N⁺), :RK5(5,5,N⁺), :Lawson5(5,5(13),N⁺), :Runge5(5,6,N⁺)]

if orderRK == 1
    algRK = :Euler
    algname = algRK
elseif orderRK == 2
    algRK = :Trapz
    algname = algRK
else
    if orderRK == 3
        if is_explicit
            if rsRK == 3
                algRK = :Kutta3
            elseif rsRK == 4
                algRK = :Heun3
            else
                eegerhb
            end
            algname = algRK
        else
            if orderEmbeded == 1
                algEmbeded = :Euler
            elseif orderEmbeded == 2
                algEmbeded = :Trapz
            elseif orderEmbeded == 3
                algEmbeded = :Heun3
            else
                etrthyhnj
            end
            if rsRK == 2
                algRK = :RadauIA3
                algname = string(string(algRK) * "",iterRK)
                algname = Symbol(algname * "",algEmbeded)
            elseif rsRK == 3
                algRK = :RadauIIA3
                algname = string(string(algRK) * "",iterRK)
                algname = Symbol(algname * "",algEmbeded)
            else
                eegerhb
            end
        end
    elseif orderRK == 4
        if is_explicit
            defgbn
        else
            if rsRK == 3
                if orderEmbeded == 3
                    if rsEmbeded == 3
                        algEmbeded = :Kutta3
                    else
                        etrthyhnj
                    end
                elseif orderEmbeded == 4
                    if rsEmbeded == 3
                        algEmbeded = :RK416
                    elseif rsEmbeded == 5
                        algEmbeded = :RK42
                    elseif rsEmbeded == 4
                        algEmbeded = :RK438
                    else
                        etrthyhnj
                    end
                else
                    etrthyhnj
                end
                algRK = :LobattoIIIA4
            elseif rsRK == 2
                algRK = :GL4
            else
                eegerhb
            end
            algname = Symbol(string(algRK) * "",algEmbeded)
        end
    elseif orderRk == 6
    else
        eefefeg
    end
end
@show algname

