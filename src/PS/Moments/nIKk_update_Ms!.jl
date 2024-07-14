


"""

  Inputs:
    nak1::Vector{Int64} = nak, which will be changed
    Iak1: = zeros(Int64,nsk1)
    Kak1: = zeros(Int64,nsk1)
    vathk1::Vector{Int64} = vathk, which will be changed
    vathk:
    Mck1::AbstractArray{T,3}
    Rck1: = Rck,   i = 0, the explicit Euler method
          = Rck1i, i ≥ 1, the implicit Euler method
          = (Rck + Rck1i)/2, i ≥ 1, the Trapezoidal method
    Rck1[njMs+1,1,:]                    # `w3k1 = Rdtvath = vₜₕ⁻¹∂ₜvₜₕ = 𝒲 / 3`
    Rck[njMs+1,1,:]                     # `w3k

  Outputs:
    nIKT_update!(nak1,vathk1,Iak1,Kak1,δvathk1,mak1,nsk1,Mck1;
                is_vth_ode=is_vth_ode,is_corrections_na=is_corrections_na)
    nIKT_update!(nak,vathk,Iak,Kak,mak,nsk,Mck;
                is_vth_ode=is_vth_ode,is_corrections_na=is_corrections_na)
    nIKT_update!(vathk,mak,nak,nsk,Mck)
"""

# [], vathk1 = deepcopy(Mck1[njMs+1,1,:])
function nIKT_update!(nak1::AbstractVector{T},vathk1::AbstractVector{T},
    Iak1::AbstractVector{T},Kak1::AbstractVector{T},δvathk1::AbstractVector{T},
    mak1::AbstractVector{T},nsk1::Int64,Mck1::AbstractArray{T,N};
    is_corrections_na::Bool=true,is_vth_ode::Bool=false) where{T,N}
    
    nsp_vec = 1:nsk1

    # # # # Updating the conservative momentums `n, I, K`
    if is_corrections_na == false
        for isp in nsp_vec
            nak1[isp] = Mck1[1, 1, isp] / mak1[isp]
        end
    else
        for isp in nsp_vec
            Mck1[1, 1, isp] = nak1[isp] * mak1[isp]
        end
    end
    ρk1 = mak1 .* nak1

    for isp in nsp_vec
        Iak1[isp] = Mck1[1, 2, isp]
        Kak1[isp] = Mck1[2, 1, isp] * CMcKa       # Mck1[2, 1, isp] * 3 / 4
    end

    if is_vth_ode
        vathk1[:] = Mck1[njMs+1,1,:]
        # @show Mck1[2, 1, :]
        if is_check_vth
            for isp in nsp_vec
                if Iak1[isp] == 0.0
                    δvathk1[isp] = (Mck1[2, 1, isp] / ρk1[isp])^0.5 / vathk1[isp] - 1
                else
                    δvathk1[isp] = ((Mck1[2, 1, isp] / ρk1[isp] - 2/3 * (Iak1[isp] / ρk1[isp])^2))^0.5 / vathk1[isp] - 1
                end
            end
            norm(δvathk1) ≤ epsT1000 || @warn("ode: The initial values of `Iak1, Kak1` and `vathk1` are not consistent!",δvathk1)
        end
        for isp in nsp_vec
            if Iak1[isp] == 0.0
                Mck1[2, 1, isp] = ρk1[isp] * vathk1[isp]^2
            else
                Mck1[2, 1, isp] = ρk1[isp] * (vathk1[isp]^2 + 2/3 * (Iak1[isp] / ρk1[isp])^2)
            end
        end
        # @show Mck1[2, 1, :]
    else
        # dsgfgvb
        for isp in nsp_vec
            if Iak1[isp] == 0.0
                vathk1[isp] = (Mck1[2, 1, isp] / ρk1[isp])^0.5
            else
                vathk1[isp] = ((Mck1[2, 1, isp] / ρk1[isp] - 2/3 * (Iak1[isp] / ρk1[isp])^2))^0.5
            end
        end
        if is_check_vth
            for isp in nsp_vec
                if Iak1[isp] == 0.0
                    δvathk1[isp] = Mck1[njMs+1,1,isp] / vathk1[isp] - 1
                else
                    δvathk1[isp] = Mck1[njMs+1,1,isp] / vathk1[isp] - 1
                end
            end
            # norm(δvathk1) ≤ epsT1000 || @warn("The constraint of `Iak1, Kak1` and `vathk1` are not consistent!",δvathk1)
        end
    end
end

function nIKT_update!(nak::AbstractVector{T},vathk::AbstractVector{T},
    Iak::AbstractVector{T},Kak::AbstractVector{T},
    mak::AbstractVector{T},nsk::Int64,Mck::AbstractArray{T,N};
    is_corrections_na::Bool=true,is_vth_ode::Bool=false) where{T,N}
    
    nsp_vec = 1:nsk

    # # # # Updating the conservative momentums `n, I, K`
    if is_corrections_na == false
        for isp in nsp_vec
            nak[isp] = Mck[1, 1, isp] / mak[isp]
        end
    else
        for isp in nsp_vec
            Mck[1, 1, isp] = nak[isp] * mak[isp]
        end
    end
    ρk = mak .* nak

    for isp in nsp_vec
        Iak[isp] = Mck[1, 2, isp]
        Kak[isp] = Mck[2, 1, isp] * CMcKa       # Mck[2, 1, isp] * 3 / 4
    end

    if is_vth_ode
        vathk[:] = Mck[njMs+1,1,:]
        for isp in nsp_vec
            if Iak1[isp] == 0.0
                Mck1[2, 1, isp] = ρk1[isp] * vathk1[isp]^2
            else
                Mck1[2, 1, isp] = ρk1[isp] * (vathk1[isp]^2 + 2/3 * (Iak1[isp] / ρk1[isp])^2)
            end
        end
    else
        sdgffgffg
        for isp in nsp_vec
            if Iak[isp] == 0.0
                vathk[isp] = (Mck[2, 1, isp] / ρk[isp])^0.5
            else
                vathk[isp] = ((Mck[2, 1, isp] / ρk[isp] - 2/3 * (Iak[isp] / ρk[isp])^2))^0.5
            end
        end
    end
end

function nIKT_update!(vathk::AbstractVector{T},mak::AbstractVector{T},
    nak::AbstractVector{T},nsk::Int64,Mck::AbstractArray{T,N}) where{T,N}
    
    for isp in 1:nsk
        ρk = mak[isp] .* nak[isp]
        Mck[1, 1, isp] = ρk
        if Iak[isp] == 0.0
            vathk[isp] = (Mck[2, 1, isp] / ρk)^0.5
        else
            vathk[isp] = ((Mck[2, 1, isp] / ρk - 2/3 * (Mck[1, 2, isp] / ρk)^2))^0.5
        end
    end
end
