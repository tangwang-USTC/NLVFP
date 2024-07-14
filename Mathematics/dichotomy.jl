"""
  isNorm::Bool=true,
  mode: [:norm, :max]


  Outputs:
    iA, alp = dichotomyR(Rx;Rnamin=Rnamin,stop0=stop0,stop1=stop1,lengthx=lengthx,maxiterA=maxiterA,abstol=abstol,reltol=reltol)
    iA: maximum number of `α` process
    alp: parameter for `δx`
    Rnamin: = Rn / Mdd, the normalized residual or the maximum residual.
"""

function dichotomyR(Rx;Mdd::Real=1.0,isNorm::Bool=false,mode::Symbol=:norm,Rnamin::T=0.0,stop0::Float64=0.0,stop1::Float64=1.0,
             lengthx::Int=5,NRerr::Int=5,abstolav::Float64=1e-14,abstol::Float64=1e-6,reltol::Float64=1e-3,maxiterA::Int=15) where{T}

    if Rnamin == 0.0
        if mode == :norm
            Rnamin = norm(Rx(0.0)) / Mdd
        else
            Rnamin = maximum(abs.(Rx(0.0))) / Mdd
        end
    end
    Rnamin_up = 1Rnamin
    if isNorm == 1
        avec = [0; 10 .^ Vector(-15.0:.1:-2.0)]
        lengthx = length(avec)
    else
        avec = range(stop0,stop=stop1,length=lengthx)
    end
    Rna = zeros(lengthx)
    istop = 0
    iA = 0
    ia = 0
    alp = 0.0
    eps64 = eps(Float64)
    # abstolAR = min(abstol, Rnamin * reltol)
    while Rnamin > abstol
        iA += 1
        if ia == 0
        elseif ia == 1
            if avec[2] - stop0 < abstolav
                @warn("δα0 < abstolav, α[1]=",avec[1])
                break
            else
                stop0 = avec[1] - (avec[2] - avec[1])
                avec = range(stop0,stop= avec[2],length=lengthx)
            end
        elseif ia == lengthx
            if (stop1 - avec[lengthx-1]) < abstolav
                @warn("δα9 < abstolav, α[1]=",avec[lengthx])
                break
            else
                stop1 = avec[lengthx] + (avec[lengthx] - avec[lengthx-1])
                avec = range(avec[lengthx-1],stop= stop1,length=lengthx)
            end
        else
            avec = range(avec[ia-1],stop= avec[ia+1],length=lengthx)
        end
        if mode == :norm
            for ia in 1:lengthx
                Rna[ia] = norm(Rx(avec[ia])) / Mdd
            end
        else
            for ia in 1:lengthx
                Rna[ia] = maximum(abs.(Rx(avec[ia]))) / Mdd
            end
        end
        Rnamin, ia = findmin(Rna)
        alp = avec[ia]
        if abs(Rnamin - Rnamin_up) / (Rnamin_up + eps64) < 1e-15
            istop += 1
        else
            istop = 0
        end
        # @show (iA,ia,fmtf2.([alp,Rnamin]))
        Rnamin_up = 1Rnamin
        if iA > maxiterA
            # @warn("Alp iteration, before convergence, reach maxiterA=",maxiterA)
            break
        end
        if istop > NRerr
            # @warn(string("Alp iteration is convergent when stagnatting, istop=",istop, ",iA=,",iA,",α=",fmtf2(alp)))
            break
        end
    end
    return iA, alp, Rnamin
end

function dichotomyRp(Rx;Mdd::Real=1.0,isNorm::Bool=false,mode::Symbol=:norm,Rnamin::T=0.0,stop0::Float64=0.0,stop1::Float64=1.0,
             lengthx::Int=5,NRerr::Int=3,abstolav::Float64=1e-14,abstol::Float64=1e-6,reltol::Float64=1e-3,maxiterA::Int=15) where{T}

    if Rnamin == 0.0
        if mode == :norm
            Rnamin = norm(Rx(0.0)) / Mdd
        else
            Rnamin = maximum(abs.(Rx(0.0))) / Mdd
        end
    end
    Rnamin_up = 1Rnamin
    if isNorm == 1
        avec = [0; 10 .^ Vector(-15.0:.1:-2.0)]
        lengthx = length(avec)
    else
        avec = range(stop0,stop=stop1,length=lengthx)
    end
    Rna = zeros(lengthx)
    istop = 0
    iA = 0
    ia = 0
    alp = 0.0
    eps64 = eps(Float64)
    # abstolAR = min(abstol, Rnamin * reltol)
    while Rnamin > abstol
        iA += 1
        if ia == 0
        elseif ia == 1
            if avec[2] - stop0 < abstolav
                @warn("δα0 < abstolav, α[1]=",avec[1])
                break
            else
                stop0 = avec[1] - (avec[2] - avec[1])
                avec = range(stop0,stop= avec[2],length=lengthx)
            end
        elseif ia == lengthx
            if (stop1 - avec[lengthx-1]) < abstolav
                @warn("δα9 < abstolav, α[1]=",avec[lengthx])
                break
            else
                stop1 = avec[lengthx] + (avec[lengthx] - avec[lengthx-1])
                avec = range(avec[lengthx-1],stop= stop1,length=lengthx)
            end
        else
            avec = range(avec[ia-1],stop= avec[ia+1],length=lengthx)
        end
        if mode == :norm
            for ia in 1:lengthx
                Rna[ia] = norm(Rx(avec[ia])) / Mdd
            end
        else
            for ia in 1:lengthx
                Rna[ia] = maximum(abs.(Rx(avec[ia]))) / Mdd
            end
        end
        # label = string("iA=",iA)
        # xlabel = string("α")
        # ylabel = string("Rna")
        # pp = plot(avec,Rna,label=label,xlabel=xlabel,ylabel=ylabel,line=(3,:auto))
        # display(pp)
        # # pplog = plot(log.(abs.(avec) .+ 1e-16),Rna,label=label,line=(3,:auto))
        # # display(plot(pp,pplog,layout=(2,1)))
        Rnamin, ia = findmin(Rna)
        alp = avec[ia]
        if abs(Rnamin - Rnamin_up) / (Rnamin_up + eps64) < 1e-15
            istop += 1
        else
            istop = 0
        end
        @show (iA,ia,fmtf2.([alp,Rnamin]))
        Rnamin_up = 1Rnamin
        if iA > maxiterA
            @warn("Alp iteration, before convergence, reach maxiterA=",maxiterA)
            break
        end
        if istop > 4NRerr
            @warn(string("Alp iteration is convergent when stagnatting, istop=",istop, ",iA=,",iA))
            break
        end
    end
    return iA, alp, Rnamin
end

# log()

function dichotomyRlog(Rx;Mdd::Real=1.0,isNorm::Bool=false,mode::Symbol=:norm,Rnamin::T=0.0,stop0::Float64=0.0,stop1::Float64=1.0,
             lengthx::Int=5,abstolav::Float64=1e-14,abstol::Float64=1e-6,reltol::Float64=1e-3,maxiterA::Int=15) where{T}

    if Rnamin == 0.0
        if mode == :norm
            Rnamin = norm(Rx(0.0)) / Mdd
        else
            Rnamin = maximum(abs.(Rx(0.0))) / Mdd
        end
    end
    Rnamin_up = 1Rnamin
    if isNorm == 1
        avec = [0; 10 .^ Vector(-15.0:.1:-2.0)]
        lengthx = length(avec)
    else
        avec = range(stop0,stop=stop1,length=lengthx)
    end
    Rna = zeros(lengthx)
    istop = 0
    iA = 0
    ia = 0
    alp = 0.0
    eps64 = eps(Float64)
    abstolAR = min(abstol, Rnamin * reltol)
    while Rnamin > abstolAR
        iA += 1
        if ia == 0
        elseif ia == 1
            if avec[2] - stop0 < abstolav
                @warn("δα0 < abstolav, α[1]=",avec[1])
                break
            else
                stop0 = avec[1] - (avec[2] - avec[1])
                avec = range(stop0,stop= avec[2],length=lengthx)
            end
        elseif ia == lengthx
            if (stop1 - avec[lengthx-1]) < abstolav
                @warn("δα9 < abstolav, α[1]=",avec[lengthx])
                break
            else
                stop1 = avec[lengthx] + (avec[lengthx] - avec[lengthx-1])
                avec = range(avec[lengthx-1],stop= stop1,length=lengthx)
            end
        else
            avec = range(avec[ia-1],stop= avec[ia+1],length=lengthx)
        end
        if mode == :norm
            for ia in 1:lengthx
                Rna[ia] = norm(Rx(avec[ia])) / Mdd
            end
        else
            for ia in 1:lengthx
                Rna[ia] = maximum(abs.(Rx(avec[ia]))) / Mdd
            end
        end
        Rnamin, ia = findmin(Rna)
        alp = avec[ia]
        if abs(Rnamin - Rnamin_up) / (Rnamin_up + eps64) < 1e-15
            istop += 1
        else
            istop = 0
        end
        Rnamin_up = 1Rnamin
        if iA > maxiterA
            @warn("Alp iteration, before convergence, reach maxiterA=",maxiterA)
            break
        end
        if istop > 4
            @warn(string("Alp iteration is convergent when stagnatting, istop=",istop, ",iA=,",iA,",α=",fmtf2(alp)))
            break
        end
    end
    return iA, alp, Rnamin
end
