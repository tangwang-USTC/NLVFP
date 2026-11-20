module IsotropicKingFit
using LeastSquaresOptim
include("KingDefinition.jl")
export fit, decode, density, moment, encode

function decode(parameters, count)
    logits = vcat(parameters[1:count-1],0.)
    weights = exp.(logits.-maximum(logits)); weights ./= sum(weights)
    widths = vcat(exp.(parameters[count:2count-2]),1.)
    shifts = parameters[2count-1:3count-2].*widths
    scale = sqrt(sum(weights.*(widths.^2 .+ 2shifts.^2/3)))
    (;weights,shifts=shifts./scale,widths=widths./scale)
end

function encode(weights, shifts, widths)
    vcat(log.(weights[1:end-1]./weights[end]),log.(widths[1:end-1]./widths[end]),shifts./widths)
end

function log_kernel(radius,shift,width)
    shell=abs(shift)
    argument=2radius*shell/width^2
    normalization=1.5log(pi)+3log(width)
    if argument<1e-3
        squared=argument^2
        log_sinhc=squared/6-squared^2/180+squared^3/2835-squared^4/37800
        return -(radius^2+shift^2)/width^2+log_sinhc-normalization
    end
    -((radius-shell)/width)^2+log(-expm1(-2argument))-log(2argument)-normalization
end
kernel(radius,shift,width)=exp(log_kernel(radius,shift,width))

function log_density(model,radius)
    terms=[log(w)+log_kernel(radius,u,s) for (w,u,s) in zip(model.weights,model.shifts,model.widths)]
    peak=maximum(terms)
    peak+log(sum(exp.(terms.-peak)))
end

density(model,radius) = sum(weight*kernel(radius,shift,width) for (weight,shift,width) in zip(model.weights,model.shifts,model.widths))

function moment(model,power)
    sum(model.weights[atom]*sum(binomial(power,j)*2.0^j/
        prod(2k+1 for k in 1:j;init=1.)*model.shifts[atom]^(2j)*model.widths[atom]^(2(power-j))
        for j in 0:power) for atom in eachindex(model.weights))
end

function fit(radius, volume, target, count; warm=nothing, lower_count=nothing, maximum_iterations=300,autodiff=:central,require_convergence=false)
    length(radius)==length(volume)==length(target)||error("King fit quadrature shape mismatch")
    lower = vcat(fill(-6.,count-1),fill(-2.,count-1),zeros(count))
    upper = vcat(fill(6.,count-1),fill(2.,count-1),fill(3.,count))
    starts = Vector{Float64}[zeros(3count-2)]
    warm===nothing||push!(starts,warm)
    for spread in (0.2,0.6)
        widths=count==1 ? [1.] : exp.(collect(range(-spread,spread;length=count)))
        for shell in (0.,0.6,1.2)
            weights=fill(1/count,count)
            push!(starts,encode(weights,shell.*widths,widths))
        end
        if count>1
            pattern=collect(range(0.,1.2;length=count))
            for orientation in (pattern,reverse(pattern)),shift in 0:count-1
                push!(starts,encode(fill(1/count,count),circshift(orientation,shift).*widths,widths))
            end
        end
    end
    if lower_count!==nothing
        previous=decode(lower_count,count-1)
        index=argmax(previous.weights)
        weights=copy(previous.weights);weights[index]/=2;push!(weights,weights[index])
        widths=vcat(previous.widths,previous.widths[index]);shifts=vcat(previous.shifts,previous.shifts[index])
        push!(starts,encode(weights,shifts,widths))
        widths[index]*=.97;widths[end]*=1.03
        push!(starts,encode(weights,shifts,widths))
    end
    square_root_target=sqrt.(max.(target,0.))
    square_root_volume=sqrt.(volume)
    function residual!(out,parameters)
        model=decode(parameters,count)
        for j in eachindex(radius)
            out[j]=square_root_volume[j]*(exp(log_density(model,radius[j])/2)-square_root_target[j])
        end
    end
    best=nothing
    failures=0
    failure_messages=String[]
    converged_starts=0
    best_unconverged=Inf
    for start in starts
        candidate=clamp.(start,lower,upper)
        try
            problem=LeastSquaresProblem(x=candidate,(f!)=residual!,output_length=length(radius),autodiff=autodiff)
            result=optimize!(problem,Dogleg(LeastSquaresOptim.QR());iterations=maximum_iterations,x_tol=1e-11,f_tol=1e-14,g_tol=1e-11,
                lower,upper,show_trace=false)
            if !result.converged
                polish=LeastSquaresProblem(x=copy(result.minimizer),(f!)=residual!,output_length=length(radius),autodiff=autodiff)
                result=optimize!(polish,LevenbergMarquardt(LeastSquaresOptim.QR());iterations=maximum_iterations,x_tol=1e-11,f_tol=1e-14,g_tol=1e-11,
                    lower,upper,show_trace=false)
            end
            parameters=copy(result.minimizer)
            defect=zeros(length(radius));residual!(defect,parameters)
            value=sum(abs2,defect)
            converged_starts+=result.converged
            if !result.converged
                best_unconverged=min(best_unconverged,value)
                require_convergence&&continue
            end
            if best===nothing||value<best.squared_Hellinger
                model=decode(parameters,count)
                l1=sum(volume.*abs.([density(model,r) for r in radius].-target))
                best=(;parameters,model,squared_Hellinger=value,L1=l1,iterations=result.iterations,
                    optimizer_converged=result.converged)
            end
        catch exception
            exception isa InterruptException&&rethrow()
            failures+=1
            push!(failure_messages,sprint(showerror,exception))
        end
    end
    best===nothing&&error("All King fit starts failed: ",join(unique(failure_messages)," | "))
    (;best...,failed_starts=failures,attempted_starts=length(starts),converged_starts,best_unconverged_squared_Hellinger=best_unconverged)
end
end
