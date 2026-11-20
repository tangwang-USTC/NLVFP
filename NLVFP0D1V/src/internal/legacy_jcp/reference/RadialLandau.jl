module RadialLandau
using LinearAlgebra, FastGaussQuadrature, ForwardDiff

struct RadialSpace
    squared_speed::Vector{Float64}
    weights::Vector{Float64}
    derivative::Matrix{Float64}
end

function radial_space(order,thermal_width)
    nodes,weights=gausslaguerre(order,0.5)
    squared_speed=thermal_width^2 .* nodes
    volume=2pi*thermal_width^3 .* weights .* exp.(nodes)
    derivative=zeros(order,order)
    # Derivative of the piecewise linear log-distribution. At an interior
    # knot use the mean of its two one-sided derivatives. Linear energy
    # potentials and Maxwellian logarithms are reproduced exactly.
    for i in 1:order
        if i==1
            slope=inv(squared_speed[2]-squared_speed[1])
            derivative[i,1]=-slope;derivative[i,2]=slope
        elseif i==order
            slope=inv(squared_speed[end]-squared_speed[end-1])
            derivative[i,end-1]=-slope;derivative[i,end]=slope
        else
            left=inv(squared_speed[i]-squared_speed[i-1])/2
            right=inv(squared_speed[i+1]-squared_speed[i])/2
            derivative[i,i-1]=-left
            derivative[i,i]=left-right
            derivative[i,i+1]=right
        end
    end
    RadialSpace(squared_speed,volume,derivative)
end

struct RadialPair
    a::Int
    b::Int
    geometry::Matrix{Float64}
end

function pair_geometry(spaces,gamma)
    pairs=RadialPair[]
    for a in eachindex(spaces),b in a:length(spaces)
        xa=spaces[a].squared_speed;xb=spaces[b].squared_speed
        # Half of the angular integral of (1-mu^2)/|v-v'|^3 equals
        # 2/(3 max(r,s)^3). Radial derivatives contribute 4 r^2 s^2.
        geometry=[gamma[a,b]*8/3*spaces[a].weights[i]*spaces[b].weights[j]*xa[i]*xb[j]/max(xa[i],xb[j])^1.5 for i in eachindex(xa),j in eachindex(xb)]
        push!(pairs,RadialPair(a,b,geometry))
    end
    pairs
end

function source(logarithms,spaces,masses,pairs,offsets)
    values=[exp.(logarithms[offsets[a]+1:offsets[a+1]]) for a in eachindex(spaces)]
    gradients=[spaces[a].derivative*logarithms[offsets[a]+1:offsets[a+1]]/masses[a] for a in eachindex(spaces)]
    forces=[zero(value) for value in values]
    for pair in pairs
        a,b=pair.a,pair.b
        mobility=pair.geometry .* (values[a]*transpose(values[b]))
        flux=mobility .* (transpose(gradients[b]).-gradients[a])
        forces[a].+=vec(sum(flux;dims=2))/masses[a]
        if a!=b
            forces[b].-=vec(sum(flux;dims=1))/masses[b]
        end
    end
    reduce(vcat,[transpose(spaces[a].derivative)*forces[a] for a in eachindex(spaces)])
end

function implicit_midpoint(old,spaces,masses,pairs,offsets,step;tolerance=1e-12,maxiter=40,backward_base=nothing,log_state=nothing)
    previous=reduce(vcat,old)
    weights=reduce(vcat,[space.weights for space in spaces])
    particles=weights.*previous
    scale=max.(particles,maximum(particles)*1e-12)
    y=log_state===nothing ? log.(previous) : reduce(vcat,log_state)
    old_log=copy(y)
    function residual(candidate)
        updated=exp.(candidate)
        stage_log=backward_base===nothing ? max.(old_log,candidate).+log1p.(exp.(-abs.(old_log.-candidate))).-log(2) : candidate
        base=backward_base===nothing ? previous : backward_base
        (weights.*(updated.-base).-step.*source(stage_log,spaces,masses,pairs,offsets))./scale
    end
    history=Float64[]
    for iteration in 1:maxiter
        defect=residual(y)
        norm_residual=maximum(abs,defect)
        relative_L1_residual=sum(abs.(defect).*scale)/sum(particles)
        push!(history,norm_residual)
        if relative_L1_residual<tolerance
            return (;values=[exp.(y[offsets[a]+1:offsets[a+1]]) for a in eachindex(spaces)],logvalues=[copy(y[offsets[a]+1:offsets[a+1]]) for a in eachindex(spaces)],residual=relative_L1_residual,maximum_scaled_residual=norm_residual,iterations=iteration)
        end
        factor=svd(ForwardDiff.jacobian(residual,y))
        inverse_values=[value>maximum(factor.S)*1e-14 ? inv(value) : 0.0 for value in factor.S]
        direction=clamp.(-(factor.V*(inverse_values.*(transpose(factor.U)*defect))),-5.0,5.0)
        accepted=false
        for power in 0:20
            trial=y+2.0^(-power)*direction
            all(isfinite,trial) && maximum(abs.(trial.-y))<=20 || continue
            candidate=residual(trial)
            if all(isfinite,candidate) && sum(abs2,candidate.*scale)<sum(abs2,defect.*scale)
                y=trial;accepted=true;break
            end
        end
        accepted || error("Radial stage line search failed; particle L1 residual ",relative_L1_residual,"; scaled history ",history)
    end
    error("Radial midpoint iteration limit, residual history ",history)
end

function implicit_sdirk(old,spaces,masses,pairs,offsets,step;tolerance=1e-12,log_state=nothing)
    diagonal=1-inv(sqrt(2.0))
    previous=reduce(vcat,old)
    first=implicit_midpoint(old,spaces,masses,pairs,offsets,diagonal*step;tolerance,backward_base=previous,log_state)
    first_values=reduce(vcat,first.values)
    weights=reduce(vcat,[space.weights for space in spaces])
    second_base=previous+(1-diagonal)*step .* source(reduce(vcat,first.logvalues),spaces,masses,pairs,offsets)./weights
    second=implicit_midpoint(first.values,spaces,masses,pairs,offsets,diagonal*step;tolerance,backward_base=second_base,log_state=first.logvalues)
    (;values=second.values,logvalues=second.logvalues,residual=max(first.residual,second.residual),maximum_scaled_residual=max(first.maximum_scaled_residual,second.maximum_scaled_residual),iterations=max(first.iterations,second.iterations))
end

function log_density(squared_speed,space,logvalues)
    nodes=space.squared_speed
    index=clamp(searchsortedlast(nodes,squared_speed),1,length(nodes)-1)
    slope=(logvalues[index+1]-logvalues[index])/(nodes[index+1]-nodes[index])
    squared_speed>last(nodes) && slope>=0 && error("Radial reconstruction tail does not decay")
    logvalues[index]+slope*(squared_speed-nodes[index])
end

end
