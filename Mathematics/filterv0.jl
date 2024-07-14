
"""
  Filter for ddfL (such as dfL/v, ddGL, dGL/v) when v → 0.

  Outputs:
    ddHvL = filterddfL!(df1,va,L1;ratio_df=ratio_df,vcfilter=vcfilter)
    dHvL = filterdfL!(dHvL,va,L1;ratio_df=ratio_df,vcfilter=vcfilter)
    df2 = filterdfL!(df1,vremesh,L1;ratio_df=1.2,vcfilter=-1)

"""

function filterdfL!(dHvL::AbstractVector,va::AbstractVector,L1::Int;
                ratio_df::Real=1.2,vcfilter::Real=-1,k::Int=2,s::Float64=0.0)

    n0v = log.(va) .< vcfilter
    n1 = length(va[n0v])
    if L1 == 1
        if k < 0
            HLL = dHvL[1:n1]./va[1:n1]
            df = derivationCDS(HLL,va[1:n1])
        else
            HLL = dHvL[1:n1]./va[1:n1]
            spl = Spline1D(va[1:n1],HLL;s=s,k=k)
            df = derivative(spl,va[1:n1],1)
        end
        ######## HLL for smoothing dH when v̂ → 0
        nc0 = n1 - 1
        for i in n1-1:-1:1
            if abs(df[i]) < ratio_df * abs(df[i+1])
                nc0 = i
            else
                break
            end
        end
        HLL[1:nc0] .= HLL[nc0+1]
        dHvL[1:nc0] = HLL[1:nc0] .* va[1:nc0]
    else
        if L1 == 2
            if k < 0
                df = derivationCDS(dHvL[1:n1],va[1:n1])
            else
                spl = Spline1D(va[1:n1],dHvL[1:n1];s=s,k=k)
                df = derivative(spl,va[1:n1],1)
            end
        else
            if k < 0
                HLL = dHvL[1:n1]./va[1:n1].^(L1-2)
                df = derivationCDS(HLL,va[1:n1])
            else
                HLL = dHvL[1:n1]./va[1:n1].^(L1-2)
                spl = Spline1D(va[1:n1],HLL;s=s,k=k)
                df = derivative(spl,va[1:n1],1)
            end
        end
        ######## HLL for smoothing dH when v̂ → 0
        nc0 = n1 - 1
        for i in n1-1:-1:1
            if abs(df[i]) < ratio_df * abs(df[i+1])
                nc0 = i
            else
                break
            end
        end
        if L1 == 2
            dHvL[1:nc0] .= dHvL[nc0+1]
        else
            HLL[1:nc0] .= HLL[nc0+1]
            dHvL[1:nc0] = HLL[1:nc0] .* va[1:nc0].^(L1-2)
        end
    end
    return dHvL
end

function filterddfL!(ddHvL::AbstractVector,va::AbstractVector,L1::Int;
                ratio_df::Real=1.2,vcfilter::Real=-1,k::Int=2,s::Float64=0.0)

    n0v = log.(va) .< vcfilter
    n1 = length(va[n0v])
    if L1 == 1
        if k < 0
            df = derivationCDS(ddHvL[1:n1],va[1:n1])
        else
            spl = Spline1D(va[1:n1],ddHvL[1:n1];s=s,k=k)
            df = derivative(spl,va[1:n1],1)
        end
        ######## HLL for smoothing ddH when v̂ → 0
        nc0 = n1 - 1
        for i in n1-1:-1:1
            if abs(df[i]) < ratio_df * abs(df[i+1])
                nc0 = i
            else
                break
            end
        end
        ddHvL[1:nc0] .= ddHvL[1 + nc0]
    else
        #####################################  ddH
        if L1 == 3
            if k < 0
                df = derivationCDS(ddHvL[1:n1],va[1:n1])
            else
                spl = Spline1D(va[1:n1],ddHvL[1:n1];s=s,k=k)
                df = derivative(spl,va[1:n1],1)
            end
        elseif L1 == 2
            if k < 0
                HLL = ddHvL[1:n1]./va[1:n1]
                df = derivationCDS(HLL,va[1:n1])
            else
                HLL = ddHvL[1:n1]./va[1:n1]
                spl = Spline1D(va[1:n1],HLL;s=s,k=k)
                df = derivative(spl,va[1:n1],1)
            end
        else
            if k < 0
                HLL = ddHvL[1:n1]./va[1:n1].^(L1-3)
                df = derivationCDS(HLL,va[1:n1])
            else
                HLL = ddHvL[1:n1]./va[1:n1].^(L1-3)
                spl = Spline1D(va[1:n1],HLL;s=s,k=k)
                df = derivative(spl,va[1:n1],1)
            end
        end
        ########################## ddH when v̂ → 0
        nc0 = n1 - 1
        for i in n1-1:-1:1
            if abs(df[i]) < ratio_df * abs(df[i+1])
                nc0 = i
            else
                break
            end
        end
        if L1 == 2
            HLL[1:nc0] .= HLL[nc0+1]
            ddHvL[1:nc0] = HLL[1:nc0] .* va[1:nc0]
        elseif L1 == 3
            ddHvL[1:nc0] .= ddHvL[1+nc0]
        else
            HLL[1:nc0] .= HLL[nc0+1]
            ddHvL[1:nc0] = HLL[1:nc0] .* va[1:nc0].^(L1-3)
        end
    end
    return ddHvL
end

function filterddfL(ddHvL::AbstractVector,va::AbstractVector,L1::Int;
                ratio_df::Real=1.2,vcfilter::Real=-1,k::Int=2,s::Float64=0.0)

    n0v = log.(va) .< vcfilter
    n1 = length(va[n0v])
    if L1 == 1
        if k < 0
            df = derivationCDS(ddHvL[1:n1],va[1:n1])
        else
            spl = Spline1D(va[1:n1],ddHvL[1:n1];s=s,k=k)
            df = derivative(spl,va[1:n1],1)
        end
        ######## HLL for smoothing ddH when v̂ → 0
        nc0 = n1 - 1
        for i in n1-1:-1:1
            if abs(df[i]) < ratio_df * abs(df[i+1])
                nc0 = i
            else
                break
            end
        end
        ddHvL[1:nc0] .= ddHvL[1 + nc0]
    else
        #####################################  ddH
        if L1 == 3
            if k < 0
                df = derivationCDS(ddHvL[1:n1],va[1:n1])
            else
                spl = Spline1D(va[1:n1],ddHvL[1:n1];s=s,k=k)
                df = derivative(spl,va[1:n1],1)
            end
        elseif L1 == 2
            if k < 0
                HLL = ddHvL[1:n1]./va[1:n1]
                df = derivationCDS(HLL,va[1:n1])
            else
                HLL = ddHvL[1:n1]./va[1:n1]
                spl = Spline1D(va[1:n1],HLL;s=s,k=k)
                df = derivative(spl,va[1:n1],1)
            end
        else
            if k < 0
                HLL = ddHvL[1:n1]./va[1:n1].^(L1-3)
                df = derivationCDS(HLL,va[1:n1])
            else
                HLL = ddHvL[1:n1]./va[1:n1].^(L1-3)
                spl = Spline1D(va[1:n1],HLL;s=s,k=k)
                df = derivative(spl,va[1:n1],1)
            end
        end
        ########################## ddH when v̂ → 0
        nc0 = n1 - 1
        for i in n1-1:-1:1
            if abs(df[i]) < ratio_df * abs(df[i+1])
                nc0 = i
            else
                break
            end
        end
        if L1 == 2
            HLL[1:nc0] .= HLL[nc0+1]
            ddHvL[1:nc0] = HLL[1:nc0] .* va[1:nc0]
        elseif L1 == 3
            ddHvL[1:nc0] .= ddHvL[1+nc0]
        else
            HLL[1:nc0] .= HLL[nc0+1]
            ddHvL[1:nc0] = HLL[1:nc0] .* va[1:nc0].^(L1-3)
        end
    end
    return ddHvL
end
