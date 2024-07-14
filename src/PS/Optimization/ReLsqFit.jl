
"""
  Based on the modular `LsqFit.jl`, a restarted version is
  carried out to find the best global solution efficiently.


"""
# restartfit::Vector{Int}=[0,10]
function curve_fit_restart(model,xdata::AbstractArray, ydata::AbstractArray,
    p::AbstractArray,restartfit::Vector{Int};inplace=false,maxIter=1000,kwargs...,)

    if restartfit[1] > 0
        Numres = round(Int,maxIter/restartfit[2])
        if Numres ≤ 1
            return curve_fit(model,xdata,ydata,p;inplace=inplace,maxIter=maxIter,kwargs...,)
        else
            count = 0
            poly = curve_fit(model,xdata,ydata,p;inplace=inplace,maxIter=restartfit[2],kwargs...,)
            p0 = poly.param         # the array of best model1 parameters
            presid = poly.resid
            # To estimate errors on the fit parameters.
            sigma = stderror(poly)  # Standard error (stderror) of each parameter by estimating errors on the fit parameters.
            resids = max(maximum(abs.(presid)),maximum((sigma)))
            resids < eps0 ? isfalse = false : isfalse = true
            resids_up = deepcopy(resids)
            Dresids = 1.0
            Relresids = 1.0
            count = 1
            show_trace == 1 ? (@show Numres,count,isfalse,fmtf2.(p0)) : nothing
            while isfalse
                poly = curve_fit(model,xdata,ydata,p0::AbstractArray;inplace=inplace,maxIter=restartfit[2],kwargs...,)
                p0 = poly.param         # the array of best model1 parameters
                presid = poly.resid
                sigma = stderror(poly)
                resids = max(maximum(abs.(presid)),maximum((sigma)))
                # @show fmtf2.([resids_up,resids])
                if resids < eps0
                    isfalse = false
                else
                    isfalse = true
                    Dresids = abs(resids - resids_up)
                    # @show Dresids, Dresids / (resids + resids_up)
                    if Dresids < eps0 || Dresids / (resids + resids_up) < eps0
                        isfalse = false
                    else
                        isfalse = true
                    end
                end
                count += 1
                show_trace == 1 ? (@show Numres,count,isfalse,fmtf2.(p0)) : nothing
                resids_up = deepcopy(resids)
                count > Numres ? break : nothing
            end
            return poly
        end
    else
        @show maxIter
        return curve_fit(model,xdata,ydata,p;inplace=inplace,maxIter=maxIter,kwargs...,)
    end
end
