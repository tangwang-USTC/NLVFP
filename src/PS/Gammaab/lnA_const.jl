function lnAeA()

    return 20.02
end

# Cab
function lnA_const(spices::Vector{Symbol})

    if spices[1] == :e && spices[2] == :e
        return 16.75
    else
        if spices[1] == :e || spices[2] == :e
            if spices[1] == :e
                k = 2
                if spices[k] == :D || spices[2] == :Tri
                    return 17.09
                elseif spices[k] == :α
                    return lnAeA()
                else
                    dcfvb
                end
            else
                k = 1
                if spices[k] == :D || spices[k] == :Tri
                    return 17.09
                elseif spices[k] == :α
                    return 20.0
                else
                    dcfvb
                end
            end
        else
            if spices[1] == :D || spices[2] == :D
                if spices[1] == :D
                    if spices[2] == :α
                        return 23.74
                    else
                        return 20.96
                    end
                else
                    if spices[1] == :α
                        return 23.74
                    else
                        return 20.96
                    end
                end
            else
                if spices[1] == :Tri || spices[2] == :Tri
                    if spices[1] == :Tri
                        if spices[2] == :α
                            return 23.74
                        else
                            return 20.96
                        end
                    else
                        if spices[1] == :α
                            return 23.74
                        else
                            return 20.96
                        end
                    end
                else
                    return 26.33
                end
            end
        end
    end
end

# Caa
function lnA_const(spices::Symbol)

    if spices == :e
        return 16.75
    else
        if spices == :α
            return 26.33
        elseif spices == :D || spices == :Tri
            return 20.96
        else
            wsedfg
        end
    end
end

function lnA_const()
    
    return 17.0
end
