
if is_fvL_CP
    if is_lnA_const
        if is_enforce_errdtnIKab
            title_model = string("IK,lnA1C1,t[τ₀]")
            title_model_alg = string("IK,lnA1C1",",",algname,",",gridv_name)
        else
            title_model = string("IK,lnA1C0,t[τ₀]")
            title_model_alg = string("IK,lnA1C0",",",algname,",",gridv_name)
        end
    else
        if is_enforce_errdtnIKab
            title_model = string("IK,lnA0C1,t[τ₀]")
            title_model_alg = string("IK,lnA0C1",",",algname,",",gridv_name)
        else
            title_model = string("IK,lnA0C0,t[τ₀]")
            title_model_alg = string("IK,lnA0C0",",",algname,",",gridv_name)
        end
    end
else
    if is_lnA_const
        if is_enforce_errdtnIKab
            title_model = string("Ms,lnA1C1,t[τ₀]")
            title_model_alg = string("Ms,lnA1C1",",",algname,",",gridv_name)
        else
            title_model = string("Ms,lnA1C0,t[τ₀]")
            title_model_alg = string("Ms,lnA1C0",",",algname,",",gridv_name)
        end
    else
        if is_enforce_errdtnIKab
            title_model = string("Ms,lnA0C1,t[τ₀]")
            title_model_alg = string("Ms,lnA0C1",",",algname,",",gridv_name)
        else
            title_model = string("Ms,lnA0C0,t[τ₀]")
            title_model_alg = string("Ms,lnA0C0",",",algname,",",gridv_name)
        end
    end
end
title_alg = string(algname,",",gridv_name)