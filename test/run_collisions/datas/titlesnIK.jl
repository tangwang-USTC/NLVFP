

title_spices = string(spices0)

# titleTEk
if norm(ua) == 0.0
    title = string("T0=", T0, ",nMod=", nMod)
else
    title = string("T0=", T0, ",Ek0=", (Ek0 .* μEk), ",nMod=", nMod)
end

if norm(ua) == 0.0
    title_TEk = string("T0=", T0)
else
    title_TEk = string("T0=", T0, ",Ek0=", (Ek0 .* μEk))
end

title_nMod = string("nMod=",nMod0)

title_nv = string("nv=",(nnv0,ocp0,NKmax))
title_nv_nMod = string("nv=",(nnv0,ocp0,NKmax),",","nMod=",nMod0)

