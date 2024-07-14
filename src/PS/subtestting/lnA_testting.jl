
# In pratical system of units (PS)
meps, mpps, mDps, mTps, mAps = me/Dₐ, 1.0, 2.0,  3.0,  4.0       # [Dₐ]
Ze,   Zp,   ZD,   ZT,   ZA   = -1,    1,   1,    1,    2         # [e]
neps, npps, nDps, nTps, nAps = 1.0,   1.0, 1.0,  1.0,  0.1       # [n20]
Teps, Tpps, TDps, TTps, TAps = 10.0, 10.0, 15.0, 15.0, 1000.0     # [keV]
TDps50, TTps50, TDps100, TTps100 = 50.0, 50.0, 100.0, 100.0

# e - e

lnAg_ee(n,T) = lnAab_fM_Tk([meps ,meps],[Ze,Ze],n,T,[:e,:e])
lnAg_eD(n,T) = lnAab_fM_Tk([meps ,mDps],[Ze,ZD],n,T,[:e,:D])
lnAg_eT(n,T) = lnAab_fM_Tk([meps ,mTps],[Ze,ZT],n,T,[:e,:T])
lnAg_eA(n,T) = lnAab_fM_Tk([meps ,mAps],[Ze,ZA],n,T,[:e,:A])
lnAg_eAA(n,T) = lnAab_fM_Tk([meps ,mAps],[ZD,ZA],n,T,[:A,:A])

lnAg_DD(n,T) = lnAab_fM_Tk([mDps ,mDps],[ZD,ZD],n,T,[:D,:D])
lnAg_DT(n,T) = lnAab_fM_Tk([mDps ,mTps],[ZD,ZT],n,T,[:D,:T])
lnAg_DA(n,T) = lnAab_fM_Tk([mDps ,mAps],[ZD,ZA],n,T,[:D,:A])

lnAg_TT(n,T) = lnAab_fM_Tk([mTps ,mTps],[ZT,ZT],n,T,[:T,:T])
lnAg_TA(n,T) = lnAab_fM_Tk([mTps ,mAps],[ZT,ZA],n,T,[:T,:A])

lnAg_AA(n,T) = lnAab_fM_Tk([mAps ,mAps],[ZA,ZA],n,T,[:A,:A])

lnAg_ee([neps,neps],[Teps,Teps])      # = 16.75
lnAg_eD([neps,nDps],[Teps,TDps])      # = 17.09
lnAg_eT([neps,nTps],[Teps,TTps])      # = 17.09
lnAg_eT([neps,nTps],[Teps,TTps])      # = 17.09
lnAg_eA([neps,nAps],[Teps,TAps])      # = ? 
lnAg_eA([neps,nAps],[0.1,2TAps])      # = 20.2
lnAg_eAA([neps,nAps],[Teps,TAps])      # = 20.2

lnAg_DD([nDps,nDps],[TDps,TDps])      # = 20.96
lnAg_DT([nDps,nTps],[TDps,TTps])      # = 20.96
lnAg_DA([nDps,nAps],[TDps,TAps])      # = 23.74
lnAg_DD([nDps,nDps],[TDps50,TDps50])  # = 22.76
lnAg_DT([nDps,nTps],[TDps50,TTps50])  # = 22.76
lnAg_DA([nDps,nAps],[TDps50,TAps])    # = 24.40

lnAg_TT([nTps,nTps],[TTps,TTps])      # = 20.96
lnAg_TA([nTps,nAps],[TTps,TAps])      # = 23.98

lnAg_AA([nAps,nAps],[TAps,TAps])      # = 26.33


lnAg_eA_A(TAps) = lnAg_eA([neps,nAps],[Teps,TAps]) 
lnAg_eA_e(Teps) = lnAg_eA([neps,nAps],[Teps,TAps]) 

TAps100 = [10,20,50,100,200,500,1000]
lnAg_eA_A.(TAps100)

Teps1 = [0.02,0.05,0.1,0.2,0.5]
lnAg_eA_e.(Teps1)
